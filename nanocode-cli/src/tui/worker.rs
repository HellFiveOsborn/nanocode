use nanocode_core::agent_loop::{AgentLoop, LoopEvent};
use nanocode_core::agents::AgentPolicy;
use nanocode_core::llm::PromptFamily;
use nanocode_core::prompts::load_prompt;
use nanocode_core::tools::ToolManager;
use nanocode_core::{AgentStats, ApprovalDecision};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, SyncSender};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

use super::runtime::{is_thinking_model, RuntimeEnv};
use super::stream::{extract_thinking_blocks_and_clean, StreamSanitizer};

const THINKING_PROMPT_BOOSTER: &str = r#"
## Thinking Discipline
- Keep reasoning private and brief.
- Avoid meta narration ("hmm", "wait", "I should").
- If a tool is needed, call it quickly.
"#;

pub enum WorkerCommand {
    Submit(String),
    Shutdown,
}

pub enum WorkerEvent {
    Ready {
        model_label: String,
    },
    Busy(bool),
    Interrupted,
    ThinkingActive(bool),
    ThinkingDelta(String),
    AssistantDone(String),
    ToolCall {
        call_id: String,
        summary: String,
    },
    ApprovalRequired {
        call_id: String,
        summary: String,
        arguments: String,
        decision_tx: SyncSender<ApprovalDecision>,
    },
    ToolResult {
        call_id: String,
        success: bool,
        status_line: Option<String>,
        result: String,
    },
    CompactStart {
        old_context_tokens: u32,
        threshold: u32,
    },
    CompactEnd {
        old_context_tokens: u32,
        new_context_tokens: u32,
        summary_len: usize,
    },
    StoppedByMiddleware {
        reason: String,
    },
    Stats(AgentStats),
    Error(String),
}

fn short_preview(text: &str, n: usize) -> String {
    let mut out = String::new();
    for ch in text.chars().take(n) {
        out.push(ch);
    }
    if text.chars().count() > n {
        out.push('…');
    }
    out
}

fn short_call_id(id: &str) -> String {
    if id.len() > 8 {
        id[..8].to_string()
    } else {
        id.to_string()
    }
}

fn pretty_tool_name(name: &str) -> String {
    match name {
        "write_file" => "Write".to_string(),
        "read_file" => "Read".to_string(),
        "bash" => "Bash".to_string(),
        other => {
            let mut out = String::new();
            let mut chars = other.chars();
            if let Some(first) = chars.next() {
                out.extend(first.to_uppercase());
                out.extend(chars);
            }
            out
        }
    }
}

fn tool_target(call_name: &str, arguments: &Value) -> String {
    if call_name == "bash" {
        arguments
            .get("command")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    } else {
        arguments
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    }
}

fn tool_status_line(
    call_id: &str,
    tool_name: Option<&str>,
    elapsed_s: f32,
    success: bool,
    result: &str,
) -> Option<String> {
    let preview_raw = result.lines().next().unwrap_or("").trim();
    let preview = short_preview(preview_raw, 90);

    if success {
        match tool_name {
            Some("read_file") => Some(format!(
                "Read {} Lines ({:.2}s)",
                result.lines().count(),
                elapsed_s
            )),
            Some("write_file") => None,
            Some("bash") => {
                if preview.is_empty() {
                    Some(format!("ok {} ({:.2}s)", short_call_id(call_id), elapsed_s))
                } else {
                    Some(format!(
                        "ok {} ({:.2}s) {}",
                        short_call_id(call_id),
                        elapsed_s,
                        preview
                    ))
                }
            }
            _ => {
                if preview.is_empty() {
                    Some(format!("ok {} ({:.2}s)", short_call_id(call_id), elapsed_s))
                } else {
                    Some(format!(
                        "ok {} ({:.2}s) {}",
                        short_call_id(call_id),
                        elapsed_s,
                        preview
                    ))
                }
            }
        }
    } else if preview.is_empty() {
        Some(format!(
            "erro {} ({:.2}s)",
            short_call_id(call_id),
            elapsed_s
        ))
    } else {
        Some(format!(
            "erro {} ({:.2}s) {}",
            short_call_id(call_id),
            elapsed_s,
            preview
        ))
    }
}

pub fn spawn_worker(
    runtime: RuntimeEnv,
    agent_policy: AgentPolicy,
    mut cmd_rx: mpsc::Receiver<WorkerCommand>,
    interrupt_signal: Arc<AtomicBool>,
) -> Receiver<WorkerEvent> {
    let (evt_tx, evt_rx) = channel::<WorkerEvent>();

    tokio::spawn(async move {
        let mut effective_config = runtime.config.clone();
        if agent_policy.auto_approve {
            effective_config.auto_approve = true;
        }
        let tool_manager = ToolManager::new(&effective_config);
        tool_manager.set_enabled_tools(agent_policy.enabled_tools.clone());
        for (tool_name, permission) in &agent_policy.tool_permission_overrides {
            let _ = tool_manager.set_permission(tool_name, *permission);
        }
        let mut loop_engine = AgentLoop::new(effective_config.clone(), tool_manager);
        loop_engine.set_agent_name(agent_policy.builtin.as_str());
        loop_engine.set_approval_handler({
            let evt_tx = evt_tx.clone();
            move |request| {
                let target =
                    short_preview(&tool_target(&request.tool_name, &request.arguments), 90);
                let display_name = pretty_tool_name(&request.tool_name);
                let summary = if target.is_empty() {
                    format!("{}()", display_name)
                } else {
                    format!("{}({})", display_name, target)
                };
                let arguments = serde_json::to_string_pretty(&request.arguments)
                    .unwrap_or_else(|_| request.arguments.to_string());
                let (decision_tx, decision_rx) = std::sync::mpsc::sync_channel(1);
                if evt_tx
                    .send(WorkerEvent::ApprovalRequired {
                        call_id: request.tool_call_id,
                        summary,
                        arguments,
                        decision_tx,
                    })
                    .is_err()
                {
                    return ApprovalDecision::Deny;
                }

                decision_rx.recv().unwrap_or(ApprovalDecision::Deny)
            }
        });

        let mut system_prompt = load_prompt(PromptFamily::Qwen3, agent_policy.prompt_variant);
        if is_thinking_model(runtime.model.display_name, runtime.quant.name) {
            system_prompt = format!("{}\n\n{}", system_prompt, THINKING_PROMPT_BOOSTER);
        }
        loop_engine.add_system_message(system_prompt);

        let _ = evt_tx.send(WorkerEvent::Ready {
            model_label: runtime.model_label.clone(),
        });

        while let Some(cmd) = cmd_rx.recv().await {
            match cmd {
                WorkerCommand::Shutdown => break,
                WorkerCommand::Submit(prompt) => {
                    interrupt_signal.store(false, Ordering::Relaxed);
                    loop_engine.add_user_message(prompt);
                    let _ = evt_tx.send(WorkerEvent::Busy(true));

                    let mut stream_sanitizer = StreamSanitizer::default();
                    let mut thinking_emitted_this_turn = false;
                    let mut thinking_active = false;
                    let mut tool_phase_started = false;
                    let mut pending_tool_calls: HashMap<String, (String, Instant)> = HashMap::new();

                    let run = loop_engine
                        .act_with_events_interruptable(
                            &runtime.model_file,
                            runtime.max_tokens,
                            Some(interrupt_signal.clone()),
                            |event| match event {
                                LoopEvent::Chunk(chunk) => {
                                    let was_thinking = stream_sanitizer.is_in_thinking();
                                    let parts = stream_sanitizer.push(&chunk, false);
                                    let is_thinking = stream_sanitizer.is_in_thinking();

                                    if !was_thinking && is_thinking {
                                        if !thinking_active {
                                            let _ = evt_tx.send(WorkerEvent::ThinkingActive(true));
                                            thinking_active = true;
                                        }
                                    }
                                    if !parts.thinking.is_empty() {
                                        thinking_emitted_this_turn = true;
                                        let _ =
                                            evt_tx.send(WorkerEvent::ThinkingDelta(parts.thinking));
                                    }
                                    if was_thinking && !is_thinking {
                                        if thinking_active {
                                            let _ = evt_tx.send(WorkerEvent::ThinkingActive(false));
                                            thinking_active = false;
                                        }
                                    }

                                    if !tool_phase_started && !parts.visible.is_empty() {
                                        if !thinking_active {
                                            let _ = evt_tx.send(WorkerEvent::ThinkingActive(true));
                                            thinking_active = true;
                                        }
                                        thinking_emitted_this_turn = true;
                                        let _ =
                                            evt_tx.send(WorkerEvent::ThinkingDelta(parts.visible));
                                    }
                                }
                                LoopEvent::ToolCall(call) => {
                                    tool_phase_started = true;
                                    let target = short_preview(
                                        &tool_target(&call.name, &call.arguments),
                                        90,
                                    );
                                    let display_name = pretty_tool_name(&call.name);
                                    let summary = if target.is_empty() {
                                        format!("{}()", display_name)
                                    } else {
                                        format!("{}({})", display_name, target)
                                    };

                                    pending_tool_calls.insert(
                                        call.id.clone(),
                                        (call.name.clone(), Instant::now()),
                                    );

                                    if thinking_active {
                                        let _ = evt_tx.send(WorkerEvent::ThinkingActive(false));
                                        thinking_active = false;
                                    }
                                    let _ = evt_tx.send(WorkerEvent::ToolCall {
                                        call_id: call.id,
                                        summary,
                                    });
                                }
                                LoopEvent::ToolResult { call_id, result } => {
                                    let call_meta = pending_tool_calls.remove(&call_id);
                                    let (tool_name, elapsed_s) = call_meta
                                        .as_ref()
                                        .map(|(name, started)| {
                                            (Some(name.as_str()), started.elapsed().as_secs_f32())
                                        })
                                        .unwrap_or((None, 0.0));

                                    let first = result.lines().next().unwrap_or("sem saída");
                                    let lower = first.to_ascii_lowercase();
                                    let success = !(lower.contains("failed")
                                        || lower.contains("error")
                                        || lower.contains("denied"));

                                    let _ = evt_tx.send(WorkerEvent::ToolResult {
                                        call_id: call_id.clone(),
                                        success,
                                        status_line: tool_status_line(
                                            &call_id, tool_name, elapsed_s, success, &result,
                                        ),
                                        result,
                                    });

                                    if pending_tool_calls.is_empty() {
                                        tool_phase_started = false;
                                    }
                                }
                                LoopEvent::Stats(stats) => {
                                    let _ = evt_tx.send(WorkerEvent::Stats(stats));
                                }
                                LoopEvent::CompactStart {
                                    old_context_tokens,
                                    threshold,
                                } => {
                                    let _ = evt_tx.send(WorkerEvent::CompactStart {
                                        old_context_tokens,
                                        threshold,
                                    });
                                }
                                LoopEvent::CompactEnd {
                                    old_context_tokens,
                                    new_context_tokens,
                                    summary_len,
                                } => {
                                    let _ = evt_tx.send(WorkerEvent::CompactEnd {
                                        old_context_tokens,
                                        new_context_tokens,
                                        summary_len,
                                    });
                                }
                                LoopEvent::StoppedByMiddleware { reason } => {
                                    let _ =
                                        evt_tx.send(WorkerEvent::StoppedByMiddleware { reason });
                                }
                                LoopEvent::Message(_) | LoopEvent::Error(_) => {}
                            },
                        )
                        .await;

                    let was_thinking = stream_sanitizer.is_in_thinking();
                    let final_parts = stream_sanitizer.push("", true);
                    let is_thinking = stream_sanitizer.is_in_thinking();

                    if !was_thinking && is_thinking {
                        if !thinking_active {
                            let _ = evt_tx.send(WorkerEvent::ThinkingActive(true));
                            thinking_active = true;
                        }
                    }
                    if !final_parts.thinking.is_empty() {
                        thinking_emitted_this_turn = true;
                        let _ = evt_tx.send(WorkerEvent::ThinkingDelta(final_parts.thinking));
                    }
                    if !tool_phase_started && !final_parts.visible.is_empty() {
                        if !thinking_active {
                            let _ = evt_tx.send(WorkerEvent::ThinkingActive(true));
                            thinking_active = true;
                        }
                        thinking_emitted_this_turn = true;
                        let _ = evt_tx.send(WorkerEvent::ThinkingDelta(final_parts.visible));
                    }
                    if was_thinking || (!was_thinking && is_thinking) || thinking_active {
                        let _ = evt_tx.send(WorkerEvent::ThinkingActive(false));
                    }

                    match run {
                        Ok(response) => {
                            let (thoughts, final_text) =
                                extract_thinking_blocks_and_clean(&response);
                            if !thinking_emitted_this_turn && !thoughts.is_empty() {
                                let _ = evt_tx.send(WorkerEvent::ThinkingActive(true));
                                let _ =
                                    evt_tx.send(WorkerEvent::ThinkingDelta(thoughts.join("\n\n")));
                                let _ = evt_tx.send(WorkerEvent::ThinkingActive(false));
                            }
                            let _ = evt_tx.send(WorkerEvent::AssistantDone(final_text));
                        }
                        Err(err) => {
                            if err.to_ascii_lowercase().contains("interrupted by user") {
                                let _ = evt_tx.send(WorkerEvent::Interrupted);
                            } else {
                                let _ = evt_tx.send(WorkerEvent::Error(err));
                            }
                        }
                    }

                    let _ = evt_tx.send(WorkerEvent::Busy(false));
                }
            }
        }
    });

    evt_rx
}

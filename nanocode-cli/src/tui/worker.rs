use nanocode_core::agent_loop::{AgentLoop, LoopEvent};
use nanocode_core::llm::{PromptFamily, PromptVariant};
use nanocode_core::prompts::load_prompt;
use nanocode_core::tools::ToolManager;
use nanocode_core::AgentStats;
use nanocode_hf::THE_MODEL;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

use super::runtime::{is_thinking_model, RuntimeEnv};
use super::stream::{extract_thinking_blocks_and_clean, StreamSanitizer};

const THINKING_PROMPT_BOOSTER: &str = r#"
## Thinking Mode: CONFIDENT ACTION

**Your Mindset:**
- Simple tasks deserve simple solutions
- 80% confidence is ENOUGH to act
- Perfect is the enemy of done
- Users prefer fast answers over perfect analysis

**Decision Rule:**
If you find yourself thinking for more than 30 seconds:
→ STOP
→ Pick the most obvious tool
→ CALL IT NOW
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
    ToolResult {
        call_id: String,
        success: bool,
        status_line: Option<String>,
        result: String,
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
    mut cmd_rx: mpsc::Receiver<WorkerCommand>,
    interrupt_signal: Arc<AtomicBool>,
) -> Receiver<WorkerEvent> {
    let (evt_tx, evt_rx) = channel::<WorkerEvent>();

    tokio::spawn(async move {
        let tool_manager = ToolManager::new(&runtime.config);
        let mut loop_engine = AgentLoop::new(runtime.config.clone(), tool_manager);

        let mut system_prompt = load_prompt(PromptFamily::Qwen3, PromptVariant::AgentDefault);
        if is_thinking_model(THE_MODEL.display_name, &runtime.model_label) {
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

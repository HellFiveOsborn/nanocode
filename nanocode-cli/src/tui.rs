use anyhow::{anyhow, Result};
use crossterm::event::{
    self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEventKind,
};
use nanocode_core::agents::{AgentPolicy, BuiltinAgent};
use nanocode_core::config::ToolPermissionConfig;
use nanocode_core::interrupt::{clear_interrupt_signal, set_interrupt_signal};
use nanocode_core::session::load_session_by_id_sync;
use nanocode_core::skills::SkillManager;
use nanocode_core::types::{LlmMessage, MessageRole};
use nanocode_core::{NcConfig, QuestionAnswerSource, UserQuestionResponse};
use nanocode_hf::{
    default_model, enforce_single_quant_cache, find_any_installed_model_quant, find_model,
    find_quant_by_name, list_cached_quants, model_quantizations, models,
    preload_dynamic_quantizations, recommend, recommend_runtime_limits, DownloadProgress,
    Downloader, HardwareInfo, ModelSpec,
};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

mod attachments;
mod clipboard;
mod commands;
mod render;
mod rewind;
mod runtime;
mod state;
mod stream;
mod terminal;
mod worker;

use attachments::{AttachmentEngine, MentionSuggestion};
use clipboard::{read_clipboard_payload, ClipboardPayload};
use commands::{
    apply_selected_slash_suggestion, handle_local_command, is_builtin_command, is_exit_command,
    refresh_slash_suggestions, resolve_slash_prompt_body, try_build_skill_prompt,
};
use render::draw_ui;
use rewind::RewindManager;
use runtime::{build_runtime, runtime_snapshot};
use state::{
    AgentSwitchRequest, AppState, ApprovalOption, ChatItem, ConfigField, ConfigFieldKey,
    ConfigScreenState, DownloadProgressView, InputMode, MentionSuggestionEntry, ModelChoice,
    ModelSetupState, ModelSetupView, PendingApproval, PendingImagePaste, PendingPlanReview,
    PendingTextPaste, PendingUserQuestion, PlanReviewOption, SessionResumeRequest, SetupChoice,
    SkillEntry, ToolState, UiScreen,
};
use terminal::{restore_terminal, setup_terminal};
use worker::{spawn_worker, WorkerCommand, WorkerEvent};

struct DownloadTask {
    model_id: String,
    quant_name: String,
    quant_filename: String,
    progress_rx: mpsc::Receiver<DownloadProgress>,
    handle: tokio::task::JoinHandle<anyhow::Result<PathBuf>>,
}

enum ModelSetupAction {
    None,
    ConfirmSelected,
    CancelDownload,
    Close,
    Quit,
}

enum ConfigScreenAction {
    None,
    CloseNoSave,
    SaveAndClose,
}

fn format_duration_short(d: Duration) -> String {
    let secs = d.as_secs();
    if secs >= 60 {
        format!("{}m {:02}s", secs / 60, secs % 60)
    } else {
        format!("{}s", secs)
    }
}

fn agent_status_label(agent: BuiltinAgent) -> &'static str {
    match agent {
        BuiltinAgent::Default => "Padrão",
        BuiltinAgent::Plan => "Plano",
        BuiltinAgent::Build => "Implementação",
        BuiltinAgent::Explore => "Explorar",
    }
}

fn should_offer_plan_review(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }

    let line_count = trimmed.lines().count();
    let long_enough = trimmed.chars().count() >= 220 || line_count >= 6;
    let has_plan_shape = trimmed.lines().any(|line| {
        let l = line.trim_start();
        l.starts_with('#')
            || l.starts_with("- [ ]")
            || l.starts_with("- [x]")
            || l.starts_with("- ")
            || l.starts_with("* ")
            || l.starts_with("1.")
            || l.starts_with("2.")
            || l.starts_with("|")
    });

    long_enough && has_plan_shape
}

fn build_plan_bootstrap_prompt(plan_text: &str) -> Option<String> {
    let trimmed = plan_text.trim();
    if trimmed.is_empty() {
        return None;
    }

    Some(format!(
        "Plano aprovado para execução no modo implementação.\n\
\n\
Use exclusivamente o plano abaixo como contexto inicial.\n\
\n\
{}\n\
\n\
Implemente este plano de forma sequencial.",
        trimmed
    ))
}

fn apply_worker_event(
    app: &mut AppState,
    evt: WorkerEvent,
    tool_index_by_id: &mut HashMap<String, usize>,
    rewind_manager: &mut RewindManager,
) {
    match evt {
        WorkerEvent::SessionReady {
            session_id,
            resumed,
        } => {
            app.current_session_id = Some(session_id.clone());
            if let Err(err) = rewind_manager.set_session(&session_id) {
                app.chat.push(ChatItem::Error(format!(
                    "Falha ao inicializar armazenamento de rewind: {err}"
                )));
            }
            if resumed {
                app.status = format!("sessão retomada: {}", short_session_id(&session_id));
            }
        }
        WorkerEvent::Ready {
            model_label,
            supports_thinking,
            supports_vision,
        } => {
            app.model_label = model_label;
            app.supports_thinking = supports_thinking;
            app.supports_vision = supports_vision;
            app.thinking_enabled = supports_thinking;
            // output_expanded stays as-is; thinking is controlled by output_expanded
            app.busy = false;
            app.busy_started_at = None;
            app.status = format!("pronto · agente {}", agent_status_label(app.active_agent));
        }
        WorkerEvent::Busy(v) => {
            app.busy = v;
            if v {
                app.status = "pensando".to_string();
                app.busy_started_at = Some(Instant::now());
                app.thinking_idx = None;
                app.stream_idx = None;
                app.pending_plan_review = None;
            } else {
                app.status = format!("pronto · agente {}", agent_status_label(app.active_agent));
                app.busy_started_at = None;
                app.stream_idx = None;
                app.pending_approval = None;
                app.pending_user_question = None;
                if let Some(idx) = app.thinking_idx {
                    if let Some(ChatItem::Thinking { active, .. }) = app.chat.get_mut(idx) {
                        *active = false;
                    }
                }
                app.thinking_idx = None;
            }
        }
        WorkerEvent::Interrupted => {
            app.busy = false;
            app.busy_started_at = None;
            if let Some(idx) = app.thinking_idx {
                if let Some(ChatItem::Thinking { active, .. }) = app.chat.get_mut(idx) {
                    *active = false;
                }
            }
            app.thinking_idx = None;
            app.stream_idx = None;
            app.status = "interrompido".to_string();
        }
        WorkerEvent::ThinkingActive(active) => {
            if active {
                let current_is_active = app
                    .thinking_idx
                    .and_then(|idx| app.chat.get(idx))
                    .and_then(|item| match item {
                        ChatItem::Thinking { active, .. } => Some(*active),
                        _ => None,
                    })
                    .unwrap_or(false);

                if !current_is_active {
                    app.chat.push(ChatItem::Thinking {
                        text: String::new(),
                        active: true,
                    });
                    app.thinking_idx = Some(app.chat.len() - 1);
                }
            } else if let Some(idx) = app.thinking_idx {
                if let Some(ChatItem::Thinking { active, .. }) = app.chat.get_mut(idx) {
                    *active = false;
                }
            }
        }
        WorkerEvent::ThinkingDelta(delta) => {
            let needs_new_block = app
                .thinking_idx
                .and_then(|idx| app.chat.get(idx))
                .map(|item| !matches!(item, ChatItem::Thinking { active: true, .. }))
                .unwrap_or(true);

            if needs_new_block {
                app.chat.push(ChatItem::Thinking {
                    text: String::new(),
                    active: true,
                });
                app.thinking_idx = Some(app.chat.len() - 1);
            }
            if let Some(idx) = app.thinking_idx {
                if let Some(ChatItem::Thinking { text, .. }) = app.chat.get_mut(idx) {
                    text.push_str(&delta);
                }
            }
        }
        WorkerEvent::AssistantChunk(chunk) => {
            if app.stream_idx.is_none() {
                // Close any active thinking block before starting assistant stream
                if let Some(idx) = app.thinking_idx {
                    if let Some(ChatItem::Thinking { active, .. }) = app.chat.get_mut(idx) {
                        *active = false;
                    }
                }
                app.chat.push(ChatItem::Assistant(String::new()));
                app.stream_idx = Some(app.chat.len() - 1);
            }
            if let Some(idx) = app.stream_idx {
                if let Some(ChatItem::Assistant(text)) = app.chat.get_mut(idx) {
                    text.push_str(&chunk);
                }
            }
        }
        WorkerEvent::AssistantDone(final_text) => {
            let has_text = !final_text.trim().is_empty();
            let should_offer_review = app.active_agent == BuiltinAgent::Plan
                && app.pending_approval.is_none()
                && should_offer_plan_review(&final_text);

            if let Some(idx) = app.stream_idx {
                if let Some(ChatItem::Assistant(text)) = app.chat.get_mut(idx) {
                    if has_text {
                        *text = final_text.clone();
                    }
                }
            } else if has_text {
                app.chat.push(ChatItem::Assistant(final_text.clone()));
            }

            // Clear busy eagerly so spinner stops as soon as the response is complete.
            // The subsequent Busy(false) from the worker will be a harmless no-op.
            app.busy = false;
            app.busy_started_at = None;
            if let Some(idx) = app.thinking_idx {
                if let Some(ChatItem::Thinking { active, .. }) = app.chat.get_mut(idx) {
                    *active = false;
                }
            }
            app.thinking_idx = None;
            app.stream_idx = None;
            app.status = format!("pronto · agente {}", agent_status_label(app.active_agent));

            if should_offer_review {
                app.pending_plan_review = Some(PendingPlanReview {
                    selected_option: PlanReviewOption::ApproveAndBuild,
                    plan_text: final_text,
                });
                app.status = "Plano pronto: aprove, reprove ou peça ajustes.".to_string();
            }
        }
        WorkerEvent::ToolCall {
            call_id,
            tool_name,
            summary,
        } => {
            if let Some(idx) = app.thinking_idx {
                if let Some(ChatItem::Thinking { active, .. }) = app.chat.get_mut(idx) {
                    *active = false;
                }
            }
            app.thinking_idx = None;
            app.stream_idx = None;
            let subagent = if tool_name == "task" {
                Some(state::SubagentTracking {
                    started_at: Some(Instant::now()),
                    ..Default::default()
                })
            } else {
                None
            };
            let tool_started_at = Instant::now();
            if tool_name == "bash" {
                app.running_bash_call_ids.push((
                    call_id.clone(),
                    summary.clone(),
                    tool_started_at,
                ));
            }
            app.chat.push(ChatItem::Tool {
                tool_name,
                summary,
                stream: Some("executando".to_string()),
                output: None,
                code_path: None,
                code: None,
                diff: None,
                state: ToolState::Running,
                subagent,
                started_at: Some(tool_started_at),
            });
            tool_index_by_id.insert(call_id, app.chat.len() - 1);
        }
        WorkerEvent::ApprovalRequired {
            call_id,
            summary,
            details,
            diff_preview,
            decision_tx,
        } => {
            if let Some(idx) = tool_index_by_id.get(&call_id).copied() {
                if let Some(ChatItem::Tool { stream, .. }) = app.chat.get_mut(idx) {
                    *stream = Some("Aguardando aprovação".to_string());
                }
            }
            app.status = "Aguardando aprovação".to_string();
            app.pending_approval = Some(PendingApproval {
                summary,
                details,
                diff_preview,
                decision_tx,
                selected_option: ApprovalOption::ApproveOnce,
            });
        }
        WorkerEvent::QuestionRequired {
            call_id,
            question,
            choices,
            allow_free_text,
            placeholder,
            response_tx,
        } => {
            if let Some(idx) = tool_index_by_id.get(&call_id).copied() {
                if let Some(ChatItem::Tool { stream, .. }) = app.chat.get_mut(idx) {
                    *stream = Some("Aguardando resposta do usuário".to_string());
                }
            }
            app.status = "Aguardando resposta do usuário".to_string();
            app.pending_user_question = Some(PendingUserQuestion {
                question,
                choices,
                allow_free_text,
                placeholder,
                selected_choice: 0,
                text_input: String::new(),
                response_tx,
            });
        }
        WorkerEvent::ToolResult {
            call_id,
            success,
            status_line,
            output,
            code_path,
            code,
            diff,
            rewind_changes,
        } => {
            app.running_bash_call_ids
                .retain(|(id, _, _)| id != &call_id);
            if app.running_bash_call_ids.is_empty() && app.process_viewer_open {
                app.process_viewer_open = false;
            }
            if let Some(idx) = tool_index_by_id.get(&call_id).copied() {
                if let Some(ChatItem::Tool {
                    tool_name,
                    stream,
                    output: existing_output,
                    code_path: existing_path,
                    code: existing_code,
                    diff: existing_diff,
                    state,
                    subagent,
                    ..
                }) = app.chat.get_mut(idx)
                {
                    *stream = status_line;
                    *existing_output = output;
                    *existing_path = code_path;
                    *existing_code = code;
                    *existing_diff = diff;
                    *state = if success {
                        ToolState::Ok
                    } else {
                        ToolState::Error
                    };
                    // Finalize subagent tracking with elapsed time
                    if tool_name == "task" {
                        if let Some(ref mut tracking) = subagent {
                            let elapsed =
                                tracking.started_at.map(|s| s.elapsed()).unwrap_or_default();
                            tracking.final_tools_called = Some(tracking.tools_done);
                            // Encode elapsed in stream for rendering
                            let tools = tracking.tools_done;
                            let elapsed_str = format_duration_short(elapsed);
                            if let Some(s) = stream {
                                *s = format!("{} · {}", s, elapsed_str);
                            }
                            let _ = (tools, elapsed_str);
                        }
                    }
                }
            }
            if !rewind_changes.is_empty() {
                if let Err(err) = rewind_manager.record_change_set(rewind_changes) {
                    app.chat.push(ChatItem::Error(format!(
                        "Falha ao salvar snapshot de rewind: {err}"
                    )));
                }
            }
        }
        WorkerEvent::CompactStart {
            old_context_tokens,
        } => {
            app.status = "compactando contexto automaticamente".to_string();
            app.chat.push(ChatItem::Compact {
                active: true,
                old_tokens: old_context_tokens,
                new_tokens: None,
            });
        }
        WorkerEvent::CompactEnd {
            old_context_tokens,
            new_context_tokens,
        } => {
            // Replace the active Compact item with the completed one
            if let Some(item) = app.chat.iter_mut().rev().find(|i| {
                matches!(i, ChatItem::Compact { active: true, .. })
            }) {
                *item = ChatItem::Compact {
                    active: false,
                    old_tokens: old_context_tokens,
                    new_tokens: Some(new_context_tokens),
                };
            }
            app.status = format!("pronto · agente {}", agent_status_label(app.active_agent));
        }
        WorkerEvent::StoppedByMiddleware { reason } => {
            app.status = "interrompido por middleware".to_string();
            app.chat.push(ChatItem::Error(format!(
                "Interrupção do middleware: {}",
                reason
            )));
        }
        WorkerEvent::SubagentToolCall {
            parent_call_id,
            summary,
        } => {
            if let Some(idx) = tool_index_by_id.get(&parent_call_id).copied() {
                if let Some(ChatItem::Tool {
                    subagent: Some(ref mut tracking),
                    stream,
                    ..
                }) = app.chat.get_mut(idx)
                {
                    tracking.sub_tools.push(state::SubToolEntry {
                        summary: summary.clone(),
                        done: false,
                    });
                    *stream = Some(format!("Executando…{}", summary));
                }
            }
        }
        WorkerEvent::SubagentToolResult { parent_call_id } => {
            if let Some(idx) = tool_index_by_id.get(&parent_call_id).copied() {
                if let Some(ChatItem::Tool {
                    subagent: Some(ref mut tracking),
                    ..
                }) = app.chat.get_mut(idx)
                {
                    tracking.tools_done += 1;
                    // Mark the latest sub-tool as done
                    if let Some(last) = tracking.sub_tools.last_mut() {
                        last.done = true;
                    }
                }
            }
        }
        WorkerEvent::Stats(stats) => {
            app.stats = stats;
        }
        WorkerEvent::Error(err) => {
            app.chat.push(ChatItem::Error(err));
        }
    }
}

fn refresh_skill_catalog(app: &mut AppState, config: &NcConfig) {
    let manager = SkillManager::new(config);
    app.skills = manager
        .available_skills()
        .into_iter()
        .map(|(name, info)| {
            (
                name.clone(),
                SkillEntry {
                    name,
                    description: info.description,
                    skill_path: info.skill_path.display().to_string(),
                    user_invocable: info.user_invocable,
                },
            )
        })
        .collect();
    app.skills_count = app.skills.len();
}

const PASTE_SUMMARY_MIN_LINES: usize = 8;
const PASTE_SUMMARY_MIN_CHARS: usize = 320;

fn clear_mention_suggestions(app: &mut AppState) {
    app.mention_suggestions.clear();
    app.mention_selected = 0;
    app.mention_replace_range = None;
}

fn refresh_input_suggestions(app: &mut AppState, attachment_engine: &mut AttachmentEngine) {
    refresh_slash_suggestions(app);
    if app.input_mode == InputMode::Slash {
        clear_mention_suggestions(app);
        return;
    }

    let completion = attachment_engine.complete_for_input(&app.input);
    let suggestions = completion
        .suggestions
        .into_iter()
        .map(|s: MentionSuggestion| MentionSuggestionEntry {
            replacement: s.replacement,
            display: s.display,
            description: s.description,
            is_directory: s.is_directory,
        })
        .collect::<Vec<_>>();

    app.mention_replace_range = completion.replace_range;
    app.mention_suggestions = suggestions;
    if app.mention_suggestions.is_empty() {
        app.mention_selected = 0;
    } else if app.mention_selected >= app.mention_suggestions.len() {
        app.mention_selected = 0;
    }
}

fn apply_selected_mention_suggestion(app: &mut AppState) -> bool {
    let Some((start, end)) = app.mention_replace_range else {
        return false;
    };
    let Some(selected) = app.mention_suggestions.get(app.mention_selected) else {
        return false;
    };
    if start >= end || end > app.input.len() {
        return false;
    }

    let mut new_input = String::with_capacity(app.input.len() + selected.replacement.len());
    new_input.push_str(&app.input[..start]);
    new_input.push_str(&selected.replacement);
    new_input.push_str(&app.input[end..]);
    app.input = new_input;
    true
}

fn clear_pending_pastes(app: &mut AppState) {
    app.pending_text_pastes.clear();
    app.pending_image_pastes.clear();
}

fn prune_detached_placeholders(app: &mut AppState) {
    app.pending_text_pastes
        .retain(|p| app.input.contains(&p.token));
    app.pending_image_pastes
        .retain(|p| app.input.contains(&p.token));
}

fn normalize_paste_text(raw: &str) -> String {
    raw.replace("\r\n", "\n").replace('\r', "\n")
}

fn should_summarize_paste(text: &str) -> bool {
    text.chars().count() >= PASTE_SUMMARY_MIN_CHARS
        || text.lines().count() >= PASTE_SUMMARY_MIN_LINES
}

fn apply_text_paste(app: &mut AppState, raw: &str, attachment_engine: &mut AttachmentEngine) {
    let normalized = normalize_paste_text(raw);
    if normalized.is_empty() {
        return;
    }

    if should_summarize_paste(&normalized) {
        let line_count = normalized.lines().count().max(1);
        app.paste_sequence = app.paste_sequence.saturating_add(1);
        let token = format!("[Colado ~{} linhas #{}]", line_count, app.paste_sequence);
        app.pending_text_pastes.push(PendingTextPaste {
            token: token.clone(),
            full_text: normalized,
        });
        app.input.push_str(&token);
        app.status = format!("paste resumido ({} linhas)", line_count);
    } else {
        app.input.push_str(&normalized);
        app.status = "texto colado".to_string();
    }

    prune_detached_placeholders(app);
    refresh_input_suggestions(app, attachment_engine);
}

fn apply_image_paste_data_url(
    app: &mut AppState,
    data_url: String,
    attachment_engine: &mut AttachmentEngine,
) {
    if !app.supports_vision {
        app.status = "modelo atual não suporta visão; imagem ignorada".to_string();
        return;
    }

    app.paste_sequence = app.paste_sequence.saturating_add(1);
    let token = format!("[Imagem {}]", app.paste_sequence);
    app.pending_image_pastes.push(PendingImagePaste {
        token: token.clone(),
        data_url,
    });
    app.input.push_str(&token);
    app.status = "imagem colada do clipboard".to_string();
    refresh_input_suggestions(app, attachment_engine);
}

fn apply_clipboard_shortcut_paste(app: &mut AppState, attachment_engine: &mut AttachmentEngine) {
    match read_clipboard_payload() {
        Ok(ClipboardPayload::ImageDataUrl(data_url)) => {
            apply_image_paste_data_url(app, data_url, attachment_engine)
        }
        Ok(ClipboardPayload::Text(text)) => apply_text_paste(app, &text, attachment_engine),
        Err(err) => {
            app.status = format!("falha ao colar do clipboard: {err}");
        }
    }
}

fn expand_prompt_with_pastes(prompt: String, app: &AppState) -> String {
    let mut expanded = prompt;
    for pending in &app.pending_text_pastes {
        expanded = expanded.replace(&pending.token, &pending.full_text);
    }
    expanded
}

fn collect_pending_image_payloads(app: &AppState) -> Vec<String> {
    app.pending_image_pastes
        .iter()
        .filter(|pending| app.input.contains(&pending.token))
        .map(|pending| pending.data_url.clone())
        .collect()
}

fn short_session_id(session_id: &str) -> &str {
    let max = 12usize;
    if session_id.len() > max {
        &session_id[..max]
    } else {
        session_id
    }
}

fn restore_chat_from_session_messages(
    app: &mut AppState,
    loaded_messages: &[LlmMessage],
    session_id: &str,
) {
    app.chat.clear();
    app.chat.push(ChatItem::Banner);

    for msg in loaded_messages {
        match msg.role {
            MessageRole::User => {
                let text = msg.content.to_plain_text_lossy();
                if !text.trim().is_empty() {
                    app.chat.push(ChatItem::User(text));
                }
            }
            MessageRole::Assistant => {
                if msg.tool_calls.is_some() && msg.content.to_plain_text_lossy().trim().is_empty() {
                    continue;
                }
                let text = msg.content.to_plain_text_lossy();
                if !text.trim().is_empty() {
                    app.chat.push(ChatItem::Assistant(text));
                }
            }
            MessageRole::System | MessageRole::Tool => {}
        }
    }

    if app.chat.len() == 1 {
        app.chat.push(ChatItem::Assistant(
            "Sessão carregada. Nenhuma mensagem visível ao usuário foi encontrada.".to_string(),
        ));
    }

    app.chat.push(ChatItem::Assistant(format!(
        "Sessão `{}` retomada.",
        short_session_id(session_id)
    )));
    app.status = format!("sessão retomada: {}", short_session_id(session_id));
    app.chat_scroll = 0;
    app.stream_idx = None;
    app.thinking_idx = None;
}

async fn submit_prompt(
    app: &mut AppState,
    attachment_engine: &mut AttachmentEngine,
    cmd_tx: &mpsc::Sender<WorkerCommand>,
    tool_index_by_id: &mut HashMap<String, usize>,
    rewind_manager: &mut RewindManager,
) -> bool {
    prune_detached_placeholders(app);

    let mut prompt_body = app.input.trim().to_string();
    if app.input_mode == InputMode::Slash {
        prompt_body = resolve_slash_prompt_body(app, &prompt_body);
    }

    if prompt_body.is_empty() {
        return false;
    }

    let prompt = match app.input_mode {
        InputMode::Default => prompt_body,
        InputMode::Slash => format!("/{}", prompt_body),
    };
    let was_slash = app.input_mode == InputMode::Slash;
    let prompt_for_model = if was_slash {
        String::new()
    } else {
        expand_prompt_with_pastes(prompt.clone(), app)
    };
    let image_payloads = if was_slash {
        Vec::new()
    } else {
        collect_pending_image_payloads(app)
    };

    app.input_mode = InputMode::Default;
    app.input.clear();
    clear_pending_pastes(app);
    // output_expanded preserved across turns
    clear_mention_suggestions(app);
    refresh_input_suggestions(app, attachment_engine);

    app.chat.push(ChatItem::User(prompt.clone()));

    if was_slash {
        if is_builtin_command(&prompt) {
            return handle_local_command(app, &prompt, tool_index_by_id, rewind_manager);
        }

        match try_build_skill_prompt(app, &prompt) {
            Ok(Some(invocation)) => {
                app.status = format!("skill `{}` carregada", invocation.skill_name);
                let _ = cmd_tx
                    .send(WorkerCommand::Submit {
                        prompt: invocation.model_prompt,
                        image_data_urls: Vec::new(),
                    })
                    .await;
            }
            Ok(None) => {
                app.chat.push(ChatItem::Error(format!(
                    "Comando desconhecido: {}. Digite /help.",
                    prompt.trim()
                )));
            }
            Err(err) => {
                app.chat.push(ChatItem::Error(err));
            }
        }
        return false;
    }

    if is_exit_command(&prompt) {
        return true;
    }

    let expansion = attachment_engine.expand_prompt(prompt_for_model);
    for line in expansion.status_lines {
        // Strip leading "⎿ " prefix — AttachmentStatus renders its own prefix.
        let cleaned = line.strip_prefix("⎿ ").unwrap_or(&line);
        app.chat.push(ChatItem::AttachmentStatus(cleaned.to_string()));
    }
    for err in expansion.errors {
        app.chat.push(ChatItem::Error(err));
    }
    let prompt_for_model = expansion.prompt_for_model;
    let mut image_data_urls = image_payloads;
    image_data_urls.extend(expansion.image_data_urls);
    if !image_data_urls.is_empty() && !app.supports_vision {
        app.chat.push(ChatItem::Error(
            "Este modelo não suporta visão. Remova os anexos de imagem ou troque o modelo."
                .to_string(),
        ));
        app.status = "envio bloqueado: modelo sem visão".to_string();
        return false;
    }

    let _ = cmd_tx
        .send(WorkerCommand::Submit {
            prompt: prompt_for_model,
            image_data_urls,
        })
        .await;
    false
}

fn queue_agent_cycle(app: &mut AppState, reverse: bool) {
    let next_agent = app.active_agent.cycle_primary(reverse);
    if next_agent == app.active_agent {
        return;
    }
    if next_agent == BuiltinAgent::Plan {
        app.yolo_mode = false;
    }
    app.requested_agent_switch = Some(AgentSwitchRequest {
        target: next_agent,
        bootstrap_prompt: None,
    });
    app.status = format!("alternando agente -> {}", agent_status_label(next_agent));
}

async fn handle_key_event(
    key: KeyEvent,
    app: &mut AppState,
    attachment_engine: &mut AttachmentEngine,
    cmd_tx: Option<&mpsc::Sender<WorkerCommand>>,
    interrupt_signal: &Arc<AtomicBool>,
    bash_kill_signal: &nanocode_core::tools::bash::BashKillSignal,
    tool_index_by_id: &mut HashMap<String, usize>,
    rewind_manager: &mut RewindManager,
) -> bool {
    if key.kind != KeyEventKind::Press {
        return false;
    }

    if let Some(pending) = app.pending_approval.take() {
        let mut pending = pending;
        let decision = match (key.code, key.modifiers) {
            (KeyCode::Char('1'), _)
            | (KeyCode::Char('y'), KeyModifiers::NONE | KeyModifiers::SHIFT) => {
                Some(ApprovalOption::ApproveOnce.to_decision())
            }
            (KeyCode::Char('2'), _)
            | (KeyCode::Char('a'), KeyModifiers::NONE | KeyModifiers::SHIFT) => {
                Some(ApprovalOption::ApproveAlwaysToolSession.to_decision())
            }
            (KeyCode::Char('3'), _)
            | (KeyCode::Char('n'), KeyModifiers::NONE | KeyModifiers::SHIFT)
            | (KeyCode::Esc, _) => Some(ApprovalOption::Deny.to_decision()),
            (KeyCode::Up, KeyModifiers::NONE) => {
                pending.selected_option = pending.selected_option.prev();
                None
            }
            (KeyCode::Down, KeyModifiers::NONE) => {
                pending.selected_option = pending.selected_option.next();
                None
            }
            (KeyCode::Left, _) => {
                pending.selected_option = pending.selected_option.prev();
                None
            }
            (KeyCode::Right, _) | (KeyCode::Tab, _) => {
                pending.selected_option = pending.selected_option.next();
                None
            }
            (KeyCode::PageUp, _) => {
                app.chat_scroll = app.chat_scroll.saturating_add(5);
                None
            }
            (KeyCode::PageDown, _) => {
                app.chat_scroll = app.chat_scroll.saturating_sub(5);
                None
            }
            (KeyCode::Home, _) => {
                app.chat_scroll = u16::MAX;
                None
            }
            (KeyCode::End, _) => {
                app.chat_scroll = 0;
                None
            }
            (KeyCode::Up, KeyModifiers::SHIFT) | (KeyCode::Up, KeyModifiers::CONTROL) => {
                app.chat_scroll = app.chat_scroll.saturating_add(5);
                None
            }
            (KeyCode::Down, KeyModifiers::SHIFT) | (KeyCode::Down, KeyModifiers::CONTROL) => {
                app.chat_scroll = app.chat_scroll.saturating_sub(5);
                None
            }
            (KeyCode::Enter, _) => Some(pending.selected_option.to_decision()),
            _ => None,
        };

        if let Some(decision) = decision {
            let _ = pending.decision_tx.send(decision);
            app.status = match decision {
                nanocode_core::ApprovalDecision::ApproveAlwaysToolSession => {
                    "executando ferramenta (sempre nesta sessão)".to_string()
                }
                nanocode_core::ApprovalDecision::ApproveOnce => "executando ferramenta".to_string(),
                nanocode_core::ApprovalDecision::Deny => "ferramenta negada".to_string(),
            };
        } else {
            app.pending_approval = Some(pending);
        }
        return false;
    }

    if let Some(pending) = app.pending_user_question.take() {
        let mut pending = pending;
        let mut response: Option<UserQuestionResponse> = None;

        match (key.code, key.modifiers) {
            (KeyCode::Esc, _) => {
                response = Some(UserQuestionResponse::cancelled());
            }
            (KeyCode::Char(ch), KeyModifiers::NONE)
                if ('1'..='9').contains(&ch) && !pending.choices.is_empty() =>
            {
                let idx = (ch as u8 - b'1') as usize;
                if idx < pending.choices.len() {
                    pending.selected_choice = idx;
                }
            }
            (KeyCode::Up, KeyModifiers::NONE) if !pending.choices.is_empty() => {
                let count = pending.choices.len();
                pending.selected_choice = (pending.selected_choice + count - 1) % count;
            }
            (KeyCode::Down, KeyModifiers::NONE) if !pending.choices.is_empty() => {
                let count = pending.choices.len();
                pending.selected_choice = (pending.selected_choice + 1) % count;
            }
            (KeyCode::Backspace, _) if pending.allow_free_text => {
                pending.text_input.pop();
            }
            (KeyCode::Char(ch), KeyModifiers::NONE | KeyModifiers::SHIFT)
                if pending.allow_free_text =>
            {
                pending.text_input.push(ch);
            }
            (KeyCode::Enter, _) => {
                let typed = pending.text_input.trim().to_string();
                if pending.allow_free_text && !typed.is_empty() {
                    response = Some(UserQuestionResponse {
                        answer: typed,
                        choice_index: None,
                        source: QuestionAnswerSource::Text,
                        cancelled: false,
                    });
                } else if let Some(choice) = pending.choices.get(pending.selected_choice) {
                    response = Some(UserQuestionResponse {
                        answer: choice.clone(),
                        choice_index: Some(pending.selected_choice),
                        source: QuestionAnswerSource::Choice,
                        cancelled: false,
                    });
                } else if pending.allow_free_text {
                    response = Some(UserQuestionResponse {
                        answer: typed,
                        choice_index: None,
                        source: QuestionAnswerSource::Text,
                        cancelled: false,
                    });
                } else {
                    response = Some(UserQuestionResponse::cancelled());
                }
            }
            (KeyCode::PageUp, _) => {
                app.chat_scroll = app.chat_scroll.saturating_add(5);
            }
            (KeyCode::PageDown, _) => {
                app.chat_scroll = app.chat_scroll.saturating_sub(5);
            }
            (KeyCode::Home, _) => {
                app.chat_scroll = u16::MAX;
            }
            (KeyCode::End, _) => {
                app.chat_scroll = 0;
            }
            (KeyCode::Up, KeyModifiers::SHIFT) | (KeyCode::Up, KeyModifiers::CONTROL) => {
                app.chat_scroll = app.chat_scroll.saturating_add(5);
            }
            (KeyCode::Down, KeyModifiers::SHIFT) | (KeyCode::Down, KeyModifiers::CONTROL) => {
                app.chat_scroll = app.chat_scroll.saturating_sub(5);
            }
            _ => {}
        }

        if let Some(answer) = response {
            let _ = pending.response_tx.send(answer.clone());
            app.status = if answer.cancelled {
                "pergunta cancelada".to_string()
            } else {
                "resposta enviada".to_string()
            };
        } else {
            app.pending_user_question = Some(pending);
        }
        return false;
    }

    if let Some(pending) = app.pending_plan_review.take() {
        let mut pending = pending;
        let decision = match (key.code, key.modifiers) {
            (KeyCode::Char('1'), _)
            | (KeyCode::Char('y'), KeyModifiers::NONE | KeyModifiers::SHIFT) => {
                Some(PlanReviewOption::ApproveAndBuild)
            }
            (KeyCode::Char('2'), _)
            | (KeyCode::Char('n'), KeyModifiers::NONE | KeyModifiers::SHIFT)
            | (KeyCode::Esc, _) => Some(PlanReviewOption::Disapprove),
            (KeyCode::Char('3'), _)
            | (KeyCode::Char('r'), KeyModifiers::NONE | KeyModifiers::SHIFT) => {
                Some(PlanReviewOption::ReworkWithSuggestion)
            }
            (KeyCode::Up, KeyModifiers::NONE) => {
                pending.selected_option = pending.selected_option.prev();
                None
            }
            (KeyCode::Down, KeyModifiers::NONE) => {
                pending.selected_option = pending.selected_option.next();
                None
            }
            (KeyCode::Left, _) | (KeyCode::BackTab, _) => {
                pending.selected_option = pending.selected_option.prev();
                None
            }
            (KeyCode::Right, _) => {
                pending.selected_option = pending.selected_option.next();
                None
            }
            (KeyCode::Tab, KeyModifiers::NONE) => {
                if pending.selected_option == PlanReviewOption::ReworkWithSuggestion {
                    Some(PlanReviewOption::ReworkWithSuggestion)
                } else {
                    pending.selected_option = pending.selected_option.next();
                    None
                }
            }
            (KeyCode::Tab, KeyModifiers::SHIFT) => {
                pending.selected_option = pending.selected_option.prev();
                None
            }
            (KeyCode::PageUp, _) => {
                app.chat_scroll = app.chat_scroll.saturating_add(5);
                None
            }
            (KeyCode::PageDown, _) => {
                app.chat_scroll = app.chat_scroll.saturating_sub(5);
                None
            }
            (KeyCode::Home, _) => {
                app.chat_scroll = u16::MAX;
                None
            }
            (KeyCode::End, _) => {
                app.chat_scroll = 0;
                None
            }
            (KeyCode::Up, KeyModifiers::SHIFT) | (KeyCode::Up, KeyModifiers::CONTROL) => {
                app.chat_scroll = app.chat_scroll.saturating_add(5);
                None
            }
            (KeyCode::Down, KeyModifiers::SHIFT) | (KeyCode::Down, KeyModifiers::CONTROL) => {
                app.chat_scroll = app.chat_scroll.saturating_sub(5);
                None
            }
            (KeyCode::Enter, _) => Some(pending.selected_option),
            _ => None,
        };

        if let Some(decision) = decision {
            match decision {
                PlanReviewOption::ApproveAndBuild => {
                    app.chat.push(ChatItem::Assistant(
                        "Plano aprovado. Mudando para modo implementação com contexto limpo do plano."
                            .to_string(),
                    ));
                    app.requested_agent_switch = Some(AgentSwitchRequest {
                        target: BuiltinAgent::Build,
                        bootstrap_prompt: build_plan_bootstrap_prompt(&pending.plan_text),
                    });
                    app.status = "Plano aprovado. Alternando para implementação.".to_string();
                }
                PlanReviewOption::Disapprove => {
                    app.chat.push(ChatItem::Assistant(
                        "Plano reprovado. Envie os ajustes desejados para gerar um novo plano."
                            .to_string(),
                    ));
                    app.status = "Plano reprovado.".to_string();
                }
                PlanReviewOption::ReworkWithSuggestion => {
                    if app.input.trim().is_empty() {
                        app.input =
                            "Refaça o plano considerando os seguintes ajustes: ".to_string();
                    }
                    app.chat.push(ChatItem::Assistant(
                        "Digite os ajustes no input e pressione Enter para refazer o plano."
                            .to_string(),
                    ));
                    app.status =
                        "Digite sua resposta e pressione Enter para continuar.".to_string();
                }
            }
        } else {
            app.pending_plan_review = Some(pending);
        }
        return false;
    }

    if let Some(pending) = app.pending_resume_selection.take() {
        let mut pending = pending;
        let mut selected_session: Option<String> = None;
        let mut cancelled = false;

        match (key.code, key.modifiers) {
            (KeyCode::Esc, _)
            | (KeyCode::Char('c'), KeyModifiers::CONTROL)
            | (KeyCode::Char('q'), KeyModifiers::NONE | KeyModifiers::SHIFT) => {
                cancelled = true;
                app.status = "retomada cancelada".to_string();
            }
            (KeyCode::Up, KeyModifiers::NONE) => {
                if !pending.sessions.is_empty() {
                    let count = pending.sessions.len();
                    pending.selected_idx = (pending.selected_idx + count - 1) % count;
                }
            }
            (KeyCode::Down, KeyModifiers::NONE) => {
                if !pending.sessions.is_empty() {
                    let count = pending.sessions.len();
                    pending.selected_idx = (pending.selected_idx + 1) % count;
                }
            }
            (KeyCode::Enter, _) => {
                if let Some(session) = pending.sessions.get(pending.selected_idx) {
                    selected_session = Some(session.id.clone());
                }
            }
            _ => {}
        }

        if let Some(session_id) = selected_session {
            app.requested_resume_session = Some(SessionResumeRequest {
                session_id_query: session_id.clone(),
            });
            app.status = format!("retomando sessão {}", short_session_id(&session_id));
        } else if !cancelled {
            app.pending_resume_selection = Some(pending);
        }
        return false;
    }

    if !app.busy
        && matches!(key.code, KeyCode::Char('v') | KeyCode::Char('V'))
        && key.modifiers.contains(KeyModifiers::CONTROL)
    {
        apply_clipboard_shortcut_paste(app, attachment_engine);
        return false;
    }

    match (key.code, key.modifiers) {
        (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
            if app.busy {
                set_interrupt_signal(interrupt_signal);
                app.status = "interrompendo".to_string();
            } else if !app.input.is_empty() {
                app.input.clear();
                clear_pending_pastes(app);
                app.input_mode = InputMode::Default;
                refresh_input_suggestions(app, attachment_engine);
            }
        }
        (KeyCode::Char('d'), KeyModifiers::CONTROL) => {
            return true;
        }
        (KeyCode::Enter, _) if app.screen == UiScreen::Welcome => {
            app.screen = if app.needs_model_setup {
                UiScreen::ModelSetup
            } else {
                UiScreen::Chat
            };
        }
        _ if app.screen == UiScreen::Welcome => {}
        (KeyCode::Char('o'), KeyModifiers::CONTROL) => {
            app.output_expanded = !app.output_expanded;
            app.status = if app.output_expanded {
                "saída expandida (Ctrl+O para recolher)".to_string()
            } else {
                "saída recolhida (Ctrl+O para expandir)".to_string()
            };
        }
        (KeyCode::Char('t'), KeyModifiers::CONTROL) => {
            app.tasklist_open = !app.tasklist_open;
            app.status = if app.tasklist_open {
                "tasklist aberta (Ctrl+T para fechar)".to_string()
            } else {
                format!("pronto · agente {}", agent_status_label(app.active_agent))
            };
        }
        (KeyCode::Char('t'), KeyModifiers::ALT) => {
            if !app.supports_thinking {
                app.status = "modelo atual não suporta raciocínio".to_string();
            } else {
                app.thinking_enabled = !app.thinking_enabled;
                if let Some(cmd_tx) = cmd_tx {
                    let _ = cmd_tx.send(WorkerCommand::SetThinkingEnabled(app.thinking_enabled)).await;
                }
                app.status = if app.thinking_enabled {
                    "raciocínio do modelo ligado".to_string()
                } else {
                    "raciocínio do modelo desligado".to_string()
                };
            }
        }
        (KeyCode::Char('b'), KeyModifiers::CONTROL) => {
            if !app.running_bash_call_ids.is_empty() {
                app.process_viewer_open = !app.process_viewer_open;
                app.status = if app.process_viewer_open {
                    "Processos ativos (K para encerrar, Esc para fechar)".to_string()
                } else {
                    format!("pronto · agente {}", agent_status_label(app.active_agent))
                };
            } else if app.process_viewer_open {
                app.process_viewer_open = false;
                app.status =
                    format!("pronto · agente {}", agent_status_label(app.active_agent));
            }
        }
        (KeyCode::PageUp, _) => {
            app.chat_scroll = app.chat_scroll.saturating_add(5);
        }
        (KeyCode::PageDown, _) => {
            app.chat_scroll = app.chat_scroll.saturating_sub(5);
        }
        (KeyCode::Home, _) => {
            app.chat_scroll = u16::MAX;
        }
        (KeyCode::End, _) => {
            app.chat_scroll = 0;
        }
        (KeyCode::Up, KeyModifiers::SHIFT) => {
            app.chat_scroll = app.chat_scroll.saturating_add(5);
        }
        (KeyCode::Down, KeyModifiers::SHIFT) => {
            app.chat_scroll = app.chat_scroll.saturating_sub(5);
        }
        (KeyCode::Up, KeyModifiers::NONE)
            if app.input_mode == InputMode::Slash && !app.slash_suggestions.is_empty() =>
        {
            let count = app.slash_suggestions.len();
            app.slash_selected = (app.slash_selected + count - 1) % count;
        }
        (KeyCode::Down, KeyModifiers::NONE)
            if app.input_mode == InputMode::Slash && !app.slash_suggestions.is_empty() =>
        {
            let count = app.slash_suggestions.len();
            app.slash_selected = (app.slash_selected + 1) % count;
        }
        (KeyCode::Up, KeyModifiers::NONE) if !app.busy && !app.mention_suggestions.is_empty() => {
            let count = app.mention_suggestions.len();
            app.mention_selected = (app.mention_selected + count - 1) % count;
        }
        (KeyCode::Down, KeyModifiers::NONE) if !app.busy && !app.mention_suggestions.is_empty() => {
            let count = app.mention_suggestions.len();
            app.mention_selected = (app.mention_selected + 1) % count;
        }
        (KeyCode::Up, KeyModifiers::NONE) => {
            app.chat_scroll = app.chat_scroll.saturating_add(1);
        }
        (KeyCode::Down, KeyModifiers::NONE) => {
            app.chat_scroll = app.chat_scroll.saturating_sub(1);
        }
        (KeyCode::Esc, _) if app.process_viewer_open => {
            app.process_viewer_open = false;
            app.status =
                format!("pronto · agente {}", agent_status_label(app.active_agent));
        }
        (KeyCode::Char('k') | KeyCode::Char('K'), _)
            if app.process_viewer_open && !app.running_bash_call_ids.is_empty() =>
        {
            bash_kill_signal.store(true, std::sync::atomic::Ordering::Relaxed);
            app.status = "Encerrando processo bash...".to_string();
        }
        (KeyCode::Esc, _) if app.tasklist_open => {
            app.tasklist_open = false;
            app.status = format!("pronto · agente {}", agent_status_label(app.active_agent));
        }
        (KeyCode::Esc, _) if app.busy => {
            set_interrupt_signal(interrupt_signal);
            app.status = "interrompendo".to_string();
        }
        (KeyCode::Esc, _) if !app.busy && app.input_mode == InputMode::Slash => {
            if app.input.is_empty() {
                app.input_mode = InputMode::Default;
            }
            refresh_input_suggestions(app, attachment_engine);
        }
        (KeyCode::Esc, _) if !app.busy && !app.mention_suggestions.is_empty() => {
            clear_mention_suggestions(app);
        }
        (KeyCode::Char('l'), KeyModifiers::CONTROL) if !app.busy => {
            app.chat.clear();
            app.chat.push(ChatItem::Banner);
            tool_index_by_id.clear();
            app.stream_idx = None;
            app.thinking_idx = None;
        }
        (KeyCode::Tab, _) if !app.busy && app.input_mode == InputMode::Slash => {
            apply_selected_slash_suggestion(app);
        }
        (KeyCode::Tab, _) if !app.busy && !app.mention_suggestions.is_empty() => {
            if apply_selected_mention_suggestion(app) {
                refresh_input_suggestions(app, attachment_engine);
            }
        }
        (KeyCode::BackTab, KeyModifiers::SHIFT | KeyModifiers::NONE)
            if !app.busy && app.input_mode == InputMode::Default =>
        {
            queue_agent_cycle(app, false);
        }
        (KeyCode::Char('?'), _)
            if matches!(
                app.active_agent,
                BuiltinAgent::Default | BuiltinAgent::Build
            ) && app.input.is_empty()
                && !app.busy =>
        {
            app.yolo_mode = !app.yolo_mode;
            if let Some(tx) = cmd_tx {
                let _ = tx.send(WorkerCommand::SetAutoApprove(app.yolo_mode)).await;
            }
        }
        (KeyCode::Enter, KeyModifiers::SHIFT) if !app.busy => {
            app.input.push('\n');
        }
        (KeyCode::Enter, _) if !app.busy => {
            if let Some(tx) = cmd_tx {
                if submit_prompt(app, attachment_engine, tx, tool_index_by_id, rewind_manager).await
                {
                    return true;
                }
            } else {
                app.chat.push(ChatItem::Error(
                    "O modelo ainda não foi inicializado. Conclua o setup primeiro.".to_string(),
                ));
            }
        }
        (KeyCode::Backspace, _) if !app.busy => {
            if app.input.is_empty() {
                app.input_mode = InputMode::Default;
            } else {
                app.input.pop();
            }
            prune_detached_placeholders(app);
            refresh_input_suggestions(app, attachment_engine);
        }
        (KeyCode::Char(ch), KeyModifiers::NONE | KeyModifiers::SHIFT) if !app.busy => {
            if ch == '/' && app.input.is_empty() && app.input_mode == InputMode::Default {
                app.input_mode = InputMode::Slash;
                refresh_input_suggestions(app, attachment_engine);
            } else {
                app.input.push(ch);
                refresh_input_suggestions(app, attachment_engine);
            }
        }
        _ => {}
    }

    false
}

fn resolve_selected_model(config: &NcConfig) -> &'static ModelSpec {
    config
        .active_model
        .as_deref()
        .and_then(find_model)
        .unwrap_or_else(default_model)
}

fn build_variant_choices(
    config: &NcConfig,
    hw: &HardwareInfo,
    model: &'static ModelSpec,
) -> Vec<SetupChoice> {
    let cached_quants = list_cached_quants(&NcConfig::models_dir(), model);
    let cached_names: HashSet<&str> = cached_quants.iter().map(|q| q.name).collect();
    let active_name = if config.active_model.as_deref() == Some(model.id) {
        config.active_quant.as_deref()
    } else {
        None
    };
    let recommended_name = recommend(hw, model).map(|q| q.name);

    model_quantizations(model)
        .iter()
        .map(|quant| SetupChoice {
            quant_name: quant.name.to_string(),
            size_human: quant.size_human(),
            quality_label: quant.quality.label().to_string(),
            notes: quant.notes.map(str::to_string),
            recommended: recommended_name == Some(quant.name),
            cached: cached_names.contains(quant.name),
            active: active_name == Some(quant.name),
        })
        .collect()
}

fn resolve_variant_selection_idx(choices: &[SetupChoice], preferred_quant: Option<&str>) -> usize {
    preferred_quant
        .and_then(|name| choices.iter().position(|c| c.quant_name == name))
        .or_else(|| choices.iter().position(|c| c.active))
        .or_else(|| choices.iter().position(|c| c.recommended))
        .or_else(|| choices.iter().position(|c| c.cached))
        .unwrap_or(0)
}

fn refresh_model_setup_selection(
    setup: &mut ModelSetupState,
    config: &NcConfig,
    hw: &HardwareInfo,
    preferred_quant: Option<&str>,
) {
    let Some(model_choice) = setup.models.get(setup.selected_model_idx) else {
        setup.current_model_id.clear();
        setup.current_model_display_name.clear();
        setup.choices.clear();
        setup.selected_idx = 0;
        setup.status_line = "Nenhum modelo disponível no catálogo.".to_string();
        return;
    };

    let model = find_model(&model_choice.model_id).unwrap_or_else(default_model);
    let choices = build_variant_choices(config, hw, model);
    let selected_idx = resolve_variant_selection_idx(&choices, preferred_quant);

    setup.current_model_id = model.id.to_string();
    setup.current_model_display_name = model.display_name.to_string();
    setup.choices = choices;
    setup.selected_idx = selected_idx;
}

fn update_model_setup_status(setup: &mut ModelSetupState) {
    if setup.models.is_empty() {
        setup.status_line = "Nenhum modelo disponível no catálogo.".to_string();
        return;
    }

    let Some(model) = setup.models.get(setup.selected_model_idx) else {
        setup.status_line = "Selecione um modelo.".to_string();
        return;
    };

    setup.status_line = match setup.view {
        ModelSetupView::Models => {
            if let Some(active_quant) = &model.active_quant {
                format!(
                    "Selecionado: {} · ativo {} · {} variante(s) em cache.",
                    model.model_display_name,
                    active_quant,
                    model.cached_quants.len()
                )
            } else if model.cached_quants.is_empty() {
                format!(
                    "Selecionado: {}. Ainda não há variantes em cache. Pressione Enter para escolher uma.",
                    model.model_display_name
                )
            } else {
                format!(
                    "Selecionado: {} · {} variante(s) em cache. Pressione Enter para escolher a variante.",
                    model.model_display_name,
                    model.cached_quants.len()
                )
            }
        }
        ModelSetupView::Variants => {
            if let Some(choice) = setup.choices.get(setup.selected_idx) {
                if choice.cached {
                    format!(
                        "A variante {} está em cache. Enter ativa e remove as outras variantes em cache.",
                        choice.quant_name
                    )
                } else {
                    format!(
                        "A variante {} não está em cache. Enter baixa e ativa.",
                        choice.quant_name
                    )
                }
            } else {
                "Nenhuma quantização disponível para o modelo selecionado.".to_string()
            }
        }
    };
}

fn build_model_setup_state(config: &NcConfig, hw: &HardwareInfo) -> ModelSetupState {
    let model_choices: Vec<ModelChoice> = models()
        .iter()
        .map(|model| {
            let cached_quants = list_cached_quants(&NcConfig::models_dir(), model)
                .iter()
                .map(|q| q.name.to_string())
                .collect::<Vec<_>>();
            let active_quant = if config.active_model.as_deref() == Some(model.id) {
                config
                    .active_quant
                    .as_deref()
                    .and_then(|name| find_quant_by_name(model, name))
                    .map(|q| q.name.to_string())
            } else {
                None
            };
            ModelChoice {
                model_id: model.id.to_string(),
                model_display_name: model.display_name.to_string(),
                category_label: model.category.label().to_string(),
                supports_thinking: model.supports_thinking,
                supports_vision: model.supports_vision,
                max_context_tokens: model.max_context_size,
                recommended_context_general: model.recommended_context_general,
                recommended_context_coding: model.recommended_context_coding,
                cached_quants,
                active_quant,
                recommended_quant: recommend(hw, model).map(|q| q.name.to_string()),
            }
        })
        .collect();

    let selected_model_idx = if model_choices.is_empty() {
        0
    } else {
        config
            .active_model
            .as_deref()
            .and_then(|id| model_choices.iter().position(|m| m.model_id == id))
            .or_else(|| model_choices.iter().position(|m| m.active_quant.is_some()))
            .or_else(|| {
                model_choices
                    .iter()
                    .position(|m| !m.cached_quants.is_empty())
            })
            .unwrap_or(0)
    };

    let mut state = ModelSetupState {
        view: ModelSetupView::Models,
        hardware_display: hw.display(),
        models: model_choices,
        selected_model_idx,
        current_model_id: String::new(),
        current_model_display_name: String::new(),
        choices: Vec::new(),
        selected_idx: 0,
        downloading: false,
        status_line: String::new(),
        progress_line: None,
        download_progress: None,
        error_line: None,
    };
    refresh_model_setup_selection(&mut state, config, hw, None);
    update_model_setup_status(&mut state);
    state
}

fn permission_to_value(permission: &ToolPermissionConfig) -> &'static str {
    match permission {
        ToolPermissionConfig::Always => "sempre",
        ToolPermissionConfig::Never => "nunca",
        ToolPermissionConfig::Ask => "perguntar",
    }
}

fn parse_permission(value: &str) -> ToolPermissionConfig {
    match value {
        "sempre" => ToolPermissionConfig::Always,
        "nunca" => ToolPermissionConfig::Never,
        _ => ToolPermissionConfig::Ask,
    }
}

fn cycle_config_value(state: &mut ConfigScreenState, step: i32) {
    let Some(field) = state.fields.get_mut(state.selected_idx) else {
        return;
    };
    if field.options.is_empty() {
        return;
    }

    let current_idx = field
        .options
        .iter()
        .position(|v| v == &field.value)
        .unwrap_or(0);
    let count = field.options.len() as i32;
    let next_idx = ((current_idx as i32 + step).rem_euclid(count)) as usize;
    field.value = field.options[next_idx].clone();
    state.error_line = None;
    state.status_line = if state.is_dirty() {
        "Alterações não salvas. Pressione Esc para salvar e fechar.".to_string()
    } else {
        "Sem alterações pendentes. Pressione Esc para voltar ao chat.".to_string()
    };
}

fn build_config_screen_state(
    config: &NcConfig,
    hw: &HardwareInfo,
    ctk_override: Option<String>,
    ctv_override: Option<String>,
    fallback_model_label: &str,
    fallback_context_tokens: u32,
) -> ConfigScreenState {
    let runtime = runtime_snapshot(config, ctk_override, ctv_override);
    let model_label = runtime
        .as_ref()
        .map(|r| r.model_label.clone())
        .unwrap_or_else(|| fallback_model_label.to_string());
    let runtime_context_tokens = runtime
        .as_ref()
        .map(|r| r.context_size)
        .unwrap_or(fallback_context_tokens);
    let runtime_max_tokens = runtime
        .as_ref()
        .map(|r| r.max_tokens)
        .unwrap_or(config.model.max_tokens);

    let context_value = config
        .model
        .context_size
        .map(|v| v.to_string())
        .unwrap_or_else(|| "Automático".to_string());
    let kv_k_value = config
        .model
        .kv_cache_type_k
        .clone()
        .unwrap_or_else(|| "Automático".to_string());
    let kv_v_value = config
        .model
        .kv_cache_type_v
        .clone()
        .unwrap_or_else(|| "Automático".to_string());

    let mut fields = vec![
        ConfigField {
            key: ConfigFieldKey::NGpuLayers,
            label: "Camadas de GPU".to_string(),
            value: if config.model.n_gpu_layers == 0 {
                "Somente CPU (0)".to_string()
            } else {
                "Automático (adaptativo)".to_string()
            },
            initial_value: if config.model.n_gpu_layers == 0 {
                "Somente CPU (0)".to_string()
            } else {
                "Automático (adaptativo)".to_string()
            },
            options: vec![
                "Automático (adaptativo)".to_string(),
                "Somente CPU (0)".to_string(),
            ],
        },
        ConfigField {
            key: ConfigFieldKey::ContextSize,
            label: "Tamanho de contexto".to_string(),
            value: context_value.clone(),
            initial_value: context_value,
            options: vec![
                "Automático".to_string(),
                "8192".to_string(),
                "16384".to_string(),
                "32768".to_string(),
                "49152".to_string(),
                "65536".to_string(),
                "81920".to_string(),
            ],
        },
        ConfigField {
            key: ConfigFieldKey::KvCacheTypeK,
            label: "Cache KV K".to_string(),
            value: kv_k_value.clone(),
            initial_value: kv_k_value,
            options: vec![
                "Automático".to_string(),
                "q8_0".to_string(),
                "q4_0".to_string(),
                "f16".to_string(),
            ],
        },
        ConfigField {
            key: ConfigFieldKey::KvCacheTypeV,
            label: "Cache KV V".to_string(),
            value: kv_v_value.clone(),
            initial_value: kv_v_value,
            options: vec![
                "Automático".to_string(),
                "q8_0".to_string(),
                "q4_0".to_string(),
                "f16".to_string(),
            ],
        },
        ConfigField {
            key: ConfigFieldKey::BashPermission,
            label: "Permissão do bash".to_string(),
            value: permission_to_value(&config.tools.bash.permission).to_string(),
            initial_value: permission_to_value(&config.tools.bash.permission).to_string(),
            options: vec![
                "perguntar".to_string(),
                "sempre".to_string(),
                "nunca".to_string(),
            ],
        },
        ConfigField {
            key: ConfigFieldKey::WriteFilePermission,
            label: "Permissão do write_file".to_string(),
            value: permission_to_value(&config.tools.write_file.permission).to_string(),
            initial_value: permission_to_value(&config.tools.write_file.permission).to_string(),
            options: vec![
                "perguntar".to_string(),
                "sempre".to_string(),
                "nunca".to_string(),
            ],
        },
        ConfigField {
            key: ConfigFieldKey::GrepPermission,
            label: "Permissão do grep".to_string(),
            value: permission_to_value(&config.tools.grep.permission).to_string(),
            initial_value: permission_to_value(&config.tools.grep.permission).to_string(),
            options: vec![
                "perguntar".to_string(),
                "sempre".to_string(),
                "nunca".to_string(),
            ],
        },
        ConfigField {
            key: ConfigFieldKey::ReadFilePermission,
            label: "Permissão do read_file".to_string(),
            value: permission_to_value(&config.tools.read_file.permission).to_string(),
            initial_value: permission_to_value(&config.tools.read_file.permission).to_string(),
            options: vec![
                "perguntar".to_string(),
                "sempre".to_string(),
                "nunca".to_string(),
            ],
        },
    ];

    // Guard against stale values that are not in current options.
    for field in &mut fields {
        if !field.options.iter().any(|v| v == &field.value) {
            field.value = field.options[0].clone();
        }
        if !field.options.iter().any(|v| v == &field.initial_value) {
            field.initial_value = field.value.clone();
        }
    }

    ConfigScreenState {
        model_label,
        hardware_display: hw.display(),
        runtime_context_tokens,
        runtime_max_tokens,
        config_path: NcConfig::config_path().display().to_string(),
        models_path: NcConfig::models_dir().display().to_string(),
        sessions_path: NcConfig::sessions_dir().display().to_string(),
        fields,
        selected_idx: 0,
        status_line: "Sem alterações pendentes. Pressione Esc para voltar ao chat.".to_string(),
        error_line: None,
    }
}

fn apply_config_screen_changes(config: &mut NcConfig, state: &ConfigScreenState) -> bool {
    let mut changed = false;

    for field in &state.fields {
        match field.key {
            ConfigFieldKey::NGpuLayers => {
                let value = if field.value == "Somente CPU (0)" {
                    0
                } else {
                    -1
                };
                if config.model.n_gpu_layers != value {
                    config.model.n_gpu_layers = value;
                    changed = true;
                }
            }
            ConfigFieldKey::ContextSize => {
                let value = if field.value == "Automático" {
                    None
                } else {
                    field.value.parse::<u32>().ok()
                };
                if config.model.context_size != value {
                    config.model.context_size = value;
                    changed = true;
                }
            }
            ConfigFieldKey::KvCacheTypeK => {
                let value = if field.value == "Automático" {
                    None
                } else {
                    Some(field.value.clone())
                };
                if config.model.kv_cache_type_k != value {
                    config.model.kv_cache_type_k = value;
                    changed = true;
                }
            }
            ConfigFieldKey::KvCacheTypeV => {
                let value = if field.value == "Automático" {
                    None
                } else {
                    Some(field.value.clone())
                };
                if config.model.kv_cache_type_v != value {
                    config.model.kv_cache_type_v = value;
                    changed = true;
                }
            }
            ConfigFieldKey::BashPermission => {
                let value = parse_permission(&field.value);
                if std::mem::discriminant(&config.tools.bash.permission)
                    != std::mem::discriminant(&value)
                {
                    config.tools.bash.permission = value;
                    changed = true;
                }
            }
            ConfigFieldKey::WriteFilePermission => {
                let value = parse_permission(&field.value);
                if std::mem::discriminant(&config.tools.write_file.permission)
                    != std::mem::discriminant(&value)
                {
                    config.tools.write_file.permission = value;
                    changed = true;
                }
            }
            ConfigFieldKey::GrepPermission => {
                let value = parse_permission(&field.value);
                if std::mem::discriminant(&config.tools.grep.permission)
                    != std::mem::discriminant(&value)
                {
                    config.tools.grep.permission = value;
                    changed = true;
                }
            }
            ConfigFieldKey::ReadFilePermission => {
                let value = parse_permission(&field.value);
                if std::mem::discriminant(&config.tools.read_file.permission)
                    != std::mem::discriminant(&value)
                {
                    config.tools.read_file.permission = value;
                    changed = true;
                }
            }
        }
    }

    changed
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else {
        format!("{} KB", bytes / KB)
    }
}

fn handle_model_setup_key(
    key: KeyEvent,
    app: &mut AppState,
    config: &NcConfig,
    hw: &HardwareInfo,
) -> ModelSetupAction {
    if key.kind != KeyEventKind::Press {
        return ModelSetupAction::None;
    }

    let downloading = app
        .model_setup
        .as_ref()
        .map(|setup| setup.downloading)
        .unwrap_or(false);

    match (key.code, key.modifiers) {
        (KeyCode::Esc, _) => {
            if downloading {
                return ModelSetupAction::None;
            }
            if let Some(setup) = app.model_setup.as_mut() {
                if setup.view == ModelSetupView::Variants {
                    setup.view = ModelSetupView::Models;
                    setup.error_line = None;
                    setup.progress_line = None;
                    setup.download_progress = None;
                    update_model_setup_status(setup);
                    return ModelSetupAction::None;
                }
            }
            return if app.model_setup_can_cancel {
                ModelSetupAction::Close
            } else {
                ModelSetupAction::Quit
            };
        }
        (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
            if downloading {
                return ModelSetupAction::CancelDownload;
            }
            if let Some(setup) = app.model_setup.as_mut() {
                if setup.view == ModelSetupView::Variants {
                    setup.view = ModelSetupView::Models;
                    setup.error_line = None;
                    setup.progress_line = None;
                    setup.download_progress = None;
                    update_model_setup_status(setup);
                    return ModelSetupAction::None;
                }
            }
            return if app.model_setup_can_cancel {
                ModelSetupAction::Close
            } else {
                ModelSetupAction::Quit
            };
        }
        (KeyCode::Up, _) => {
            if let Some(setup) = app.model_setup.as_mut() {
                if !setup.downloading {
                    match setup.view {
                        ModelSetupView::Models => {
                            if !setup.models.is_empty() {
                                let count = setup.models.len();
                                setup.selected_model_idx =
                                    (setup.selected_model_idx + count - 1) % count;
                                refresh_model_setup_selection(setup, config, hw, None);
                                setup.error_line = None;
                                setup.progress_line = None;
                                setup.download_progress = None;
                                update_model_setup_status(setup);
                            }
                        }
                        ModelSetupView::Variants => {
                            if !setup.choices.is_empty() {
                                let count = setup.choices.len();
                                setup.selected_idx = (setup.selected_idx + count - 1) % count;
                                setup.error_line = None;
                                setup.progress_line = None;
                                setup.download_progress = None;
                                update_model_setup_status(setup);
                            }
                        }
                    }
                }
            }
        }
        (KeyCode::Down, _) => {
            if let Some(setup) = app.model_setup.as_mut() {
                if !setup.downloading {
                    match setup.view {
                        ModelSetupView::Models => {
                            if !setup.models.is_empty() {
                                let count = setup.models.len();
                                setup.selected_model_idx = (setup.selected_model_idx + 1) % count;
                                refresh_model_setup_selection(setup, config, hw, None);
                                setup.error_line = None;
                                setup.progress_line = None;
                                setup.download_progress = None;
                                update_model_setup_status(setup);
                            }
                        }
                        ModelSetupView::Variants => {
                            if !setup.choices.is_empty() {
                                let count = setup.choices.len();
                                setup.selected_idx = (setup.selected_idx + 1) % count;
                                setup.error_line = None;
                                setup.progress_line = None;
                                setup.download_progress = None;
                                update_model_setup_status(setup);
                            }
                        }
                    }
                }
            }
        }
        (KeyCode::Enter, _) => {
            if let Some(setup) = app.model_setup.as_mut() {
                if !setup.downloading {
                    if setup.view == ModelSetupView::Models {
                        setup.view = ModelSetupView::Variants;
                        setup.error_line = None;
                        setup.progress_line = None;
                        setup.download_progress = None;
                        update_model_setup_status(setup);
                    } else {
                        return ModelSetupAction::ConfirmSelected;
                    }
                }
            }
        }
        _ => {}
    }

    ModelSetupAction::None
}

fn handle_config_screen_key(key: KeyEvent, app: &mut AppState) -> ConfigScreenAction {
    if key.kind != KeyEventKind::Press {
        return ConfigScreenAction::None;
    }

    let Some(config_screen) = app.config_screen.as_mut() else {
        return ConfigScreenAction::CloseNoSave;
    };

    match (key.code, key.modifiers) {
        (KeyCode::Esc, _) | (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
            if config_screen.is_dirty() {
                ConfigScreenAction::SaveAndClose
            } else {
                ConfigScreenAction::CloseNoSave
            }
        }
        (KeyCode::Up, _) => {
            if !config_screen.fields.is_empty() {
                let count = config_screen.fields.len();
                config_screen.selected_idx = (config_screen.selected_idx + count - 1) % count;
            }
            ConfigScreenAction::None
        }
        (KeyCode::Down, _) => {
            if !config_screen.fields.is_empty() {
                let count = config_screen.fields.len();
                config_screen.selected_idx = (config_screen.selected_idx + 1) % count;
            }
            ConfigScreenAction::None
        }
        (KeyCode::Left, _) => {
            cycle_config_value(config_screen, -1);
            ConfigScreenAction::None
        }
        (KeyCode::Right, _) | (KeyCode::Enter, _) | (KeyCode::Char(' '), _) => {
            cycle_config_value(config_screen, 1);
            ConfigScreenAction::None
        }
        _ => ConfigScreenAction::None,
    }
}

fn start_chat_worker(
    config: &NcConfig,
    ctk_override: Option<String>,
    ctv_override: Option<String>,
    active_agent: BuiltinAgent,
    interrupt_signal: Arc<AtomicBool>,
    resume_session_id: Option<String>,
    initial_messages: Vec<LlmMessage>,
) -> Result<(
    mpsc::Sender<WorkerCommand>,
    Receiver<WorkerEvent>,
    HardwareInfo,
    u32,
    nanocode_core::tools::bash::BashKillSignal,
)> {
    let runtime = build_runtime(config, ctk_override, ctv_override)
        .map_err(|e| anyhow!("Falha ao inicializar runtime: {e}"))?;
    let telemetry_hw = runtime.hardware.clone();
    let max_context_tokens = runtime.config.model.context_size.unwrap_or(200_000);

    let (cmd_tx, cmd_rx) = mpsc::channel::<WorkerCommand>(8);
    let agent_policy = AgentPolicy::from_builtin(active_agent);
    let (evt_rx, bash_kill_signal) = spawn_worker(
        runtime,
        agent_policy,
        cmd_rx,
        interrupt_signal,
        resume_session_id,
        initial_messages,
    );
    Ok((cmd_tx, evt_rx, telemetry_hw, max_context_tokens, bash_kill_signal))
}

fn start_download_task(model: &'static ModelSpec, quant_name: &str) -> Result<DownloadTask> {
    let quant = find_quant_by_name(model, quant_name)
        .ok_or_else(|| anyhow!("Quantização selecionada desconhecida: {}", quant_name))?;

    let downloader = Downloader::new();
    let (progress_tx, progress_rx) = mpsc::channel(100);
    let dest_dir = NcConfig::models_dir();

    let handle = tokio::spawn(async move {
        downloader
            .download(model, quant, &dest_dir, progress_tx)
            .await
    });

    Ok(DownloadTask {
        model_id: model.id.to_string(),
        quant_name: quant.name.to_string(),
        quant_filename: quant.filename.to_string(),
        progress_rx,
        handle,
    })
}

async fn restart_chat_worker(
    worker_cmd_tx: &mut Option<mpsc::Sender<WorkerCommand>>,
    worker_evt_rx: &mut Option<Receiver<WorkerEvent>>,
    telemetry_hw: &mut HardwareInfo,
    bash_kill_signal_out: &mut nanocode_core::tools::bash::BashKillSignal,
    app: &mut AppState,
    config: &NcConfig,
    ctk_override: Option<String>,
    ctv_override: Option<String>,
    interrupt_signal: Arc<AtomicBool>,
    resume_session_id: Option<String>,
    initial_messages: Vec<LlmMessage>,
) -> Result<()> {
    if let Some(cmd_tx) = worker_cmd_tx.take() {
        let _ = cmd_tx.send(WorkerCommand::Shutdown).await;
    }
    *worker_evt_rx = None;
    clear_interrupt_signal(&interrupt_signal);

    let (cmd_tx, evt_rx, hw, max_context_tokens, bash_kill) = start_chat_worker(
        config,
        ctk_override,
        ctv_override,
        app.active_agent,
        interrupt_signal,
        resume_session_id,
        initial_messages,
    )?;
    *worker_cmd_tx = Some(cmd_tx);
    *worker_evt_rx = Some(evt_rx);
    *telemetry_hw = hw;
    *bash_kill_signal_out = bash_kill;
    app.max_context_tokens = max_context_tokens;
    app.running_bash_call_ids.clear();
    app.status = format!("pronto · agente {}", agent_status_label(app.active_agent));
    app.busy = false;
    app.busy_started_at = None;
    Ok(())
}

pub async fn run_tui(
    config: &mut NcConfig,
    ctk_override: Option<String>,
    ctv_override: Option<String>,
    setup_only: bool,
    initial_agent: BuiltinAgent,
    resume_session_id: Option<String>,
) -> Result<()> {
    preload_dynamic_quantizations().await;

    let initial_hw = HardwareInfo::detect();
    let selected_model = resolve_selected_model(config);
    let fallback_limits = recommend_runtime_limits(
        initial_hw.vram_mb.unwrap_or(initial_hw.ram_mb),
        selected_model,
        true,
    );

    let has_installed_model = find_any_installed_model_quant(&NcConfig::models_dir()).is_some();

    let mut app = AppState::new(
        fallback_limits.context_size,
        initial_hw.sample_runtime_telemetry(),
        initial_agent,
    );
    refresh_skill_catalog(&mut app, config);
    app.mcp_servers_count = config.mcp_servers.len();
    let mut attachment_engine =
        AttachmentEngine::new(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    app.setup_only = setup_only;
    app.needs_model_setup = setup_only || !has_installed_model;

    let mut bootstrap_resume_id: Option<String> = None;
    let mut bootstrap_messages: Vec<LlmMessage> = Vec::new();
    if let Some(session_id_query) = resume_session_id.as_deref() {
        let Some((resolved_id, loaded_messages)) =
            load_session_by_id_sync(&NcConfig::sessions_dir(), session_id_query)?
        else {
            return Err(anyhow!(
                "Sessão '{}' não encontrada em {}",
                session_id_query,
                NcConfig::sessions_dir().display()
            ));
        };
        restore_chat_from_session_messages(&mut app, &loaded_messages, &resolved_id);
        bootstrap_resume_id = Some(resolved_id);
        bootstrap_messages = loaded_messages;
    }

    if app.needs_model_setup {
        app.model_setup = Some(build_model_setup_state(config, &initial_hw));
        app.model_setup_can_cancel = false;
        app.status = "setup necessário".to_string();
        app.busy = false;
        app.busy_started_at = None;
        if setup_only {
            app.screen = UiScreen::ModelSetup;
        }
    }

    let mut worker_cmd_tx: Option<mpsc::Sender<WorkerCommand>> = None;
    let mut worker_evt_rx: Option<Receiver<WorkerEvent>> = None;
    let mut rewind_manager = RewindManager::default();
    let interrupt_signal = Arc::new(AtomicBool::new(false));
    let mut telemetry_hw = initial_hw;
    let mut bash_kill_signal = nanocode_core::tools::bash::new_kill_signal();

    if !app.needs_model_setup {
        restart_chat_worker(
            &mut worker_cmd_tx,
            &mut worker_evt_rx,
            &mut telemetry_hw,
            &mut bash_kill_signal,
            &mut app,
            config,
            ctk_override.clone(),
            ctv_override.clone(),
            interrupt_signal.clone(),
            bootstrap_resume_id,
            bootstrap_messages,
        )
        .await?;
    }

    let mut terminal = setup_terminal()?;
    let mut should_quit = false;
    let mut tool_index_by_id: HashMap<String, usize> = HashMap::new();
    let telemetry_refresh_interval = Duration::from_millis(900);
    let spinner_interval = Duration::from_millis(40);
    let poll_busy_interval = Duration::from_millis(8);
    let poll_idle_interval = Duration::from_millis(80);
    let download_poll_interval = Duration::from_millis(25);
    let mut last_telemetry_refresh = Instant::now()
        .checked_sub(Duration::from_secs(2))
        .unwrap_or_else(Instant::now);
    let mut last_spinner_tick = Instant::now();
    let mut download_task: Option<DownloadTask> = None;
    let mut needs_redraw = true;

    while !should_quit {
        let now = Instant::now();

        if now.duration_since(last_telemetry_refresh) >= telemetry_refresh_interval {
            app.telemetry = telemetry_hw.sample_runtime_telemetry();
            last_telemetry_refresh = now;
            needs_redraw = true;
        }

        if let Some(evt_rx) = worker_evt_rx.as_ref() {
            loop {
                match evt_rx.try_recv() {
                    Ok(evt) => {
                        apply_worker_event(
                            &mut app,
                            evt,
                            &mut tool_index_by_id,
                            &mut rewind_manager,
                        );
                        needs_redraw = true;
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        // Worker died — force busy off so UI doesn't hang.
                        if app.busy {
                            app.busy = false;
                            app.busy_started_at = None;
                            app.status = "worker encerrado inesperadamente".to_string();
                            needs_redraw = true;
                        }
                        break;
                    }
                }
            }
        }

        if let Some(task) = download_task.as_mut() {
            while let Ok(progress) = task.progress_rx.try_recv() {
                if let Some(setup) = app.model_setup.as_mut() {
                    let percent = if progress.total > 0 {
                        (progress.downloaded as f64 / progress.total as f64 * 100.0) as u32
                    } else {
                        0
                    };
                    setup.progress_line = Some(format!(
                        "{}% · {} / {} · {} MB/s · ~{}s",
                        percent,
                        format_size(progress.downloaded),
                        format_size(progress.total),
                        progress.speed_bps / (1024 * 1024),
                        progress.eta_seconds
                    ));
                    setup.download_progress = Some(DownloadProgressView {
                        filename: task.quant_filename.clone(),
                        downloaded: progress.downloaded,
                        total: progress.total,
                        speed_bps: progress.speed_bps,
                        eta_seconds: progress.eta_seconds,
                    });
                    needs_redraw = true;
                }
            }

            if task.handle.is_finished() {
                needs_redraw = true;
                let completed = download_task.take().expect("download task should exist");
                let model_id = completed.model_id.clone();
                let quant_name = completed.quant_name.clone();
                let result = completed
                    .handle
                    .await
                    .map_err(|e| anyhow!("Tarefa de download falhou: {e}"))?;

                match result {
                    Ok(_path) => {
                        let model = find_model(&model_id).unwrap_or_else(default_model);
                        if let Err(err) =
                            enforce_single_quant_cache(&NcConfig::models_dir(), model, &quant_name)
                        {
                            if let Some(setup) = app.model_setup.as_mut() {
                                setup.downloading = false;
                                setup.error_line = Some(format!(
                                    "Falha ao limpar variantes anteriores em cache: {err}"
                                ));
                                setup.status_line =
                                    "Download concluído, mas a limpeza de cache falhou. Seleção não aplicada."
                                        .to_string();
                                setup.progress_line = None;
                                setup.download_progress = None;
                            }
                            continue;
                        }

                        let changed = config.active_model.as_deref() != Some(model_id.as_str())
                            || config.active_quant.as_deref() != Some(quant_name.as_str());
                        config.active_model = Some(model.id.to_string());
                        config.active_quant = Some(quant_name);
                        config.save()?;

                        app.needs_model_setup = false;
                        if app.setup_only {
                            should_quit = true;
                        } else {
                            if changed || worker_cmd_tx.is_none() {
                                if let Err(err) = restart_chat_worker(
                                    &mut worker_cmd_tx,
                                    &mut worker_evt_rx,
                                    &mut telemetry_hw,
                                    &mut bash_kill_signal,
                                    &mut app,
                                    config,
                                    ctk_override.clone(),
                                    ctv_override.clone(),
                                    interrupt_signal.clone(),
                                    None,
                                    Vec::new(),
                                )
                                .await
                                {
                                    if let Some(setup) = app.model_setup.as_mut() {
                                        setup.downloading = false;
                                        setup.error_line =
                                            Some(format!("Falha ao inicializar modelo: {err}"));
                                        setup.status_line =
                                            "Modelo pronto no disco, mas a inicialização falhou. Tente outra quantização."
                                                .to_string();
                                        setup.progress_line = None;
                                        setup.download_progress = None;
                                    }
                                    app.model_setup_can_cancel = true;
                                    app.screen = UiScreen::ModelSetup;
                                    app.status = "setup de modelo".to_string();
                                    app.busy = false;
                                    app.busy_started_at = None;
                                    continue;
                                }
                            }
                            app.model_setup = None;
                            app.model_setup_can_cancel = false;
                            app.screen = UiScreen::Chat;
                        }
                    }
                    Err(err) => {
                        if let Some(setup) = app.model_setup.as_mut() {
                            setup.downloading = false;
                            setup.error_line = Some(err.to_string());
                            setup.status_line =
                                "Falha no download. Selecione uma quantização e tente novamente."
                                    .to_string();
                            setup.progress_line = None;
                            setup.download_progress = None;
                        }
                    }
                }
            }
        }

        if app.busy || download_task.is_some() || matches!(app.screen, UiScreen::Welcome) {
            if now.duration_since(last_spinner_tick) >= spinner_interval {
                app.spinner_idx = app.spinner_idx.wrapping_add(1);
                last_spinner_tick = now;
                needs_redraw = true;
            }
        }

        if needs_redraw {
            draw_ui(&mut terminal, &app)?;
            needs_redraw = false;
        }

        let mut poll_timeout = if app.busy
            || app.pending_approval.is_some()
            || app.pending_user_question.is_some()
            || app.pending_plan_review.is_some()
            || app.pending_resume_selection.is_some()
        {
            poll_busy_interval
        } else {
            poll_idle_interval
        };

        if download_task.is_some() && poll_timeout > download_poll_interval {
            poll_timeout = download_poll_interval;
        }

        if app.busy {
            let elapsed_since_spinner = now.duration_since(last_spinner_tick);
            if elapsed_since_spinner < spinner_interval {
                let until_next_spinner = spinner_interval - elapsed_since_spinner;
                if until_next_spinner < poll_timeout {
                    poll_timeout = until_next_spinner;
                }
            } else {
                poll_timeout = Duration::from_millis(0);
            }
        }

        if event::poll(poll_timeout)? {
            match event::read()? {
                Event::Key(key) => match app.screen {
                    UiScreen::ModelSetup => {
                        match handle_model_setup_key(key, &mut app, config, &telemetry_hw) {
                            ModelSetupAction::Close => {
                                app.model_setup = None;
                                app.model_setup_can_cancel = false;
                                app.screen = UiScreen::Chat;
                                app.status = "pronto".to_string();
                                app.busy = false;
                                app.busy_started_at = None;
                            }
                            ModelSetupAction::Quit => {
                                should_quit = true;
                            }
                            ModelSetupAction::CancelDownload => {
                                if let Some(task) = download_task.take() {
                                    task.handle.abort();
                                }
                                if let Some(state) = app.model_setup.as_mut() {
                                    state.downloading = false;
                                    state.error_line = None;
                                    state.status_line =
                                        "Download cancelado. Arquivo parcial mantido para retomada."
                                            .to_string();
                                    state.progress_line = None;
                                    state.download_progress = None;
                                }
                            }
                            ModelSetupAction::ConfirmSelected => {
                                let Some(setup) = app.model_setup.as_ref() else {
                                    continue;
                                };
                                if setup.view != ModelSetupView::Variants {
                                    continue;
                                }
                                if setup.choices.is_empty() {
                                    if let Some(state) = app.model_setup.as_mut() {
                                        state.error_line =
                                            Some("Nenhuma quantização disponível".to_string());
                                    }
                                    continue;
                                }

                                let quant_name =
                                    setup.choices[setup.selected_idx].quant_name.clone();
                                let selected_was_cached = setup.choices[setup.selected_idx].cached;
                                let model = find_model(&setup.current_model_id)
                                    .unwrap_or_else(default_model);

                                if selected_was_cached {
                                    if let Err(err) = enforce_single_quant_cache(
                                        &NcConfig::models_dir(),
                                        model,
                                        &quant_name,
                                    ) {
                                        if let Some(state) = app.model_setup.as_mut() {
                                            state.error_line = Some(format!(
                                                "Falha ao limpar variantes anteriores em cache: {err}"
                                            ));
                                            state.status_line =
                                                "A limpeza de cache falhou. A variante não foi ativada."
                                                    .to_string();
                                            state.progress_line = None;
                                            state.download_progress = None;
                                        }
                                        continue;
                                    }

                                    let changed = config.active_model.as_deref() != Some(model.id)
                                        || config.active_quant.as_deref()
                                            != Some(quant_name.as_str());
                                    config.active_model = Some(model.id.to_string());
                                    config.active_quant = Some(quant_name.clone());
                                    config.save()?;
                                    app.needs_model_setup = false;

                                    if app.setup_only {
                                        should_quit = true;
                                    } else {
                                        if changed || worker_cmd_tx.is_none() {
                                            if let Err(err) = restart_chat_worker(
                                                &mut worker_cmd_tx,
                                                &mut worker_evt_rx,
                                                &mut telemetry_hw,
                                                &mut bash_kill_signal,
                                                &mut app,
                                                config,
                                                ctk_override.clone(),
                                                ctv_override.clone(),
                                                interrupt_signal.clone(),
                                                None,
                                                Vec::new(),
                                            )
                                            .await
                                            {
                                                if let Some(state) = app.model_setup.as_mut() {
                                                    state.downloading = false;
                                                    state.error_line = Some(format!(
                                                        "Falha ao inicializar modelo: {err}"
                                                    ));
                                                    state.status_line =
                                                        "Não foi possível ativar o modelo selecionado. Escolha outra quantização."
                                                            .to_string();
                                                    state.progress_line = None;
                                                    state.download_progress = None;
                                                }
                                                app.model_setup_can_cancel = true;
                                                app.screen = UiScreen::ModelSetup;
                                                app.status = "setup de modelo".to_string();
                                                app.busy = false;
                                                app.busy_started_at = None;
                                                continue;
                                            }
                                        } else {
                                            app.status = "pronto".to_string();
                                            app.busy = false;
                                            app.busy_started_at = None;
                                        }
                                        app.model_setup = None;
                                        app.model_setup_can_cancel = false;
                                        app.screen = UiScreen::Chat;
                                    }
                                } else {
                                    let task = start_download_task(model, &quant_name)?;
                                    if let Some(state) = app.model_setup.as_mut() {
                                        let filename = find_quant_by_name(model, &quant_name)
                                            .map(|q| q.filename.to_string())
                                            .unwrap_or_else(|| format!("{quant_name}.gguf"));
                                        state.downloading = true;
                                        state.status_line = format!(
                                            "Baixando {} ({}) do Hugging Face...",
                                            state.current_model_display_name, quant_name
                                        );
                                        state.progress_line =
                                            Some("Iniciando download...".to_string());
                                        state.download_progress = Some(DownloadProgressView {
                                            filename,
                                            downloaded: 0,
                                            total: 0,
                                            speed_bps: 0,
                                            eta_seconds: 0,
                                        });
                                        state.error_line = None;
                                    }
                                    download_task = Some(task);
                                }
                            }
                            ModelSetupAction::None => {}
                        }
                    }
                    UiScreen::Config => match handle_config_screen_key(key, &mut app) {
                        ConfigScreenAction::CloseNoSave => {
                            app.config_screen = None;
                            app.screen = UiScreen::Chat;
                            app.status = "pronto".to_string();
                            app.busy = false;
                            app.busy_started_at = None;
                        }
                        ConfigScreenAction::SaveAndClose => {
                            let Some(config_screen) = app.config_screen.as_ref() else {
                                app.screen = UiScreen::Chat;
                                continue;
                            };

                            let previous_config = config.clone();
                            let mut next_config = config.clone();
                            let changed =
                                apply_config_screen_changes(&mut next_config, config_screen);
                            if !changed {
                                app.config_screen = None;
                                app.screen = UiScreen::Chat;
                                app.chat.push(ChatItem::Assistant(
                                    "Configuração fechada (sem alterações).".to_string(),
                                ));
                                app.status = "pronto".to_string();
                                app.busy = false;
                                app.busy_started_at = None;
                                continue;
                            }

                            *config = next_config;
                            if let Err(err) = config.save() {
                                *config = previous_config;
                                if let Some(state) = app.config_screen.as_mut() {
                                    state.error_line = Some(format!(
                                        "Falha ao salvar arquivo de configuração: {err}"
                                    ));
                                    state.status_line =
                                        "Falha ao salvar. Ajuste as configurações e tente novamente."
                                            .to_string();
                                }
                                continue;
                            }

                            if let Err(err) = restart_chat_worker(
                                &mut worker_cmd_tx,
                                &mut worker_evt_rx,
                                &mut telemetry_hw,
                                &mut bash_kill_signal,
                                &mut app,
                                config,
                                ctk_override.clone(),
                                ctv_override.clone(),
                                interrupt_signal.clone(),
                                None,
                                Vec::new(),
                            )
                            .await
                            {
                                let restart_err = err.to_string();
                                *config = previous_config;
                                let _ = config.save();
                                let _ = restart_chat_worker(
                                    &mut worker_cmd_tx,
                                    &mut worker_evt_rx,
                                    &mut telemetry_hw,
                                    &mut bash_kill_signal,
                                    &mut app,
                                    config,
                                    ctk_override.clone(),
                                    ctv_override.clone(),
                                    interrupt_signal.clone(),
                                    None,
                                    Vec::new(),
                                )
                                .await;

                                if let Some(state) = app.config_screen.as_mut() {
                                    state.error_line = Some(format!(
                                        "Falha ao aplicar configuração: {restart_err}"
                                    ));
                                    state.status_line =
                                        "Configuração revertida. Ajuste os valores e tente novamente."
                                            .to_string();
                                }
                                continue;
                            }

                            app.config_screen = None;
                            app.screen = UiScreen::Chat;
                            app.chat.push(ChatItem::Assistant(
                                "Configuração salva e aplicada.".to_string(),
                            ));
                        }
                        ConfigScreenAction::None => {}
                    },
                    UiScreen::Welcome | UiScreen::Chat => {
                        should_quit = handle_key_event(
                            key,
                            &mut app,
                            &mut attachment_engine,
                            worker_cmd_tx.as_ref(),
                            &interrupt_signal,
                            &bash_kill_signal,
                            &mut tool_index_by_id,
                            &mut rewind_manager,
                        )
                        .await;

                        if let Some(resume_request) = app.requested_resume_session.take() {
                            match load_session_by_id_sync(
                                &NcConfig::sessions_dir(),
                                &resume_request.session_id_query,
                            ) {
                                Ok(Some((resolved_id, loaded_messages))) => {
                                    restore_chat_from_session_messages(
                                        &mut app,
                                        &loaded_messages,
                                        &resolved_id,
                                    );
                                    app.pending_resume_selection = None;
                                    app.input.clear();
                                    app.input_mode = InputMode::Default;
                                    clear_pending_pastes(&mut app);
                                    refresh_input_suggestions(&mut app, &mut attachment_engine);
                                    tool_index_by_id.clear();

                                    match restart_chat_worker(
                                        &mut worker_cmd_tx,
                                        &mut worker_evt_rx,
                                        &mut telemetry_hw,
                                        &mut bash_kill_signal,
                                        &mut app,
                                        config,
                                        ctk_override.clone(),
                                        ctv_override.clone(),
                                        interrupt_signal.clone(),
                                        Some(resolved_id.clone()),
                                        loaded_messages,
                                    )
                                    .await
                                    {
                                        Ok(()) => {
                                            app.status = format!(
                                                "sessão retomada: {}",
                                                short_session_id(&resolved_id)
                                            );
                                        }
                                        Err(err) => {
                                            app.chat.push(ChatItem::Error(format!(
                                                "Falha ao retomar sessão `{}`: {}",
                                                resolved_id, err
                                            )));
                                            app.status = "falha ao retomar".to_string();
                                        }
                                    }
                                }
                                Ok(None) => {
                                    app.chat.push(ChatItem::Error(format!(
                                        "Sessão `{}` não encontrada.",
                                        resume_request.session_id_query
                                    )));
                                    app.status = "sessão não encontrada".to_string();
                                }
                                Err(err) => {
                                    app.chat.push(ChatItem::Error(format!(
                                        "Falha ao carregar sessão `{}`: {}",
                                        resume_request.session_id_query, err
                                    )));
                                    app.status = "falha ao retomar".to_string();
                                }
                            }
                        }

                        if app.open_settings_requested {
                            app.open_settings_requested = false;
                            app.config_screen = Some(build_config_screen_state(
                                config,
                                &telemetry_hw,
                                ctk_override.clone(),
                                ctv_override.clone(),
                                &app.model_label,
                                app.max_context_tokens,
                            ));
                            app.model_setup = None;
                            app.model_setup_can_cancel = false;
                            app.screen = UiScreen::Config;
                            app.status = "configuração".to_string();
                            app.busy = false;
                            app.busy_started_at = None;
                        }

                        if app.open_model_setup_requested {
                            app.open_model_setup_requested = false;
                            app.model_setup = Some(build_model_setup_state(config, &telemetry_hw));
                            app.config_screen = None;
                            app.model_setup_can_cancel = true;
                            app.screen = UiScreen::ModelSetup;
                            app.needs_model_setup = false;
                            app.status = "setup de modelo".to_string();
                            app.busy = false;
                            app.busy_started_at = None;
                        }

                        if app.compact_requested {
                            app.compact_requested = false;
                            if let Some(cmd_tx) = worker_cmd_tx.as_ref() {
                                let _ = cmd_tx.send(WorkerCommand::Compact).await;
                            }
                        }

                        if app.reload_requested {
                            app.reload_requested = false;
                            match NcConfig::load() {
                                Ok(new_config) => {
                                    *config = new_config;
                                    refresh_skill_catalog(&mut app, config);
                                    app.mcp_servers_count = config.mcp_servers.len();
                                    refresh_input_suggestions(&mut app, &mut attachment_engine);
                                    match restart_chat_worker(
                                        &mut worker_cmd_tx,
                                        &mut worker_evt_rx,
                                        &mut telemetry_hw,
                                        &mut bash_kill_signal,
                                        &mut app,
                                        config,
                                        ctk_override.clone(),
                                        ctv_override.clone(),
                                        interrupt_signal.clone(),
                                        None,
                                        Vec::new(),
                                    )
                                    .await
                                    {
                                        Ok(()) => {
                                            app.chat.push(ChatItem::Assistant(
                                                "Configuração recarregada e aplicada.".to_string(),
                                            ));
                                        }
                                        Err(err) => {
                                            app.chat.push(ChatItem::Error(format!(
                                                "Falha ao recarregar: {err}"
                                            )));
                                        }
                                    }
                                }
                                Err(err) => {
                                    app.chat.push(ChatItem::Error(format!(
                                        "Falha ao carregar configuração: {err}"
                                    )));
                                    app.status = "falha ao recarregar".to_string();
                                }
                            }
                        }

                        if let Some(switch_request) = app.requested_agent_switch.take() {
                            let next_agent = switch_request.target;
                            let bootstrap_prompt = switch_request.bootstrap_prompt;
                            let previous_agent = app.active_agent;
                            app.active_agent = next_agent;

                            if app.needs_model_setup {
                                app.chat.push(ChatItem::Assistant(format!(
                                    "Agente definido para `{}`. Será aplicado após o setup do modelo.",
                                    agent_status_label(app.active_agent)
                                )));
                                app.status = format!(
                                    "pronto · agente {}",
                                    agent_status_label(app.active_agent)
                                );
                                app.busy = false;
                                app.busy_started_at = None;
                                continue;
                            }

                            match restart_chat_worker(
                                &mut worker_cmd_tx,
                                &mut worker_evt_rx,
                                &mut telemetry_hw,
                                &mut bash_kill_signal,
                                &mut app,
                                config,
                                ctk_override.clone(),
                                ctv_override.clone(),
                                interrupt_signal.clone(),
                                None,
                                Vec::new(),
                            )
                            .await
                            {
                                Ok(()) => {
                                    if let Some(prompt) = bootstrap_prompt {
                                        if let Some(cmd_tx) = worker_cmd_tx.as_ref() {
                                            let _ = cmd_tx
                                                .send(WorkerCommand::Submit {
                                                    prompt,
                                                    image_data_urls: Vec::new(),
                                                })
                                                .await;
                                            app.status =
                                                "Modo implementação iniciado com contexto exclusivo do plano."
                                                    .to_string();
                                        } else {
                                            app.status = format!(
                                                "pronto · agente {}",
                                                agent_status_label(app.active_agent)
                                            );
                                        }
                                    } else {
                                        app.status = format!(
                                            "pronto · agente {}",
                                            agent_status_label(app.active_agent)
                                        );
                                    }
                                }
                                Err(err) => {
                                    app.active_agent = previous_agent;
                                    app.chat.push(ChatItem::Error(format!(
                                        "Falha ao trocar agente para `{}`: {}",
                                        next_agent.as_str(),
                                        err
                                    )));
                                }
                            }
                        }
                    }
                },
                Event::Mouse(mouse) => match mouse.kind {
                    MouseEventKind::ScrollUp => {
                        app.chat_scroll = app.chat_scroll.saturating_add(3);
                    }
                    MouseEventKind::ScrollDown => {
                        app.chat_scroll = app.chat_scroll.saturating_sub(3);
                    }
                    _ => continue,
                },
                Event::Paste(pasted) => {
                    if matches!(app.screen, UiScreen::Welcome | UiScreen::Chat)
                        && !app.busy
                        && app.pending_approval.is_none()
                        && app.pending_user_question.is_none()
                        && app.pending_plan_review.is_none()
                        && app.pending_resume_selection.is_none()
                    {
                        apply_text_paste(&mut app, &pasted, &mut attachment_engine);
                    }
                }
                _ => {}
            }
            needs_redraw = true;
        }
    }

    if let Some(cmd_tx) = worker_cmd_tx.as_ref() {
        let _ = cmd_tx.send(WorkerCommand::Shutdown).await;
    }
    let session_resume_hint = app.current_session_id.clone();
    restore_terminal(terminal)?;
    if let Some(session_id) = session_resume_hint {
        println!();
        println!(
            "A sessão pode ser retomada com 'nanocode --resume {}'.",
            session_id
        );
    }
    Ok(())
}

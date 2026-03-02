use anyhow::{anyhow, Result};
use crossterm::event::{
    self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEvent, MouseEventKind,
};
use nanocode_core::agents::{AgentPolicy, BuiltinAgent};
use nanocode_core::config::ToolPermissionConfig;
use nanocode_core::NcConfig;
use nanocode_hf::{
    default_model, enforce_single_quant_cache, find_any_installed_model_quant, find_model,
    find_quant_by_name, list_cached_quants, models, recommend, recommend_runtime_limits,
    DownloadProgress, Downloader, HardwareInfo, ModelSpec,
};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

mod commands;
mod render;
mod runtime;
mod state;
mod stream;
mod terminal;
mod worker;

use commands::{
    apply_selected_slash_suggestion, handle_local_command, is_exit_command,
    refresh_slash_suggestions, resolve_slash_prompt_body,
};
use render::draw_ui;
use runtime::{build_runtime, runtime_snapshot};
use state::{
    AppState, ApprovalOption, ChatItem, ConfigField, ConfigFieldKey, ConfigScreenState,
    DownloadProgressView, InputMode, ModelChoice, ModelSetupState, ModelSetupView, PendingApproval,
    SetupChoice, ToolState, UiScreen,
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

fn apply_worker_event(
    app: &mut AppState,
    evt: WorkerEvent,
    tool_index_by_id: &mut HashMap<String, usize>,
) {
    match evt {
        WorkerEvent::Ready { model_label } => {
            app.model_label = model_label;
            app.busy = false;
            app.busy_started_at = None;
            app.status = format!("ready · agent {}", app.active_agent.as_str());
        }
        WorkerEvent::Busy(v) => {
            app.busy = v;
            if v {
                app.status = "thinking".to_string();
                app.busy_started_at = Some(Instant::now());
                app.thinking_idx = None;
                app.stream_idx = None;
            } else {
                app.status = format!("ready · agent {}", app.active_agent.as_str());
                app.busy_started_at = None;
                app.stream_idx = None;
                app.pending_approval = None;
                if let Some(idx) = app.thinking_idx {
                    if let Some(ChatItem::Thinking { active, .. }) = app.chat.get_mut(idx) {
                        *active = false;
                    }
                }
                app.thinking_idx = None;
            }
        }
        WorkerEvent::Interrupted => {
            app.status = "interrupted".to_string();
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
        WorkerEvent::AssistantDone(final_text) => {
            if let Some(idx) = app.stream_idx {
                if let Some(ChatItem::Assistant(text)) = app.chat.get_mut(idx) {
                    if !final_text.trim().is_empty() {
                        *text = final_text;
                    }
                }
            } else if !final_text.trim().is_empty() {
                app.chat.push(ChatItem::Assistant(final_text));
            }
        }
        WorkerEvent::ToolCall { call_id, summary } => {
            if let Some(idx) = app.thinking_idx {
                if let Some(ChatItem::Thinking { active, .. }) = app.chat.get_mut(idx) {
                    *active = false;
                }
            }
            app.thinking_idx = None;
            app.chat.push(ChatItem::Tool {
                summary,
                stream: Some("running".to_string()),
                detail: None,
                state: ToolState::Running,
            });
            tool_index_by_id.insert(call_id, app.chat.len() - 1);
        }
        WorkerEvent::ApprovalRequired {
            call_id,
            summary,
            arguments,
            decision_tx,
        } => {
            app.status = "awaiting approval".to_string();
            app.pending_approval = Some(PendingApproval {
                call_id,
                summary,
                arguments,
                decision_tx,
                selected_option: ApprovalOption::ApproveOnce,
            });
        }
        WorkerEvent::ToolResult {
            call_id,
            success,
            status_line,
            result,
        } => {
            if let Some(idx) = tool_index_by_id.get(&call_id).copied() {
                if let Some(ChatItem::Tool {
                    stream,
                    detail,
                    state,
                    ..
                }) = app.chat.get_mut(idx)
                {
                    *stream = status_line;
                    *detail = Some(result);
                    *state = if success {
                        ToolState::Ok
                    } else {
                        ToolState::Error
                    };
                }
            }
        }
        WorkerEvent::CompactStart {
            old_context_tokens,
            threshold,
        } => {
            app.status = "auto-compacting context".to_string();
            app.chat.push(ChatItem::Assistant(format!(
                "[compact] Triggered at {} tokens (threshold {}).",
                old_context_tokens, threshold
            )));
        }
        WorkerEvent::CompactEnd {
            old_context_tokens,
            new_context_tokens,
            summary_len,
        } => {
            app.chat.push(ChatItem::Assistant(format!(
                "[compact] Completed: {} -> {} tokens (summary {} chars).",
                old_context_tokens, new_context_tokens, summary_len
            )));
            app.status = format!("ready · agent {}", app.active_agent.as_str());
        }
        WorkerEvent::StoppedByMiddleware { reason } => {
            app.status = "stopped by middleware".to_string();
            app.chat
                .push(ChatItem::Error(format!("Middleware stop: {}", reason)));
        }
        WorkerEvent::Stats(stats) => {
            app.stats = stats;
        }
        WorkerEvent::Error(err) => {
            app.chat.push(ChatItem::Error(err));
        }
    }
}

async fn submit_prompt(
    app: &mut AppState,
    cmd_tx: &mpsc::Sender<WorkerCommand>,
    tool_index_by_id: &mut HashMap<String, usize>,
) -> bool {
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

    app.input_mode = InputMode::Default;
    app.input.clear();
    app.thinking_collapsed = true;
    refresh_slash_suggestions(app);

    app.chat.push(ChatItem::User(prompt.clone()));

    if was_slash {
        return handle_local_command(app, &prompt, tool_index_by_id);
    }

    if is_exit_command(&prompt) {
        return true;
    }

    let _ = cmd_tx.send(WorkerCommand::Submit(prompt)).await;
    false
}

async fn handle_key_event(
    key: KeyEvent,
    app: &mut AppState,
    cmd_tx: Option<&mpsc::Sender<WorkerCommand>>,
    interrupt_signal: &Arc<AtomicBool>,
    tool_index_by_id: &mut HashMap<String, usize>,
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
            (KeyCode::Left, _) | (KeyCode::Up, _) => {
                pending.selected_option = pending.selected_option.prev();
                None
            }
            (KeyCode::Right, _) | (KeyCode::Down, _) | (KeyCode::Tab, _) => {
                pending.selected_option = pending.selected_option.next();
                None
            }
            (KeyCode::Enter, _) => Some(pending.selected_option.to_decision()),
            _ => None,
        };

        if let Some(decision) = decision {
            let _ = pending.decision_tx.send(decision);
            app.status = match decision {
                nanocode_core::ApprovalDecision::ApproveAlwaysToolSession => {
                    "running tool (tool set to always for this session)".to_string()
                }
                nanocode_core::ApprovalDecision::ApproveOnce => "running tool".to_string(),
                nanocode_core::ApprovalDecision::Deny => "tool denied".to_string(),
            };
        } else {
            app.pending_approval = Some(pending);
        }
        return false;
    }

    match (key.code, key.modifiers) {
        (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
            if app.input.is_empty() {
                return true;
            }
            app.input.clear();
            app.input_mode = InputMode::Default;
            refresh_slash_suggestions(app);
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
            if app.busy && app.thinking_idx.is_some() {
                app.thinking_collapsed = !app.thinking_collapsed;
            } else if app.input_mode == InputMode::Slash {
                app.slash_details_expanded = !app.slash_details_expanded;
            } else {
                app.tools_collapsed = !app.tools_collapsed;
            }
        }
        (KeyCode::PageUp, _) => {
            app.chat_scroll = app.chat_scroll.saturating_add(5);
        }
        (KeyCode::PageDown, _) => {
            app.chat_scroll = app.chat_scroll.saturating_sub(5);
        }
        (KeyCode::Up, KeyModifiers::SHIFT) => {
            app.chat_scroll = app.chat_scroll.saturating_add(5);
        }
        (KeyCode::Down, KeyModifiers::SHIFT) => {
            app.chat_scroll = app.chat_scroll.saturating_sub(5);
        }
        (KeyCode::Esc, _) if app.busy => {
            interrupt_signal.store(true, Ordering::Relaxed);
            app.status = "interrupting".to_string();
        }
        (KeyCode::Esc, _) if !app.busy && app.input_mode == InputMode::Slash => {
            if app.input.is_empty() {
                app.input_mode = InputMode::Default;
                app.slash_details_expanded = false;
            }
            refresh_slash_suggestions(app);
        }
        (KeyCode::Char('l'), KeyModifiers::CONTROL) if !app.busy => {
            app.chat.clear();
            app.chat.push(ChatItem::Banner);
            tool_index_by_id.clear();
            app.stream_idx = None;
            app.thinking_idx = None;
        }
        (KeyCode::Up, _) if !app.busy && app.input_mode == InputMode::Slash => {
            if !app.slash_suggestions.is_empty() {
                let count = app.slash_suggestions.len();
                app.slash_selected = (app.slash_selected + count - 1) % count;
            }
        }
        (KeyCode::Down, _) if !app.busy && app.input_mode == InputMode::Slash => {
            if !app.slash_suggestions.is_empty() {
                let count = app.slash_suggestions.len();
                app.slash_selected = (app.slash_selected + 1) % count;
            }
        }
        (KeyCode::Tab, _) if !app.busy && app.input_mode == InputMode::Slash => {
            apply_selected_slash_suggestion(app);
        }
        (KeyCode::Enter, KeyModifiers::CONTROL | KeyModifiers::ALT) if !app.busy => {
            app.input.push('\n');
        }
        (KeyCode::Enter, _) if !app.busy => {
            if let Some(tx) = cmd_tx {
                if submit_prompt(app, tx, tool_index_by_id).await {
                    return true;
                }
            } else {
                app.chat.push(ChatItem::Error(
                    "Model is not initialized yet. Complete setup first.".to_string(),
                ));
            }
        }
        (KeyCode::Backspace, _) if !app.busy => {
            if app.input.is_empty() {
                app.input_mode = InputMode::Default;
            } else {
                app.input.pop();
            }
            refresh_slash_suggestions(app);
        }
        (KeyCode::Char(ch), KeyModifiers::NONE | KeyModifiers::SHIFT) if !app.busy => {
            if ch == '/' && app.input.is_empty() && app.input_mode == InputMode::Default {
                app.input_mode = InputMode::Slash;
                app.slash_details_expanded = false;
                refresh_slash_suggestions(app);
            } else {
                app.input.push(ch);
                refresh_slash_suggestions(app);
            }
        }
        _ => {}
    }

    false
}

fn handle_mouse_event(mouse: MouseEvent, app: &mut AppState) {
    if app.screen != UiScreen::Chat {
        return;
    }

    match mouse.kind {
        MouseEventKind::ScrollUp => {
            app.chat_scroll = app.chat_scroll.saturating_add(3);
        }
        MouseEventKind::ScrollDown => {
            app.chat_scroll = app.chat_scroll.saturating_sub(3);
        }
        _ => {}
    }
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

    model
        .quantizations
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
        setup.status_line = "No models available in catalog.".to_string();
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
        setup.status_line = "No models available in catalog.".to_string();
        return;
    }

    let Some(model) = setup.models.get(setup.selected_model_idx) else {
        setup.status_line = "Select a model.".to_string();
        return;
    };

    setup.status_line = match setup.view {
        ModelSetupView::Models => {
            if let Some(active_quant) = &model.active_quant {
                format!(
                    "Selected: {} · active {} · {} cached variant(s).",
                    model.model_display_name,
                    active_quant,
                    model.cached_quants.len()
                )
            } else if model.cached_quants.is_empty() {
                format!(
                    "Selected: {}. No cached variants yet. Press Enter to choose one.",
                    model.model_display_name
                )
            } else {
                format!(
                    "Selected: {} · {} cached variant(s). Press Enter to choose variant.",
                    model.model_display_name,
                    model.cached_quants.len()
                )
            }
        }
        ModelSetupView::Variants => {
            if let Some(choice) = setup.choices.get(setup.selected_idx) {
                if choice.cached {
                    format!(
                        "Variant {} is cached. Enter activates it and removes other cached variants.",
                        choice.quant_name
                    )
                } else {
                    format!(
                        "Variant {} is not cached. Enter downloads and activates it.",
                        choice.quant_name
                    )
                }
            } else {
                "No quantizations available for selected model.".to_string()
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
        ToolPermissionConfig::Always => "always",
        ToolPermissionConfig::Never => "never",
        ToolPermissionConfig::Ask => "ask",
    }
}

fn parse_permission(value: &str) -> ToolPermissionConfig {
    match value {
        "always" => ToolPermissionConfig::Always,
        "never" => ToolPermissionConfig::Never,
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
        "Unsaved changes. Press Esc to save and close.".to_string()
    } else {
        "No pending changes. Press Esc to return to chat.".to_string()
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
        .unwrap_or_else(|| "Auto".to_string());
    let kv_k_value = config
        .model
        .kv_cache_type_k
        .clone()
        .unwrap_or_else(|| "Auto".to_string());
    let kv_v_value = config
        .model
        .kv_cache_type_v
        .clone()
        .unwrap_or_else(|| "Auto".to_string());

    let mut fields = vec![
        ConfigField {
            key: ConfigFieldKey::NGpuLayers,
            label: "GPU Layers".to_string(),
            value: if config.model.n_gpu_layers == 0 {
                "CPU only (0)".to_string()
            } else {
                "Auto (-1)".to_string()
            },
            initial_value: if config.model.n_gpu_layers == 0 {
                "CPU only (0)".to_string()
            } else {
                "Auto (-1)".to_string()
            },
            options: vec!["Auto (-1)".to_string(), "CPU only (0)".to_string()],
        },
        ConfigField {
            key: ConfigFieldKey::ContextSize,
            label: "Context Size".to_string(),
            value: context_value.clone(),
            initial_value: context_value,
            options: vec![
                "Auto".to_string(),
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
            label: "KV Cache K".to_string(),
            value: kv_k_value.clone(),
            initial_value: kv_k_value,
            options: vec![
                "Auto".to_string(),
                "q8_0".to_string(),
                "q4_0".to_string(),
                "f16".to_string(),
            ],
        },
        ConfigField {
            key: ConfigFieldKey::KvCacheTypeV,
            label: "KV Cache V".to_string(),
            value: kv_v_value.clone(),
            initial_value: kv_v_value,
            options: vec![
                "Auto".to_string(),
                "q8_0".to_string(),
                "q4_0".to_string(),
                "f16".to_string(),
            ],
        },
        ConfigField {
            key: ConfigFieldKey::BashPermission,
            label: "bash permission".to_string(),
            value: permission_to_value(&config.tools.bash.permission).to_string(),
            initial_value: permission_to_value(&config.tools.bash.permission).to_string(),
            options: vec!["ask".to_string(), "always".to_string(), "never".to_string()],
        },
        ConfigField {
            key: ConfigFieldKey::WriteFilePermission,
            label: "write_file permission".to_string(),
            value: permission_to_value(&config.tools.write_file.permission).to_string(),
            initial_value: permission_to_value(&config.tools.write_file.permission).to_string(),
            options: vec!["ask".to_string(), "always".to_string(), "never".to_string()],
        },
        ConfigField {
            key: ConfigFieldKey::GrepPermission,
            label: "grep permission".to_string(),
            value: permission_to_value(&config.tools.grep.permission).to_string(),
            initial_value: permission_to_value(&config.tools.grep.permission).to_string(),
            options: vec!["ask".to_string(), "always".to_string(), "never".to_string()],
        },
        ConfigField {
            key: ConfigFieldKey::ReadFilePermission,
            label: "read_file permission".to_string(),
            value: permission_to_value(&config.tools.read_file.permission).to_string(),
            initial_value: permission_to_value(&config.tools.read_file.permission).to_string(),
            options: vec!["ask".to_string(), "always".to_string(), "never".to_string()],
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
        status_line: "No pending changes. Press Esc to return to chat.".to_string(),
        error_line: None,
    }
}

fn apply_config_screen_changes(config: &mut NcConfig, state: &ConfigScreenState) -> bool {
    let mut changed = false;

    for field in &state.fields {
        match field.key {
            ConfigFieldKey::NGpuLayers => {
                let value = if field.value == "CPU only (0)" { 0 } else { -1 };
                if config.model.n_gpu_layers != value {
                    config.model.n_gpu_layers = value;
                    changed = true;
                }
            }
            ConfigFieldKey::ContextSize => {
                let value = if field.value == "Auto" {
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
                let value = if field.value == "Auto" {
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
                let value = if field.value == "Auto" {
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
) -> Result<(
    mpsc::Sender<WorkerCommand>,
    Receiver<WorkerEvent>,
    HardwareInfo,
    u32,
)> {
    let runtime = build_runtime(config, ctk_override, ctv_override)
        .map_err(|e| anyhow!("Failed to initialize runtime: {e}"))?;
    let telemetry_hw = runtime.hardware.clone();
    let max_context_tokens = runtime.config.model.context_size.unwrap_or(200_000);

    let (cmd_tx, cmd_rx) = mpsc::channel::<WorkerCommand>(8);
    let agent_policy = AgentPolicy::from_builtin(active_agent);
    let evt_rx = spawn_worker(runtime, agent_policy, cmd_rx, interrupt_signal);
    Ok((cmd_tx, evt_rx, telemetry_hw, max_context_tokens))
}

fn start_download_task(model: &'static ModelSpec, quant_name: &str) -> Result<DownloadTask> {
    let quant = find_quant_by_name(model, quant_name)
        .ok_or_else(|| anyhow!("Unknown quantization selected: {}", quant_name))?;

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
    app: &mut AppState,
    config: &NcConfig,
    ctk_override: Option<String>,
    ctv_override: Option<String>,
    interrupt_signal: Arc<AtomicBool>,
) -> Result<()> {
    if let Some(cmd_tx) = worker_cmd_tx.take() {
        let _ = cmd_tx.send(WorkerCommand::Shutdown).await;
    }
    *worker_evt_rx = None;
    interrupt_signal.store(false, Ordering::Relaxed);

    let (cmd_tx, evt_rx, hw, max_context_tokens) = start_chat_worker(
        config,
        ctk_override,
        ctv_override,
        app.active_agent,
        interrupt_signal,
    )?;
    *worker_cmd_tx = Some(cmd_tx);
    *worker_evt_rx = Some(evt_rx);
    *telemetry_hw = hw;
    app.max_context_tokens = max_context_tokens;
    app.status = "initializing model...".to_string();
    app.busy = true;
    app.busy_started_at = Some(Instant::now());
    Ok(())
}

pub async fn run_tui(
    config: &mut NcConfig,
    ctk_override: Option<String>,
    ctv_override: Option<String>,
    setup_only: bool,
    initial_agent: BuiltinAgent,
) -> Result<()> {
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
    app.setup_only = setup_only;
    app.needs_model_setup = setup_only || !has_installed_model;
    if app.needs_model_setup {
        app.model_setup = Some(build_model_setup_state(config, &initial_hw));
        app.model_setup_can_cancel = false;
        app.status = "setup required".to_string();
        app.busy = false;
        app.busy_started_at = None;
        if setup_only {
            app.screen = UiScreen::ModelSetup;
        }
    }

    let mut worker_cmd_tx: Option<mpsc::Sender<WorkerCommand>> = None;
    let mut worker_evt_rx: Option<Receiver<WorkerEvent>> = None;
    let interrupt_signal = Arc::new(AtomicBool::new(false));
    let mut telemetry_hw = initial_hw;

    if !app.needs_model_setup {
        restart_chat_worker(
            &mut worker_cmd_tx,
            &mut worker_evt_rx,
            &mut telemetry_hw,
            &mut app,
            config,
            ctk_override.clone(),
            ctv_override.clone(),
            interrupt_signal.clone(),
        )
        .await?;
    }

    let mut terminal = setup_terminal()?;
    let mut should_quit = false;
    let mut tool_index_by_id: HashMap<String, usize> = HashMap::new();
    let mut last_telemetry_refresh = Instant::now()
        .checked_sub(Duration::from_secs(2))
        .unwrap_or_else(Instant::now);
    let mut download_task: Option<DownloadTask> = None;

    while !should_quit {
        if last_telemetry_refresh.elapsed() >= Duration::from_millis(900) {
            app.telemetry = telemetry_hw.sample_runtime_telemetry();
            last_telemetry_refresh = Instant::now();
        }

        if let Some(evt_rx) = worker_evt_rx.as_ref() {
            while let Ok(evt) = evt_rx.try_recv() {
                apply_worker_event(&mut app, evt, &mut tool_index_by_id);
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
                }
            }

            if task.handle.is_finished() {
                let completed = download_task.take().expect("download task should exist");
                let model_id = completed.model_id.clone();
                let quant_name = completed.quant_name.clone();
                let result = completed
                    .handle
                    .await
                    .map_err(|e| anyhow!("Download task failed: {e}"))?;

                match result {
                    Ok(_path) => {
                        let model = find_model(&model_id).unwrap_or_else(default_model);
                        if let Err(err) =
                            enforce_single_quant_cache(&NcConfig::models_dir(), model, &quant_name)
                        {
                            if let Some(setup) = app.model_setup.as_mut() {
                                setup.downloading = false;
                                setup.error_line = Some(format!(
                                    "Failed to clean previous cached variants: {err}"
                                ));
                                setup.status_line =
                                    "Download finished, but cache cleanup failed. Selection not applied."
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
                                    &mut app,
                                    config,
                                    ctk_override.clone(),
                                    ctv_override.clone(),
                                    interrupt_signal.clone(),
                                )
                                .await
                                {
                                    if let Some(setup) = app.model_setup.as_mut() {
                                        setup.downloading = false;
                                        setup.error_line =
                                            Some(format!("Failed to initialize model: {err}"));
                                        setup.status_line =
                                            "Model ready on disk, but runtime failed to start. Try another quantization."
                                                .to_string();
                                        setup.progress_line = None;
                                        setup.download_progress = None;
                                    }
                                    app.model_setup_can_cancel = true;
                                    app.screen = UiScreen::ModelSetup;
                                    app.status = "model setup".to_string();
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
                                "Download failed. Select a quantization and try again.".to_string();
                            setup.progress_line = None;
                            setup.download_progress = None;
                        }
                    }
                }
            }
        }

        app.spinner_idx = app.spinner_idx.wrapping_add(1);
        draw_ui(&mut terminal, &app)?;

        if event::poll(Duration::from_millis(50))? {
            match event::read()? {
                Event::Key(key) => match app.screen {
                    UiScreen::ModelSetup => {
                        match handle_model_setup_key(key, &mut app, config, &telemetry_hw) {
                            ModelSetupAction::Close => {
                                app.model_setup = None;
                                app.model_setup_can_cancel = false;
                                app.screen = UiScreen::Chat;
                                app.status = "ready".to_string();
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
                                        "Download canceled. Partial file kept for resume."
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
                                            Some("No quantizations available".to_string());
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
                                                "Failed to clean previous cached variants: {err}"
                                            ));
                                            state.status_line =
                                                "Cache cleanup failed. Variant was not activated."
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
                                                &mut app,
                                                config,
                                                ctk_override.clone(),
                                                ctv_override.clone(),
                                                interrupt_signal.clone(),
                                            )
                                            .await
                                            {
                                                if let Some(state) = app.model_setup.as_mut() {
                                                    state.downloading = false;
                                                    state.error_line = Some(format!(
                                                        "Failed to initialize model: {err}"
                                                    ));
                                                    state.status_line =
                                                        "Could not activate selected model. Pick another quantization."
                                                            .to_string();
                                                    state.progress_line = None;
                                                    state.download_progress = None;
                                                }
                                                app.model_setup_can_cancel = true;
                                                app.screen = UiScreen::ModelSetup;
                                                app.status = "model setup".to_string();
                                                app.busy = false;
                                                app.busy_started_at = None;
                                                continue;
                                            }
                                        } else {
                                            app.status = "ready".to_string();
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
                                            "Downloading {} ({}) from Hugging Face...",
                                            state.current_model_display_name, quant_name
                                        );
                                        state.progress_line =
                                            Some("Starting download...".to_string());
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
                            app.status = "ready".to_string();
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
                                    "Configuration closed (no changes).".to_string(),
                                ));
                                app.status = "ready".to_string();
                                app.busy = false;
                                app.busy_started_at = None;
                                continue;
                            }

                            *config = next_config;
                            if let Err(err) = config.save() {
                                *config = previous_config;
                                if let Some(state) = app.config_screen.as_mut() {
                                    state.error_line =
                                        Some(format!("Failed to save config file: {err}"));
                                    state.status_line =
                                        "Save failed. Fix settings and try again.".to_string();
                                }
                                continue;
                            }

                            if let Err(err) = restart_chat_worker(
                                &mut worker_cmd_tx,
                                &mut worker_evt_rx,
                                &mut telemetry_hw,
                                &mut app,
                                config,
                                ctk_override.clone(),
                                ctv_override.clone(),
                                interrupt_signal.clone(),
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
                                    &mut app,
                                    config,
                                    ctk_override.clone(),
                                    ctv_override.clone(),
                                    interrupt_signal.clone(),
                                )
                                .await;

                                if let Some(state) = app.config_screen.as_mut() {
                                    state.error_line = Some(format!(
                                        "Failed to apply configuration: {restart_err}"
                                    ));
                                    state.status_line =
                                        "Configuration reverted. Adjust values and try again."
                                            .to_string();
                                }
                                continue;
                            }

                            app.config_screen = None;
                            app.screen = UiScreen::Chat;
                            app.chat.push(ChatItem::Assistant(
                                "Configuration saved and applied.".to_string(),
                            ));
                        }
                        ConfigScreenAction::None => {}
                    },
                    UiScreen::Welcome | UiScreen::Chat => {
                        should_quit = handle_key_event(
                            key,
                            &mut app,
                            worker_cmd_tx.as_ref(),
                            &interrupt_signal,
                            &mut tool_index_by_id,
                        )
                        .await;

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
                            app.status = "configuration".to_string();
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
                            app.status = "model setup".to_string();
                            app.busy = false;
                            app.busy_started_at = None;
                        }

                        if let Some(next_agent) = app.requested_agent_switch.take() {
                            let previous_agent = app.active_agent;
                            app.active_agent = next_agent;

                            if app.needs_model_setup {
                                app.chat.push(ChatItem::Assistant(format!(
                                    "Agent set to `{}`. It will be applied after model setup.",
                                    app.active_agent.as_str()
                                )));
                                app.status = format!("ready · agent {}", app.active_agent.as_str());
                                app.busy = false;
                                app.busy_started_at = None;
                                continue;
                            }

                            match restart_chat_worker(
                                &mut worker_cmd_tx,
                                &mut worker_evt_rx,
                                &mut telemetry_hw,
                                &mut app,
                                config,
                                ctk_override.clone(),
                                ctv_override.clone(),
                                interrupt_signal.clone(),
                            )
                            .await
                            {
                                Ok(()) => {
                                    app.chat.push(ChatItem::Assistant(format!(
                                        "Active agent: `{}` (session scope).",
                                        app.active_agent.as_str()
                                    )));
                                }
                                Err(err) => {
                                    app.active_agent = previous_agent;
                                    app.chat.push(ChatItem::Error(format!(
                                        "Failed to switch agent to `{}`: {}",
                                        next_agent.as_str(),
                                        err
                                    )));
                                }
                            }
                        }
                    }
                },
                Event::Mouse(mouse) => handle_mouse_event(mouse, &mut app),
                _ => {}
            }
        }
    }

    if let Some(cmd_tx) = worker_cmd_tx.as_ref() {
        let _ = cmd_tx.send(WorkerCommand::Shutdown).await;
    }
    restore_terminal(terminal)?;
    Ok(())
}

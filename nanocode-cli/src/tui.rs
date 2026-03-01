use anyhow::Result;
use crossterm::event::{
    self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEvent, MouseEventKind,
};
use nanocode_core::NcConfig;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
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
use runtime::build_runtime;
use state::{AppState, ChatItem, InputMode, ToolState, UiScreen};
use terminal::{restore_terminal, setup_terminal};
use worker::{spawn_worker, WorkerCommand, WorkerEvent};

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
            app.status = "ready".to_string();
        }
        WorkerEvent::Busy(v) => {
            app.busy = v;
            if v {
                app.status = "thinking".to_string();
                app.busy_started_at = Some(Instant::now());
                app.thinking_idx = None;
                app.stream_idx = None;
            } else {
                app.status = "ready".to_string();
                app.busy_started_at = None;
                app.stream_idx = None;
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
    cmd_tx: &mpsc::Sender<WorkerCommand>,
    interrupt_signal: &Arc<AtomicBool>,
    tool_index_by_id: &mut HashMap<String, usize>,
) -> bool {
    if key.kind != KeyEventKind::Press {
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
            app.screen = UiScreen::Chat;
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
            if submit_prompt(app, cmd_tx, tool_index_by_id).await {
                return true;
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

pub async fn run_tui(
    config: &NcConfig,
    ctk_override: Option<String>,
    ctv_override: Option<String>,
) -> Result<()> {
    let runtime = build_runtime(config, ctk_override, ctv_override);
    let telemetry_hw = runtime.hardware.clone();
    let max_context_tokens = runtime.config.model.context_size.unwrap_or(200_000);

    let (cmd_tx, cmd_rx) = mpsc::channel::<WorkerCommand>(8);
    let interrupt_signal = Arc::new(AtomicBool::new(false));
    let evt_rx = spawn_worker(runtime, cmd_rx, interrupt_signal.clone());

    let mut terminal = setup_terminal()?;
    let mut app = AppState::new(max_context_tokens, telemetry_hw.sample_runtime_telemetry());
    let mut should_quit = false;
    let mut tool_index_by_id: HashMap<String, usize> = HashMap::new();
    let mut last_telemetry_refresh =
        Instant::now().checked_sub(Duration::from_secs(2)).unwrap_or_else(Instant::now);

    while !should_quit {
        if last_telemetry_refresh.elapsed() >= Duration::from_millis(900) {
            app.telemetry = telemetry_hw.sample_runtime_telemetry();
            last_telemetry_refresh = Instant::now();
        }

        while let Ok(evt) = evt_rx.try_recv() {
            apply_worker_event(&mut app, evt, &mut tool_index_by_id);
        }

        app.spinner_idx = app.spinner_idx.wrapping_add(1);
        draw_ui(&mut terminal, &app)?;

        if event::poll(Duration::from_millis(50))? {
            match event::read()? {
                Event::Key(key) => {
                    should_quit = handle_key_event(
                        key,
                        &mut app,
                        &cmd_tx,
                        &interrupt_signal,
                        &mut tool_index_by_id,
                    )
                    .await;
                }
                Event::Mouse(mouse) => handle_mouse_event(mouse, &mut app),
                _ => {}
            }
        }
    }

    let _ = cmd_tx.send(WorkerCommand::Shutdown).await;
    restore_terminal(terminal)?;
    Ok(())
}

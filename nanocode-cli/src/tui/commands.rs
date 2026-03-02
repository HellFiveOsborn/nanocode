use std::collections::HashMap;

use nanocode_core::agents::BuiltinAgent;
use nanocode_core::NcConfig;

use super::state::{AppState, ChatItem, InputMode, SlashCommandEntry};

pub const MAX_SLASH_SUGGESTIONS: usize = 8;

const SLASH_COMMANDS: [SlashCommandEntry; 14] = [
    SlashCommandEntry {
        alias: "/help",
        description: "Show help message",
    },
    SlashCommandEntry {
        alias: "/config",
        description: "Open configuration settings",
    },
    SlashCommandEntry {
        alias: "/model",
        description: "Open model/quantization setup",
    },
    SlashCommandEntry {
        alias: "/setup",
        description: "Alias for /model",
    },
    SlashCommandEntry {
        alias: "/reload",
        description: "Reload configuration from disk",
    },
    SlashCommandEntry {
        alias: "/clear",
        description: "Clear conversation history",
    },
    SlashCommandEntry {
        alias: "/log",
        description: "Show sessions directory path",
    },
    SlashCommandEntry {
        alias: "/compact",
        description: "Compact conversation history",
    },
    SlashCommandEntry {
        alias: "/terminal-setup",
        description: "Show Shift+Enter setup hint",
    },
    SlashCommandEntry {
        alias: "/status",
        description: "Display agent statistics",
    },
    SlashCommandEntry {
        alias: "/agent",
        description: "Show or switch active agent profile",
    },
    SlashCommandEntry {
        alias: "/proxy-setup",
        description: "Show proxy setup hint",
    },
    SlashCommandEntry {
        alias: "/quit",
        description: "Exit the application",
    },
    SlashCommandEntry {
        alias: "/exit",
        description: "Exit the application",
    },
];

pub fn is_exit_command(input: &str) -> bool {
    matches!(
        input.trim().to_ascii_lowercase().as_str(),
        "/quit" | "/exit"
    )
}

fn slash_query(input: &str) -> &str {
    input
        .trim_start()
        .trim_start_matches('/')
        .split_whitespace()
        .next()
        .unwrap_or("")
}

pub fn slash_command_suggestions(input: &str) -> Vec<SlashCommandEntry> {
    let query = slash_query(input).to_ascii_lowercase();
    let mut matches: Vec<SlashCommandEntry> = SLASH_COMMANDS
        .iter()
        .copied()
        .filter(|entry| {
            if query.is_empty() {
                return true;
            }
            entry.alias.trim_start_matches('/').starts_with(&query)
        })
        .collect();
    matches.sort_by_key(|entry| entry.alias);
    matches
}

pub fn refresh_slash_suggestions(app: &mut AppState) {
    if app.input_mode != InputMode::Slash {
        app.slash_suggestions.clear();
        app.slash_selected = 0;
        return;
    }

    app.slash_suggestions = slash_command_suggestions(&app.input);
    if app.slash_suggestions.is_empty() {
        app.slash_selected = 0;
    } else if app.slash_selected >= app.slash_suggestions.len() {
        app.slash_selected = 0;
    }
}

pub fn apply_selected_slash_suggestion(app: &mut AppState) -> bool {
    let Some(selected) = app.slash_suggestions.get(app.slash_selected).copied() else {
        return false;
    };
    app.input = selected.alias.trim_start_matches('/').to_string();
    refresh_slash_suggestions(app);
    true
}

pub fn resolve_slash_prompt_body(app: &AppState, prompt_body: &str) -> String {
    let Some(selected) = app.slash_suggestions.get(app.slash_selected).copied() else {
        return prompt_body.to_string();
    };

    let trailing = prompt_body
        .split_once(char::is_whitespace)
        .map(|(_, rest)| rest.trim_start())
        .unwrap_or("");

    let selected_body = selected.alias.trim_start_matches('/');
    if trailing.is_empty() {
        selected_body.to_string()
    } else {
        format!("{} {}", selected_body, trailing)
    }
}

fn slash_help_text() -> String {
    let mut lines = vec![
        "### Commands".to_string(),
        "".to_string(),
        "Use `/` to open command mode. Press `Tab` to autocomplete and `↑/↓` to navigate."
            .to_string(),
        "".to_string(),
    ];
    for entry in SLASH_COMMANDS {
        lines.push(format!("- `{}`: {}", entry.alias, entry.description));
    }
    lines.join("\n")
}

fn slash_status_text(app: &AppState) -> String {
    format!(
        "Status: {}\nAgent: {}\nModel: {}\nSession messages: {}\nContext tokens: {}\nTotal tokens: {}",
        app.status,
        app.active_agent.as_str(),
        app.model_label,
        app.chat.len(),
        app.stats.context_tokens,
        app.stats.tokens_used
    )
}

pub fn handle_local_command(
    app: &mut AppState,
    prompt: &str,
    tool_index_by_id: &mut HashMap<String, usize>,
) -> bool {
    let command = prompt
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();

    match command.as_str() {
        "/help" => {
            app.chat.push(ChatItem::Assistant(slash_help_text()));
        }
        "/clear" => {
            app.chat.clear();
            app.chat.push(ChatItem::Banner);
            tool_index_by_id.clear();
        }
        "/status" => {
            app.chat.push(ChatItem::Assistant(slash_status_text(app)));
        }
        "/agent" => {
            let mut parts = prompt.split_whitespace();
            let _ = parts.next();
            if let Some(target_name) = parts.next() {
                if let Some(target_agent) = BuiltinAgent::parse(target_name) {
                    if target_agent == app.active_agent {
                        app.chat.push(ChatItem::Assistant(format!(
                            "Agent `{}` is already active.",
                            target_agent.as_str()
                        )));
                    } else {
                        app.requested_agent_switch = Some(target_agent);
                        app.chat.push(ChatItem::Assistant(format!(
                            "Switching agent to `{}` for this session...",
                            target_agent.as_str()
                        )));
                    }
                } else {
                    app.chat.push(ChatItem::Error(format!(
                        "Unknown agent `{}`. Available: {}",
                        target_name,
                        BuiltinAgent::available_names().join(", ")
                    )));
                }
            } else {
                app.chat.push(ChatItem::Assistant(format!(
                    "Active agent: `{}`. Available: {}",
                    app.active_agent.as_str(),
                    BuiltinAgent::available_names().join(", ")
                )));
            }
        }
        "/model" | "/models" => {
            app.open_model_setup_requested = true;
            app.chat
                .push(ChatItem::Assistant("Opening model setup...".to_string()));
        }
        "/setup" => {
            app.open_model_setup_requested = true;
            app.chat
                .push(ChatItem::Assistant("Opening model setup...".to_string()));
        }
        "/config" => {
            app.open_settings_requested = true;
            app.chat.push(ChatItem::Assistant(
                "Opening configuration settings...".to_string(),
            ));
        }
        "/reload" => {
            app.chat.push(ChatItem::Assistant(
                "Reload is not available yet in the ratatui port.".to_string(),
            ));
        }
        "/compact" => {
            app.chat.push(ChatItem::Assistant(
                "Compact is not available yet in the ratatui port.".to_string(),
            ));
        }
        "/log" => {
            app.chat.push(ChatItem::Assistant(format!(
                "Sessions directory: {}",
                NcConfig::sessions_dir().display()
            )));
        }
        "/terminal-setup" => {
            app.chat.push(ChatItem::Assistant(
                "Use Shift+Enter or Ctrl+Enter to insert a new line.".to_string(),
            ));
        }
        "/proxy-setup" => {
            app.chat.push(ChatItem::Assistant(
                "Set HTTP_PROXY / HTTPS_PROXY in your shell environment.".to_string(),
            ));
        }
        cmd if is_exit_command(cmd) => return true,
        _ => {
            app.chat.push(ChatItem::Error(format!(
                "Unknown command: {}. Type /help.",
                prompt.trim()
            )));
        }
    }

    false
}

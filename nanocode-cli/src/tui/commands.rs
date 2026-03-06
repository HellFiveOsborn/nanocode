use std::collections::HashMap;

use nanocode_core::agents::BuiltinAgent;
use nanocode_core::session::list_sessions_sync;
use nanocode_core::NcConfig;

use super::rewind::{RewindManager, RewindOutcome};
use super::state::{
    AgentSwitchRequest, AppState, ChatItem, InputMode, PendingResumeSelection,
    SessionResumeRequest, SlashCommandEntry,
};

pub const MAX_SLASH_SUGGESTIONS: usize = 8;

const BUILTIN_SLASH_COMMANDS: [(&str, &str); 17] = [
    ("/help", "Mostrar ajuda e atalhos de teclado"),
    ("/plan", "Trocar para modo plan (somente leitura)"),
    ("/build", "Trocar para modo build (implementação)"),
    ("/agent", "Mostrar ou trocar agente (default/plan/build)"),
    ("/model", "Abrir setup de modelo/quantização"),
    ("/config", "Abrir configurações"),
    ("/compact", "Compactar histórico da conversa"),
    ("/clear", "Limpar histórico da conversa"),
    (
        "/rewind",
        "Reverter a última edição de arquivo desta sessão",
    ),
    ("/status", "Mostrar estatísticas do agente e da sessão"),
    ("/log", "Mostrar caminho da pasta de sessões"),
    ("/resume", "Listar e retomar uma sessão anterior"),
    ("/continue", "Alias de /resume"),
    ("/reload", "Recarregar configuração do disco"),
    ("/setup", "Alias de /model"),
    ("/terminal-setup", "Configurar Shift+Enter para nova linha"),
    ("/quit", "Sair da aplicação (/exit)"),
];

pub struct SkillInvocation {
    pub skill_name: String,
    pub model_prompt: String,
}

pub fn is_exit_command(input: &str) -> bool {
    matches!(
        input.trim().to_ascii_lowercase().as_str(),
        "/quit" | "/exit"
    )
}

pub fn is_builtin_command(input: &str) -> bool {
    let command = slash_command_name(input);
    BUILTIN_SLASH_COMMANDS
        .iter()
        .any(|(alias, _)| alias.eq_ignore_ascii_case(command.as_str()))
        || is_exit_command(&command)
}

fn slash_command_name(input: &str) -> String {
    input
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase()
}

fn slash_query(input: &str) -> &str {
    input
        .trim_start()
        .trim_start_matches('/')
        .split_whitespace()
        .next()
        .unwrap_or("")
}

fn builtin_command_entries() -> Vec<SlashCommandEntry> {
    BUILTIN_SLASH_COMMANDS
        .iter()
        .map(|(alias, description)| SlashCommandEntry {
            alias: (*alias).to_string(),
            description: (*description).to_string(),
            is_skill: false,
        })
        .collect()
}

pub fn slash_command_suggestions(app: &AppState, input: &str) -> Vec<SlashCommandEntry> {
    let query = slash_query(input).to_ascii_lowercase();
    let mut entries = builtin_command_entries();

    entries.extend(
        app.skills
            .values()
            .filter(|skill| skill.user_invocable)
            .map(|skill| SlashCommandEntry {
                alias: format!("/{}", skill.name),
                description: skill.description.clone(),
                is_skill: true,
            }),
    );

    let mut matches: Vec<SlashCommandEntry> = entries
        .into_iter()
        .filter(|entry| {
            if query.is_empty() {
                return true;
            }
            entry.alias.trim_start_matches('/').starts_with(&query)
        })
        .collect();

    matches.sort_by(|a, b| a.alias.cmp(&b.alias));
    matches
}

pub fn refresh_slash_suggestions(app: &mut AppState) {
    if app.input_mode != InputMode::Slash {
        app.slash_suggestions.clear();
        app.slash_selected = 0;
        return;
    }

    app.slash_suggestions = slash_command_suggestions(app, &app.input);
    if app.slash_suggestions.is_empty() {
        app.slash_selected = 0;
    } else if app.slash_selected >= app.slash_suggestions.len() {
        app.slash_selected = 0;
    }
}

pub fn apply_selected_slash_suggestion(app: &mut AppState) -> bool {
    let Some(selected) = app.slash_suggestions.get(app.slash_selected).cloned() else {
        return false;
    };
    app.input = selected.alias.trim_start_matches('/').to_string();
    refresh_slash_suggestions(app);
    true
}

pub fn resolve_slash_prompt_body(app: &AppState, prompt_body: &str) -> String {
    let Some(selected) = app.slash_suggestions.get(app.slash_selected).cloned() else {
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

pub fn try_build_skill_prompt(
    app: &AppState,
    slash_prompt: &str,
) -> Result<Option<SkillInvocation>, String> {
    let trimmed = slash_prompt.trim();
    if !trimmed.starts_with('/') {
        return Ok(None);
    }

    let body = trimmed.trim_start_matches('/');
    let mut parts = body.splitn(2, char::is_whitespace);
    let skill_name = parts.next().unwrap_or("").trim().to_ascii_lowercase();
    if skill_name.is_empty() {
        return Ok(None);
    }

    let Some(skill) = app.skills.get(&skill_name) else {
        return Ok(None);
    };

    if !skill.user_invocable {
        return Ok(None);
    }

    let skill_content = std::fs::read_to_string(&skill.skill_path)
        .map_err(|err| format!("Falha ao ler skill '{}': {}", skill.name, err))?;

    let trailing = parts.next().map(str::trim).unwrap_or("");
    let model_prompt = if trailing.is_empty() {
        skill_content
    } else {
        format!("{skill_content}\n\n## Contexto de Invocação da Skill\nArgs do usuário: {trailing}")
    };

    Ok(Some(SkillInvocation {
        skill_name,
        model_prompt,
    }))
}

fn slash_help_text(app: &AppState) -> String {
    let mut lines = vec![
        "### Atalhos de Teclado".to_string(),
        "".to_string(),
        "- `Ctrl+C` / `Esc`: interromper operação atual (tool call ou geração)".to_string(),
        "- `Ctrl+D`: sair do nanocode".to_string(),
        "- `Ctrl+T`: mostrar/ocultar tasklist".to_string(),
        "- `Ctrl+O`: expandir/recolher saída (ferramentas, thinking, código)".to_string(),
        "- `Shift+Tab`: alternar modo do agente (default → plan → build)".to_string(),
        "- `Alt+T`: ligar/desligar raciocínio do modelo".to_string(),
        "- `Enter`: enviar mensagem".to_string(),
        "- `Shift+Enter`: nova linha".to_string(),
        "- `Ctrl+V`: colar (texto ou imagem)".to_string(),
        "- `Tab`: autocompletar em opções".to_string(),
        "- `Esc`: cancelar em menus".to_string(),
        "- `↑/↓`: navegar sugestões/opções".to_string(),
        "- `←/→`: navegar tabs de questionários".to_string(),
        "- `@arquivo` / `@pasta|busca`: anexar contexto de arquivos/pastas".to_string(),
        "- `Ctrl+L`: limpar histórico do chat".to_string(),
        "- `?` (input vazio, Default/Build): alternar modo YOLO".to_string(),
        "- `Shift+↑/↓`, `PageUp/PageDown`: rolar histórico do chat".to_string(),
        "".to_string(),
        "### Comandos Slash".to_string(),
        "".to_string(),
        "Digite `/` para abrir o menu de comandos. Use `↑/↓` para navegar, `Tab` para completar e `Enter` para enviar."
            .to_string(),
        "Use `/resume` para listar sessões e pressione `↑/↓` + `Enter` para retomar uma."
            .to_string(),
        "".to_string(),
    ];

    for (alias, description) in BUILTIN_SLASH_COMMANDS {
        lines.push(format!("- `{}` — {}", alias, description));
    }

    let mut invocable_skills: Vec<_> = app
        .skills
        .values()
        .filter(|skill| skill.user_invocable)
        .collect();
    invocable_skills.sort_by(|a, b| a.name.cmp(&b.name));

    if !invocable_skills.is_empty() {
        lines.push("".to_string());
        lines.push("### Skills".to_string());
        lines.push("".to_string());
        for skill in invocable_skills {
            lines.push(format!("- `/{}` — {}", skill.name, skill.description));
        }
    }

    lines.join("\n")
}

fn slash_status_text(app: &AppState) -> String {
    let yolo = if app.yolo_mode { " (YOLO)" } else { "" };
    format!(
        "### Status\n\n- Agente: `{}{}`\n- Modelo: `{}`\n- Mensagens da sessão: {}\n- Turnos: {}\n- Tokens de contexto: {} / {}\n- Total de tokens usados: {}\n- Ferramentas chamadas: {}\n- Skills: {}",
        app.active_agent.as_str(),
        yolo,
        app.model_label,
        app.chat.len(),
        app.stats.turns,
        app.stats.context_tokens,
        app.max_context_tokens,
        app.stats.tokens_used,
        app.stats.tools_called,
        app.skills_count,
    )
}

fn available_agent_names_for_display() -> String {
    BuiltinAgent::available_names().join(", ")
}

fn switch_agent(app: &mut AppState, target: BuiltinAgent) {
    if target == app.active_agent {
        app.chat.push(ChatItem::Assistant(format!(
            "O agente `{}` já está ativo.",
            target.as_str()
        )));
        return;
    }
    if target == BuiltinAgent::Plan {
        app.yolo_mode = false;
    }
    app.requested_agent_switch = Some(AgentSwitchRequest {
        target,
        bootstrap_prompt: None,
    });
    app.chat.push(ChatItem::Assistant(format!(
        "Trocando para o agente `{}` nesta sessão...",
        target.as_str()
    )));
}

pub fn handle_local_command(
    app: &mut AppState,
    prompt: &str,
    tool_index_by_id: &mut HashMap<String, usize>,
    rewind_manager: &mut RewindManager,
) -> bool {
    let command = prompt
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();

    match command.as_str() {
        "/help" => {
            app.chat.push(ChatItem::Assistant(slash_help_text(app)));
        }
        "/clear" => {
            app.chat.clear();
            app.chat.push(ChatItem::Banner);
            tool_index_by_id.clear();
        }
        "/rewind" => match rewind_manager.rewind_once() {
            Ok(RewindOutcome::Empty) => {
                app.chat.push(ChatItem::Assistant(
                    "Ainda não há nada para desfazer com /rewind nesta sessão.".to_string(),
                ));
                app.status = "rewind: sem snapshots disponíveis".to_string();
            }
            Ok(RewindOutcome::Applied { files, remaining }) => {
                let mut preview = files
                    .iter()
                    .take(3)
                    .map(|path| format!("`{}`", path))
                    .collect::<Vec<_>>()
                    .join(", ");
                if files.len() > 3 {
                    preview.push_str(&format!(", +{} a mais", files.len() - 3));
                }
                app.chat.push(ChatItem::Assistant(format!(
                    "Rewind aplicado em {} arquivo(s): {}",
                    files.len(),
                    preview
                )));
                app.status = format!("rewind concluído · {} snapshot(s) restantes", remaining);
            }
            Err(err) => {
                app.chat
                    .push(ChatItem::Error(format!("Falha no rewind: {err}")));
                app.status = "falha no rewind".to_string();
            }
        },
        "/status" => {
            app.chat.push(ChatItem::Assistant(slash_status_text(app)));
        }
        "/agent" => {
            let mut parts = prompt.split_whitespace();
            let _ = parts.next();
            if let Some(target_name) = parts.next() {
                if let Some(target_agent) = BuiltinAgent::parse(target_name) {
                    if target_agent == BuiltinAgent::Explore {
                        app.chat.push(ChatItem::Error(
                            "Explore é um subagente interno. Use default, plan ou build."
                                .to_string(),
                        ));
                    } else {
                        switch_agent(app, target_agent);
                    }
                } else {
                    app.chat.push(ChatItem::Error(format!(
                        "Agente `{}` desconhecido. Disponíveis: {}",
                        target_name,
                        available_agent_names_for_display()
                    )));
                }
            } else {
                app.chat.push(ChatItem::Assistant(format!(
                    "Agente ativo: `{}`. Disponíveis: {}",
                    app.active_agent.as_str(),
                    available_agent_names_for_display()
                )));
            }
        }
        "/plan" => {
            switch_agent(app, BuiltinAgent::Plan);
        }
        "/build" => {
            switch_agent(app, BuiltinAgent::Build);
        }
        "/model" | "/models" => {
            app.open_model_setup_requested = true;
            app.chat.push(ChatItem::Assistant(
                "Abrindo setup de modelo...".to_string(),
            ));
        }
        "/setup" => {
            app.open_model_setup_requested = true;
            app.chat.push(ChatItem::Assistant(
                "Abrindo setup de modelo...".to_string(),
            ));
        }
        "/config" => {
            app.open_settings_requested = true;
            app.chat
                .push(ChatItem::Assistant("Abrindo configurações...".to_string()));
        }
        "/reload" => {
            app.reload_requested = true;
            app.chat.push(ChatItem::Assistant(
                "Recarregando configuração do disco...".to_string(),
            ));
        }
        "/compact" => {
            app.compact_requested = true;
            // The CompactStart event from the worker will push the ChatItem::Compact.
        }
        "/log" => {
            app.chat.push(ChatItem::Assistant(format!(
                "Pasta de sessões: {}",
                NcConfig::sessions_dir().display()
            )));
        }
        "/resume" | "/continue" => {
            let mut parts = prompt.split_whitespace();
            let _ = parts.next();
            if let Some(session_id_query) = parts.next() {
                app.pending_resume_selection = None;
                app.requested_resume_session = Some(SessionResumeRequest {
                    session_id_query: session_id_query.to_string(),
                });
                app.chat.push(ChatItem::Assistant(format!(
                    "Retomando sessão `{}`...",
                    session_id_query
                )));
                app.status = "retomando sessão...".to_string();
            } else {
                match list_sessions_sync(&NcConfig::sessions_dir()) {
                    Ok(sessions) if sessions.is_empty() => {
                        app.chat.push(ChatItem::Assistant(
                            "Nenhuma sessão anterior encontrada.".to_string(),
                        ));
                        app.status = "nenhuma sessão encontrada".to_string();
                    }
                    Ok(sessions) => {
                        app.pending_resume_selection = Some(PendingResumeSelection {
                            sessions,
                            selected_idx: 0,
                        });
                        app.status = "use ↑/↓ e Enter para retomar uma sessão".to_string();
                    }
                    Err(err) => {
                        app.chat
                            .push(ChatItem::Error(format!("Falha ao listar sessões: {err}")));
                        app.status = "falha ao retomar".to_string();
                    }
                }
            }
        }
        "/terminal-setup" => {
            app.chat.push(ChatItem::Assistant(
                "Use Shift+Enter para inserir uma nova linha.".to_string(),
            ));
        }
        cmd if is_exit_command(cmd) => return true,
        _ => {
            app.chat.push(ChatItem::Error(format!(
                "Comando desconhecido: {}. Digite /help.",
                prompt.trim()
            )));
        }
    }

    false
}

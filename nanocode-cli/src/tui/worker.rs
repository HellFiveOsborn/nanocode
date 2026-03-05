use nanocode_core::agent_loop::{AgentLoop, LoopEvent};
use nanocode_core::agents::{AgentPolicy, BuiltinAgent};
use nanocode_core::interrupt::{clear_interrupt_signal, is_user_interrupted_error};
use nanocode_core::llm::{LlmEngineHandle, LoadedModel, PromptFamily};
use nanocode_core::prompts::load_prompt;
use nanocode_core::session::SessionLogger;
use nanocode_core::skills::SkillManager;
use nanocode_core::tools::bash::{new_kill_signal, BashKillSignal};
use nanocode_core::tools::ToolManager;
use nanocode_core::types::{LlmMessage, MessageRole};
use nanocode_core::{AgentStats, ApprovalDecision, NcConfig, UserQuestionResponse};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{channel, Receiver, SyncSender};
use std::sync::Arc;
use tokio::sync::mpsc;

use super::rewind::RewindChange;
use super::runtime::{is_thinking_model, RuntimeEnv};
use super::stream::{extract_thinking_blocks_and_clean, StreamSanitizer};

const THINKING_PROMPT_BOOSTER: &str = r#"
## Thinking Discipline
- Keep reasoning private and brief.
- Emit reasoning only inside `<think>...</think>`.
- Put the final user-facing answer outside `<think>` blocks.
- Avoid meta narration ("hmm", "wait", "I should").
- If a tool is needed, call it quickly.
"#;

pub enum WorkerCommand {
    Submit {
        prompt: String,
        image_data_urls: Vec<String>,
    },
    Compact,
    SetAutoApprove(bool),
    Shutdown,
}

pub enum WorkerEvent {
    SessionReady {
        session_id: String,
        resumed: bool,
    },
    Ready {
        model_label: String,
        supports_thinking: bool,
        supports_vision: bool,
    },
    Busy(bool),
    Interrupted,
    ThinkingActive(bool),
    ThinkingDelta(String),
    AssistantChunk(String),
    AssistantDone(String),
    ToolCall {
        call_id: String,
        tool_name: String,
        summary: String,
    },
    ApprovalRequired {
        call_id: String,
        summary: String,
        details: Vec<String>,
        diff_preview: Option<String>,
        decision_tx: SyncSender<ApprovalDecision>,
    },
    QuestionRequired {
        call_id: String,
        question: String,
        choices: Vec<String>,
        allow_free_text: bool,
        placeholder: Option<String>,
        response_tx: SyncSender<UserQuestionResponse>,
    },
    ToolResult {
        call_id: String,
        success: bool,
        status_line: Option<String>,
        output: Option<String>,
        code_path: Option<String>,
        code: Option<String>,
        diff: Option<String>,
        rewind_changes: Vec<RewindChange>,
    },
    /// Subagent started a sub-tool call.
    SubagentToolCall {
        parent_call_id: String,
        summary: String,
    },
    /// Subagent completed a sub-tool call.
    SubagentToolResult {
        parent_call_id: String,
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

struct PendingToolCall {
    name: String,
    arguments: Value,
    path: Option<String>,
    before_exists: bool,
    before_bytes: Option<Vec<u8>>,
    before_content: Option<String>,
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

fn pretty_tool_name(name: &str) -> String {
    match name {
        "write_file" => "Escrever".to_string(),
        "read_file" => "Ler".to_string(),
        "search_replace" => "Atualizar".to_string(),
        "grep" => "Buscar".to_string(),
        "bash" => "Bash".to_string(),
        "task" => "Explorar".to_string(),
        "ask_user_question" => "Pergunta".to_string(),
        other => {
            let mut out = String::new();
            for part in other.split('_') {
                if part.is_empty() {
                    continue;
                }
                let mut chars = part.chars();
                if let Some(first) = chars.next() {
                    out.extend(first.to_uppercase());
                    out.extend(chars);
                }
            }
            out
        }
    }
}

fn tool_target(call_name: &str, arguments: &Value) -> String {
    match call_name {
        "bash" => arguments
            .get("command")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        "grep" => arguments
            .get("path")
            .and_then(Value::as_str)
            .or_else(|| arguments.get("pattern").and_then(Value::as_str))
            .unwrap_or("")
            .to_string(),
        "task" => arguments
            .get("task")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        "ask_user_question" => arguments
            .get("question")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        _ => arguments
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
    }
}

fn format_tool_summary(tool_name: &str, arguments: &Value) -> String {
    let target = short_preview(&tool_target(tool_name, arguments), 90);
    let display_name = pretty_tool_name(tool_name);
    if target.is_empty() {
        format!("{}()", display_name)
    } else {
        format!("{}({})", display_name, target)
    }
}

fn path_from_arguments(tool_name: &str, arguments: &Value) -> Option<String> {
    match tool_name {
        "read_file" | "write_file" | "search_replace" => arguments
            .get("path")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        _ => None,
    }
}

fn snapshot_file(path: &str) -> Option<String> {
    const MAX_SNAPSHOT_BYTES: u64 = 256 * 1024;
    let file_path = Path::new(path);
    let metadata = std::fs::metadata(file_path).ok()?;
    if metadata.len() > MAX_SNAPSHOT_BYTES {
        return None;
    }
    std::fs::read_to_string(file_path).ok()
}

fn snapshot_file_bytes(path: &str) -> Option<Vec<u8>> {
    std::fs::read(Path::new(path)).ok()
}

fn resolve_absolute_path(path: &str) -> String {
    let candidate = Path::new(path);
    let absolute = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(candidate)
    };
    absolute.to_string_lossy().to_string()
}

/// Build a diff preview for the approval modal (write_file / search_replace).
fn build_approval_diff(tool_name: &str, arguments: &Value) -> Option<String> {
    let path = arguments.get("path").and_then(Value::as_str)?;
    match tool_name {
        "write_file" => {
            let new_content = arguments.get("content").and_then(Value::as_str)?;
            let old_content = snapshot_file(path).unwrap_or_default();
            if old_content.is_empty() {
                // New file — show first lines as all-added
                let preview: String = new_content
                    .lines()
                    .take(60)
                    .map(|l| format!("     0 +{}\n", l))
                    .collect();
                if preview.is_empty() {
                    return None;
                }
                let total = new_content.lines().count();
                if total > 60 {
                    return Some(format!("{}       ... (+{} lines)", preview, total - 60));
                }
                Some(preview.trim_end().to_string())
            } else {
                build_unified_diff(&old_content, new_content, 3, 60)
            }
        }
        "search_replace" => {
            let search = arguments.get("search").and_then(Value::as_str)?;
            let replace = arguments.get("replace").and_then(Value::as_str)?;
            let old_content = snapshot_file(path)?;
            let global = arguments
                .get("global")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let new_content = if global {
                old_content.replace(search, replace)
            } else {
                old_content.replacen(search, replace, 1)
            };
            build_unified_diff(&old_content, &new_content, 3, 60)
        }
        _ => None,
    }
}

fn format_bool(v: bool) -> &'static str {
    if v {
        "sim"
    } else {
        "não"
    }
}

fn build_approval_details(tool_name: &str, arguments: &Value) -> Vec<String> {
    let mut details = Vec::new();
    match tool_name {
        "bash" => {
            if let Some(command) = arguments.get("command").and_then(Value::as_str) {
                details.push(format!("comando: {}", short_preview(command, 180)));
            }
            if let Some(workdir) = arguments.get("workdir").and_then(Value::as_str) {
                if !workdir.trim().is_empty() {
                    details.push(format!(
                        "diretório de trabalho: {}",
                        short_preview(workdir, 120)
                    ));
                }
            }
        }
        "read_file" => {
            if let Some(path) = arguments.get("path").and_then(Value::as_str) {
                details.push(format!("caminho: {}", short_preview(path, 160)));
            }
            if let Some(offset) = arguments.get("offset").and_then(Value::as_u64) {
                details.push(format!("deslocamento: {}", offset));
            }
            if let Some(limit) = arguments.get("limit").and_then(Value::as_u64) {
                details.push(format!("limite: {}", limit));
            }
        }
        "write_file" => {
            if let Some(path) = arguments.get("path").and_then(Value::as_str) {
                details.push(format!("caminho: {}", short_preview(path, 160)));
            }
            if let Some(content) = arguments.get("content").and_then(Value::as_str) {
                details.push(format!(
                    "conteúdo: {} linhas · {} caracteres",
                    content.lines().count(),
                    content.chars().count()
                ));
            }
            if let Some(append) = arguments.get("append").and_then(Value::as_bool) {
                details.push(format!("append: {}", format_bool(append)));
            }
        }
        "search_replace" => {
            if let Some(path) = arguments.get("path").and_then(Value::as_str) {
                details.push(format!("caminho: {}", short_preview(path, 160)));
            }
            if let Some(search) = arguments.get("search").and_then(Value::as_str) {
                details.push(format!("busca: {}", short_preview(search, 120)));
            }
            if let Some(replace) = arguments.get("replace").and_then(Value::as_str) {
                details.push(format!("substituição: {}", short_preview(replace, 120)));
            }
            let global = arguments
                .get("global")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            details.push(format!("global: {}", format_bool(global)));
        }
        "grep" => {
            if let Some(pattern) = arguments.get("pattern").and_then(Value::as_str) {
                details.push(format!("padrão: {}", short_preview(pattern, 140)));
            }
            if let Some(path) = arguments.get("path").and_then(Value::as_str) {
                details.push(format!("caminho: {}", short_preview(path, 140)));
            }
            if let Some(include) = arguments.get("include").and_then(Value::as_str) {
                details.push(format!("incluir: {}", short_preview(include, 100)));
            }
            if let Some(exclude) = arguments.get("exclude").and_then(Value::as_str) {
                details.push(format!("excluir: {}", short_preview(exclude, 100)));
            }
        }
        "ask_user_question" => {
            if let Some(question) = arguments.get("question").and_then(Value::as_str) {
                details.push(format!("pergunta: {}", short_preview(question, 180)));
            }
            if let Some(choices) = arguments.get("choices").and_then(Value::as_array) {
                if !choices.is_empty() {
                    details.push(format!("opções: {}", choices.len()));
                }
            }
            if let Some(allow_free_text) = arguments.get("allow_free_text").and_then(Value::as_bool)
            {
                details.push(format!(
                    "permitir_texto_livre: {}",
                    format_bool(allow_free_text)
                ));
            }
        }
        "task" => {
            if let Some(agent) = arguments.get("agent").and_then(Value::as_str) {
                details.push(format!("agente: {}", short_preview(agent, 80)));
            }
            if let Some(task) = arguments.get("task").and_then(Value::as_str) {
                details.push(format!("tarefa: {}", short_preview(task, 180)));
            }
        }
        _ => {
            let mut pairs = Vec::new();
            if let Some(obj) = arguments.as_object() {
                for (idx, (k, v)) in obj.iter().enumerate() {
                    if idx >= 4 {
                        break;
                    }
                    let preview = match v {
                        Value::String(s) => short_preview(s, 80),
                        _ => short_preview(&v.to_string(), 80),
                    };
                    pairs.push(format!("{}: {}", k, preview));
                }
            }
            details.extend(pairs);
        }
    }
    details
}

fn truncate_lines(text: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() <= max_lines {
        return text.to_string();
    }
    let mut out = lines[..max_lines].join("\n");
    out.push_str(&format!(
        "\n... (+{} linhas omitidas)",
        lines.len() - max_lines
    ));
    out
}

fn parse_read_output_content(result: &str) -> String {
    if let Some((head, tail)) = result.split_once('\n') {
        if head.to_ascii_lowercase().contains("lines from line") {
            return tail.to_string();
        }
    }
    result.to_string()
}

fn summarize_bash(command: &str, result: &str) -> String {
    let cmd = command.trim().to_ascii_lowercase();
    if cmd.starts_with("ls") {
        let entries = result
            .lines()
            .filter(|line| !line.trim().is_empty())
            .filter(|line| !line.trim_start().starts_with("total "))
            .count();
        return format!("Listou {} caminho(s)", entries);
    }
    let first = result.lines().next().unwrap_or("").trim();
    if first.is_empty() || first == "[Command completed with no output]" {
        "Comando concluído sem saída".to_string()
    } else {
        short_preview(first, 90)
    }
}

fn summarize_grep(result: &str) -> Option<String> {
    let first = result.lines().next()?.trim();
    if first.to_ascii_lowercase().starts_with("found ") {
        Some(first.to_string())
    } else {
        None
    }
}

fn summarize_ask_user_question(result: &str) -> Option<String> {
    let parsed = serde_json::from_str::<serde_json::Value>(result).ok()?;
    if parsed
        .get("cancelled")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        return Some("Usuário cancelou".to_string());
    }

    let source = parsed
        .get("source")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("text");
    let answer = parsed
        .get("answer")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .trim();
    if answer.is_empty() {
        return Some("Respondido".to_string());
    }

    let prefix = match source {
        "choice" => "Escolha",
        "text" => "Resposta",
        _ => "Resposta",
    };
    Some(format!("{}: {}", prefix, short_preview(answer, 80)))
}

fn render_ask_user_question_output(result: &str) -> Option<String> {
    let parsed = serde_json::from_str::<serde_json::Value>(result).ok()?;
    if parsed
        .get("cancelled")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        return Some("cancelado: sim".to_string());
    }

    let answer = parsed
        .get("answer")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .trim();
    let source = parsed
        .get("source")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("text");
    let choice_index = parsed
        .get("choice_index")
        .and_then(serde_json::Value::as_u64)
        .map(|v| (v + 1).to_string())
        .unwrap_or_else(|| "-".to_string());

    let mut lines = vec![
        format!("origem: {}", source),
        format!("escolha: {}", choice_index),
    ];
    if !answer.is_empty() {
        lines.push(format!("resposta: {}", short_preview(answer, 200)));
    }
    Some(lines.join("\n"))
}

fn render_grep_output(result: &str) -> Option<String> {
    // Keep summary header + first matches visible in expanded tool details.
    let trimmed = result.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(truncate_lines(trimmed, 40))
}

/// Build a unified diff with context lines and line numbers (Claude Code style).
///
/// Format per line:
/// - `{num:>6}  {content}` — context
/// - `{num:>6} -{content}` — removed (old line number)
/// - `{num:>6} +{content}` — added (new line number)
/// - `  ...` — hunk separator
fn build_unified_diff(
    before: &str,
    after: &str,
    context: usize,
    max_lines: usize,
) -> Option<String> {
    let old: Vec<&str> = before.lines().collect();
    let new: Vec<&str> = after.lines().collect();
    if old == new {
        return None;
    }

    // Build edit ops via greedy sequential matching with lookahead.
    enum Op {
        Equal((), usize), // (_, new_idx)
        Remove(usize),
        Add(usize),
    }

    let mut ops: Vec<Op> = Vec::new();
    let (mut oi, mut ni) = (0usize, 0usize);

    while oi < old.len() && ni < new.len() {
        if old[oi] == new[ni] {
            ops.push(Op::Equal((), ni));
            oi += 1;
            ni += 1;
            continue;
        }
        let max_look = 12;
        let mut found = false;
        for look in 1..=max_look {
            if ni + look < new.len() && old[oi] == new[ni + look] {
                for k in ni..ni + look {
                    ops.push(Op::Add(k));
                }
                ni += look;
                found = true;
                break;
            }
            if oi + look < old.len() && old[oi + look] == new[ni] {
                for k in oi..oi + look {
                    ops.push(Op::Remove(k));
                }
                oi += look;
                found = true;
                break;
            }
        }
        if !found {
            ops.push(Op::Remove(oi));
            ops.push(Op::Add(ni));
            oi += 1;
            ni += 1;
        }
    }
    while oi < old.len() {
        ops.push(Op::Remove(oi));
        oi += 1;
    }
    while ni < new.len() {
        ops.push(Op::Add(ni));
        ni += 1;
    }

    // Group change indices into hunks (merge nearby changes).
    let is_change = |op: &Op| !matches!(op, Op::Equal(_, _));
    let mut hunks: Vec<(usize, usize)> = Vec::new();
    let mut in_change = false;
    let mut cs = 0usize;

    for (i, op) in ops.iter().enumerate() {
        if is_change(op) {
            if !in_change {
                cs = i;
                in_change = true;
            }
        } else if in_change {
            if let Some(last) = hunks.last_mut() {
                if cs <= last.1 + context * 2 + 1 {
                    last.1 = i - 1;
                } else {
                    hunks.push((cs, i - 1));
                }
            } else {
                hunks.push((cs, i - 1));
            }
            in_change = false;
        }
    }
    if in_change {
        let end = ops.len() - 1;
        if let Some(last) = hunks.last_mut() {
            if cs <= last.1 + context * 2 + 1 {
                last.1 = end;
            } else {
                hunks.push((cs, end));
            }
        } else {
            hunks.push((cs, end));
        }
    }

    if hunks.is_empty() {
        return None;
    }

    // Render hunks with context.
    let mut out: Vec<String> = Vec::new();
    for (hi, &(start, end)) in hunks.iter().enumerate() {
        let ctx_start = start.saturating_sub(context);
        let ctx_end = (end + context + 1).min(ops.len());

        if hi > 0 {
            out.push("  ...".to_string());
        }

        for i in ctx_start..ctx_end {
            match &ops[i] {
                Op::Equal(_, ni) => {
                    out.push(format!("{:>6}  {}", ni + 1, new[*ni]));
                }
                Op::Remove(oi) => {
                    out.push(format!("{:>6} -{}", oi + 1, old[*oi]));
                }
                Op::Add(ni) => {
                    out.push(format!("{:>6} +{}", ni + 1, new[*ni]));
                }
            }
            if out.len() >= max_lines {
                out.push("  ...".to_string());
                return Some(out.join("\n"));
            }
        }
    }

    if out.is_empty() {
        None
    } else {
        Some(out.join("\n"))
    }
}

fn count_diff_changes(diff: &str) -> (usize, usize) {
    let mut additions = 0usize;
    let mut removals = 0usize;
    for line in diff.lines() {
        // Unified format: "{num:>6} {sign}{content}" — sign at byte index 7
        if line.len() >= 8 {
            match line.as_bytes()[7] {
                b'+' => additions += 1,
                b'-' => removals += 1,
                _ => {}
            }
        }
    }
    (additions, removals)
}

fn tool_status_line(
    tool: &PendingToolCall,
    success: bool,
    result: &str,
    diff: Option<&str>,
) -> String {
    if !success {
        let first = short_preview(result.lines().next().unwrap_or("Ferramenta falhou"), 90);
        return format!("Falhou: {}", first);
    }

    match tool.name.as_str() {
        "read_file" => {
            let content = parse_read_output_content(result);
            format!("Leu {} linhas", content.lines().count())
        }
        "write_file" => {
            if let Some(d) = diff {
                let (additions, removals) = count_diff_changes(d);
                if additions > 0 || removals > 0 {
                    return format!(
                        "Adicionou {} linhas, removeu {} linhas",
                        additions, removals
                    );
                }
            }
            if tool.before_exists {
                "Atualizado".to_string()
            } else {
                "Criado".to_string()
            }
        }
        "search_replace" => {
            if let Some(d) = diff {
                let (additions, removals) = count_diff_changes(d);
                if additions > 0 || removals > 0 {
                    return format!(
                        "Adicionou {} linhas, removeu {} linhas",
                        additions, removals
                    );
                }
            }
            "Atualizado".to_string()
        }
        "grep" => summarize_grep(result).unwrap_or_else(|| "Busca concluída".to_string()),
        "task" => {
            // Parse the JSON result to extract stats
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(result) {
                let tools = v.get("tools_called").and_then(|v| v.as_u64()).unwrap_or(0);
                let tokens = v.get("tokens_used").and_then(|v| v.as_u64()).unwrap_or(0);
                let completed = v
                    .get("completed")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let status = if completed {
                    "Concluído"
                } else {
                    "Interrompido"
                };
                let tokens_display = if tokens >= 1000 {
                    format!("{:.1}k tokens", tokens as f64 / 1000.0)
                } else {
                    format!("{} tokens", tokens)
                };
                format!(
                    "{} ({} usos de ferramenta · {})",
                    status, tools, tokens_display
                )
            } else {
                "Concluído".to_string()
            }
        }
        "bash" => {
            let command = tool
                .arguments
                .get("command")
                .and_then(Value::as_str)
                .unwrap_or("");
            summarize_bash(command, result)
        }
        "ask_user_question" => {
            summarize_ask_user_question(result).unwrap_or_else(|| "Respondido".to_string())
        }
        _ => short_preview(result.lines().next().unwrap_or("Concluído"), 90),
    }
}

pub fn spawn_worker(
    runtime: RuntimeEnv,
    agent_policy: AgentPolicy,
    mut cmd_rx: mpsc::Receiver<WorkerCommand>,
    interrupt_signal: Arc<AtomicBool>,
    resume_session_id: Option<String>,
    initial_messages: Vec<LlmMessage>,
) -> (Receiver<WorkerEvent>, BashKillSignal) {
    let (evt_tx, evt_rx) = channel::<WorkerEvent>();
    let bash_kill_signal = new_kill_signal();
    let bash_kill_signal_inner = bash_kill_signal.clone();

    tokio::spawn(async move {
        let effective_config = runtime.config.clone();

        // Load model once into memory (blocking). Reused for the entire session.
        let model_file = runtime.model_file.clone();
        let load_config = effective_config.clone();
        let loaded_model = tokio::task::spawn_blocking(move || {
            LoadedModel::load(&model_file, &load_config)
        })
        .await;
        let llm_engine: Option<Arc<LlmEngineHandle>> = match loaded_model {
            Ok(Ok(loaded)) => Some(Arc::new(LlmEngineHandle::from_loaded(loaded))),
            Ok(Err(err)) => {
                let _ = evt_tx.send(WorkerEvent::Error(format!(
                    "Falha ao carregar modelo: {err}"
                )));
                None
            }
            Err(err) => {
                let _ = evt_tx.send(WorkerEvent::Error(format!(
                    "Falha ao carregar modelo (join): {err}"
                )));
                None
            }
        };

        let tool_manager = ToolManager::new(&effective_config).await;
        apply_agent_tool_filter(&tool_manager, &agent_policy);
        for (tool_name, permission) in &agent_policy.tool_permission_overrides {
            let _ = tool_manager.set_permission(tool_name, *permission);
        }
        let mut loop_engine = AgentLoop::new(effective_config.clone(), tool_manager);
        if let Some(engine) = &llm_engine {
            loop_engine.set_llm_engine(engine.clone());
        }
        loop_engine.set_bash_kill_signal(bash_kill_signal_inner);
        loop_engine.set_agent_name(agent_policy.builtin.as_str());
        let session_logger = match resume_session_id.as_deref() {
            Some(session_id) => SessionLogger::resume(&NcConfig::sessions_dir(), session_id),
            None => SessionLogger::new(
                &NcConfig::sessions_dir(),
                &runtime.model_label,
                agent_policy.builtin.as_str(),
            ),
        };
        let mut session_logger = match session_logger {
            Ok(logger) => logger,
            Err(err) => {
                let _ = evt_tx.send(WorkerEvent::Error(format!(
                    "Falha ao inicializar logger da sessão: {err}"
                )));
                return;
            }
        };
        let resumed = resume_session_id.is_some();
        let _ = evt_tx.send(WorkerEvent::SessionReady {
            session_id: session_logger.session_id().to_string(),
            resumed,
        });

        // Subagent progress channel: task tool sends events here, we forward to TUI.
        let (sub_progress_tx, sub_progress_rx) =
            std::sync::mpsc::channel::<(String, nanocode_core::SubagentProgress)>();
        loop_engine.set_subagent_progress_tx(sub_progress_tx);

        // Spawn drainer thread that forwards subagent progress to evt_tx.
        {
            let evt_tx = evt_tx.clone();
            std::thread::spawn(move || {
                while let Ok((parent_id, progress)) = sub_progress_rx.recv() {
                    let event = match progress {
                        nanocode_core::SubagentProgress::ToolCall { summary, .. } => {
                            WorkerEvent::SubagentToolCall {
                                parent_call_id: parent_id,
                                summary,
                            }
                        }
                        nanocode_core::SubagentProgress::ToolResult { .. } => {
                            WorkerEvent::SubagentToolResult {
                                parent_call_id: parent_id,
                            }
                        }
                    };
                    if evt_tx.send(event).is_err() {
                        break;
                    }
                }
            });
        }
        loop_engine.set_approval_handler({
            let evt_tx = evt_tx.clone();
            move |request| {
                let summary = format_tool_summary(&request.tool_name, &request.arguments);
                let details = build_approval_details(&request.tool_name, &request.arguments);
                let diff_preview = build_approval_diff(&request.tool_name, &request.arguments);
                let (decision_tx, decision_rx) = std::sync::mpsc::sync_channel(1);
                if evt_tx
                    .send(WorkerEvent::ApprovalRequired {
                        call_id: request.tool_call_id,
                        summary,
                        details,
                        diff_preview,
                        decision_tx,
                    })
                    .is_err()
                {
                    return ApprovalDecision::Deny;
                }

                decision_rx.recv().unwrap_or(ApprovalDecision::Deny)
            }
        });
        loop_engine.set_question_handler({
            let evt_tx = evt_tx.clone();
            move |request| {
                let (response_tx, response_rx) = std::sync::mpsc::sync_channel(1);
                if evt_tx
                    .send(WorkerEvent::QuestionRequired {
                        call_id: request.tool_call_id,
                        question: request.question,
                        choices: request.choices,
                        allow_free_text: request.allow_free_text,
                        placeholder: request.placeholder,
                        response_tx,
                    })
                    .is_err()
                {
                    return UserQuestionResponse::cancelled();
                }

                response_rx
                    .recv()
                    .unwrap_or_else(|_| UserQuestionResponse::cancelled())
            }
        });

        let skill_manager = SkillManager::new(&runtime.config);
        let mut system_prompt = load_prompt(PromptFamily::Qwen3, agent_policy.prompt_variant);
        let skills_section = skill_manager.available_skills_prompt_section();
        if !skills_section.is_empty() {
            system_prompt = format!("{system_prompt}\n\n{skills_section}");
        }
        let thinking_enabled_by_default = runtime.model.supports_thinking
            || is_thinking_model(runtime.model.display_name, runtime.quant.name);
        if thinking_enabled_by_default {
            system_prompt = format!("{}\n\n{}", system_prompt, THINKING_PROMPT_BOOSTER);
        }
        loop_engine.add_system_message(system_prompt);
        if !resumed {
            if let Some(system_message) = loop_engine.messages().last() {
                if let Err(err) = session_logger.append(system_message).await {
                    let _ = evt_tx.send(WorkerEvent::Error(format!(
                        "Falha ao persistir bootstrap da sessão: {err}"
                    )));
                }
            }
        }
        if !initial_messages.is_empty() {
            loop_engine.extend_messages(
                initial_messages
                    .into_iter()
                    .filter(|msg| msg.role != MessageRole::System),
            );
        }

        let _ = evt_tx.send(WorkerEvent::Ready {
            model_label: runtime.model_label.clone(),
            supports_thinking: thinking_enabled_by_default,
            supports_vision: runtime.model.supports_vision,
        });

        while let Some(cmd) = cmd_rx.recv().await {
            match cmd {
                WorkerCommand::Shutdown => {
                    let _ = session_logger.finish();
                    break;
                }
                WorkerCommand::SetAutoApprove(val) => {
                    loop_engine.set_auto_approve(val);
                }
                WorkerCommand::Compact => {
                    let old_ctx = loop_engine.stats().context_tokens;
                    let _ = evt_tx.send(WorkerEvent::CompactStart {
                        old_context_tokens: old_ctx,
                        threshold: 0,
                    });
                    match loop_engine
                        .compact(
                            &runtime.model_file,
                            runtime.max_tokens,
                            Some(interrupt_signal.clone()),
                        )
                        .await
                    {
                        Ok(summary) => {
                            let new_ctx = loop_engine.stats().context_tokens;
                            let _ = evt_tx.send(WorkerEvent::CompactEnd {
                                old_context_tokens: old_ctx,
                                new_context_tokens: new_ctx,
                                summary_len: summary.len(),
                            });
                        }
                        Err(err) => {
                            let _ = evt_tx
                                .send(WorkerEvent::Error(format!("Falha na compactação: {err}")));
                        }
                    }
                }
                WorkerCommand::Submit {
                    prompt,
                    image_data_urls,
                } => {
                    clear_interrupt_signal(&interrupt_signal);
                    let before_len = loop_engine.messages().len();
                    if image_data_urls.is_empty() {
                        loop_engine.add_user_message(prompt);
                    } else {
                        loop_engine.add_user_message_with_images(prompt, image_data_urls);
                    }
                    let _ = evt_tx.send(WorkerEvent::Busy(true));

                    let mut stream_sanitizer = StreamSanitizer::default();
                    if thinking_enabled_by_default {
                        stream_sanitizer.start_in_thinking();
                    }
                    let mut thinking_emitted_this_turn = false;
                    let mut thinking_active = thinking_enabled_by_default;
                    let mut tool_phase_started = false;
                    let mut pending_tool_calls: HashMap<String, PendingToolCall> = HashMap::new();

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

                                    if !parts.visible.is_empty() {
                                        let _ =
                                            evt_tx.send(WorkerEvent::AssistantChunk(parts.visible));
                                    }
                                }
                                LoopEvent::ToolCall(call) => {
                                    tool_phase_started = true;
                                    let summary = format_tool_summary(&call.name, &call.arguments);
                                    let tracked_path =
                                        path_from_arguments(&call.name, &call.arguments)
                                            .map(|path| resolve_absolute_path(&path));
                                    let before_exists = tracked_path
                                        .as_deref()
                                        .map(|path| Path::new(path).exists())
                                        .unwrap_or(false);

                                    pending_tool_calls.insert(
                                        call.id.clone(),
                                        PendingToolCall {
                                            path: tracked_path.clone(),
                                            before_exists,
                                            before_bytes: tracked_path.as_deref().and_then(
                                                |path| {
                                                    if call.name == "write_file"
                                                        || call.name == "search_replace"
                                                    {
                                                        snapshot_file_bytes(path)
                                                    } else {
                                                        None
                                                    }
                                                },
                                            ),
                                            before_content: tracked_path.as_deref().and_then(
                                                |path| {
                                                    if call.name == "write_file"
                                                        || call.name == "search_replace"
                                                    {
                                                        snapshot_file(path)
                                                    } else {
                                                        None
                                                    }
                                                },
                                            ),
                                            name: call.name.clone(),
                                            arguments: call.arguments.clone(),
                                        },
                                    );

                                    if thinking_active {
                                        let _ = evt_tx.send(WorkerEvent::ThinkingActive(false));
                                        thinking_active = false;
                                    }
                                    let _ = evt_tx.send(WorkerEvent::ToolCall {
                                        call_id: call.id,
                                        tool_name: call.name,
                                        summary,
                                    });
                                }
                                LoopEvent::ToolResult { call_id, result } => {
                                    let call_meta = pending_tool_calls.remove(&call_id);

                                    let first = result.lines().next().unwrap_or("sem saída");
                                    let lower = first.to_ascii_lowercase();
                                    let success = !(lower.contains("failed")
                                        || lower.contains("error")
                                        || lower.contains("denied"));

                                    let (
                                        status_line,
                                        output,
                                        code_path,
                                        code,
                                        diff,
                                        rewind_changes,
                                    ) = if let Some(meta) = call_meta.as_ref() {
                                        let code_path = meta.path.clone();
                                        let after_content = code_path.as_deref().and_then(|path| {
                                            if meta.name == "write_file"
                                                || meta.name == "search_replace"
                                            {
                                                snapshot_file(path)
                                            } else {
                                                None
                                            }
                                        });
                                        let after_exists = code_path
                                            .as_deref()
                                            .map(|path| Path::new(path).exists())
                                            .unwrap_or(false);
                                        let after_bytes = code_path.as_deref().and_then(|path| {
                                            if meta.name == "write_file"
                                                || meta.name == "search_replace"
                                            {
                                                snapshot_file_bytes(path)
                                            } else {
                                                None
                                            }
                                        });
                                        let diff = match (
                                            meta.before_content.as_deref(),
                                            after_content.as_deref(),
                                        ) {
                                            (Some(before), Some(after)) => {
                                                build_unified_diff(before, after, 3, 120)
                                            }
                                            _ => None,
                                        };

                                        let status_line = Some(tool_status_line(
                                            meta,
                                            success,
                                            &result,
                                            diff.as_deref(),
                                        ));
                                        let output = match meta.name.as_str() {
                                            "bash" => Some(truncate_lines(&result, 80)),
                                            "grep" => render_grep_output(&result),
                                            "ask_user_question" => {
                                                render_ask_user_question_output(&result)
                                            }
                                            _ => None,
                                        };
                                        let code = match meta.name.as_str() {
                                            "read_file" => {
                                                let read_content =
                                                    parse_read_output_content(&result);
                                                if read_content.trim().is_empty() {
                                                    None
                                                } else {
                                                    Some(truncate_lines(&read_content, 220))
                                                }
                                            }
                                            "write_file" | "search_replace" => after_content
                                                .map(|content| truncate_lines(&content, 220)),
                                            _ => None,
                                        };
                                        let diff = diff.map(|value| truncate_lines(&value, 220));
                                        let rewind_changes = if success
                                            && matches!(
                                                meta.name.as_str(),
                                                "write_file" | "search_replace"
                                            ) {
                                            let changed = meta.before_exists != after_exists
                                                || meta.before_bytes.as_deref()
                                                    != after_bytes.as_deref();
                                            if changed {
                                                if let Some(path) = code_path.clone() {
                                                    if meta.before_exists
                                                        && meta.before_bytes.is_none()
                                                    {
                                                        Vec::new()
                                                    } else {
                                                        vec![RewindChange {
                                                            path,
                                                            existed_before: meta.before_exists,
                                                            before_content: if meta.before_exists {
                                                                meta.before_bytes.clone()
                                                            } else {
                                                                None
                                                            },
                                                        }]
                                                    }
                                                } else {
                                                    Vec::new()
                                                }
                                            } else {
                                                Vec::new()
                                            }
                                        } else {
                                            Vec::new()
                                        };
                                        (status_line, output, code_path, code, diff, rewind_changes)
                                    } else {
                                        (
                                            Some(short_preview(
                                                result.lines().next().unwrap_or("Concluído"),
                                                90,
                                            )),
                                            Some(truncate_lines(&result, 80)),
                                            None,
                                            None,
                                            None,
                                            Vec::new(),
                                        )
                                    };

                                    let _ = evt_tx.send(WorkerEvent::ToolResult {
                                        call_id: call_id.clone(),
                                        success,
                                        status_line,
                                        output,
                                        code_path,
                                        code,
                                        diff,
                                        rewind_changes,
                                    });

                                    if pending_tool_calls.is_empty() {
                                        tool_phase_started = false;
                                        if thinking_enabled_by_default {
                                            stream_sanitizer.start_in_thinking();
                                            thinking_active = true;
                                        }
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
                    if !final_parts.visible.is_empty() {
                        let _ = evt_tx.send(WorkerEvent::AssistantChunk(final_parts.visible));
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
                            if is_user_interrupted_error(&err) {
                                let _ = evt_tx.send(WorkerEvent::Interrupted);
                            } else {
                                let _ = evt_tx.send(WorkerEvent::Error(err));
                            }
                        }
                    }

                    let persisted_messages = loop_engine
                        .messages()
                        .iter()
                        .skip(before_len)
                        .cloned()
                        .collect::<Vec<_>>();
                    for message in persisted_messages {
                        if let Err(err) = session_logger.append(&message).await {
                            let _ = evt_tx.send(WorkerEvent::Error(format!(
                                "Falha ao persistir sessão: {err}"
                            )));
                            break;
                        }
                    }

                    let _ = evt_tx.send(WorkerEvent::Busy(false));
                }
            }
        }

        let _ = session_logger.finish();
    });

    (evt_rx, bash_kill_signal)
}

fn apply_agent_tool_filter(tool_manager: &ToolManager, agent_policy: &AgentPolicy) {
    let mut enabled = agent_policy.enabled_tools.clone();
    if matches!(
        agent_policy.builtin,
        BuiltinAgent::Default | BuiltinAgent::Build
    ) {
        for name in tool_manager.list_tool_names() {
            if name.starts_with("mcp_") {
                enabled.insert(name);
            }
        }
    }
    tool_manager.set_enabled_tools(enabled);
}

//! Nano Code CLI - Main entry point

use anyhow::Result;
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use nanocode_core::agent_loop::{AgentLoop, LoopEvent};
use nanocode_core::llm::{PromptFamily, PromptVariant};
use nanocode_core::prompts::load_prompt;
use nanocode_core::tools::ToolManager;
use nanocode_core::NcConfig;
use nanocode_hf::{
    find_installed_quant, find_quant_by_name, recommend_runtime_limits, HardwareInfo, THE_MODEL,
};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use syntect::easy::HighlightLines;
use syntect::highlighting::{Theme, ThemeSet};
use syntect::parsing::{SyntaxReference, SyntaxSet};
use syntect::util::as_24_bit_terminal_escaped;

mod setup;
mod tui;

#[derive(Parser)]
#[command(name = "nanocode")]
#[command(about = "AI coding assistant with local LLM", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Prompt to execute (programmatic mode)
    #[arg(short, long)]
    prompt: Option<String>,

    /// Auto-approve all tool calls
    #[arg(long)]
    auto_approve: bool,

    /// Resume last session
    #[arg(short, long)]
    continue_session: bool,

    /// Agent profile (default, plan, accept-edits, auto-approve)
    #[arg(long, default_value = "default")]
    agent: String,

    /// KV cache type K (ex: q8_0, q4_0, f16)
    #[arg(long = "ctk")]
    ctk: Option<String>,

    /// KV cache type V (ex: q8_0, q4_0, f16)
    #[arg(long = "ctv")]
    ctv: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Setup or reconfigure the model
    Setup,
    /// List available sessions
    Sessions,
    /// Show configuration
    Config,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HiddenStreamState {
    None,
    ThinkTag,
    ThinkBracket,
    ToolCallXml,
}

#[derive(Debug, Default)]
struct StreamSanitizer {
    pending: String,
    hidden: HiddenStreamState,
}

impl Default for HiddenStreamState {
    fn default() -> Self {
        Self::None
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("nanocode=info,warn")
        .init();

    let normalized_args = std::env::args().map(|arg| match arg.as_str() {
        "-ctk" => "--ctk".to_string(),
        "-ctv" => "--ctv".to_string(),
        other => other.to_string(),
    });
    let cli = Cli::parse_from(normalized_args);

    let mut config = NcConfig::load().unwrap_or_default();
    if cli.auto_approve {
        config.auto_approve = true;
    }

    let is_setup_command = matches!(&cli.command, Some(Commands::Setup));
    if !NcConfig::is_model_installed() && !is_setup_command {
        println!("No model found. Running first-time setup...");
        setup::run_first_time_setup(&mut config).await?;
    }

    match &cli.command {
        Some(Commands::Setup) => setup::run_first_time_setup(&mut config).await?,
        Some(Commands::Sessions) => list_sessions().await?,
        Some(Commands::Config) => println!("{}", toml::to_string_pretty(&config)?),
        None => {
            if let Some(prompt) = cli.prompt {
                run_prompt(&prompt, &config, cli.ctk.clone(), cli.ctv.clone()).await?;
            } else {
                tui::run_tui(&config, cli.ctk.clone(), cli.ctv.clone()).await?;
            }
        }
    }

    Ok(())
}

async fn list_sessions() -> Result<()> {
    use nanocode_core::session::list_sessions;
    let sessions = list_sessions(&NcConfig::sessions_dir()).await?;
    if sessions.is_empty() {
        println!("No sessions found.");
    } else {
        for session in sessions {
            println!("{} - {} messages", session.id, session.message_count);
        }
    }
    Ok(())
}

fn find_ci(haystack: &str, needle: &str) -> Option<usize> {
    haystack
        .to_ascii_lowercase()
        .find(&needle.to_ascii_lowercase())
}

fn split_keep_tail_chars(s: &str, tail_chars: usize) -> (&str, &str) {
    let total_chars = s.chars().count();
    if total_chars <= tail_chars {
        return ("", s);
    }
    let keep_from_char = total_chars - tail_chars;
    let keep_from_byte = s
        .char_indices()
        .nth(keep_from_char)
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    (&s[..keep_from_byte], &s[keep_from_byte..])
}

impl StreamSanitizer {
    fn push(&mut self, chunk: &str, finalize: bool) -> String {
        const START_THINK: &str = "<think>";
        const END_THINK: &str = "</think>";
        const START_THINK_BRACKET: &str = "[start thinking]";
        const END_THINK_BRACKET: &str = "[end thinking]";
        const START_TOOL_XML: &str = "<tool_call>";
        const END_TOOL_XML: &str = "</tool_call>";
        const IM_START_ASSISTANT: &str = "<|im_start|>assistant";
        const IM_END: &str = "<|im_end|>";
        const TAIL_CHARS: usize = 24;

        self.pending.push_str(chunk);
        let mut visible = String::new();

        loop {
            match self.hidden {
                HiddenStreamState::ThinkTag => {
                    if let Some(end_idx) = find_ci(&self.pending, END_THINK) {
                        let drain_to = end_idx + END_THINK.len();
                        self.pending.drain(..drain_to);
                        self.hidden = HiddenStreamState::None;
                        continue;
                    }
                    if !finalize {
                        let (_, tail) = split_keep_tail_chars(&self.pending, END_THINK.len() - 1);
                        self.pending = tail.to_string();
                    } else {
                        self.pending.clear();
                    }
                    break;
                }
                HiddenStreamState::ThinkBracket => {
                    if let Some(end_idx) = find_ci(&self.pending, END_THINK_BRACKET) {
                        let drain_to = end_idx + END_THINK_BRACKET.len();
                        self.pending.drain(..drain_to);
                        self.hidden = HiddenStreamState::None;
                        continue;
                    }
                    if !finalize {
                        let (_, tail) =
                            split_keep_tail_chars(&self.pending, END_THINK_BRACKET.len() - 1);
                        self.pending = tail.to_string();
                    } else {
                        self.pending.clear();
                    }
                    break;
                }
                HiddenStreamState::ToolCallXml => {
                    if let Some(end_idx) = find_ci(&self.pending, END_TOOL_XML) {
                        let drain_to = end_idx + END_TOOL_XML.len();
                        self.pending.drain(..drain_to);
                        self.hidden = HiddenStreamState::None;
                        continue;
                    }
                    if !finalize {
                        let (_, tail) =
                            split_keep_tail_chars(&self.pending, END_TOOL_XML.len() - 1);
                        self.pending = tail.to_string();
                    } else {
                        self.pending.clear();
                    }
                    break;
                }
                HiddenStreamState::None => {
                    let mut candidates: Vec<(usize, &'static str, HiddenStreamState)> = Vec::new();
                    if let Some(i) = find_ci(&self.pending, START_THINK) {
                        candidates.push((i, START_THINK, HiddenStreamState::ThinkTag));
                    }
                    if let Some(i) = find_ci(&self.pending, START_THINK_BRACKET) {
                        candidates.push((i, START_THINK_BRACKET, HiddenStreamState::ThinkBracket));
                    }
                    if let Some(i) = find_ci(&self.pending, START_TOOL_XML) {
                        candidates.push((i, START_TOOL_XML, HiddenStreamState::ToolCallXml));
                    }
                    if let Some(i) = find_ci(&self.pending, END_THINK) {
                        candidates.push((i, END_THINK, HiddenStreamState::None));
                    }
                    if let Some(i) = find_ci(&self.pending, END_THINK_BRACKET) {
                        candidates.push((i, END_THINK_BRACKET, HiddenStreamState::None));
                    }
                    if let Some(i) = find_ci(&self.pending, END_TOOL_XML) {
                        candidates.push((i, END_TOOL_XML, HiddenStreamState::None));
                    }
                    if let Some(i) = find_ci(&self.pending, IM_START_ASSISTANT) {
                        candidates.push((i, IM_START_ASSISTANT, HiddenStreamState::None));
                    }
                    if let Some(i) = find_ci(&self.pending, IM_END) {
                        candidates.push((i, IM_END, HiddenStreamState::None));
                    }

                    if let Some((start_idx, marker, next_state)) =
                        candidates.into_iter().min_by_key(|(i, _, _)| *i)
                    {
                        if start_idx > 0 {
                            visible.push_str(&self.pending[..start_idx]);
                        }
                        let drain_to = start_idx + marker.len();
                        self.pending.drain(..drain_to);
                        self.hidden = next_state;
                        continue;
                    }

                    if finalize {
                        visible.push_str(&self.pending);
                        self.pending.clear();
                    } else {
                        let (emit, tail) = split_keep_tail_chars(&self.pending, TAIL_CHARS);
                        visible.push_str(emit);
                        self.pending = tail.to_string();
                    }
                    break;
                }
            }
        }

        visible
    }
}

fn extract_thinking_blocks_and_clean(text: &str) -> (Vec<String>, String) {
    let mut thoughts = Vec::new();
    let mut out = text.to_string();

    loop {
        let lower = out.to_ascii_lowercase();
        let Some(start) = lower.find("<think>") else {
            break;
        };
        let Some(end_rel) = lower[start..].find("</think>") else {
            let thought = out[start + "<think>".len()..].trim();
            if !thought.is_empty() {
                thoughts.push(thought.to_string());
            }
            out.truncate(start);
            break;
        };
        let end = start + end_rel;
        let thought = out[start + "<think>".len()..end].trim();
        if !thought.is_empty() {
            thoughts.push(thought.to_string());
        }
        out = format!("{}{}", &out[..start], &out[end + "</think>".len()..]);
    }

    loop {
        let lower = out.to_ascii_lowercase();
        let Some(start) = lower.find("[start thinking]") else {
            break;
        };
        let Some(end_rel) = lower[start..].find("[end thinking]") else {
            let thought = out[start + "[start thinking]".len()..].trim();
            if !thought.is_empty() {
                thoughts.push(thought.to_string());
            }
            out.truncate(start);
            break;
        };
        let end = start + end_rel;
        let thought = out[start + "[start thinking]".len()..end].trim();
        if !thought.is_empty() {
            thoughts.push(thought.to_string());
        }
        out = format!("{}{}", &out[..start], &out[end + "[end thinking]".len()..]);
    }

    while let Some(idx) = out.to_ascii_lowercase().find("</think>") {
        let end = idx + "</think>".len();
        out = format!("{}{}", &out[..idx], &out[end..]);
    }

    while let Some(start) = out.to_ascii_lowercase().find("<tool_call>") {
        if let Some(end_rel) = out.to_ascii_lowercase()[start..].find("</tool_call>") {
            let end = start + end_rel + "</tool_call>".len();
            out.replace_range(start..end, "");
        } else {
            out.truncate(start);
            break;
        }
    }

    (
        thoughts,
        out.replace("<|im_start|>assistant", "")
            .replace("<|im_end|>", "")
            .trim()
            .to_string(),
    )
}

fn style_shimmer_text(text: &str, frame: usize) -> String {
    let palette = [226, 214, 208, 202, 196];
    let mut out = String::new();
    for (idx, ch) in text.chars().enumerate() {
        let color = palette[(idx + frame) % palette.len()];
        out.push_str(&format!("\x1b[38;5;{}m{}\x1b[0m", color, ch));
    }
    out
}

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

**Ambiguity Handling:**
- Missing filename? → List directory (`ls -l`)
- Unclear path? → Use current directory
- Multiple interpretations? → Pick simplest, act
- User can clarify AFTER seeing results

**Token Awareness:**
- Every "Wait..." costs 10 tokens
- Every "Hmm..." costs 5 tokens
- Budget: 150 tokens for simple tasks
- At 100 tokens? → Force tool call

**Action Mantras:**
- "Good enough → Execute"
- "List first, filter later"
- "Show results, then ask"
- "Action creates clarity"

**Language:**
- Match user language in final response
- Tool calls in English
- No translation loops
"#;

fn is_thinking_model(display_name: &str, quant_name: &str) -> bool {
    let combined = format!(
        "{} {}",
        display_name.to_ascii_lowercase(),
        quant_name.to_ascii_lowercase()
    );
    combined.contains("thinking") || combined.contains("r1")
}

fn thinking_symbol(frame: usize) -> char {
    if frame.is_multiple_of(2) {
        '∗'
    } else {
        '⋇'
    }
}

fn tool_dot(name: &str) -> &'static str {
    let n = name.to_ascii_lowercase();
    if n.contains("delete") || n.contains("remove") {
        "\x1b[31m●\x1b[0m"
    } else if n.contains("read") {
        "\x1b[90m●\x1b[0m"
    } else if n.contains("write") || n.contains("edit") || n.contains("create") {
        "\x1b[32m●\x1b[0m"
    } else {
        "●"
    }
}

fn emit_thinking_line(line: &str, thinking_frame: &mut usize, thinking_banner_shown: &mut bool) {
    if !*thinking_banner_shown {
        println!();
        let symbol = thinking_symbol(*thinking_frame);
        let shimmer = style_shimmer_text("Thinking...", *thinking_frame);
        println!("{} {}", symbol, shimmer);
        *thinking_frame = thinking_frame.wrapping_add(1);
        *thinking_banner_shown = true;
    }
    println!("    │ {}", line);
}

fn syntax_highlight_line(path: &str, line: &str) -> String {
    fn supports_ansi_color() -> bool {
        if std::env::var_os("NO_COLOR").is_some() {
            return false;
        }
        !matches!(std::env::var("TERM"), Ok(term) if term == "dumb")
    }

    if !supports_ansi_color() {
        return line.to_string();
    }

    struct HighlightAssets {
        ps: SyntaxSet,
        ts: ThemeSet,
    }

    static ASSETS: OnceLock<HighlightAssets> = OnceLock::new();
    let assets = ASSETS.get_or_init(|| HighlightAssets {
        ps: SyntaxSet::load_defaults_newlines(),
        ts: ThemeSet::load_defaults(),
    });

    let extension = Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase());

    let syntax: &SyntaxReference = extension
        .as_deref()
        .and_then(|ext| assets.ps.find_syntax_by_extension(ext))
        .unwrap_or_else(|| assets.ps.find_syntax_plain_text());

    if line.trim_start().starts_with('+') {
        return format!("\x1b[30;42m{}\x1b[0m", line);
    }
    if line.trim_start().starts_with('-') {
        return format!("\x1b[37;41m{}\x1b[0m", line);
    }

    let theme: &Theme = assets
        .ts
        .themes
        .get("base16-ocean.dark")
        .or_else(|| assets.ts.themes.get("Monokai Extended"))
        .or_else(|| assets.ts.themes.get("Monokai Extended Bright"))
        .or_else(|| assets.ts.themes.get("Solarized (dark)"))
        .or_else(|| assets.ts.themes.get("Monokai"))
        .or_else(|| assets.ts.themes.values().next())
        .expect("syntect theme set should contain at least one theme");

    let mut h = HighlightLines::new(syntax, theme);
    match h.highlight_line(line, &assets.ps) {
        // Avoid forcing token background colors. This keeps highlight readable
        // in terminals/themes where light backgrounds make code illegible.
        Ok(ranges) => as_24_bit_terminal_escaped(&ranges, false),
        Err(_) => line.to_string(),
    }
}

async fn run_prompt(
    prompt_text: &str,
    config: &NcConfig,
    ctk_override: Option<String>,
    ctv_override: Option<String>,
) -> Result<()> {
    let model_path = NcConfig::models_dir();
    let quant = if let Some(active_name) = config.active_quant.as_deref() {
        if let Some(active_quant) = find_quant_by_name(active_name) {
            let active_path = model_path.join(active_quant.filename);
            if active_path.exists() {
                active_quant
            } else {
                find_installed_quant(&model_path).expect("No model installed")
            }
        } else {
            find_installed_quant(&model_path).expect("No model installed")
        }
    } else {
        find_installed_quant(&model_path).expect("No model installed")
    };

    let model_file = model_path.join(quant.filename);
    let hw = HardwareInfo::detect();
    let memory_mb = hw.vram_mb.unwrap_or(hw.ram_mb);
    let runtime_limits = recommend_runtime_limits(memory_mb, &THE_MODEL, true);

    let mut runtime_config = config.clone();
    runtime_config.model.context_size = Some(
        runtime_config
            .model
            .context_size
            .unwrap_or(runtime_limits.context_size)
            .min(THE_MODEL.max_context_size)
            .max(8_192),
    );
    if let Some(ctk) = ctk_override {
        runtime_config.model.kv_cache_type_k = Some(ctk);
    }
    if let Some(ctv) = ctv_override {
        runtime_config.model.kv_cache_type_v = Some(ctv);
    }

    let mut system_prompt = load_prompt(PromptFamily::Qwen3, PromptVariant::AgentDefault);
    if is_thinking_model(THE_MODEL.display_name, quant.name) {
        system_prompt = format!("{}\n\n{}", system_prompt, THINKING_PROMPT_BOOSTER);
    }
    if quant.name == "Q2_K" {
        eprintln!("Aviso: Q2_K pode não produzir resposta final em modelos 'thinking'. Considere usar 'nanocode setup' e selecionar Q4_K_M ou Q5_K_M.");
    }

    let max_tokens = config
        .model
        .max_tokens
        .min(runtime_limits.max_tokens)
        .clamp(512, 8192);

    let tool_manager = ToolManager::new(&runtime_config);
    let mut loop_engine = AgentLoop::new(runtime_config.clone(), tool_manager);
    loop_engine.add_system_message(system_prompt);
    loop_engine.add_user_message(prompt_text);

    let total_start = Instant::now();
    let mut first_token_at: Option<Instant> = None;
    let mut thinking_frame = 0usize;
    let mut thinking_banner_shown = false;
    let mut tool_phase_started = false;
    let mut thinking_tools_gap_printed = false;
    let mut pending_tool_calls: HashMap<String, (String, Value)> = HashMap::new();
    let mut stream_sanitizer = StreamSanitizer::default();
    let mut stream_line_buf = String::new();
    let mut last_tool_call_started_at: Option<Instant> = None;

    let pretty_tool_name = |name: &str| -> String {
        match name {
            "write_file" => "Write".to_string(),
            "read_file" => "Read".to_string(),
            "bash" => "Bash".to_string(),
            other => other.to_string(),
        }
    };

    let short_call_id = |id: &str| -> String {
        if id.len() > 8 {
            id[..8].to_string()
        } else {
            id.to_string()
        }
    };

    let truncate_preview = |text: &str, max_chars: usize| {
        let mut out = String::new();
        for ch in text.chars().take(max_chars) {
            out.push(ch);
        }
        if text.chars().count() > max_chars {
            out.push('…');
        }
        out
    };

    let split_tool_result = |result: &str| {
        let preview_raw = result.lines().next().unwrap_or("").trim();
        let lower = preview_raw.to_ascii_lowercase();
        let success =
            !(lower.contains("failed") || lower.contains("error") || lower.contains("denied"));
        (success, truncate_preview(preview_raw, 90))
    };

    let line_count = |s: &str| -> usize { s.lines().count() };
    let horizontal_rule = |w: usize| -> String { "─".repeat(w) };
    let render_tool_preview = |name: &str, arguments: &Value| {
        if name == "write_file" {
            let path = arguments
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let content = arguments
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let title = format!("Create File {}", path);
            let width = 82usize;
            println!("{}", horizontal_rule(width));
            println!("{}", title);
            println!("{}", horizontal_rule(width));
            for (idx, line) in content.lines().take(12).enumerate() {
                println!("{}. {}", idx + 1, syntax_highlight_line(path, line));
            }
            if line_count(content) > 12 {
                println!("…");
            }
            println!("{}", horizontal_rule(width));
        }
    };
    let render_bash_result_preview = |result: &str| {
        let max_lines = 4usize;
        let lines: Vec<&str> = result.lines().collect();
        if lines.is_empty() {
            println!("    ⌊ (no output)");
            return;
        }
        for (idx, line) in lines.iter().take(max_lines).enumerate() {
            if idx == 0 {
                println!("    ⌊ {}", truncate_preview(line, 110));
            } else {
                println!("      {}", truncate_preview(line, 110));
            }
        }
        if lines.len() > max_lines {
            println!("      ... +{} lines", lines.len() - max_lines);
        }
    };

    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::with_template("{spinner} {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_spinner())
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    spinner.set_message(format!("Carregando modelo {}...", quant.name));
    spinner.enable_steady_tick(Duration::from_millis(90));

    let run_result = loop_engine
        .act_with_events(&model_file, max_tokens, |event| match event {
            LoopEvent::Stats(_) => {
                if first_token_at.is_none() {
                    let elapsed = total_start.elapsed().as_secs_f32();
                    spinner.finish_with_message(format!("Modelo carregado em {:.2}s", elapsed));
                    first_token_at = Some(Instant::now());
                }
            }
            LoopEvent::ToolCall(call) => {
                if !tool_phase_started && !stream_line_buf.trim().is_empty() {
                    emit_thinking_line(
                        stream_line_buf.trim_end(),
                        &mut thinking_frame,
                        &mut thinking_banner_shown,
                    );
                    stream_line_buf.clear();
                }
                if !thinking_tools_gap_printed && thinking_banner_shown {
                    println!();
                    thinking_tools_gap_printed = true;
                }
                tool_phase_started = true;
                let display_name = pretty_tool_name(&call.name);
                last_tool_call_started_at = Some(Instant::now());
                pending_tool_calls
                    .insert(call.id.clone(), (call.name.clone(), call.arguments.clone()));

                let target = if call.name == "bash" {
                    call.arguments
                        .get("command")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                } else {
                    call.arguments
                        .get("path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                };
                if target.is_empty() {
                    println!("{} {}()", tool_dot(&call.name), display_name);
                } else {
                    println!(
                        "{} {}({})",
                        tool_dot(&call.name),
                        display_name,
                        truncate_preview(target, 90)
                    );
                }
                render_tool_preview(&call.name, &call.arguments);
            }
            LoopEvent::ToolResult { call_id, result } => {
                let elapsed_s = last_tool_call_started_at
                    .take()
                    .map(|t| t.elapsed().as_secs_f32())
                    .unwrap_or(0.0);
                let resolved_tool_call = pending_tool_calls.remove(&call_id);

                let (success, preview) = split_tool_result(&result);
                if success {
                    let mut printed = false;
                    if let Some((tool_name, _args)) = resolved_tool_call.as_ref() {
                        if tool_name == "read_file" {
                            let lines = line_count(&result);
                            println!("    ⌊ Read {} Lines ({:.2}s)", lines, elapsed_s);
                            printed = true;
                        } else if tool_name == "bash" {
                            render_bash_result_preview(&result);
                            printed = true;
                        } else if tool_name == "write_file" {
                            // write_file already has a rich preview block; suppress
                            // redundant generic "ok ... file written" noise.
                            printed = true;
                        }
                    }
                    if !printed {
                        println!(
                            "  ok {} ({:.2}s) {}",
                            short_call_id(&call_id),
                            elapsed_s,
                            preview
                        );
                    }
                } else {
                    println!(
                        "  erro {} ({:.2}s) {}",
                        short_call_id(&call_id),
                        elapsed_s,
                        preview
                    );
                }

                // Re-enable streamed thinking between tool rounds.
                // `tool_phase_started` is raised on ToolCall and should only gate
                // output while a call is actively pending.
                if pending_tool_calls.is_empty() {
                    tool_phase_started = false;
                    thinking_tools_gap_printed = false;
                    thinking_banner_shown = false;
                }
            }
            LoopEvent::Chunk(chunk) => {
                if first_token_at.is_none() {
                    let elapsed = total_start.elapsed().as_secs_f32();
                    spinner.finish_with_message(format!("Modelo carregado em {:.2}s", elapsed));
                    first_token_at = Some(Instant::now());
                }
                let delta = stream_sanitizer.push(&chunk, false);
                if !tool_phase_started && !delta.trim().is_empty() {
                    stream_line_buf.push_str(&delta);
                    while let Some(pos) = stream_line_buf.find('\n') {
                        let line = &stream_line_buf[..pos];
                        let printable = line.trim_end();
                        if !printable.trim().is_empty() {
                            emit_thinking_line(
                                printable,
                                &mut thinking_frame,
                                &mut thinking_banner_shown,
                            );
                        }
                        stream_line_buf.drain(..=pos);
                    }
                    let wrap_at = 120usize;
                    while stream_line_buf.chars().count() >= wrap_at {
                        let candidate: String = stream_line_buf.chars().take(wrap_at).collect();
                        let split_at = candidate
                            .rfind(char::is_whitespace)
                            .filter(|idx| *idx > 20)
                            .unwrap_or(candidate.len());
                        let (head, tail) = stream_line_buf.split_at(split_at);
                        let printable = head.trim_end();
                        if !printable.trim().is_empty() {
                            emit_thinking_line(
                                printable,
                                &mut thinking_frame,
                                &mut thinking_banner_shown,
                            );
                        }
                        let tail = tail.trim_start_matches(char::is_whitespace).to_string();
                        stream_line_buf = tail;
                    }
                }
            }
            LoopEvent::Message(_) | LoopEvent::Error(_) => {}
        })
        .await;

    if first_token_at.is_none() {
        spinner.finish_and_clear();
    }

    match run_result {
        Ok(response) => {
            let final_delta = stream_sanitizer.push("", true);
            if !tool_phase_started && !final_delta.trim().is_empty() {
                stream_line_buf.push_str(&final_delta);
            }
            if !tool_phase_started && !stream_line_buf.trim().is_empty() {
                emit_thinking_line(
                    stream_line_buf.trim_end(),
                    &mut thinking_frame,
                    &mut thinking_banner_shown,
                );
                stream_line_buf.clear();
            }

            let (_thoughts, final_response) = extract_thinking_blocks_and_clean(&response);
            println!();
            println!("{}", final_response);
        }
        Err(e) => {
            if e == "No output generated" && quant.name == "Q2_K" {
                eprintln!(
                    "Error: {}. A quantização Q2_K costuma degradar o raciocínio; execute 'nanocode setup' e selecione Q4_K_M ou Q5_K_M.",
                    e
                );
            } else {
                eprintln!("Error: {}", e);
            }
        }
    }

    let total_elapsed = total_start.elapsed().as_secs_f32();
    let infer_elapsed = first_token_at
        .map(|t| t.elapsed().as_secs_f32())
        .unwrap_or(0.0)
        .max(0.001);
    let stats = loop_engine.stats();
    let tps = stats.tokens_out as f32 / infer_elapsed;

    println!();
    println!("• Model: {} ({})", THE_MODEL.display_name, quant.name);
    println!("• Input Tokens: {}", stats.tokens_in);
    println!("• Output Tokens: {}", stats.tokens_out);
    println!("• Total Tokens: {}", stats.tokens_used);
    println!(
        "• Tempo: load={:.2}s inferência={:.2}s total={:.2}s ({:.1} tok/s)",
        first_token_at
            .map(|_| total_elapsed - infer_elapsed)
            .unwrap_or(total_elapsed),
        infer_elapsed,
        total_elapsed,
        tps
    );

    Ok(())
}

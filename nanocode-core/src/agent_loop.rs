//! Agent loop implementation

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crate::config::NcConfig;
use crate::interrupt::{check_interrupt_signal, is_user_interrupted_error};
use crate::llm::{
    build_available_tools_schema, chat_via_engine_streaming, chat_via_openai_server_streaming,
    parse_tool_calls, tools_to_json, LlmEngineHandle,
};
use crate::middleware::{
    AutoCompactMiddleware, ContextWarningMiddleware, ConversationContext, MiddlewareAction,
    MiddlewarePipeline, PlanAgentMiddleware, ResetReason, TurnLimitMiddleware,
};
use crate::tools::ToolManager;
use crate::types::*;

const DEFAULT_MAX_TURNS: u32 = 24;
const COMPACT_SUMMARY_PROMPT: &str = r#"Create a continuation summary for this coding session.

Write a concise brief with these sections:
## Goal
- Current user objective and success criteria.

## Constraints
- Important user constraints, style requirements, and environment limits.

## Decisions
- Key technical decisions already made and why.

## Progress
- What was completed, including important tool outcomes.

## Pending
- Open tasks, blockers, and the immediate next action.

Rules:
- Focus on information needed to continue in the SAME session.
- Do not include verbose logs or repeated details.
- Mention file paths only when relevant to pending work.
- Keep the total summary under 450 words."#;
const COMPACT_PROTECTED_USER_TURNS: u32 = 2;
const COMPACT_INPUT_RESERVE_TOKENS: u32 = 2_048;
const COMPACT_OLD_TOOL_MAX_CHARS: usize = 700;
const COMPACT_OLD_TEXT_MAX_CHARS: usize = 1_400;
const FALLBACK_TAIL_MESSAGES: usize = 6;
const FALLBACK_TOOL_MAX_CHARS: usize = 280;
const FALLBACK_TEXT_MAX_CHARS: usize = 700;
const SUMMARY_SOFT_TOKEN_CAP: u32 = 1_200;

fn estimate_tokens(text: &str) -> u32 {
    let words = text.split_whitespace().count() as u32;
    let chars = text.chars().count() as u32;
    let char_based_estimate = chars.saturating_add(3) / 4;
    words.max(char_based_estimate).max(1)
}

fn estimate_messages_tokens(messages: &[LlmMessage]) -> u32 {
    messages
        .iter()
        .map(|msg| estimate_tokens(&msg.content.to_plain_text_lossy()))
        .sum::<u32>()
}

fn merge_consecutive_user_messages(messages: &[LlmMessage]) -> Vec<LlmMessage> {
    let mut merged: Vec<LlmMessage> = Vec::with_capacity(messages.len());

    for msg in messages {
        if msg.role == MessageRole::User {
            if let Some(last) = merged.last_mut() {
                if last.role == MessageRole::User {
                    let prev = last.content.as_text().map(str::trim);
                    let curr = msg.content.as_text().map(str::trim);
                    if let (Some(prev), Some(curr)) = (prev, curr) {
                        last.content = if prev.is_empty() {
                            MessageContent::text(curr.to_string())
                        } else if curr.is_empty() {
                            MessageContent::text(prev.to_string())
                        } else {
                            MessageContent::text(format!("{}\n\n{}", prev, curr))
                        };
                        continue;
                    }
                }
            }
        }
        merged.push(msg.clone());
    }

    merged
}

fn looks_like_tool_call_attempt(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.contains("<tool_call>")
        || lower.contains("\"tool_calls\"")
        || (lower.contains("\"name\"") && lower.contains("\"arguments\""))
}

fn repair_excerpt(text: &str) -> String {
    let excerpt = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("")
        .chars()
        .take(120)
        .collect::<String>();
    excerpt.replace('"', "'")
}

fn malformed_tool_call_repair_prompt(assistant_text: &str) -> String {
    format!(
        "[runtime] Tool contract mismatch after: \"{}\". Next response must be valid OpenAI JSON only: {{\"tool_calls\":[{{\"id\":\"call_...\",\"type\":\"function\",\"function\":{{\"name\":\"...\",\"arguments\":\"{{\\\"field\\\":\\\"value\\\"}}\"}}}}]}}",
        repair_excerpt(assistant_text)
    )
}

fn deferred_execution_repair_prompt(assistant_text: &str) -> String {
    format!(
        "[runtime] Intention without execution detected after: \"{}\". Next turn must contain one concrete action now (tool call or final answer), not future-tense planning.",
        repair_excerpt(assistant_text)
    )
}

fn repetition_repair_prompt(assistant_text: &str) -> String {
    format!(
        "[runtime] Repetition detected around: \"{}\". Continue with a new concrete step and do not repeat the same planning prefix.",
        repair_excerpt(assistant_text)
    )
}

fn looks_like_deferred_execution(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    let has_future_phrase = [
        "vou ", "vou\n", "i will", "i'll", "let me", "deixe-me", "deixa eu", "irei ", "farei ",
    ]
    .iter()
    .any(|needle| lower.contains(needle));

    let has_action_verb = [
        "corrig",
        "alter",
        "edit",
        "atualiz",
        "updat",
        "modific",
        "criar",
        "create",
        "escrev",
        "write",
        "implement",
        "fix",
        "patch",
        "adicion",
        "change",
        "execut",
        "rodar",
    ]
    .iter()
    .any(|needle| lower.contains(needle));

    has_future_phrase && has_action_verb
}

fn normalize_repetition_line(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }

    let trimmed = trimmed.trim_start_matches(|c: char| {
        c.is_ascii_digit() || c.is_whitespace() || matches!(c, '.' | ')' | '-' | '*' | '•' | ':')
    });
    if trimmed.is_empty() {
        return None;
    }

    let words: Vec<String> = trimmed
        .split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_ascii_lowercase()
        })
        .filter(|w| !w.is_empty())
        .take(6)
        .collect();

    if words.len() < 4 {
        return None;
    }

    Some(words.join(" "))
}

fn looks_like_repetition_loop(text: &str) -> bool {
    let mut signatures: HashMap<String, u8> = HashMap::new();

    for line in text.lines() {
        let Some(signature) = normalize_repetition_line(line) else {
            continue;
        };
        let count = signatures.entry(signature).or_insert(0);
        *count = count.saturating_add(1);
        if *count >= 3 {
            return true;
        }
    }

    false
}

fn model_context_limit(config: &NcConfig) -> u32 {
    config
        .model
        .context_size
        .unwrap_or(32_768)
        .clamp(512, 262_144)
}

fn summary_max_tokens_for_compaction(max_tokens: u32) -> u32 {
    (max_tokens / 3).clamp(192, 768)
}

fn compaction_input_budget(config: &NcConfig, summary_max_tokens: u32) -> u32 {
    let context_limit = model_context_limit(config);
    let available = context_limit
        .saturating_sub(summary_max_tokens)
        .saturating_sub(COMPACT_INPUT_RESERVE_TOKENS);
    let floor = context_limit.saturating_div(4).max(512);
    available.max(floor)
}

fn compact_text_for_context(
    text: &str,
    max_chars: usize,
    head_lines: usize,
    tail_lines: usize,
) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let total_chars = trimmed.chars().count();
    if total_chars <= max_chars {
        return trimmed.to_string();
    }

    let lines: Vec<&str> = trimmed.lines().collect();
    let head = head_lines.min(lines.len());
    let tail = tail_lines.min(lines.len().saturating_sub(head));
    let omitted = lines.len().saturating_sub(head + tail);

    let mut out = String::new();
    for line in lines.iter().take(head) {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(line);
    }

    if omitted > 0 {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&format!("[...{} lines compacted...]", omitted));
    }

    if tail > 0 {
        for line in lines.iter().skip(lines.len() - tail) {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(line);
        }
    }

    let truncated = if out.chars().count() > max_chars {
        out.chars()
            .take(max_chars.saturating_sub(24))
            .collect::<String>()
    } else {
        out
    };

    format!("{}\n[...compacted...]", truncated.trim_end())
}

fn compact_message_content(
    msg: &mut LlmMessage,
    max_chars: usize,
    head_lines: usize,
    tail_lines: usize,
) {
    let plain = msg.content.to_plain_text_lossy();
    let compacted = compact_text_for_context(&plain, max_chars, head_lines, tail_lines);
    if compacted != plain.trim() {
        msg.content = MessageContent::text(compacted);
    }
}

fn prepare_messages_for_compaction(
    messages: &[LlmMessage],
    config: &NcConfig,
    summary_max_tokens: u32,
) -> Vec<LlmMessage> {
    let mut prepared = messages.to_vec();
    let mut seen_recent_user_turns = 0u32;

    for msg in prepared.iter_mut().rev() {
        if msg.role == MessageRole::User {
            seen_recent_user_turns = seen_recent_user_turns.saturating_add(1);
        }

        if seen_recent_user_turns <= COMPACT_PROTECTED_USER_TURNS {
            continue;
        }

        match msg.role {
            MessageRole::Tool => {
                let original = msg.content.to_plain_text_lossy();
                compact_message_content(msg, COMPACT_OLD_TOOL_MAX_CHARS, 5, 3);
                if msg.content.to_plain_text_lossy() != original.trim() {
                    msg.content = MessageContent::text(format!(
                        "[tool output compacted: {} chars]\n{}",
                        original.chars().count(),
                        msg.content.to_plain_text_lossy()
                    ));
                }
            }
            MessageRole::Assistant | MessageRole::User => {
                compact_message_content(msg, COMPACT_OLD_TEXT_MAX_CHARS, 10, 4);
            }
            MessageRole::System => {}
        }
    }

    let budget = compaction_input_budget(config, summary_max_tokens);
    if estimate_messages_tokens(&prepared) <= budget {
        return prepared;
    }

    retain_recent_messages_within_budget(prepared, budget)
}

fn retain_recent_messages_within_budget(messages: Vec<LlmMessage>, budget: u32) -> Vec<LlmMessage> {
    if messages.is_empty() {
        return Vec::new();
    }

    let has_system = messages
        .first()
        .map(|m| m.role == MessageRole::System)
        .unwrap_or(false);
    let body_start = if has_system { 1 } else { 0 };
    let body = &messages[body_start..];

    let mut out = Vec::with_capacity(messages.len());
    if has_system {
        out.push(messages[0].clone());
    }
    if body.is_empty() {
        return out;
    }

    let mut used = 0u32;
    let mut kept_rev: Vec<LlmMessage> = Vec::new();
    for msg in body.iter().rev() {
        let msg_tokens = estimate_tokens(&msg.content.to_plain_text_lossy());
        if !kept_rev.is_empty() && used.saturating_add(msg_tokens) > budget {
            break;
        }
        used = used.saturating_add(msg_tokens);
        kept_rev.push(msg.clone());
    }

    if kept_rev.is_empty() {
        kept_rev.push(body[body.len() - 1].clone());
    }

    kept_rev.reverse();
    out.extend(kept_rev);
    out
}

fn trim_summary_for_budget(summary: &str) -> String {
    if estimate_tokens(summary) <= SUMMARY_SOFT_TOKEN_CAP {
        return summary.trim().to_string();
    }

    compact_text_for_context(summary, (SUMMARY_SOFT_TOKEN_CAP as usize) * 4, 20, 8)
}

fn is_prompt_too_long_error(err: &str) -> bool {
    let lower = err.to_ascii_lowercase();
    lower.contains("prompt too long")
        || lower.contains("n_ctx")
        || lower.contains("context exceeds")
}

/// Agent loop that orchestrates LLM calls and tool execution.
pub struct AgentLoop {
    config: NcConfig,
    messages: Vec<LlmMessage>,
    stats: AgentStats,
    agent_name: String,
    middleware_pipeline: MiddlewarePipeline,
    tool_manager: Arc<ToolManager>,
    approval_handler: Option<Arc<dyn Fn(ApprovalRequest) -> ApprovalDecision + Send + Sync>>,
    question_handler:
        Option<Arc<dyn Fn(UserQuestionRequest) -> UserQuestionResponse + Send + Sync>>,
    subagent_progress_tx: Option<std::sync::mpsc::Sender<(String, crate::types::SubagentProgress)>>,
    /// Shared pre-loaded LLM engine. When set, the model stays in memory across turns.
    llm_engine: Option<Arc<LlmEngineHandle>>,
    /// Kill signal for the currently running bash process.
    bash_kill_signal: Option<crate::tools::bash::BashKillSignal>,
}

impl AgentLoop {
    pub fn new(config: NcConfig, tool_manager: ToolManager) -> Self {
        let mut loop_state = Self {
            config,
            messages: Vec::new(),
            stats: AgentStats::default(),
            agent_name: "default".to_string(),
            middleware_pipeline: MiddlewarePipeline::new(),
            tool_manager: Arc::new(tool_manager),
            approval_handler: None,
            question_handler: None,
            subagent_progress_tx: None,
            llm_engine: None,
            bash_kill_signal: None,
        };
        loop_state.setup_middleware();
        loop_state
    }

    /// Set a shared pre-loaded LLM engine so the model stays in memory across turns.
    pub fn set_llm_engine(&mut self, engine: Arc<LlmEngineHandle>) {
        self.llm_engine = Some(engine);
    }

    /// Get the shared LLM engine, if set.
    pub fn llm_engine(&self) -> Option<&Arc<LlmEngineHandle>> {
        self.llm_engine.as_ref()
    }

    /// Set a shared bash kill signal used by bash tool invocations.
    pub fn set_bash_kill_signal(&mut self, signal: crate::tools::bash::BashKillSignal) {
        self.bash_kill_signal = Some(signal);
    }

    fn setup_middleware(&mut self) {
        self.middleware_pipeline.clear();
        self.middleware_pipeline
            .add(TurnLimitMiddleware::new(DEFAULT_MAX_TURNS));

        if self.config.auto_compact_threshold > 0 {
            self.middleware_pipeline.add(AutoCompactMiddleware::new(
                self.config.auto_compact_threshold,
            ));
            self.middleware_pipeline.add(ContextWarningMiddleware::new(
                self.config.auto_compact_threshold,
            ));
        }

        self.middleware_pipeline.add(PlanAgentMiddleware::new());
    }

    fn refresh_context_tokens(&mut self) {
        self.stats.context_tokens = estimate_messages_tokens(&self.messages);
    }

    fn conversation_context(&self) -> ConversationContext {
        ConversationContext {
            messages: self.messages.clone(),
            stats: self.stats.clone(),
            agent_name: self.agent_name.clone(),
        }
    }

    fn fallback_compact(&mut self) {
        if self.messages.len() <= 2 {
            return;
        }

        let system = self.messages.first().cloned();
        let mut tail: Vec<_> = self
            .messages
            .iter()
            .skip(1)
            .rev()
            .take(FALLBACK_TAIL_MESSAGES)
            .cloned()
            .collect();
        tail.reverse();
        for msg in &mut tail {
            match msg.role {
                MessageRole::Tool => compact_message_content(msg, FALLBACK_TOOL_MAX_CHARS, 4, 2),
                MessageRole::Assistant | MessageRole::User => {
                    compact_message_content(msg, FALLBACK_TEXT_MAX_CHARS, 8, 3)
                }
                MessageRole::System => {}
            }
        }

        self.messages.clear();
        if let Some(system_message) = system {
            self.messages.push(system_message);
        }
        self.messages.extend(tail);
    }

    /// Add a system message.
    pub fn add_system_message(&mut self, content: impl Into<String>) {
        self.messages.push(LlmMessage::system(content));
        self.refresh_context_tokens();
    }

    /// Add a user message.
    pub fn add_user_message(&mut self, content: impl Into<String>) {
        self.messages.push(LlmMessage::user(content));
        self.refresh_context_tokens();
    }

    /// Add a user message with image attachments encoded as data URLs.
    pub fn add_user_message_with_images(
        &mut self,
        content: impl Into<String>,
        image_data_urls: impl IntoIterator<Item = String>,
    ) {
        self.messages
            .push(LlmMessage::user_with_images(content, image_data_urls));
        self.refresh_context_tokens();
    }

    /// Add an assistant message.
    pub fn add_assistant_message(&mut self, content: impl Into<String>) {
        self.messages.push(LlmMessage::assistant(content));
        self.refresh_context_tokens();
    }

    /// Append an existing sequence of messages (used for session resume).
    pub fn extend_messages<I>(&mut self, messages: I)
    where
        I: IntoIterator<Item = LlmMessage>,
    {
        self.messages.extend(messages);
        self.refresh_context_tokens();
    }

    /// Set current agent profile name used by middleware.
    pub fn set_agent_name(&mut self, name: impl Into<String>) {
        self.agent_name = name.into();
    }

    /// Set auto-approve flag at runtime (YOLO mode toggle).
    pub fn set_auto_approve(&mut self, value: bool) {
        self.config.auto_approve = value;
    }

    pub fn set_subagent_progress_tx(
        &mut self,
        tx: std::sync::mpsc::Sender<(String, crate::types::SubagentProgress)>,
    ) {
        self.subagent_progress_tx = Some(tx);
    }

    /// Get messages for the LLM.
    pub fn messages(&self) -> &[LlmMessage] {
        &self.messages
    }

    /// Get current stats.
    pub fn stats(&self) -> &AgentStats {
        &self.stats
    }

    /// Get tool manager.
    pub fn tool_manager(&self) -> &Arc<ToolManager> {
        &self.tool_manager
    }

    /// Register approval handler used when a tool is configured as `ask`.
    pub fn set_approval_handler<F>(&mut self, handler: F)
    where
        F: Fn(ApprovalRequest) -> ApprovalDecision + Send + Sync + 'static,
    {
        self.approval_handler = Some(Arc::new(handler));
    }

    /// Register ask_user_question handler used by interactive UI.
    pub fn set_question_handler<F>(&mut self, handler: F)
    where
        F: Fn(UserQuestionRequest) -> UserQuestionResponse + Send + Sync + 'static,
    {
        self.question_handler = Some(Arc::new(handler));
    }

    /// Internal: dispatch LLM call via shared engine or legacy per-call path.
    async fn llm_call(
        &self,
        model_path: &Path,
        messages: &[LlmMessage],
        max_tokens: u32,
        tools: Option<serde_json::Value>,
        tool_choice: Option<serde_json::Value>,
        interrupt_signal: Option<Arc<AtomicBool>>,
    ) -> Result<String, String> {
        if let Some(engine) = &self.llm_engine {
            chat_via_engine_streaming(
                engine.clone(),
                messages,
                max_tokens,
                tools,
                tool_choice,
                interrupt_signal,
                |_| {},
            )
            .await
        } else {
            chat_via_openai_server_streaming(
                model_path,
                &self.config,
                messages,
                max_tokens,
                tools,
                tool_choice,
                interrupt_signal,
                |_| {},
            )
            .await
        }
    }

    /// Execute full multi-turn loop with tool-calling.
    pub async fn act(&mut self, model_path: &Path, max_tokens: u32) -> Result<String, String> {
        self.act_with_events(model_path, max_tokens, |_| {}).await
    }

    /// Execute the loop and emit progress events for CLI rendering.
    pub async fn act_with_events<F>(
        &mut self,
        model_path: &Path,
        max_tokens: u32,
        on_event: F,
    ) -> Result<String, String>
    where
        F: FnMut(LoopEvent),
    {
        self.act_with_events_interruptable(model_path, max_tokens, None, on_event)
            .await
    }

    /// Execute the loop and emit progress events with cooperative interruption support.
    pub async fn act_with_events_interruptable<F>(
        &mut self,
        model_path: &Path,
        max_tokens: u32,
        interrupt_signal: Option<Arc<AtomicBool>>,
        mut on_event: F,
    ) -> Result<String, String>
    where
        F: FnMut(LoopEvent),
    {
        let available_tools = build_available_tools_schema(&self.tool_manager.available_tools());
        let tools_json = tools_to_json(&available_tools);
        let tools_value = serde_json::from_str::<serde_json::Value>(&tools_json).ok();

        let mut safety_guard: u32 = 0;
        let mut deferred_execution_repairs: u8 = 0;
        let mut repetition_repairs: u8 = 0;
        loop {
            check_interrupt_signal(interrupt_signal.as_ref())?;
            safety_guard = safety_guard.saturating_add(1);
            if safety_guard > 128 {
                return Err("Agent loop safety guard exceeded".to_string());
            }
            self.refresh_context_tokens();

            let middleware_result = self
                .middleware_pipeline
                .run_before_turn(&self.conversation_context())
                .await;

            match middleware_result.action {
                MiddlewareAction::Continue => {}
                MiddlewareAction::InjectMessage => {
                    if let Some(message) = middleware_result.message {
                        let trimmed = message.trim();
                        if !trimmed.is_empty() {
                            self.messages.push(LlmMessage::user(trimmed.to_string()));
                            self.refresh_context_tokens();
                        }
                    }
                }
                MiddlewareAction::Stop => {
                    let reason = middleware_result
                        .reason
                        .unwrap_or_else(|| "Conversation stopped by middleware".to_string());
                    on_event(LoopEvent::StoppedByMiddleware {
                        reason: reason.clone(),
                    });
                    let final_message = LlmMessage::assistant(reason.clone());
                    self.messages.push(final_message.clone());
                    self.refresh_context_tokens();
                    on_event(LoopEvent::Message(final_message));
                    return Ok(reason);
                }
                MiddlewareAction::Compact => {
                    check_interrupt_signal(interrupt_signal.as_ref())?;
                    let old_tokens = middleware_result
                        .metadata
                        .get("old_tokens")
                        .and_then(serde_json::Value::as_u64)
                        .map(|v| v as u32)
                        .unwrap_or(self.stats.context_tokens);
                    let threshold = middleware_result
                        .metadata
                        .get("threshold")
                        .and_then(serde_json::Value::as_u64)
                        .map(|v| v as u32)
                        .unwrap_or(self.config.auto_compact_threshold);

                    on_event(LoopEvent::CompactStart {
                        old_context_tokens: old_tokens,
                        threshold,
                    });
                    let summary = self
                        .compact(model_path, max_tokens, interrupt_signal.clone())
                        .await?;
                    check_interrupt_signal(interrupt_signal.as_ref())?;
                    on_event(LoopEvent::CompactEnd {
                        old_context_tokens: old_tokens,
                        new_context_tokens: self.stats.context_tokens,
                        summary_len: summary.chars().count(),
                    });
                }
            }

            let tool_choice = if tools_value.is_some() {
                Some(serde_json::json!("auto"))
            } else {
                None
            };

            let effective_messages = merge_consecutive_user_messages(&self.messages);
            let prompt_tokens_estimate = estimate_messages_tokens(&effective_messages);
            self.stats.context_tokens = prompt_tokens_estimate;

            check_interrupt_signal(interrupt_signal.as_ref())?;
            let assistant_text = if let Some(engine) = &self.llm_engine {
                chat_via_engine_streaming(
                    engine.clone(),
                    &effective_messages,
                    max_tokens,
                    tools_value.clone(),
                    tool_choice,
                    interrupt_signal.clone(),
                    |chunk| {
                        on_event(LoopEvent::Chunk(chunk));
                    },
                )
                .await?
            } else {
                chat_via_openai_server_streaming(
                    model_path,
                    &self.config,
                    &effective_messages,
                    max_tokens,
                    tools_value.clone(),
                    tool_choice,
                    interrupt_signal.clone(),
                    |chunk| {
                        on_event(LoopEvent::Chunk(chunk));
                    },
                )
                .await?
            };
            check_interrupt_signal(interrupt_signal.as_ref())?;

            self.stats.turns += 1;
            self.stats.tokens_in += prompt_tokens_estimate;
            self.stats.tokens_out += estimate_tokens(&assistant_text);
            self.stats.tokens_used = self.stats.tokens_in + self.stats.tokens_out;
            on_event(LoopEvent::Stats(self.stats.clone()));

            let tool_calls = parse_tool_calls(&assistant_text);
            if tool_calls.is_empty() {
                if tools_value.is_some() && looks_like_tool_call_attempt(&assistant_text) {
                    let repair = malformed_tool_call_repair_prompt(&assistant_text);
                    self.messages.push(LlmMessage::assistant(assistant_text));
                    self.messages.push(LlmMessage::user(repair));
                    self.refresh_context_tokens();
                    continue;
                }
                if tools_value.is_some()
                    && deferred_execution_repairs < 2
                    && looks_like_deferred_execution(&assistant_text)
                {
                    let repair = deferred_execution_repair_prompt(&assistant_text);
                    self.messages.push(LlmMessage::assistant(assistant_text));
                    self.messages.push(LlmMessage::user(repair));
                    self.refresh_context_tokens();
                    deferred_execution_repairs = deferred_execution_repairs.saturating_add(1);
                    repetition_repairs = 0;
                    continue;
                }
                if repetition_repairs < 2 && looks_like_repetition_loop(&assistant_text) {
                    let repair = repetition_repair_prompt(&assistant_text);
                    self.messages.push(LlmMessage::assistant(assistant_text));
                    self.messages.push(LlmMessage::user(repair));
                    self.refresh_context_tokens();
                    repetition_repairs = repetition_repairs.saturating_add(1);
                    deferred_execution_repairs = 0;
                    continue;
                }

                let final_message = LlmMessage::assistant(assistant_text.clone());
                self.messages.push(final_message.clone());
                self.refresh_context_tokens();
                on_event(LoopEvent::Message(final_message));
                return Ok(assistant_text);
            }

            deferred_execution_repairs = 0;
            repetition_repairs = 0;

            self.messages.push(LlmMessage {
                role: MessageRole::Assistant,
                content: MessageContent::text(String::new()),
                name: None,
                tool_call_id: None,
                tool_calls: Some(tool_calls.clone()),
            });
            self.refresh_context_tokens();

            let mut seen_calls: HashSet<String> = HashSet::new();
            for call in tool_calls {
                check_interrupt_signal(interrupt_signal.as_ref())?;
                let call_signature = format!("{}:{}", call.name, call.arguments);
                if !seen_calls.insert(call_signature) {
                    continue;
                }

                on_event(LoopEvent::ToolCall(call.clone()));
                self.stats.tools_called += 1;
                let tool_result = self.execute_tool_call(&call, model_path).await;
                check_interrupt_signal(interrupt_signal.as_ref())?;
                on_event(LoopEvent::ToolResult {
                    call_id: call.id.clone(),
                    result: tool_result.clone(),
                });
                self.messages.push(LlmMessage::tool(tool_result, call.id));
                self.refresh_context_tokens();
            }
        }
    }

    async fn execute_tool_call(&self, call: &ToolCall, model_path: &Path) -> String {
        if !self.tool_manager.has_tool(&call.name) {
            return format!("Tool not found: {}", call.name);
        }
        if !self.tool_manager.is_tool_enabled(&call.name) {
            return format!(
                "Tool denied by active agent policy: {}. Tool denied, choose alternative.",
                call.name
            );
        }

        let permission = self
            .tool_manager
            .get_permission(&call.name, &call.arguments);
        match permission {
            ToolPermission::Never => {
                return format!(
                    "Tool denied by permission policy: {}. Tool denied, choose alternative.",
                    call.name
                );
            }
            ToolPermission::Ask if !self.config.auto_approve => {
                let request = ApprovalRequest {
                    tool_call_id: call.id.clone(),
                    tool_name: call.name.clone(),
                    arguments: call.arguments.clone(),
                    permission,
                };

                let decision = self
                    .approval_handler
                    .as_ref()
                    .map(|handler| handler(request))
                    .unwrap_or(ApprovalDecision::Deny);

                match decision {
                    ApprovalDecision::ApproveOnce => {}
                    ApprovalDecision::ApproveAlwaysToolSession => {
                        let _ = self
                            .tool_manager
                            .set_permission(&call.name, ToolPermission::Always);
                    }
                    ApprovalDecision::Deny => {
                        return format!(
                            "Tool denied by user decision: {}. Tool denied, choose alternative.",
                            call.name
                        );
                    }
                }
            }
            ToolPermission::Always | ToolPermission::Ask => {}
        }

        // Reset bash kill signal before each bash invocation.
        let bash_signal = if call.name == "bash" {
            if let Some(ref sig) = self.bash_kill_signal {
                sig.store(false, std::sync::atomic::Ordering::Relaxed);
                Some(sig.clone())
            } else {
                None
            }
        } else {
            None
        };

        let ctx = InvokeContext {
            tool_call_id: call.id.clone(),
            approval_tx: None,
            question_handler: self.question_handler.clone(),
            subagent_progress_tx: self.subagent_progress_tx.clone(),
            runtime_config: Some(self.config.clone()),
            runtime_model_path: Some(model_path.to_path_buf()),
            llm_engine: self.llm_engine.clone(),
            bash_kill_signal: bash_signal,
        };

        match self
            .tool_manager
            .invoke(&call.name, call.arguments.clone(), &ctx)
            .await
        {
            Ok(output) => output.into_text(),
            Err(err) => format!("Tool execution failed ({}): {}", call.name, err),
        }
    }

    /// Compact the conversation context.
    pub async fn compact(
        &mut self,
        model_path: &Path,
        max_tokens: u32,
        interrupt_signal: Option<Arc<AtomicBool>>,
    ) -> Result<String, String> {
        check_interrupt_signal(interrupt_signal.as_ref())?;
        let Some(system_message) = self.messages.first().cloned() else {
            return Err("Cannot compact empty conversation".to_string());
        };

        let summary_max_tokens = summary_max_tokens_for_compaction(max_tokens);
        let mut compact_input =
            prepare_messages_for_compaction(&self.messages, &self.config, summary_max_tokens);

        let mut summary_result = {
            let mut summary_messages = compact_input.clone();
            summary_messages.push(LlmMessage::user(COMPACT_SUMMARY_PROMPT));
            self.llm_call(
                model_path,
                &summary_messages,
                summary_max_tokens,
                None,
                None,
                interrupt_signal.clone(),
            )
            .await
        };

        if let Err(err) = &summary_result {
            if !is_user_interrupted_error(err) && is_prompt_too_long_error(err) {
                check_interrupt_signal(interrupt_signal.as_ref())?;
                let retry_budget =
                    compaction_input_budget(&self.config, summary_max_tokens).saturating_div(2);
                compact_input = retain_recent_messages_within_budget(compact_input, retry_budget);

                let mut retry_messages = compact_input;
                retry_messages.push(LlmMessage::user(COMPACT_SUMMARY_PROMPT));
                summary_result = self
                    .llm_call(
                        model_path,
                        &retry_messages,
                        summary_max_tokens,
                        None,
                        None,
                        interrupt_signal.clone(),
                    )
                    .await;
            }
        }

        let summary = match summary_result {
            Ok(summary) => trim_summary_for_budget(summary.trim()),
            Err(err) if is_user_interrupted_error(&err) => return Err(err),
            Err(_) => String::new(),
        };

        check_interrupt_signal(interrupt_signal.as_ref())?;

        if summary.is_empty() {
            self.fallback_compact();
            self.middleware_pipeline.reset(ResetReason::Compact);
            self.refresh_context_tokens();
            return Ok(String::new());
        }

        self.messages = vec![system_message, LlmMessage::user(summary.clone())];
        self.middleware_pipeline.reset(ResetReason::Compact);
        self.refresh_context_tokens();
        Ok(summary)
    }

    /// Reset the conversation.
    pub fn reset(&mut self) {
        let system = self.messages.first().cloned();
        self.messages.clear();
        if let Some(sys) = system {
            self.messages.push(sys);
        }
        self.stats = AgentStats::default();
        self.middleware_pipeline.reset(ResetReason::Stop);
        self.refresh_context_tokens();
    }
}

/// Events from agent loop.
#[derive(Debug)]
pub enum LoopEvent {
    /// Token stream chunk.
    Chunk(String),
    /// Assistant message complete.
    Message(LlmMessage),
    /// Tool call.
    ToolCall(ToolCall),
    /// Tool result.
    ToolResult { call_id: String, result: String },
    /// Auto-compact started.
    CompactStart {
        old_context_tokens: u32,
        threshold: u32,
    },
    /// Auto-compact completed.
    CompactEnd {
        old_context_tokens: u32,
        new_context_tokens: u32,
        summary_len: usize,
    },
    /// Loop stopped by middleware.
    StoppedByMiddleware { reason: String },
    /// Error.
    Error(String),
    /// Stats update.
    Stats(AgentStats),
}

#[cfg(test)]
mod tests {
    use super::{
        compact_text_for_context, estimate_tokens, looks_like_deferred_execution,
        looks_like_repetition_loop, prepare_messages_for_compaction,
        retain_recent_messages_within_budget, summary_max_tokens_for_compaction,
    };
    use crate::config::NcConfig;
    use crate::types::{LlmMessage, MessageRole};

    #[test]
    fn deferred_execution_detection_matches_pt_future_action() {
        let text = "Identifiquei o problema. Vou corrigir esses problemas no arquivo `index.html`.";
        assert!(looks_like_deferred_execution(text));
    }

    #[test]
    fn deferred_execution_detection_matches_en_future_action() {
        let text = "I will update the file now and then run tests.";
        assert!(looks_like_deferred_execution(text));
    }

    #[test]
    fn deferred_execution_detection_ignores_done_status() {
        let text = "Atualizei o arquivo e executei os testes com sucesso.";
        assert!(!looks_like_deferred_execution(text));
    }

    #[test]
    fn repetition_detection_matches_repeated_planning_prefix() {
        let text = r#"
plan loop marker one two three detail-a.
plan loop marker one two three detail-b.
plan loop marker one two three detail-c.
"#;
        assert!(looks_like_repetition_loop(text));
    }

    #[test]
    fn repetition_detection_ignores_diverse_lines() {
        let text = r#"
goal summary states desired outcome.
file finding references exact location.
validation checklist confirms expected behavior.
"#;
        assert!(!looks_like_repetition_loop(text));
    }

    #[test]
    fn token_estimate_handles_dense_code_like_text() {
        let dense = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        assert!(estimate_tokens(dense) >= 8);
    }

    #[test]
    fn compact_text_marks_truncation() {
        let text = (1..=30)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let compacted = compact_text_for_context(&text, 120, 3, 2);
        assert!(compacted.contains("[...compacted...]"));
        assert!(compacted.contains("line 1"));
    }

    #[test]
    fn retain_recent_messages_keeps_system_and_tail() {
        let messages = vec![
            LlmMessage::system("sys"),
            LlmMessage::user("old old old old old old old"),
            LlmMessage::assistant("mid mid mid mid"),
            LlmMessage::user("latest request"),
        ];
        let kept = retain_recent_messages_within_budget(messages, 6);
        assert_eq!(kept[0].role, MessageRole::System);
        assert!(kept.iter().any(|m| m
            .content
            .as_text()
            .unwrap_or_default()
            .contains("latest request")));
    }

    #[test]
    fn prepare_messages_compacts_old_tool_outputs() {
        let config = NcConfig::default();
        let summary_max = summary_max_tokens_for_compaction(4096);
        let long_tool_output = "x".repeat(4000);
        let messages = vec![
            LlmMessage::system("sys"),
            LlmMessage::user("first turn"),
            LlmMessage::tool(long_tool_output, "call_1"),
            LlmMessage::user("second turn"),
            LlmMessage::assistant("ack"),
            LlmMessage::user("third turn"),
            LlmMessage::assistant("ack 2"),
            LlmMessage::user("fourth turn"),
        ];

        let prepared = prepare_messages_for_compaction(&messages, &config, summary_max);
        let tool_msg = prepared
            .iter()
            .find(|m| m.role == MessageRole::Tool)
            .expect("tool message should exist");
        let tool_text = tool_msg.content.to_plain_text_lossy();
        assert!(tool_text.contains("tool output compacted"));
    }
}

//! Agent loop implementation

use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crate::config::NcConfig;
use crate::llm::{
    build_available_tools_schema, chat_via_openai_server_streaming, parse_tool_calls, tools_to_json,
};
use crate::middleware::{
    AutoCompactMiddleware, ContextWarningMiddleware, ConversationContext, MiddlewareAction,
    MiddlewarePipeline, PlanAgentMiddleware, ResetReason, TurnLimitMiddleware,
};
use crate::tools::ToolManager;
use crate::types::*;

const DEFAULT_MAX_TURNS: u32 = 8;
const COMPACT_SUMMARY_PROMPT: &str = "Summarize the current conversation for continuation. Preserve user goals, key constraints, decisions, pending tasks, and the latest tool outcomes. Keep it concise and actionable.";

fn estimate_tokens(text: &str) -> u32 {
    text.split_whitespace().count().max(1) as u32
}

fn estimate_messages_tokens(messages: &[LlmMessage]) -> u32 {
    messages
        .iter()
        .map(|msg| estimate_tokens(&msg.content))
        .sum::<u32>()
}

fn merge_consecutive_user_messages(messages: &[LlmMessage]) -> Vec<LlmMessage> {
    let mut merged: Vec<LlmMessage> = Vec::with_capacity(messages.len());

    for msg in messages {
        if msg.role == MessageRole::User {
            if let Some(last) = merged.last_mut() {
                if last.role == MessageRole::User {
                    let prev = last.content.trim();
                    let curr = msg.content.trim();
                    last.content = if prev.is_empty() {
                        curr.to_string()
                    } else if curr.is_empty() {
                        prev.to_string()
                    } else {
                        format!("{}\n\n{}", prev, curr)
                    };
                    continue;
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

fn malformed_tool_call_repair_prompt() -> &'static str {
    "Seu último output de tool call veio inválido, truncado ou fora do contrato. Responda SOMENTE com JSON OpenAI válido neste formato: {\"tool_calls\":[{\"id\":\"call_...\",\"type\":\"function\",\"function\":{\"name\":\"...\",\"arguments\":\"{\\\"campo\\\":\\\"valor\\\"}\"}}]}. Sem texto adicional."
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
        };
        loop_state.setup_middleware();
        loop_state
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
        if self.messages.len() <= 3 {
            return;
        }

        let system = self.messages.first().cloned();
        let last: Vec<_> = self.messages.iter().rev().take(4).cloned().collect();

        self.messages.clear();
        if let Some(system_message) = system {
            self.messages.push(system_message);
        }
        self.messages.extend(last.into_iter().rev());
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

    /// Add an assistant message.
    pub fn add_assistant_message(&mut self, content: impl Into<String>) {
        self.messages.push(LlmMessage::assistant(content));
        self.refresh_context_tokens();
    }

    /// Set current agent profile name used by middleware.
    pub fn set_agent_name(&mut self, name: impl Into<String>) {
        self.agent_name = name.into();
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
        loop {
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
                    let summary = self.compact(model_path, max_tokens).await?;
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

            let assistant_text = chat_via_openai_server_streaming(
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
            .await?;

            self.stats.turns += 1;
            self.stats.tokens_in += prompt_tokens_estimate;
            self.stats.tokens_out += estimate_tokens(&assistant_text);
            self.stats.tokens_used = self.stats.tokens_in + self.stats.tokens_out;
            on_event(LoopEvent::Stats(self.stats.clone()));

            let tool_calls = parse_tool_calls(&assistant_text);
            if tool_calls.is_empty() {
                if tools_value.is_some() && looks_like_tool_call_attempt(&assistant_text) {
                    self.messages.push(LlmMessage::assistant(assistant_text));
                    self.messages
                        .push(LlmMessage::user(malformed_tool_call_repair_prompt()));
                    self.refresh_context_tokens();
                    continue;
                }

                let final_message = LlmMessage::assistant(assistant_text.clone());
                self.messages.push(final_message.clone());
                self.refresh_context_tokens();
                on_event(LoopEvent::Message(final_message));
                return Ok(assistant_text);
            }

            self.messages.push(LlmMessage {
                role: MessageRole::Assistant,
                content: String::new(),
                name: None,
                tool_call_id: None,
                tool_calls: Some(tool_calls.clone()),
            });
            self.refresh_context_tokens();

            let mut seen_calls: HashSet<String> = HashSet::new();
            for call in tool_calls {
                let call_signature = format!("{}:{}", call.name, call.arguments);
                if !seen_calls.insert(call_signature) {
                    continue;
                }

                on_event(LoopEvent::ToolCall(call.clone()));
                self.stats.tools_called += 1;
                let tool_result = self.execute_tool_call(&call).await;
                on_event(LoopEvent::ToolResult {
                    call_id: call.id.clone(),
                    result: tool_result.clone(),
                });
                self.messages.push(LlmMessage::tool(tool_result, call.id));
                self.refresh_context_tokens();
            }
        }
    }

    async fn execute_tool_call(&self, call: &ToolCall) -> String {
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

        let ctx = InvokeContext {
            tool_call_id: call.id.clone(),
            approval_tx: None,
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
    pub async fn compact(&mut self, model_path: &Path, max_tokens: u32) -> Result<String, String> {
        let Some(system_message) = self.messages.first().cloned() else {
            return Err("Cannot compact empty conversation".to_string());
        };

        let mut summary_messages = self.messages.clone();
        summary_messages.push(LlmMessage::user(COMPACT_SUMMARY_PROMPT));

        let summary_max_tokens = max_tokens.clamp(256, 1024);
        let summary_result = chat_via_openai_server_streaming(
            model_path,
            &self.config,
            &summary_messages,
            summary_max_tokens,
            None,
            None,
            None,
            |_| {},
        )
        .await;

        let summary = summary_result
            .ok()
            .map(|s| s.trim().to_string())
            .unwrap_or_default();

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

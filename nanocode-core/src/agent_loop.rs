//! Agent loop implementation

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

fn estimate_tokens(text: &str) -> u32 {
    text.split_whitespace().count().max(1) as u32
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

use crate::config::NcConfig;
use crate::llm::{
    build_available_tools_schema, chat_via_openai_server_streaming, parse_tool_calls, tools_to_json,
};
use crate::tools::ToolManager;
use crate::types::*;

const DEFAULT_MAX_TURNS: u32 = 8;

/// Agent loop that orchestrates LLM calls and tool execution
pub struct AgentLoop {
    config: NcConfig,
    messages: Vec<LlmMessage>,
    stats: AgentStats,
    tool_manager: Arc<ToolManager>,
}

impl AgentLoop {
    pub fn new(config: NcConfig, tool_manager: ToolManager) -> Self {
        Self {
            config,
            messages: Vec::new(),
            stats: AgentStats::default(),
            tool_manager: Arc::new(tool_manager),
        }
    }

    /// Add a system message
    pub fn add_system_message(&mut self, content: impl Into<String>) {
        self.messages.push(LlmMessage::system(content));
    }

    /// Add a user message
    pub fn add_user_message(&mut self, content: impl Into<String>) {
        self.messages.push(LlmMessage::user(content));
    }

    /// Add an assistant message
    pub fn add_assistant_message(&mut self, content: impl Into<String>) {
        self.messages.push(LlmMessage::assistant(content));
    }

    /// Get messages for the LLM
    pub fn messages(&self) -> &[LlmMessage] {
        &self.messages
    }

    /// Get current stats
    pub fn stats(&self) -> &AgentStats {
        &self.stats
    }

    /// Get tool manager
    pub fn tool_manager(&self) -> &Arc<ToolManager> {
        &self.tool_manager
    }

    /// Execute full multi-turn loop with tool-calling.
    pub async fn act(
        &mut self,
        model_path: &std::path::Path,
        max_tokens: u32,
    ) -> Result<String, String> {
        self.act_with_events(model_path, max_tokens, |_| {}).await
    }

    /// Execute the loop and emit progress events for CLI rendering.
    pub async fn act_with_events<F>(
        &mut self,
        model_path: &std::path::Path,
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
        model_path: &std::path::Path,
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

        for turn_idx in 0..DEFAULT_MAX_TURNS {
            let tool_choice = if turn_idx == 0 && tools_value.is_some() {
                Some(serde_json::json!("required"))
            } else {
                Some(serde_json::json!("auto"))
            };

            let prompt_tokens_estimate: u32 = self
                .messages
                .iter()
                .map(|m| estimate_tokens(&m.content))
                .sum();

            let assistant_text = chat_via_openai_server_streaming(
                model_path,
                &self.config,
                &self.messages,
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
                    continue;
                }

                let final_message = LlmMessage::assistant(assistant_text.clone());
                self.messages.push(final_message.clone());
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
            }
        }

        Err("Agent loop reached max turns without final answer".to_string())
    }

    async fn execute_tool_call(&self, call: &ToolCall) -> String {
        if !self.tool_manager.has_tool(&call.name) {
            return format!("Tool not found: {}", call.name);
        }

        let permission = self
            .tool_manager
            .get_permission(&call.name, &call.arguments);
        if permission == ToolPermission::Never {
            return format!("Permission denied for tool: {}", call.name);
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

    /// Compact the conversation context
    pub async fn compact(&mut self) -> Result<(), String> {
        // Simplified: just keep the first (system) and last few messages
        if self.messages.len() > 3 {
            let system = self.messages[0].clone();
            let last: Vec<_> = self.messages.iter().rev().take(3).cloned().collect();
            self.messages.clear();
            self.messages.push(system);
            self.messages.extend(last.into_iter().rev());
        }
        Ok(())
    }

    /// Reset the conversation
    pub fn reset(&mut self) {
        let system = self.messages.first().cloned();
        self.messages.clear();
        if let Some(sys) = system {
            self.messages.push(sys);
        }
        self.stats = AgentStats::default();
    }
}

/// Events from agent loop
#[derive(Debug)]
pub enum LoopEvent {
    /// Token stream chunk
    Chunk(String),
    /// Assistant message complete
    Message(LlmMessage),
    /// Tool call
    ToolCall(ToolCall),
    /// Tool result
    ToolResult { call_id: String, result: String },
    /// Error
    Error(String),
    /// Stats update
    Stats(AgentStats),
}

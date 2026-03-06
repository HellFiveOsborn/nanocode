//! Core types for Nano Code

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

use crate::config::NcConfig;

/// Role in the conversation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

impl std::fmt::Display for MessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageRole::System => write!(f, "system"),
            MessageRole::User => write!(f, "user"),
            MessageRole::Assistant => write!(f, "assistant"),
            MessageRole::Tool => write!(f, "tool"),
        }
    }
}

/// A message in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmMessage {
    pub role: MessageRole,
    #[serde(default)]
    pub content: MessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl LlmMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: MessageContent::text(content),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: MessageContent::text(content),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn user_with_images(
        content: impl Into<String>,
        image_data_urls: impl IntoIterator<Item = String>,
    ) -> Self {
        Self {
            role: MessageRole::User,
            content: MessageContent::from_text_and_images(content, image_data_urls),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: MessageContent::text(content),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Tool,
            content: MessageContent::text(content),
            name: None,
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: None,
        }
    }
}

/// Message content that supports text-only and multimodal OpenAI-style parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<MessageContentPart>),
}

impl Default for MessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl MessageContent {
    pub fn text(content: impl Into<String>) -> Self {
        Self::Text(content.into())
    }

    pub fn from_text_and_images(
        content: impl Into<String>,
        image_data_urls: impl IntoIterator<Item = String>,
    ) -> Self {
        let text = content.into();
        let mut parts = Vec::new();
        if !text.is_empty() {
            parts.push(MessageContentPart::Text { text });
        }

        for data_url in image_data_urls {
            if data_url.trim().is_empty() {
                continue;
            }
            parts.push(MessageContentPart::ImageUrl {
                image_url: ImageUrlPart {
                    url: data_url,
                    detail: None,
                },
            });
        }

        if parts.is_empty() {
            Self::Text(String::new())
        } else if parts.len() == 1 {
            if let MessageContentPart::Text { text } = &parts[0] {
                Self::Text(text.clone())
            } else {
                Self::Parts(parts)
            }
        } else {
            Self::Parts(parts)
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text.as_str()),
            Self::Parts(_) => None,
        }
    }

    pub fn to_plain_text_lossy(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Parts(parts) => {
                let mut out = String::new();
                for part in parts {
                    match part {
                        MessageContentPart::Text { text } => {
                            if !out.is_empty() {
                                out.push('\n');
                            }
                            out.push_str(text);
                        }
                        MessageContentPart::ImageUrl { .. } => {
                            if !out.is_empty() {
                                out.push('\n');
                            }
                            out.push_str("[image]");
                        }
                    }
                }
                out
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrlPart },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrlPart {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// A tool call from the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Tool definition for the LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailableTool {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Statistics about the agent session
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentStats {
    pub turns: u32,
    pub context_tokens: u32,
    pub tokens_used: u32,
    pub tokens_in: u32,
    pub tokens_out: u32,
    pub tools_called: u32,
}

/// Events emitted by the agent loop
#[derive(Debug)]
pub enum AgentEvent {
    /// Streaming chunk of assistant text
    AssistantChunk(String),
    /// Final assistant message (after thinking split)
    AssistantFinal { text: String },
    /// Thinking content (if separated by tags)
    AssistantThinking { text: String },
    /// Tool call request
    ToolCall { call: ToolCall },
    /// Tool execution result
    ToolResult {
        call_id: String,
        result: String,
        success: bool,
    },
    /// Error occurred
    Error(String),
    /// Session stats update
    Stats(AgentStats),
}

/// Tool output
#[derive(Debug, Clone)]
pub enum ToolOutput {
    Text(String),
    Structured(serde_json::Value),
}

impl ToolOutput {
    pub fn into_text(self) -> String {
        match self {
            ToolOutput::Text(s) => s,
            ToolOutput::Structured(v) => serde_json::to_string_pretty(&v).unwrap_or_default(),
        }
    }
}

/// Tool error
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("Tool not found: {0}")]
    NotFound(String),
    #[error("Invalid arguments: {0}")]
    InvalidArguments(String),
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Timeout")]
    Timeout,
}

/// Tool permission policy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ToolPermission {
    Always,
    Never,
    #[default]
    Ask,
}

/// User decision for tool approval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalDecision {
    ApproveOnce,
    ApproveAlwaysToolSession,
    Deny,
}

/// Source of answer for ask_user_question tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuestionAnswerSource {
    Choice,
    Text,
    Cancelled,
}

/// Interactive question request emitted by ask_user_question tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserQuestionRequest {
    pub tool_call_id: String,
    pub question: String,
    pub choices: Vec<String>,
    pub allow_free_text: bool,
    pub placeholder: Option<String>,
}

/// Interactive question answer returned by UI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserQuestionResponse {
    pub answer: String,
    pub choice_index: Option<usize>,
    pub source: QuestionAnswerSource,
    pub cancelled: bool,
}

impl UserQuestionResponse {
    pub fn cancelled() -> Self {
        Self {
            answer: String::new(),
            choice_index: None,
            source: QuestionAnswerSource::Cancelled,
            cancelled: true,
        }
    }
}

/// Progress event from a subagent (task tool).
#[derive(Debug, Clone)]
pub enum SubagentProgress {
    /// Subagent started a tool call.
    ToolCall { tool_name: String, summary: String },
    /// Subagent completed a tool call.
    ToolResult { tool_name: String, success: bool },
}

/// Invoke context for tool execution
pub struct InvokeContext {
    pub tool_call_id: String,
    pub approval_tx: Option<flume::Sender<ApprovalRequest>>,
    pub question_handler:
        Option<Arc<dyn Fn(UserQuestionRequest) -> UserQuestionResponse + Send + Sync>>,
    pub subagent_progress_tx: Option<std::sync::mpsc::Sender<(String, SubagentProgress)>>,
    /// Runtime config from the parent loop (used by task/subagent execution).
    pub runtime_config: Option<NcConfig>,
    /// Active model path from the parent loop (used by task/subagent execution).
    pub runtime_model_path: Option<PathBuf>,
    /// Shared pre-loaded LLM engine from the parent loop (avoids reloading model for subagents).
    pub llm_engine: Option<Arc<crate::llm::LlmEngineHandle>>,
    /// Kill signal for the bash tool — when set to true, the running process is terminated.
    pub bash_kill_signal: Option<crate::tools::bash::BashKillSignal>,
    /// How the model controls thinking (inherited from parent for subagents).
    pub thinking_control: nanocode_hf::ThinkingControl,
}

/// Approval request for a tool
#[derive(Debug, Clone)]
pub struct ApprovalRequest {
    pub tool_call_id: String,
    pub tool_name: String,
    pub arguments: serde_json::Value,
    pub permission: ToolPermission,
}

//! Core types for Nano Code

use serde::{Deserialize, Serialize};

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
    pub content: String,
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
            content: content.into(),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Tool,
            content: content.into(),
            name: None,
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: None,
        }
    }
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

/// Invoke context for tool execution
pub struct InvokeContext {
    pub tool_call_id: String,
    pub approval_tx: Option<flume::Sender<ApprovalRequest>>,
}

/// Approval request for a tool
#[derive(Debug, Clone)]
pub struct ApprovalRequest {
    pub tool_call_id: String,
    pub tool_name: String,
    pub arguments: serde_json::Value,
    pub permission: ToolPermission,
}

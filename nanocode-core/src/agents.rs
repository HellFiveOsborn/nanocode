//! Agent profiles

use serde::{Deserialize, Serialize};

/// Built-in agent types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BuiltinAgent {
    Default,
    Plan,
    AcceptEdits,
    AutoApprove,
}

impl Default for BuiltinAgent {
    fn default() -> Self {
        Self::Default
    }
}

/// Agent profile configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentProfile {
    pub name: String,
    pub auto_approve: bool,
    pub readonly: bool,
    pub system_prompt_override: Option<String>,
    pub disabled_tools: Vec<String>,
}

impl AgentProfile {
    pub fn from_builtin(agent: BuiltinAgent) -> Self {
        match agent {
            BuiltinAgent::Default => Self {
                name: "default".to_string(),
                auto_approve: false,
                readonly: false,
                system_prompt_override: None,
                disabled_tools: Vec::new(),
            },
            BuiltinAgent::Plan => Self {
                name: "plan".to_string(),
                auto_approve: false,
                readonly: true,
                system_prompt_override: Some(
                    "You are in plan mode. Only perform read-only operations. \
                    Do not execute bash commands or modify files."
                        .to_string(),
                ),
                disabled_tools: vec![
                    "bash".to_string(),
                    "write_file".to_string(),
                    "search_replace".to_string(),
                ],
            },
            BuiltinAgent::AcceptEdits => Self {
                name: "accept-edits".to_string(),
                auto_approve: false,
                readonly: false,
                system_prompt_override: Some(
                    "You can automatically approve write_file and search_replace operations. \
                    Ask for confirmation before bash commands."
                        .to_string(),
                ),
                disabled_tools: Vec::new(),
            },
            BuiltinAgent::AutoApprove => Self {
                name: "auto-approve".to_string(),
                auto_approve: true,
                readonly: false,
                system_prompt_override: None,
                disabled_tools: Vec::new(),
            },
        }
    }
}

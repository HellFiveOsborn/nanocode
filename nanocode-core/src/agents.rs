//! Agent profiles

use std::collections::{HashMap, HashSet};

use crate::llm::PromptVariant;
use crate::types::ToolPermission;
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

impl BuiltinAgent {
    pub const ALL: [Self; 4] = [
        Self::Default,
        Self::Plan,
        Self::AcceptEdits,
        Self::AutoApprove,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Plan => "plan",
            Self::AcceptEdits => "accept-edits",
            Self::AutoApprove => "auto-approve",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "default" => Some(Self::Default),
            "plan" => Some(Self::Plan),
            "accept-edits" => Some(Self::AcceptEdits),
            "auto-approve" => Some(Self::AutoApprove),
            _ => None,
        }
    }

    pub fn available_names() -> Vec<&'static str> {
        Self::ALL.iter().map(|agent| agent.as_str()).collect()
    }
}

/// Resolved agent policy.
#[derive(Debug, Clone)]
pub struct AgentPolicy {
    pub builtin: BuiltinAgent,
    pub prompt_variant: PromptVariant,
    pub auto_approve: bool,
    pub enabled_tools: HashSet<String>,
    pub tool_permission_overrides: HashMap<String, ToolPermission>,
}

impl AgentPolicy {
    pub fn resolve(name: &str) -> Result<Self, String> {
        let builtin = BuiltinAgent::parse(name).ok_or_else(|| {
            format!(
                "Invalid agent '{}'. Available: {}",
                name,
                BuiltinAgent::available_names().join(", ")
            )
        })?;
        Ok(Self::from_builtin(builtin))
    }

    pub fn from_builtin(agent: BuiltinAgent) -> Self {
        let all_tools = all_tools();
        match agent {
            BuiltinAgent::Default => Self {
                builtin: agent,
                prompt_variant: PromptVariant::AgentDefault,
                auto_approve: false,
                enabled_tools: all_tools.clone(),
                tool_permission_overrides: permission_overrides(
                    &all_tools,
                    ToolPermission::Ask,
                    None,
                ),
            },
            BuiltinAgent::Plan => Self {
                builtin: agent,
                prompt_variant: PromptVariant::AgentPlan,
                auto_approve: false,
                enabled_tools: to_set(["read_file", "grep"]),
                tool_permission_overrides: permission_overrides(
                    &to_set(["read_file", "grep"]),
                    ToolPermission::Ask,
                    None,
                ),
            },
            BuiltinAgent::AcceptEdits => Self {
                builtin: agent,
                prompt_variant: PromptVariant::AgentBuild,
                auto_approve: false,
                enabled_tools: all_tools.clone(),
                tool_permission_overrides: permission_overrides(
                    &all_tools,
                    ToolPermission::Ask,
                    Some(vec![
                        ("write_file".to_string(), ToolPermission::Always),
                        ("search_replace".to_string(), ToolPermission::Always),
                    ]),
                ),
            },
            BuiltinAgent::AutoApprove => Self {
                builtin: agent,
                prompt_variant: PromptVariant::AgentDefault,
                auto_approve: true,
                enabled_tools: all_tools.clone(),
                tool_permission_overrides: permission_overrides(
                    &all_tools,
                    ToolPermission::Ask,
                    None,
                ),
            },
        }
    }
}

fn all_tools() -> HashSet<String> {
    to_set(["bash", "read_file", "write_file", "grep", "search_replace"])
}

fn to_set<const N: usize>(tools: [&str; N]) -> HashSet<String> {
    tools.iter().map(|tool| (*tool).to_string()).collect()
}

fn permission_overrides(
    enabled_tools: &HashSet<String>,
    default_permission: ToolPermission,
    explicit: Option<Vec<(String, ToolPermission)>>,
) -> HashMap<String, ToolPermission> {
    let mut map = enabled_tools
        .iter()
        .map(|tool| (tool.clone(), default_permission))
        .collect::<HashMap<_, _>>();

    if let Some(entries) = explicit {
        for (tool, permission) in entries {
            map.insert(tool, permission);
        }
    }
    map
}

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
    Build,
    Explore,
}

impl Default for BuiltinAgent {
    fn default() -> Self {
        Self::Default
    }
}

impl BuiltinAgent {
    pub const ALL: [Self; 4] = [Self::Default, Self::Plan, Self::Build, Self::Explore];

    pub const PRIMARY_CYCLE: [Self; 3] = [Self::Plan, Self::Build, Self::Default];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Plan => "plan",
            Self::Build => "build",
            Self::Explore => "explore",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "default" | "ask" => Some(Self::Default),
            "plan" => Some(Self::Plan),
            "build" | "accept-edits" => Some(Self::Build),
            "explore" => Some(Self::Explore),
            _ => None,
        }
    }

    pub fn available_names() -> Vec<&'static str> {
        vec!["default", "plan", "build"]
    }

    pub fn primary_cycle_names() -> Vec<&'static str> {
        Self::PRIMARY_CYCLE
            .iter()
            .map(|agent| agent.as_str())
            .collect()
    }

    pub fn cycle_primary(self, reverse: bool) -> Self {
        let current_idx = Self::PRIMARY_CYCLE
            .iter()
            .position(|agent| *agent == self)
            .unwrap_or_else(|| {
                Self::PRIMARY_CYCLE
                    .iter()
                    .position(|agent| *agent == Self::Default)
                    .unwrap_or(0)
            });

        let count = Self::PRIMARY_CYCLE.len();
        let next_idx = if reverse {
            (current_idx + count - 1) % count
        } else {
            (current_idx + 1) % count
        };
        Self::PRIMARY_CYCLE[next_idx]
    }
}

/// Resolved agent policy.
#[derive(Debug, Clone)]
pub struct AgentPolicy {
    pub builtin: BuiltinAgent,
    pub prompt_variant: PromptVariant,
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
                enabled_tools: to_set(["read_file", "grep", "task", "ask_user_question"]),
                tool_permission_overrides: permission_overrides(
                    &to_set(["read_file", "grep", "task", "ask_user_question"]),
                    ToolPermission::Always,
                    None,
                ),
            },
            BuiltinAgent::Build => Self {
                builtin: agent,
                prompt_variant: PromptVariant::AgentBuild,
                enabled_tools: all_tools.clone(),
                tool_permission_overrides: permission_overrides(
                    &all_tools,
                    ToolPermission::Ask,
                    None,
                ),
            },
            BuiltinAgent::Explore => {
                let enabled_tools = to_set(["read_file", "grep"]);
                Self {
                    builtin: agent,
                    prompt_variant: PromptVariant::SubagentExplore,
                    enabled_tools: enabled_tools.clone(),
                    tool_permission_overrides: permission_overrides(
                        &enabled_tools,
                        ToolPermission::Always,
                        None,
                    ),
                }
            }
        }
    }
}

fn all_tools() -> HashSet<String> {
    to_set([
        "bash",
        "read_file",
        "write_file",
        "grep",
        "search_replace",
        "task",
        "ask_user_question",
    ])
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

#[cfg(test)]
mod tests {
    use super::BuiltinAgent;

    #[test]
    fn parse_build_alias_remains_compatible() {
        assert_eq!(BuiltinAgent::parse("build"), Some(BuiltinAgent::Build));
        assert_eq!(
            BuiltinAgent::parse("accept-edits"),
            Some(BuiltinAgent::Build)
        );
    }

    #[test]
    fn primary_cycle_skips_explore() {
        // Default -> Build (forward)
        assert_eq!(
            BuiltinAgent::Default.cycle_primary(false),
            BuiltinAgent::Plan
        );
        // Default -> Plan (reverse)
        assert_eq!(
            BuiltinAgent::Default.cycle_primary(true),
            BuiltinAgent::Build
        );
        // Explore falls back to Default position
        assert_eq!(
            BuiltinAgent::Explore.cycle_primary(false),
            BuiltinAgent::Plan
        );
    }

    #[test]
    fn available_names_excludes_explore() {
        let names = BuiltinAgent::available_names();
        assert!(!names.contains(&"explore"));
        assert!(names.contains(&"default"));
        assert!(names.contains(&"plan"));
        assert!(names.contains(&"build"));
    }

    #[test]
    fn auto_approve_alias_removed() {
        assert_eq!(BuiltinAgent::parse("auto-approve"), None);
    }
}

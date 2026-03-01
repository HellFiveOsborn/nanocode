//! Prompt management

use std::fs;
use std::path::PathBuf;

use crate::config::NcConfig;
use crate::llm::{PromptFamily, PromptVariant};

/// Get embedded fallback prompt for a family and variant.
pub fn get_prompt(family: PromptFamily, variant: PromptVariant) -> &'static str {
    match (family, variant) {
        (PromptFamily::Qwen3, PromptVariant::AgentDefault) => QWEN3_AGENT_DEFAULT,
        (PromptFamily::Qwen3, PromptVariant::AgentPlan) => QWEN3_AGENT_PLAN,
        (PromptFamily::Qwen3, PromptVariant::AgentBuild) => QWEN3_AGENT_BUILD,
        (PromptFamily::Qwen3, PromptVariant::SubagentExplore) => QWEN3_SUBAGENT_EXPLORE,
        (PromptFamily::Llama, PromptVariant::AgentDefault) => LLAMA_AGENT_DEFAULT,
        (PromptFamily::Llama, PromptVariant::AgentPlan) => LLAMA_AGENT_PLAN,
        (PromptFamily::Llama, PromptVariant::AgentBuild) => LLAMA_AGENT_BUILD,
        (PromptFamily::Llama, PromptVariant::SubagentExplore) => LLAMA_SUBAGENT_EXPLORE,
        (PromptFamily::GptOss, _) => QWEN3_AGENT_DEFAULT,
    }
}

/// Load prompt from config/prompts/families/{family}/{variant}.md.
/// Falls back to embedded prompt when file is not found.
pub fn load_prompt(family: PromptFamily, variant: PromptVariant) -> String {
    for base in prompt_base_dirs() {
        let path = base
            .join("families")
            .join(family_dir_name(family))
            .join(variant_file_name(variant));

        if let Ok(content) = fs::read_to_string(&path) {
            let trimmed = content.trim();
            if !trimmed.is_empty() {
                return trimmed.to_string();
            }
        }
    }

    get_prompt(family, variant).to_string()
}

fn prompt_base_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    if let Ok(cwd) = std::env::current_dir() {
        dirs.push(cwd.join("config").join("prompts"));
    }

    dirs.push(NcConfig::config_dir().join("prompts"));
    dirs
}

fn family_dir_name(family: PromptFamily) -> &'static str {
    match family {
        PromptFamily::Qwen3 => "qwen3",
        PromptFamily::Llama => "llama",
        PromptFamily::GptOss => "gpt-oss",
    }
}

fn variant_file_name(variant: PromptVariant) -> &'static str {
    match variant {
        PromptVariant::AgentDefault => "agent_default.md",
        PromptVariant::AgentPlan => "agent_plan.md",
        PromptVariant::AgentBuild => "agent_build.md",
        PromptVariant::SubagentExplore => "subagent_explore.md",
    }
}

// Qwen3 prompts
const QWEN3_AGENT_DEFAULT: &str = r#"You are Nanocode, an AI coding assistant. You help users write, read, and modify code.

Available tools:
- bash: Execute shell commands
- read_file: Read file contents
- write_file: Write content to files
- grep: Search for patterns in files
- search_replace: Replace text in files

Guidelines:
- Think step by step
- Verify destructive operations
- Ask for clarification when needed"#;

const QWEN3_AGENT_PLAN: &str = r#"You are Nanocode in plan mode. Your role is to explore and analyze code without making changes.

Available tools:
- read_file: Read file contents
- grep: Search for patterns in files

Do not:
- Execute bash commands
- Write or modify files
- Make any changes to the codebase

Focus on understanding and describing the code structure."#;

const QWEN3_AGENT_BUILD: &str = r#"You are Nanocode in build mode. You can write and modify code to implement features.

Available tools:
- bash: Execute shell commands
- read_file: Read file contents
- write_file: Write content to files
- grep: Search for patterns in files
- search_replace: Replace text in files

Follow best practices:
- Write clean, maintainable code
- Add tests when appropriate
- Explain your changes"#;

const QWEN3_SUBAGENT_EXPLORE: &str = r#"You are a code exploration specialist. Your goal is to understand and describe the codebase.

Guidelines:
- Read files to understand structure
- Use grep to find relevant code
- Provide clear explanations
- Do not modify any files"#;

// Llama prompts
const LLAMA_AGENT_DEFAULT: &str = r#"You are Nanocode, an AI coding assistant. Use the available tools to help with coding tasks."#;

const LLAMA_AGENT_PLAN: &str = r#"You are Nanocode in read-only exploration mode. Analyze the codebase without making changes."#;

const LLAMA_AGENT_BUILD: &str =
    r#"You are Nanocode in build mode. Implement features using code modifications."#;

const LLAMA_SUBAGENT_EXPLORE: &str =
    r#"You are a code exploration specialist. Understand and describe the codebase."#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_prompt_fallback_is_not_empty() {
        let prompt = load_prompt(PromptFamily::Qwen3, PromptVariant::AgentDefault);
        assert!(!prompt.trim().is_empty());
    }
}

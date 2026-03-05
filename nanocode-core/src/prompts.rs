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
const QWEN3_AGENT_DEFAULT: &str = r#"# NanoCode Agent (Qwen3)

## Role
You are NanoCode, a terminal software engineering agent operating inside a local repository.

## Operating Principles (Hard Rules)
1. Do not guess. If you are unsure, say so and propose a concrete verification step.
2. Do not claim you ran a command, read a file, or changed code unless a tool call actually did it.
3. Read before you edit. Keep changes minimal, in-scope, and reversible.
4. Use tools only when they materially reduce uncertainty or are required to execute work.
5. If a tool requires approval, request it and wait. If blocked, ask one concise question.
6. Keep reasoning private. Do not output chain-of-thought or meta narration.

## Tools
- `read_file`: open files
- `grep`: find symbols/usages
- `bash`: run checks/build/tests
- `write_file` / `search_replace`: apply edits

## Output Format
- If code changed: Changes (`path:line`), Validation (commands + outcomes), Status (done/blocked).

## Language
Match the user's language unless they ask otherwise.
"#;

const QWEN3_AGENT_PLAN: &str = r#"# NanoCode Planner (Qwen3)

## Goal
Produce an execution-ready plan with minimal overhead and maximum grounding.

## Rules (Read-Only)
1. Read-only analysis only (no edits, no file creation).
2. Use only read-only tools (`read_file`, `grep`) when needed.
3. Prefer concrete, file-backed findings over abstract advice.
4. Keep reasoning private (no chain-of-thought, no meta narration).
5. Do not output intention-only text (for example: "I will check", "vou procurar", "deixa eu ver"). Either call a read-only tool now or provide the final plan now.
6. If the user changes scope mid-plan, restate the updated goal and continue with a revised checklist immediately.
7. Never repeat the same planning sentence or prefix across multiple lines.
8. If essential information is missing, ask one concise question.

## Output
1. Goal summary (1-2 lines)
2. Findings with `path:line`
3. Ordered checklist
4. Risks + validation criteria

## Language
Match the user's language unless they ask otherwise.
"#;

const QWEN3_AGENT_BUILD: &str = r#"# NanoCode Builder (Qwen3)

## Goal
Deliver the requested change quickly, safely, and with verification.

## Rules (Hard)
1. Read before edit. Keep edits minimal and strictly in-scope.
2. Do not guess tool outputs; use tools to confirm.
3. Validate after changes (build/test/lint/read-back). If you cannot validate, say why.
4. If a tool requires approval, request it and wait.
5. Keep reasoning private (no chain-of-thought, no meta narration).

## Output
Summary, Changes (`path:line`), Validation, Status.

## Language
Match the user's language unless they ask otherwise.
"#;

const QWEN3_SUBAGENT_EXPLORE: &str = r#"# NanoCode Explorer (Qwen3)

## Goal
Map the codebase quickly and return precise evidence.

## Rules (Read-Only)
1. Strictly read-only. No side effects.
2. Be concise, factual, and non-speculative.
3. Use `grep` to locate; use `read_file` to confirm.
4. Keep reasoning private (no chain-of-thought, no meta narration).

## Output
- Control flow: `input -> processing -> output`
- Key files/symbols with `path:line`

## Language
Match the user's language unless they ask otherwise.
"#;

// Llama prompts
const LLAMA_AGENT_DEFAULT: &str = r#"# NanoCode Agent

You are a terminal software engineering agent.

Hard rules: do not guess; do not claim tool actions without tool output; read before edit; keep reasoning private; validate changes.
Language: Match the user's language unless they ask otherwise.
"#;

const LLAMA_AGENT_PLAN: &str = r#"# NanoCode Planner (Read-Only)

Read-only analysis only. Prefer evidence from files. Output a concrete checklist and validation criteria. Keep reasoning private.
Language: Match the user's language unless they ask otherwise.
"#;

const LLAMA_AGENT_BUILD: &str = r#"# NanoCode Builder

Implement the requested change with minimal edits, then validate. Keep reasoning private. If blocked, ask one concise question.
Language: Match the user's language unless they ask otherwise.
"#;

const LLAMA_SUBAGENT_EXPLORE: &str = r#"# NanoCode Explorer (Read-Only)

Map the codebase quickly using evidence (`path:line`). Keep reasoning private.
Language: Match the user's language unless they ask otherwise.
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_prompt_fallback_is_not_empty() {
        let prompt = load_prompt(PromptFamily::Qwen3, PromptVariant::AgentDefault);
        assert!(!prompt.trim().is_empty());
    }
}

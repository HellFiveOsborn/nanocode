//! Task tool (subagent runner)

use super::base::Tool;
use super::manager::ToolManager;
use crate::agent_loop::{AgentLoop, LoopEvent};
use crate::agents::{AgentPolicy, BuiltinAgent};
use crate::config::NcConfig;
use crate::llm::PromptFamily;
use crate::prompts::load_prompt;
use crate::types::{InvokeContext, SubagentProgress, ToolError, ToolOutput, ToolPermission};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct TaskArgs {
    task: String,
    #[serde(default)]
    agent: Option<String>,
}

pub struct TaskTool;

impl TaskTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TaskTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for TaskTool {
    fn name(&self) -> &str {
        "task"
    }

    fn description(&self) -> &str {
        "Run a restricted subagent task (default: explore, read-only)."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Subtask prompt for the subagent"
                },
                "agent": {
                    "type": "string",
                    "description": "Subagent name (only 'explore' is supported)"
                }
            },
            "required": ["task"]
        })
    }

    async fn invoke(
        &self,
        args: serde_json::Value,
        ctx: &InvokeContext,
    ) -> Result<ToolOutput, ToolError> {
        let args: TaskArgs =
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        let task = args.task.trim();
        if task.is_empty() {
            return Err(ToolError::InvalidArguments(
                "task must not be empty".to_string(),
            ));
        }

        let subagent = resolve_subagent(args.agent.as_deref())?;

        let (runtime_config, model_path) = resolve_runtime_config_and_model(ctx)?;
        let policy = AgentPolicy::from_builtin(subagent);

        let tool_manager = ToolManager::new(&runtime_config).await;
        tool_manager.set_enabled_tools(policy.enabled_tools.clone());
        for (tool_name, permission) in &policy.tool_permission_overrides {
            let _ = tool_manager.set_permission(tool_name, *permission);
        }

        let mut subloop = AgentLoop::new(runtime_config.clone(), tool_manager);
        subloop.set_agent_name(policy.builtin.as_str());
        subloop.add_system_message(load_prompt(PromptFamily::Qwen3, policy.prompt_variant));
        subloop.add_user_message(task.to_string());

        let progress_tx = ctx.subagent_progress_tx.clone();
        let parent_call_id = ctx.tool_call_id.clone();
        let mut running_calls: HashMap<String, String> = HashMap::new();

        let mut stopped = false;
        let run_result = subloop
            .act_with_events(
                &model_path,
                runtime_config.model.max_tokens.clamp(256, 8192),
                |event| match &event {
                    LoopEvent::StoppedByMiddleware { .. } => {
                        stopped = true;
                    }
                    LoopEvent::ToolCall(call) => {
                        running_calls.insert(call.id.clone(), call.name.clone());
                        if let Some(tx) = &progress_tx {
                            let summary = format_subagent_tool_summary(&call.name, &call.arguments);
                            let _ = tx.send((
                                parent_call_id.clone(),
                                SubagentProgress::ToolCall {
                                    tool_name: call.name.clone(),
                                    summary,
                                },
                            ));
                        }
                    }
                    LoopEvent::ToolResult { call_id, result } => {
                        if let Some(tx) = &progress_tx {
                            let first = result.lines().next().unwrap_or("");
                            let lower = first.to_ascii_lowercase();
                            let success = !(lower.contains("failed")
                                || lower.contains("error")
                                || lower.contains("denied"));
                            let tool_name = running_calls.remove(call_id).unwrap_or_default();
                            let _ = tx.send((
                                parent_call_id.clone(),
                                SubagentProgress::ToolResult { tool_name, success },
                            ));
                        }
                    }
                    _ => {}
                },
            )
            .await;

        let turns_used = subloop.stats().turns;
        let tools_called = subloop.stats().tools_called;
        let tokens_used = subloop.stats().tokens_used;
        let (response, completed) = match run_result {
            Ok(response) => (response, !stopped),
            Err(err) => (err, false),
        };

        Ok(ToolOutput::Structured(json!({
            "response": response,
            "turns_used": turns_used,
            "tools_called": tools_called,
            "tokens_used": tokens_used,
            "completed": completed
        })))
    }

    fn permission(&self) -> ToolPermission {
        ToolPermission::Ask
    }
}

fn resolve_subagent(value: Option<&str>) -> Result<BuiltinAgent, ToolError> {
    let raw = value.unwrap_or("explore");
    let Some(agent) = BuiltinAgent::parse(raw) else {
        return Err(ToolError::InvalidArguments(format!(
            "Invalid subagent '{}'. Supported: explore",
            raw
        )));
    };

    if agent != BuiltinAgent::Explore {
        return Err(ToolError::InvalidArguments(format!(
            "Subagent '{}' is not supported in this stage. Supported: explore",
            raw
        )));
    }

    Ok(agent)
}

fn resolve_runtime_config_and_model(ctx: &InvokeContext) -> Result<(NcConfig, PathBuf), ToolError> {
    if let (Some(config), Some(model_path)) =
        (ctx.runtime_config.clone(), ctx.runtime_model_path.clone())
    {
        return Ok((config, model_path));
    }

    let config = ctx
        .runtime_config
        .clone()
        .or_else(|| NcConfig::load().ok())
        .unwrap_or_default();

    if let Some(model_path) = ctx.runtime_model_path.clone() {
        return Ok((config, model_path));
    }

    let models_dir = NcConfig::models_dir();

    let candidates = collect_gguf_files(&models_dir)?;
    let Some(model_path) = select_model_path(&config, &candidates) else {
        return Err(ToolError::ExecutionFailed(format!(
            "No .gguf model found in {}",
            models_dir.display()
        )));
    };

    Ok((config, model_path))
}

fn collect_gguf_files(models_dir: &Path) -> Result<Vec<PathBuf>, ToolError> {
    let entries = std::fs::read_dir(models_dir).map_err(|err| {
        ToolError::ExecutionFailed(format!(
            "Failed to read models directory {}: {}",
            models_dir.display(),
            err
        ))
    })?;

    let mut files = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        })
        .collect::<Vec<_>>();
    files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    Ok(files)
}

fn select_model_path(config: &NcConfig, candidates: &[PathBuf]) -> Option<PathBuf> {
    if candidates.is_empty() {
        return None;
    }

    let active_quant = config.active_quant.as_deref().map(str::to_ascii_lowercase);
    if let Some(quant) = active_quant {
        if let Some(found) = candidates.iter().find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.to_ascii_lowercase().contains(&quant))
                .unwrap_or(false)
        }) {
            return Some(found.clone());
        }
    }

    let active_model = config.active_model.as_deref().map(str::to_ascii_lowercase);
    if let Some(model) = active_model {
        if let Some(found) = candidates.iter().find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.to_ascii_lowercase().contains(&model))
                .unwrap_or(false)
        }) {
            return Some(found.clone());
        }
    }

    candidates.first().cloned()
}

fn format_subagent_tool_summary(tool_name: &str, arguments: &serde_json::Value) -> String {
    let display = match tool_name {
        "read_file" => "Read",
        "grep" => "Grep",
        "bash" => "Bash",
        other => other,
    };
    let target = match tool_name {
        "bash" => arguments
            .get("command")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(""),
        "grep" => arguments
            .get("pattern")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(""),
        _ => arguments
            .get("path")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(""),
    };
    let target_short: String = target.chars().take(100).collect();
    if target_short.is_empty() {
        format!("{}()", display)
    } else {
        format!("{}({})", display, target_short)
    }
}

#[cfg(test)]
mod tests {
    use super::{resolve_runtime_config_and_model, resolve_subagent, select_model_path};
    use crate::agents::BuiltinAgent;
    use crate::config::NcConfig;
    use crate::types::InvokeContext;
    use std::path::PathBuf;

    #[test]
    fn subagent_defaults_to_explore() {
        let result = resolve_subagent(None);
        assert!(matches!(result, Ok(BuiltinAgent::Explore)));
    }

    #[test]
    fn subagent_rejects_non_explore_names() {
        assert!(resolve_subagent(Some("build")).is_err());
        let invalid = resolve_subagent(Some("unknown"));
        assert!(invalid.is_err());
    }

    #[test]
    fn model_selection_prefers_active_quant_when_possible() {
        let mut config = NcConfig::default();
        config.active_quant = Some("Q4_K_M".to_string());

        let candidates = vec![
            PathBuf::from("other-model-Q2_K.gguf"),
            PathBuf::from("my-model-Q4_K_M.gguf"),
        ];

        let selected = select_model_path(&config, &candidates)
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()));

        assert_eq!(selected.as_deref(), Some("my-model-Q4_K_M.gguf"));
    }

    #[test]
    fn runtime_resolution_prefers_parent_loop_context() {
        let mut config = NcConfig::default();
        config.active_model = Some("from-context".to_string());
        let ctx = InvokeContext {
            tool_call_id: "call_1".to_string(),
            approval_tx: None,
            question_handler: None,
            subagent_progress_tx: None,
            runtime_config: Some(config.clone()),
            runtime_model_path: Some(PathBuf::from("/tmp/from-parent.gguf")),
        };

        let (resolved_config, resolved_model) =
            resolve_runtime_config_and_model(&ctx).expect("context runtime should resolve");
        assert_eq!(resolved_config.active_model, config.active_model);
        assert_eq!(resolved_model, PathBuf::from("/tmp/from-parent.gguf"));
    }
}

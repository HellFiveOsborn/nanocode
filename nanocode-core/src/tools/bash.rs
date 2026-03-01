//! Bash tool

use super::base::Tool;
use crate::types::ToolPermission;
use crate::types::{InvokeContext, ToolError, ToolOutput};
use async_trait::async_trait;
use serde::Deserialize;
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;
use tokio::time::timeout;

const DEFAULT_TIMEOUT: u64 = 300; // 5 minutes
const MAX_OUTPUT_BYTES: usize = 16 * 1024; // 16KB

#[derive(Debug, Deserialize)]
pub struct BashArgs {
    pub command: String,
    #[serde(default)]
    pub timeout: Option<u64>,
    #[serde(default)]
    pub working_dir: Option<String>,
}

pub struct BashTool {
    timeout: Duration,
}

impl BashTool {
    pub fn new() -> Self {
        Self {
            timeout: Duration::from_secs(DEFAULT_TIMEOUT),
        }
    }
}

impl Default for BashTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a bash command. Returns stdout and stderr."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds (default: 300)"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory for the command"
                }
            },
            "required": ["command"]
        })
    }

    async fn invoke(
        &self,
        args: serde_json::Value,
        _ctx: &InvokeContext,
    ) -> Result<ToolOutput, ToolError> {
        let args: BashArgs =
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        let timeout_secs = args.timeout.unwrap_or_else(|| self.timeout.as_secs());

        // Build the command
        let mut cmd = Command::new("bash");
        cmd.args(["-c", &args.command])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("CI", "true")
            .env("NONINTERACTIVE", "1")
            .env("PAGER", "cat");

        // Set working directory if specified
        if let Some(dir) = args.working_dir {
            cmd.current_dir(dir);
        }

        // Execute with timeout
        let result = timeout(Duration::from_secs(timeout_secs), cmd.output()).await;

        match result {
            Ok(Ok(output)) => {
                let mut result = String::new();

                // Add stdout
                if !output.stdout.is_empty() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let trimmed = stdout.trim();
                    if trimmed.len() > MAX_OUTPUT_BYTES {
                        result.push_str(&trimmed[..MAX_OUTPUT_BYTES]);
                        result.push_str("\n\n[Output truncated]");
                    } else {
                        result.push_str(trimmed);
                    }
                }

                // Add stderr
                if !output.stderr.is_empty() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let trimmed = stderr.trim();
                    if !trimmed.is_empty() {
                        if !result.is_empty() {
                            result.push_str("\n\n");
                        }
                        result.push_str("[stderr]: ");
                        result.push_str(trimmed);
                    }
                }

                // Add exit code if non-zero
                if !output.status.success() {
                    if !result.is_empty() {
                        result.push_str("\n\n");
                    }
                    result.push_str(&format!(
                        "[Exit code: {}]",
                        output.status.code().unwrap_or(-1)
                    ));
                }

                if result.is_empty() {
                    result = "[Command completed with no output]".to_string();
                }

                Ok(ToolOutput::Text(result))
            }
            Ok(Err(e)) => Err(ToolError::ExecutionFailed(e.to_string())),
            Err(_) => Err(ToolError::Timeout),
        }
    }

    fn permission(&self) -> ToolPermission {
        ToolPermission::Ask
    }
}

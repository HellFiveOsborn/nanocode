//! Bash tool

use super::base::Tool;
use crate::types::ToolPermission;
use crate::types::{InvokeContext, ToolError, ToolOutput};
use async_trait::async_trait;
use serde::Deserialize;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::AsyncReadExt;
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

/// Shared kill signal for a running bash process.
/// Set to `true` to request termination from outside.
pub type BashKillSignal = Arc<AtomicBool>;

/// Creates a new kill signal (initially false).
pub fn new_kill_signal() -> BashKillSignal {
    Arc::new(AtomicBool::new(false))
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
        ctx: &InvokeContext,
    ) -> Result<ToolOutput, ToolError> {
        let args: BashArgs =
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        let timeout_secs = args.timeout.unwrap_or_else(|| self.timeout.as_secs());

        // Build the command
        let mut cmd = Command::new("bash");
        cmd.args(["-c", &args.command])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .env("CI", "true")
            .env("NONINTERACTIVE", "1")
            .env("PAGER", "cat");

        // Set working directory if specified
        if let Some(dir) = args.working_dir {
            cmd.current_dir(dir);
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        // Register the kill signal so external code can terminate this process.
        let kill_signal = ctx
            .bash_kill_signal
            .clone()
            .unwrap_or_else(new_kill_signal);

        // Wait for completion, timeout, or kill signal.
        let result = tokio::select! {
            res = async {
                // Read stdout and stderr concurrently while waiting for exit.
                let mut stdout_buf = Vec::new();
                let mut stderr_buf = Vec::new();
                if let Some(mut stdout) = child.stdout.take() {
                    let mut stderr = child.stderr.take();
                    let (stdout_res, stderr_res) = tokio::join!(
                        async { stdout.read_to_end(&mut stdout_buf).await },
                        async {
                            if let Some(ref mut se) = stderr {
                                se.read_to_end(&mut stderr_buf).await
                            } else {
                                Ok(0)
                            }
                        }
                    );
                    let _ = stdout_res;
                    let _ = stderr_res;
                }
                let status = child.wait().await;
                (status, stdout_buf, stderr_buf)
            } => {
                Ok(res)
            }
            _ = timeout(Duration::from_secs(timeout_secs), std::future::pending::<()>()) => {
                let _ = child.kill().await;
                Err(ToolError::Timeout)
            }
            _ = poll_kill_signal(&kill_signal) => {
                let _ = child.kill().await;
                // Collect partial output after kill
                let mut partial_out = Vec::new();
                let mut partial_err = Vec::new();
                if let Some(mut stdout) = child.stdout.take() {
                    let _ = tokio::time::timeout(
                        Duration::from_millis(200),
                        stdout.read_to_end(&mut partial_out),
                    ).await;
                }
                if let Some(mut stderr) = child.stderr.take() {
                    let _ = tokio::time::timeout(
                        Duration::from_millis(200),
                        stderr.read_to_end(&mut partial_err),
                    ).await;
                }
                let _ = child.wait().await;
                let mut result = format_output(&partial_out, &partial_err, false);
                if !result.is_empty() {
                    result.push_str("\n\n");
                }
                result.push_str("[Process terminated by user]");
                return Ok(ToolOutput::Text(result));
            }
        };

        match result {
            Ok((Ok(status), stdout_buf, stderr_buf)) => {
                let result = format_output(&stdout_buf, &stderr_buf, status.success());
                if result.is_empty() {
                    Ok(ToolOutput::Text(
                        "[Command completed with no output]".to_string(),
                    ))
                } else {
                    Ok(ToolOutput::Text(result))
                }
            }
            Ok((Err(e), _, _)) => Err(ToolError::ExecutionFailed(e.to_string())),
            Err(e) => Err(e),
        }
    }

    fn permission(&self) -> ToolPermission {
        ToolPermission::Ask
    }
}

fn format_output(stdout_buf: &[u8], stderr_buf: &[u8], success: bool) -> String {
    let mut result = String::new();

    if !stdout_buf.is_empty() {
        let stdout = String::from_utf8_lossy(stdout_buf);
        let trimmed = stdout.trim();
        if trimmed.len() > MAX_OUTPUT_BYTES {
            result.push_str(&trimmed[..MAX_OUTPUT_BYTES]);
            result.push_str("\n\n[Output truncated]");
        } else {
            result.push_str(trimmed);
        }
    }

    if !stderr_buf.is_empty() {
        let stderr = String::from_utf8_lossy(stderr_buf);
        let trimmed = stderr.trim();
        if !trimmed.is_empty() {
            if !result.is_empty() {
                result.push_str("\n\n");
            }
            result.push_str("[stderr]: ");
            result.push_str(trimmed);
        }
    }

    if !success {
        if !result.is_empty() {
            result.push_str("\n\n");
        }
        result.push_str("[Exit code: non-zero]");
    }

    result
}

async fn poll_kill_signal(signal: &AtomicBool) {
    loop {
        if signal.load(Ordering::Relaxed) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

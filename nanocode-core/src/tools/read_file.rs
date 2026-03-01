//! Read file tool

use super::base::Tool;
use crate::types::ToolPermission;
use crate::types::{InvokeContext, ToolError, ToolOutput};
use async_trait::async_trait;
use serde::Deserialize;
use std::path::PathBuf;
use tokio::fs;

const MAX_FILE_SIZE: u64 = 64 * 1024; // 64KB

#[derive(Debug, Deserialize)]
pub struct ReadFileArgs {
    pub path: String,
    #[serde(default)]
    pub offset: Option<usize>,
    #[serde(default)]
    pub limit: Option<usize>,
}

pub struct ReadFileTool;

impl ReadFileTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ReadFileTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read contents of a file. Supports offset and limit for partial reads."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "offset": {
                    "type": "number",
                    "description": "Line number to start reading from (1-indexed)"
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of lines to read"
                }
            },
            "required": ["path"]
        })
    }

    async fn invoke(
        &self,
        args: serde_json::Value,
        _ctx: &InvokeContext,
    ) -> Result<ToolOutput, ToolError> {
        let args: ReadFileArgs =
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        // Validate path (prevent traversal)
        let path = validate_path(&args.path)?;

        // Check file exists
        if !path.exists() {
            return Err(ToolError::NotFound(format!(
                "File not found: {}",
                args.path
            )));
        }

        // Check file size
        let metadata = fs::metadata(&path).await?;
        if metadata.len() > MAX_FILE_SIZE {
            return Err(ToolError::ExecutionFailed(format!(
                "File too large: {} bytes (max: {})",
                metadata.len(),
                MAX_FILE_SIZE
            )));
        }

        // Read the file
        let content = fs::read_to_string(&path).await?;

        // Apply offset and limit
        let lines: Vec<&str> = content.lines().collect();
        let offset = args.offset.unwrap_or(1).saturating_sub(1); // Convert to 0-indexed
        let limit = args.limit.unwrap_or(usize::MAX);

        let selected: Vec<&str> = lines.iter().skip(offset).take(limit).copied().collect();

        let result = if let Some(limit) = args.limit {
            format!(
                "{} lines from line {}:\n{}",
                limit.min(selected.len()),
                offset + 1,
                selected.join("\n")
            )
        } else {
            selected.join("\n")
        };

        Ok(ToolOutput::Text(result))
    }

    fn permission(&self) -> ToolPermission {
        ToolPermission::Always
    }
}

/// Validate path to prevent traversal attacks
fn validate_path(path: &str) -> Result<PathBuf, ToolError> {
    // Absolute path
    if path.starts_with('/') {
        return Err(ToolError::InvalidArguments(
            "Absolute paths not allowed".to_string(),
        ));
    }

    // Check for traversal
    let path_buf = PathBuf::from(path);
    let normalized = path_buf.normalize();

    if normalized.to_string_lossy().contains("..") {
        return Err(ToolError::InvalidArguments(
            "Path traversal not allowed".to_string(),
        ));
    }

    Ok(normalized)
}

trait Normalize {
    fn normalize(&self) -> Self;
}

impl Normalize for PathBuf {
    fn normalize(&self) -> Self {
        let mut result = Self::new();
        for component in self.components() {
            match component {
                std::path::Component::ParentDir => {
                    result.pop();
                }
                std::path::Component::Normal(name) => {
                    result.push(name);
                }
                _ => {}
            }
        }
        result
    }
}

//! Write file tool

use super::base::Tool;
use crate::types::ToolPermission;
use crate::types::{InvokeContext, ToolError, ToolOutput};
use async_trait::async_trait;
use serde::Deserialize;
use std::path::PathBuf;
use tokio::fs;

const MAX_FILE_SIZE: u64 = 1024 * 1024; // 1MB

#[derive(Debug, Deserialize)]
pub struct WriteFileArgs {
    pub path: String,
    pub content: String,
    #[serde(default)]
    pub append: Option<bool>,
}

pub struct WriteFileTool;

impl WriteFileTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for WriteFileTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write content to a file. Creates parent directories if needed."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                },
                "append": {
                    "type": "boolean",
                    "description": "Append to file instead of overwriting (default: false)"
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn invoke(
        &self,
        args: serde_json::Value,
        _ctx: &InvokeContext,
    ) -> Result<ToolOutput, ToolError> {
        let args: WriteFileArgs =
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        // Validate path
        let path = validate_path(&args.path)?;

        // Check content size
        if args.content.len() as u64 > MAX_FILE_SIZE {
            return Err(ToolError::ExecutionFailed(format!(
                "Content too large: {} bytes (max: {})",
                args.content.len(),
                MAX_FILE_SIZE
            )));
        }

        // Create parent directories
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }

        // Write atomically using temp file
        let temp_path = path.with_extension(".tmp");

        if args.append.unwrap_or(false) {
            // Append mode: read existing, then append
            let existing = fs::read_to_string(&path).await.unwrap_or_default();
            let new_content = format!("{}{}", existing, args.content);
            fs::write(&temp_path, new_content).await?;
        } else {
            // Write mode
            fs::write(&temp_path, &args.content).await?;
        }

        // Rename to final location (atomic on most filesystems)
        fs::rename(&temp_path, &path).await?;

        Ok(ToolOutput::Text(format!(
            "File written successfully: {}",
            args.path
        )))
    }

    fn permission(&self) -> ToolPermission {
        ToolPermission::Ask
    }
}

/// Validate path to prevent traversal attacks
fn validate_path(path: &str) -> Result<PathBuf, ToolError> {
    if path.starts_with('/') {
        return Err(ToolError::InvalidArguments(
            "Absolute paths not allowed".to_string(),
        ));
    }

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

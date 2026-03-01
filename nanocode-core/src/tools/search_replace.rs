//! Search and replace tool

use super::base::Tool;
use crate::types::ToolPermission;
use crate::types::{InvokeContext, ToolError, ToolOutput};
use async_trait::async_trait;
use serde::Deserialize;
use std::path::PathBuf;
use tokio::fs;

#[derive(Debug, Deserialize)]
pub struct SearchReplaceArgs {
    pub path: String,
    pub search: String,
    pub replace: String,
    #[serde(default)]
    pub global: Option<bool>,
}

pub struct SearchReplaceTool;

impl SearchReplaceTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SearchReplaceTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for SearchReplaceTool {
    fn name(&self) -> &str {
        "search_replace"
    }

    fn description(&self) -> &str {
        "Replace exact text in a file. Replaces one occurrence by default."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file"
                },
                "search": {
                    "type": "string",
                    "description": "Text to search for"
                },
                "replace": {
                    "type": "string",
                    "description": "Text to replace with"
                },
                "global": {
                    "type": "boolean",
                    "description": "Replace all occurrences (default: false)"
                }
            },
            "required": ["path", "search", "replace"]
        })
    }

    async fn invoke(
        &self,
        args: serde_json::Value,
        _ctx: &InvokeContext,
    ) -> Result<ToolOutput, ToolError> {
        let args: SearchReplaceArgs =
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        // Validate path
        let path = validate_path(&args.path)?;

        if !path.exists() {
            return Err(ToolError::NotFound(format!(
                "File not found: {}",
                args.path
            )));
        }

        // Read file
        let content = fs::read_to_string(&path).await?;

        // Count occurrences
        let occurrences = content.matches(&args.search).count();

        if occurrences == 0 {
            return Err(ToolError::ExecutionFailed(format!(
                "Text not found in file: {}",
                args.search
            )));
        }

        // Check for multiple occurrences if not global
        if !args.global.unwrap_or(false) && occurrences > 1 {
            return Err(ToolError::ExecutionFailed(format!(
                "Found {} occurrences. Use global=true to replace all.",
                occurrences
            )));
        }

        // Replace
        let new_content = if args.global.unwrap_or(false) {
            content.replace(&args.search, &args.replace)
        } else {
            content.replacen(&args.search, &args.replace, 1)
        };

        // Write atomically
        let temp_path = path.with_extension(".tmp");
        fs::write(&temp_path, &new_content).await?;
        fs::rename(&temp_path, &path).await?;

        let replacements = if args.global.unwrap_or(false) {
            occurrences
        } else {
            1
        };
        Ok(ToolOutput::Text(format!(
            "Replaced {} occurrence(s) in {}",
            replacements, args.path
        )))
    }

    fn permission(&self) -> ToolPermission {
        ToolPermission::Ask
    }
}

fn validate_path(path: &str) -> Result<PathBuf, ToolError> {
    if path.starts_with('/') {
        return Err(ToolError::InvalidArguments(
            "Absolute paths not allowed".to_string(),
        ));
    }

    let path_buf = PathBuf::from(path);
    let mut result = PathBuf::new();

    for component in path_buf.components() {
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

    if result.to_string_lossy().contains("..") {
        return Err(ToolError::InvalidArguments(
            "Path traversal not allowed".to_string(),
        ));
    }

    Ok(result)
}

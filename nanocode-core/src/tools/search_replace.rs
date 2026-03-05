//! Search and replace tool

use super::base::Tool;
use crate::types::ToolPermission;
use crate::types::{InvokeContext, ToolError, ToolOutput};
use async_trait::async_trait;
use serde::Deserialize;
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

        // Validate path (prevent traversal, resolve absolute/relative)
        let path = super::path_utils::validate_and_resolve(&args.path)?;

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


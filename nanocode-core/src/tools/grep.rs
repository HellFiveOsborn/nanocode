//! Grep tool

use super::base::Tool;
use crate::types::ToolPermission;
use crate::types::{InvokeContext, ToolError, ToolOutput};
use async_trait::async_trait;
use regex::Regex;
use serde::Deserialize;
use std::path::PathBuf;

const MAX_RESULTS: usize = 1000;

#[derive(Debug, Deserialize)]
pub struct GrepArgs {
    pub pattern: String,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub include: Option<String>,
    #[serde(default)]
    pub exclude: Option<String>,
    #[serde(default)]
    pub context: Option<usize>,
}

pub struct GrepTool;

impl GrepTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GrepTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search for patterns in files using regex."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: current directory)"
                },
                "include": {
                    "type": "string",
                    "description": "Glob pattern for files to include (e.g., '*.rs')"
                },
                "exclude": {
                    "type": "string",
                    "description": "Glob pattern for files to exclude"
                },
                "context": {
                    "type": "number",
                    "description": "Number of context lines to show"
                }
            },
            "required": ["pattern"]
        })
    }

    async fn invoke(
        &self,
        args: serde_json::Value,
        _ctx: &InvokeContext,
    ) -> Result<ToolOutput, ToolError> {
        let args: GrepArgs =
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        // Compile regex
        let regex = Regex::new(&args.pattern)
            .map_err(|e| ToolError::InvalidArguments(format!("Invalid regex: {}", e)))?;

        // Determine search path
        let search_path = args
            .path
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));

        if !search_path.exists() {
            return Err(ToolError::NotFound(format!(
                "Path not found: {:?}",
                search_path
            )));
        }

        // Get files to search
        let files = collect_files(
            &search_path,
            args.include.as_deref(),
            args.exclude.as_deref(),
        )?;

        // Search in files
        let context = args.context.unwrap_or(0);
        let mut results: Vec<GrepResult> = Vec::new();

        for file in &files {
            let file_results = search_file(file, &regex, context);
            results.extend(file_results);

            if results.len() >= MAX_RESULTS {
                break;
            }
        }

        if results.is_empty() {
            return Ok(ToolOutput::Text("No matches found.".to_string()));
        }

        // Format results
        let output: String = results
            .iter()
            .map(|r| format!("{}:{}: {}", r.line_number, r.column, r.line))
            .collect::<Vec<_>>()
            .join("\n");

        let unique_files: std::collections::HashSet<_> = results.iter().map(|r| &r.file).collect();

        let summary = format!(
            "Found {} matches in {} files:\n\n{}",
            results.len(),
            unique_files.len(),
            output
        );

        Ok(ToolOutput::Text(summary))
    }

    fn permission(&self) -> ToolPermission {
        ToolPermission::Always
    }
}

#[derive(Debug, Clone)]
struct GrepResult {
    file: String,
    line_number: usize,
    column: usize,
    line: String,
}

fn collect_files(
    path: &PathBuf,
    include: Option<&str>,
    exclude: Option<&str>,
) -> Result<Vec<PathBuf>, ToolError> {
    let mut files = Vec::new();

    fn walk_dir(
        dir: &PathBuf,
        include: Option<&str>,
        exclude: Option<&str>,
        files: &mut Vec<PathBuf>,
    ) -> std::io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            // Skip common ignore patterns
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name == ".git" || name == "node_modules" || name == "target" || name == ".venv" {
                continue;
            }

            if path.is_dir() {
                walk_dir(&path, include, exclude, files)?;
            } else if path.is_file() {
                // Check include/exclude
                let should_include = match (include, exclude) {
                    (Some(inc), _) => glob_match(inc, name),
                    (_, Some(exc)) => !glob_match(exc, name),
                    _ => true,
                };

                if should_include {
                    files.push(path);
                }
            }
        }
        Ok(())
    }

    walk_dir(path, include, exclude, &mut files)?;
    Ok(files)
}

fn glob_match(pattern: &str, name: &str) -> bool {
    // Simple glob matching for common patterns
    let pattern = pattern.replace("*.", ".*\\.");
    Regex::new(&format!("^{}$", pattern))
        .map(|r| r.is_match(name))
        .unwrap_or(false)
}

fn search_file(path: &PathBuf, regex: &Regex, _context: usize) -> Vec<GrepResult> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let file_str = path.to_string_lossy().to_string();
    let mut results = Vec::new();

    for (line_num, line) in content.lines().enumerate() {
        if regex.is_match(line) {
            // Find column
            let column = regex.find(line).map(|m| m.start() + 1).unwrap_or(1);

            results.push(GrepResult {
                file: file_str.clone(),
                line_number: line_num + 1,
                column,
                line: line.chars().take(200).collect(),
            });
        }
    }

    results
}

//! Tool manager

use super::base::Tool;
use crate::config::NcConfig;
use crate::types::{AvailableTool, InvokeContext, ToolError, ToolOutput, ToolPermission};
use std::collections::HashMap;
use std::sync::Arc;

pub struct ToolManager {
    tools: HashMap<String, Arc<dyn Tool>>,
    configs: HashMap<String, ToolConfig>,
}

#[derive(Clone)]
pub struct ToolConfig {
    pub permission: ToolPermission,
    pub allowlist: Vec<String>,
    pub denylist: Vec<String>,
}

impl ToolManager {
    pub fn new(config: &NcConfig) -> Self {
        let mut tools = HashMap::new();
        let mut configs = HashMap::new();

        // Add built-in tools
        #[cfg(feature = "tool-bash")]
        {
            tools.insert(
                "bash".to_string(),
                Arc::new(crate::tools::bash::BashTool::new()) as Arc<dyn Tool>,
            );
            let tool_config = &config.tools.bash;
            configs.insert(
                "bash".to_string(),
                ToolConfig {
                    permission: match tool_config.permission {
                        crate::config::ToolPermissionConfig::Always => ToolPermission::Always,
                        crate::config::ToolPermissionConfig::Never => ToolPermission::Never,
                        crate::config::ToolPermissionConfig::Ask => ToolPermission::Ask,
                    },
                    allowlist: tool_config.allowlist.clone().unwrap_or_default(),
                    denylist: tool_config.denylist.clone().unwrap_or_default(),
                },
            );
        }

        #[cfg(feature = "tool-read")]
        {
            tools.insert(
                "read_file".to_string(),
                Arc::new(crate::tools::read_file::ReadFileTool::new()) as Arc<dyn Tool>,
            );
            let tool_config = &config.tools.read_file;
            configs.insert(
                "read_file".to_string(),
                ToolConfig {
                    permission: match tool_config.permission {
                        crate::config::ToolPermissionConfig::Always => ToolPermission::Always,
                        crate::config::ToolPermissionConfig::Never => ToolPermission::Never,
                        crate::config::ToolPermissionConfig::Ask => ToolPermission::Ask,
                    },
                    allowlist: tool_config.allowlist.clone().unwrap_or_default(),
                    denylist: tool_config.denylist.clone().unwrap_or_default(),
                },
            );
        }

        #[cfg(feature = "tool-write")]
        {
            tools.insert(
                "write_file".to_string(),
                Arc::new(crate::tools::write_file::WriteFileTool::new()) as Arc<dyn Tool>,
            );
            let tool_config = &config.tools.write_file;
            configs.insert(
                "write_file".to_string(),
                ToolConfig {
                    permission: match tool_config.permission {
                        crate::config::ToolPermissionConfig::Always => ToolPermission::Always,
                        crate::config::ToolPermissionConfig::Never => ToolPermission::Never,
                        crate::config::ToolPermissionConfig::Ask => ToolPermission::Ask,
                    },
                    allowlist: tool_config.allowlist.clone().unwrap_or_default(),
                    denylist: tool_config.denylist.clone().unwrap_or_default(),
                },
            );
        }

        #[cfg(feature = "tool-grep")]
        {
            tools.insert(
                "grep".to_string(),
                Arc::new(crate::tools::grep::GrepTool::new()) as Arc<dyn Tool>,
            );
            let tool_config = &config.tools.grep;
            configs.insert(
                "grep".to_string(),
                ToolConfig {
                    permission: match tool_config.permission {
                        crate::config::ToolPermissionConfig::Always => ToolPermission::Always,
                        crate::config::ToolPermissionConfig::Never => ToolPermission::Never,
                        crate::config::ToolPermissionConfig::Ask => ToolPermission::Ask,
                    },
                    allowlist: tool_config.allowlist.clone().unwrap_or_default(),
                    denylist: tool_config.denylist.clone().unwrap_or_default(),
                },
            );
        }

        #[cfg(feature = "tool-write")]
        {
            tools.insert(
                "search_replace".to_string(),
                Arc::new(crate::tools::search_replace::SearchReplaceTool::new()) as Arc<dyn Tool>,
            );
            configs.insert(
                "search_replace".to_string(),
                ToolConfig {
                    permission: ToolPermission::Ask,
                    allowlist: Vec::new(),
                    denylist: Vec::new(),
                },
            );
        }

        Self { tools, configs }
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    pub fn get_arc(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    pub async fn invoke(
        &self,
        name: &str,
        args: serde_json::Value,
        ctx: &InvokeContext,
    ) -> Result<ToolOutput, ToolError> {
        let tool = self
            .get_arc(name)
            .ok_or_else(|| ToolError::NotFound(name.to_string()))?;
        tool.invoke(args, ctx).await
    }

    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    pub fn available_tools(&self) -> Vec<&dyn Tool> {
        self.tools.values().map(|t| t.as_ref()).collect()
    }

    pub fn get_available_tools_schema(&self) -> Vec<AvailableTool> {
        self.tools
            .values()
            .map(|t| {
                let t = t.as_ref();
                AvailableTool {
                    name: t.name().to_string(),
                    description: t.description().to_string(),
                    parameters: t.parameters_schema(),
                }
            })
            .collect()
    }

    pub fn get_permission(&self, name: &str, args: &serde_json::Value) -> ToolPermission {
        // First check tool-specific allowlist/denylist
        if let Some(tool) = self.get(name) {
            if let Some(override_perm) = tool.check_allowlist_denylist(args) {
                return override_perm;
            }
        }

        // Then check global config
        self.configs
            .get(name)
            .map(|c| c.permission)
            .unwrap_or(ToolPermission::Ask)
    }
}

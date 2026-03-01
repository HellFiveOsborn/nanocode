//! Tool system base

use crate::types::{InvokeContext, ToolError, ToolOutput, ToolPermission};
use async_trait::async_trait;

/// Base trait for tools
#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool name
    fn name(&self) -> &str;

    /// Tool description
    fn description(&self) -> &str;

    /// JSON Schema for parameters
    fn parameters_schema(&self) -> serde_json::Value;

    /// Execute the tool
    async fn invoke(
        &self,
        args: serde_json::Value,
        ctx: &InvokeContext,
    ) -> Result<ToolOutput, ToolError>;

    /// Default permission
    fn permission(&self) -> ToolPermission {
        ToolPermission::Ask
    }

    /// Check allowlist/denylist
    fn check_allowlist_denylist(&self, _args: &serde_json::Value) -> Option<ToolPermission> {
        None
    }
}

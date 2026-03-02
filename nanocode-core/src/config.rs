//! Configuration for Nano Code

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NcConfig {
    /// Currently active model id
    pub active_model: Option<String>,

    /// Currently active quantization
    pub active_quant: Option<String>,

    /// Auto-approve tool calls
    #[serde(default)]
    pub auto_approve: bool,

    /// Auto-compaction threshold (in tokens)
    #[serde(default = "default_compact_threshold")]
    pub auto_compact_threshold: u32,

    /// Model configuration
    #[serde(default)]
    pub model: ModelConfig,

    /// Tool configurations
    #[serde(default)]
    pub tools: ToolsConfig,

    /// UI configuration
    #[serde(default)]
    pub ui: UiConfig,
}

fn default_compact_threshold() -> u32 {
    32_000
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Number of GPU layers (-1 = all, 0 = CPU only)
    #[serde(default = "default_gpu_layers")]
    pub n_gpu_layers: i32,

    /// Context size override
    #[serde(default)]
    pub context_size: Option<u32>,

    /// Temperature
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Top-k sampling
    #[serde(default = "default_top_k")]
    pub top_k: u32,

    /// Top-p sampling
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Min-p sampling (0.0 disables)
    #[serde(default = "default_min_p")]
    pub min_p: f32,

    /// Repetition penalty (1.0 disables)
    #[serde(default = "default_repeat_penalty")]
    pub repeat_penalty: f32,

    /// Number of recent tokens to apply repetition penalty
    #[serde(default = "default_repeat_last_n")]
    pub repeat_last_n: i32,

    /// Frequency penalty
    #[serde(default = "default_frequency_penalty")]
    pub frequency_penalty: f32,

    /// Presence penalty
    #[serde(default = "default_presence_penalty")]
    pub presence_penalty: f32,

    /// Max tokens
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// KV cache type for K tensors (ex: q8_0, q4_0, f16)
    #[serde(default)]
    pub kv_cache_type_k: Option<String>,

    /// KV cache type for V tensors (ex: q8_0, q4_0, f16)
    #[serde(default)]
    pub kv_cache_type_v: Option<String>,
}

fn default_gpu_layers() -> i32 {
    -1
}

fn default_temperature() -> f32 {
    0.6
}

fn default_top_k() -> u32 {
    20
}

fn default_top_p() -> f32 {
    0.95
}

fn default_min_p() -> f32 {
    0.0
}

fn default_repeat_penalty() -> f32 {
    1.05
}

fn default_repeat_last_n() -> i32 {
    128
}

fn default_frequency_penalty() -> f32 {
    0.0
}

fn default_presence_penalty() -> f32 {
    0.0
}

fn default_max_tokens() -> u32 {
    4096
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            n_gpu_layers: -1,
            context_size: None,
            temperature: 0.6,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.0,
            repeat_penalty: 1.05,
            repeat_last_n: 128,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            max_tokens: 4096,
            kv_cache_type_k: None,
            kv_cache_type_v: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsConfig {
    #[serde(default)]
    pub bash: ToolPolicyConfig,

    #[serde(default)]
    pub read_file: ToolPolicyConfig,

    #[serde(default)]
    pub write_file: ToolPolicyConfig,

    #[serde(default)]
    pub grep: ToolPolicyConfig,
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            bash: ToolPolicyConfig {
                permission: ToolPermissionConfig::Ask,
                allowlist: Some(vec![
                    "ls".to_string(),
                    "git diff".to_string(),
                    "git log".to_string(),
                    "git status".to_string(),
                    "cat".to_string(),
                    "echo".to_string(),
                ]),
                denylist: Some(vec![
                    "rm -rf".to_string(),
                    "sudo".to_string(),
                    "passwd".to_string(),
                ]),
            },
            read_file: ToolPolicyConfig {
                permission: ToolPermissionConfig::Ask,
                allowlist: None,
                denylist: None,
            },
            write_file: ToolPolicyConfig {
                permission: ToolPermissionConfig::Ask,
                allowlist: None,
                denylist: None,
            },
            grep: ToolPolicyConfig {
                permission: ToolPermissionConfig::Ask,
                allowlist: None,
                denylist: None,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolPermissionConfig {
    Always,
    Never,
    Ask,
}

impl Default for ToolPermissionConfig {
    fn default() -> Self {
        Self::Ask
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolPolicyConfig {
    #[serde(default)]
    pub permission: ToolPermissionConfig,

    #[serde(default)]
    pub allowlist: Option<Vec<String>>,

    #[serde(default)]
    pub denylist: Option<Vec<String>>,
}

impl Default for ToolPolicyConfig {
    fn default() -> Self {
        Self {
            permission: ToolPermissionConfig::default(),
            allowlist: None,
            denylist: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiConfig {
    /// Show thinking content in chat
    #[serde(default)]
    pub show_thinking: bool,

    /// Show tool logs panel
    #[serde(default)]
    pub show_tool_logs: bool,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            show_thinking: false,
            show_tool_logs: true,
        }
    }
}

impl Default for NcConfig {
    fn default() -> Self {
        Self {
            active_model: None,
            active_quant: None,
            auto_approve: false,
            auto_compact_threshold: 32_000,
            model: ModelConfig::default(),
            tools: ToolsConfig::default(),
            ui: UiConfig::default(),
        }
    }
}

impl NcConfig {
    /// Get the config directory
    pub fn config_dir() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("nanocode")
    }

    /// Get the data directory
    pub fn data_dir() -> PathBuf {
        dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("nanocode")
    }

    /// Get the models directory
    pub fn models_dir() -> PathBuf {
        Self::data_dir().join("models")
    }

    /// Get the sessions directory
    pub fn sessions_dir() -> PathBuf {
        Self::data_dir().join("sessions")
    }

    /// Get the config file path
    pub fn config_path() -> PathBuf {
        Self::config_dir().join("config.toml")
    }

    /// Load config from file
    pub fn load() -> Result<Self> {
        let path = Self::config_path();
        if path.exists() {
            let content = std::fs::read_to_string(&path)?;
            let config: NcConfig = toml::from_str(&content)?;
            Ok(config)
        } else {
            Ok(NcConfig::default())
        }
    }

    /// Save config to file
    pub fn save(&self) -> Result<()> {
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Check if a model is installed
    pub fn is_model_installed() -> bool {
        Self::models_dir()
            .read_dir()
            .map(|mut entries| entries.next().is_some())
            .unwrap_or(false)
    }
}

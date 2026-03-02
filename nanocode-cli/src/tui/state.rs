use nanocode_core::agents::BuiltinAgent;
use nanocode_core::AgentStats;
use nanocode_core::ApprovalDecision;
use nanocode_hf::RuntimeTelemetry;
use std::sync::mpsc::SyncSender;
use std::time::Instant;

#[derive(Clone)]
pub enum ToolState {
    Running,
    Ok,
    Error,
}

#[derive(Clone)]
pub enum ChatItem {
    Banner,
    User(String),
    Thinking {
        text: String,
        active: bool,
    },
    Assistant(String),
    Tool {
        summary: String,
        stream: Option<String>,
        detail: Option<String>,
        state: ToolState,
    },
    Error(String),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum UiScreen {
    Welcome,
    ModelSetup,
    Config,
    Chat,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    Default,
    Slash,
}

#[derive(Clone, Copy)]
pub struct SlashCommandEntry {
    pub alias: &'static str,
    pub description: &'static str,
}

pub struct SetupChoice {
    pub quant_name: String,
    pub size_human: String,
    pub quality_label: String,
    pub notes: Option<String>,
    pub recommended: bool,
    pub cached: bool,
    pub active: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ModelSetupView {
    Models,
    Variants,
}

pub struct ModelChoice {
    pub model_id: String,
    pub model_display_name: String,
    pub category_label: String,
    pub supports_thinking: bool,
    pub supports_vision: bool,
    pub max_context_tokens: u32,
    pub recommended_context_general: u32,
    pub recommended_context_coding: u32,
    pub cached_quants: Vec<String>,
    pub active_quant: Option<String>,
    pub recommended_quant: Option<String>,
}

pub struct ModelSetupState {
    pub view: ModelSetupView,
    pub hardware_display: String,
    pub models: Vec<ModelChoice>,
    pub selected_model_idx: usize,
    pub current_model_id: String,
    pub current_model_display_name: String,
    pub choices: Vec<SetupChoice>,
    pub selected_idx: usize,
    pub downloading: bool,
    pub status_line: String,
    pub progress_line: Option<String>,
    pub download_progress: Option<DownloadProgressView>,
    pub error_line: Option<String>,
}

pub struct DownloadProgressView {
    pub filename: String,
    pub downloaded: u64,
    pub total: u64,
    pub speed_bps: u64,
    pub eta_seconds: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ConfigFieldKey {
    NGpuLayers,
    ContextSize,
    KvCacheTypeK,
    KvCacheTypeV,
    BashPermission,
    WriteFilePermission,
    GrepPermission,
    ReadFilePermission,
}

pub struct ConfigField {
    pub key: ConfigFieldKey,
    pub label: String,
    pub value: String,
    pub initial_value: String,
    pub options: Vec<String>,
}

impl ConfigField {
    pub fn is_dirty(&self) -> bool {
        self.value != self.initial_value
    }
}

pub struct ConfigScreenState {
    pub model_label: String,
    pub hardware_display: String,
    pub runtime_context_tokens: u32,
    pub runtime_max_tokens: u32,
    pub config_path: String,
    pub models_path: String,
    pub sessions_path: String,
    pub fields: Vec<ConfigField>,
    pub selected_idx: usize,
    pub status_line: String,
    pub error_line: Option<String>,
}

impl ConfigScreenState {
    pub fn is_dirty(&self) -> bool {
        self.fields.iter().any(ConfigField::is_dirty)
    }
}

pub struct PendingApproval {
    pub call_id: String,
    pub summary: String,
    pub arguments: String,
    pub decision_tx: SyncSender<ApprovalDecision>,
    pub selected_option: ApprovalOption,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ApprovalOption {
    ApproveOnce,
    ApproveAlwaysToolSession,
    Deny,
}

impl ApprovalOption {
    pub fn next(self) -> Self {
        match self {
            Self::ApproveOnce => Self::ApproveAlwaysToolSession,
            Self::ApproveAlwaysToolSession => Self::Deny,
            Self::Deny => Self::ApproveOnce,
        }
    }

    pub fn prev(self) -> Self {
        match self {
            Self::ApproveOnce => Self::Deny,
            Self::ApproveAlwaysToolSession => Self::ApproveOnce,
            Self::Deny => Self::ApproveAlwaysToolSession,
        }
    }

    pub fn to_decision(self) -> ApprovalDecision {
        match self {
            Self::ApproveOnce => ApprovalDecision::ApproveOnce,
            Self::ApproveAlwaysToolSession => ApprovalDecision::ApproveAlwaysToolSession,
            Self::Deny => ApprovalDecision::Deny,
        }
    }
}

pub struct AppState {
    pub screen: UiScreen,
    pub needs_model_setup: bool,
    pub setup_only: bool,
    pub model_setup_can_cancel: bool,
    pub open_settings_requested: bool,
    pub open_model_setup_requested: bool,
    pub input_mode: InputMode,
    pub input: String,
    pub slash_suggestions: Vec<SlashCommandEntry>,
    pub slash_selected: usize,
    pub model_setup: Option<ModelSetupState>,
    pub config_screen: Option<ConfigScreenState>,
    pub chat: Vec<ChatItem>,
    pub status: String,
    pub model_label: String,
    pub telemetry: RuntimeTelemetry,
    pub max_context_tokens: u32,
    pub busy: bool,
    pub tools_collapsed: bool,
    pub thinking_collapsed: bool,
    pub slash_details_expanded: bool,
    pub chat_scroll: u16,
    pub spinner_idx: usize,
    pub stream_idx: Option<usize>,
    pub thinking_idx: Option<usize>,
    pub pending_approval: Option<PendingApproval>,
    pub active_agent: BuiltinAgent,
    pub requested_agent_switch: Option<BuiltinAgent>,
    pub busy_started_at: Option<Instant>,
    pub stats: AgentStats,
}

impl AppState {
    pub fn new(
        max_context_tokens: u32,
        telemetry: RuntimeTelemetry,
        active_agent: BuiltinAgent,
    ) -> Self {
        Self {
            screen: UiScreen::Welcome,
            needs_model_setup: false,
            setup_only: false,
            model_setup_can_cancel: false,
            open_settings_requested: false,
            open_model_setup_requested: false,
            input_mode: InputMode::Default,
            input: String::new(),
            slash_suggestions: Vec::new(),
            slash_selected: 0,
            model_setup: None,
            config_screen: None,
            chat: vec![ChatItem::Banner],
            status: "initializing model...".to_string(),
            model_label: "loading".to_string(),
            telemetry,
            max_context_tokens,
            busy: true,
            tools_collapsed: true,
            thinking_collapsed: true,
            slash_details_expanded: false,
            chat_scroll: 0,
            spinner_idx: 0,
            stream_idx: None,
            thinking_idx: None,
            pending_approval: None,
            active_agent,
            requested_agent_switch: None,
            busy_started_at: Some(Instant::now()),
            stats: AgentStats::default(),
        }
    }
}

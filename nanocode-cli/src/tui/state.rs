use nanocode_core::agents::BuiltinAgent;
use nanocode_core::session::SessionInfo;
use nanocode_core::AgentStats;
use nanocode_core::ApprovalDecision;
use nanocode_core::UserQuestionResponse;
use nanocode_hf::RuntimeTelemetry;
use std::collections::BTreeMap;
use std::sync::mpsc::SyncSender;
use std::time::Instant;

#[derive(Clone)]
pub enum ToolState {
    Running,
    Ok,
    Error,
}

#[derive(Clone)]
pub struct SubToolEntry {
    pub summary: String,
    pub done: bool,
}

#[derive(Clone, Default)]
pub struct SubagentTracking {
    /// Sub-tool calls made by the subagent (newest last).
    pub sub_tools: Vec<SubToolEntry>,
    /// Total tool uses completed.
    pub tools_done: u32,
    /// Start time for elapsed calculation.
    pub started_at: Option<Instant>,
    /// Final stats (set when subagent finishes).
    pub final_tools_called: Option<u32>,
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
        tool_name: String,
        summary: String,
        stream: Option<String>,
        output: Option<String>,
        code_path: Option<String>,
        code: Option<String>,
        diff: Option<String>,
        state: ToolState,
        subagent: Option<SubagentTracking>,
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

#[derive(Clone)]
pub struct SlashCommandEntry {
    pub alias: String,
    pub description: String,
    pub is_skill: bool,
}

#[derive(Clone)]
pub struct SkillEntry {
    pub name: String,
    pub description: String,
    pub skill_path: String,
    pub user_invocable: bool,
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
    pub summary: String,
    pub details: Vec<String>,
    pub diff_preview: Option<String>,
    pub decision_tx: SyncSender<ApprovalDecision>,
    pub selected_option: ApprovalOption,
}

pub struct PendingPlanReview {
    pub selected_option: PlanReviewOption,
    pub plan_text: String,
}

pub struct PendingUserQuestion {
    pub question: String,
    pub choices: Vec<String>,
    pub allow_free_text: bool,
    pub placeholder: Option<String>,
    pub selected_choice: usize,
    pub text_input: String,
    pub response_tx: SyncSender<UserQuestionResponse>,
}

pub struct PendingResumeSelection {
    pub sessions: Vec<SessionInfo>,
    pub selected_idx: usize,
}

pub struct SessionResumeRequest {
    pub session_id_query: String,
}

pub struct AgentSwitchRequest {
    pub target: BuiltinAgent,
    pub bootstrap_prompt: Option<String>,
}

pub struct PendingTextPaste {
    pub token: String,
    pub full_text: String,
}

pub struct PendingImagePaste {
    pub token: String,
    pub data_url: String,
}

#[derive(Clone)]
pub struct MentionSuggestionEntry {
    pub replacement: String,
    pub display: String,
    pub description: String,
    pub is_directory: bool,
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PlanReviewOption {
    ApproveAndBuild,
    Disapprove,
    ReworkWithSuggestion,
}

impl PlanReviewOption {
    pub fn next(self) -> Self {
        match self {
            Self::ApproveAndBuild => Self::Disapprove,
            Self::Disapprove => Self::ReworkWithSuggestion,
            Self::ReworkWithSuggestion => Self::ApproveAndBuild,
        }
    }

    pub fn prev(self) -> Self {
        match self {
            Self::ApproveAndBuild => Self::ReworkWithSuggestion,
            Self::Disapprove => Self::ApproveAndBuild,
            Self::ReworkWithSuggestion => Self::Disapprove,
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
    pub compact_requested: bool,
    pub reload_requested: bool,
    pub input_mode: InputMode,
    pub input: String,
    pub supports_thinking: bool,
    pub supports_vision: bool,
    pub pending_text_pastes: Vec<PendingTextPaste>,
    pub pending_image_pastes: Vec<PendingImagePaste>,
    pub paste_sequence: u32,
    pub skills: BTreeMap<String, SkillEntry>,
    pub skills_count: usize,
    pub mcp_servers_count: usize,
    pub slash_suggestions: Vec<SlashCommandEntry>,
    pub slash_selected: usize,
    pub mention_suggestions: Vec<MentionSuggestionEntry>,
    pub mention_selected: usize,
    pub mention_replace_range: Option<(usize, usize)>,
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
    pub code_blocks_collapsed: bool,
    pub chat_scroll: u16,
    pub spinner_idx: usize,
    pub stream_idx: Option<usize>,
    pub thinking_idx: Option<usize>,
    pub pending_approval: Option<PendingApproval>,
    pub pending_user_question: Option<PendingUserQuestion>,
    pub pending_plan_review: Option<PendingPlanReview>,
    pub pending_resume_selection: Option<PendingResumeSelection>,
    pub active_agent: BuiltinAgent,
    pub yolo_mode: bool,
    pub requested_agent_switch: Option<AgentSwitchRequest>,
    pub requested_resume_session: Option<SessionResumeRequest>,
    pub current_session_id: Option<String>,
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
            compact_requested: false,
            reload_requested: false,
            input_mode: InputMode::Default,
            input: String::new(),
            supports_thinking: false,
            supports_vision: false,
            pending_text_pastes: Vec::new(),
            pending_image_pastes: Vec::new(),
            paste_sequence: 0,
            skills: BTreeMap::new(),
            skills_count: 0,
            mcp_servers_count: 0,
            slash_suggestions: Vec::new(),
            slash_selected: 0,
            mention_suggestions: Vec::new(),
            mention_selected: 0,
            mention_replace_range: None,
            model_setup: None,
            config_screen: None,
            chat: vec![ChatItem::Banner],
            status: "inicializando modelo...".to_string(),
            model_label: "carregando".to_string(),
            telemetry,
            max_context_tokens,
            busy: true,
            tools_collapsed: false,
            thinking_collapsed: true,
            code_blocks_collapsed: false,
            chat_scroll: 0,
            spinner_idx: 0,
            stream_idx: None,
            thinking_idx: None,
            pending_approval: None,
            pending_user_question: None,
            pending_plan_review: None,
            pending_resume_selection: None,
            active_agent,
            yolo_mode: false,
            requested_agent_switch: None,
            requested_resume_session: None,
            current_session_id: None,
            busy_started_at: Some(Instant::now()),
            stats: AgentStats::default(),
        }
    }
}

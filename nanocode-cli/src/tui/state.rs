use nanocode_core::AgentStats;
use nanocode_hf::RuntimeTelemetry;
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

pub struct AppState {
    pub screen: UiScreen,
    pub input_mode: InputMode,
    pub input: String,
    pub slash_suggestions: Vec<SlashCommandEntry>,
    pub slash_selected: usize,
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
    pub busy_started_at: Option<Instant>,
    pub stats: AgentStats,
}

impl AppState {
    pub fn new(max_context_tokens: u32, telemetry: RuntimeTelemetry) -> Self {
        Self {
            screen: UiScreen::Welcome,
            input_mode: InputMode::Default,
            input: String::new(),
            slash_suggestions: Vec::new(),
            slash_selected: 0,
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
            busy_started_at: Some(Instant::now()),
            stats: AgentStats::default(),
        }
    }
}

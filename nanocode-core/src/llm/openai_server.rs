use crate::config::NcConfig;
use crate::llm::parse_tool_calls;
use crate::types::{LlmMessage, MessageRole};
use std::net::SocketAddr;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

use super::inference::{LlmEngine, LlmEngineHandle};

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Clone)]
struct AppState {
    engine: Arc<LlmEngineHandle>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    messages: Vec<OpenAiMessage>,
    max_tokens: Option<u32>,
    tools: Option<serde_json::Value>,
    tool_choice: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct OpenAiMessage {
    role: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    tool_call_id: Option<String>,
    #[serde(default)]
    tool_calls: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct OpenAiError {
    message: String,
    r#type: &'static str,
}

#[derive(Debug, Serialize)]
struct OpenAiErrorResponse {
    error: OpenAiError,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct Choice {
    index: u32,
    message: OpenAiMessage,
    finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

pub struct LocalOpenAiServer {
    addr: SocketAddr,
    shutdown_tx: Option<oneshot::Sender<()>>,
    handle: tokio::task::JoinHandle<()>,
}

impl LocalOpenAiServer {
    pub async fn start(model_path: &Path, config: &NcConfig) -> Result<Self, String> {
        let engine = LlmEngine::new(model_path, config)?;
        let state = AppState {
            engine: Arc::new(LlmEngineHandle::new(engine)),
        };

        let app = Router::new()
            .route("/health", get(|| async { "ok" }))
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state);

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .map_err(|e| format!("failed to bind local OpenAI server: {e}"))?;
        let addr = listener
            .local_addr()
            .map_err(|e| format!("failed to read local server addr: {e}"))?;

        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = shutdown_rx.await;
                })
                .await;
        });

        Ok(Self {
            addr,
            shutdown_tx: Some(shutdown_tx),
            handle,
        })
    }

    pub fn chat_completions_url(&self) -> String {
        format!("http://{}/v1/chat/completions", self.addr)
    }

    pub async fn shutdown(mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        let _ = self.handle.await;
    }
}

pub async fn chat_via_openai_server(
    model_path: &Path,
    config: &NcConfig,
    messages: &[LlmMessage],
    max_tokens: u32,
    tools: Option<serde_json::Value>,
    tool_choice: Option<serde_json::Value>,
) -> Result<String, String> {
    chat_via_openai_server_streaming(
        model_path,
        config,
        messages,
        max_tokens,
        tools,
        tool_choice,
        None,
        |_| {},
    )
    .await
}

pub async fn chat_via_openai_server_streaming<F>(
    model_path: &Path,
    config: &NcConfig,
    messages: &[LlmMessage],
    max_tokens: u32,
    tools: Option<serde_json::Value>,
    tool_choice: Option<serde_json::Value>,
    interrupt_signal: Option<Arc<AtomicBool>>,
    mut on_chunk: F,
) -> Result<String, String>
where
    F: FnMut(String),
{
    let model_path = model_path.to_path_buf();
    let config = config.clone();
    let messages = messages.to_vec();
    let tools_owned = tools.clone();
    let tool_choice_owned = tool_choice.clone();
    let interrupt_signal_owned = interrupt_signal.clone();
    let (chunk_tx, mut chunk_rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    let mut generate_task = tokio::task::spawn_blocking(move || -> Result<String, String> {
        let engine = LlmEngine::new(&model_path, &config)?;
        let handle = LlmEngineHandle::new(engine);
        let mut callback = |chunk: &str| -> bool {
            if !chunk.is_empty() {
                let _ = chunk_tx.send(chunk.to_string());
            }
            if let Some(signal) = interrupt_signal_owned.as_ref() {
                return !signal.load(Ordering::Relaxed);
            }
            true
        };
        handle.generate_with_chunk_callback(
            &messages,
            max_tokens,
            tools_owned,
            tool_choice_owned,
            &mut callback,
        )
    });

    loop {
        tokio::select! {
            maybe_chunk = chunk_rx.recv() => {
                if let Some(chunk) = maybe_chunk {
                    on_chunk(chunk);
                }
            }
            task_result = &mut generate_task => {
                return task_result.map_err(|e| format!("inference task join failed: {e}"))?;
            }
        }
    }
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, Json<OpenAiErrorResponse>)> {
    let mut converted = Vec::with_capacity(req.messages.len());
    for m in req.messages {
        let role = match m.role.as_str() {
            "system" => MessageRole::System,
            "user" => MessageRole::User,
            "assistant" => MessageRole::Assistant,
            "tool" => MessageRole::Tool,
            other => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(OpenAiErrorResponse {
                        error: OpenAiError {
                            message: format!("invalid role: {other}"),
                            r#type: "invalid_request_error",
                        },
                    }),
                ))
            }
        };

        let parsed_tool_calls = m.tool_calls.as_ref().and_then(|raw| {
            if let Some(items) = raw.as_array() {
                let parsed = items
                    .iter()
                    .filter_map(|tc| {
                        let id = tc.get("id").and_then(|v| v.as_str())?;
                        let name = tc
                            .get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|n| n.as_str())?;
                        let arguments = tc
                            .get("function")
                            .and_then(|f| f.get("arguments"))
                            .and_then(|v| v.as_str())?;

                        let parsed_args =
                            serde_json::from_str::<serde_json::Value>(arguments).ok()?;
                        if !parsed_args.is_object() {
                            return None;
                        }

                        Some(crate::types::ToolCall {
                            id: id.to_string(),
                            name: name.to_string(),
                            arguments: parsed_args,
                        })
                    })
                    .collect::<Vec<_>>();

                if !parsed.is_empty() {
                    return Some(parsed);
                }
            }
            None
        });

        converted.push(LlmMessage {
            role,
            content: m.content,
            name: m.name,
            tool_call_id: m.tool_call_id,
            tool_calls: parsed_tool_calls,
        });
    }

    let max_tokens = req.max_tokens.unwrap_or(512);
    let text = state
        .engine
        .generate(&converted, max_tokens, req.tools, req.tool_choice)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(OpenAiErrorResponse {
                    error: OpenAiError {
                        message: e,
                        r#type: "server_error",
                    },
                }),
            )
        })?;

    let parsed_calls = parse_tool_calls(&text);
    let (assistant_content, assistant_tool_calls, finish_reason) = if parsed_calls.is_empty() {
        (text, None, "stop")
    } else {
        let tool_calls = parsed_calls
            .into_iter()
            .map(|tc| {
                json!({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments.to_string(),
                    }
                })
            })
            .collect::<Vec<_>>();

        (String::new(), Some(json!(tool_calls)), "tool_calls")
    };

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let resp = ChatCompletionResponse {
        id: format!("chatcmpl-{}", now),
        object: "chat.completion",
        created: now,
        model: "nanocode-local".to_string(),
        choices: vec![Choice {
            index: 0,
            message: OpenAiMessage {
                role: "assistant".to_string(),
                content: assistant_content,
                name: None,
                tool_call_id: None,
                tool_calls: assistant_tool_calls,
            },
            finish_reason,
        }],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    };

    Ok(Json(resp))
}

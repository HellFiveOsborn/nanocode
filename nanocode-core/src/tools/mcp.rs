//! MCP proxy tools (stdio/http/streamable-http).

use super::base::Tool;
use crate::config::{
    McpHttpServerConfig, McpServerConfig, McpStdioServerConfig, NcConfig,
};
use crate::types::{InvokeContext, ToolError, ToolOutput, ToolPermission};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::time::{timeout, Duration};

const MCP_PROTOCOL_VERSION: &str = "2024-11-05";
const MCP_CLIENT_NAME: &str = "nanocode";

#[derive(Debug, Clone)]
pub struct RemoteToolSpec {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Value,
}

#[derive(Debug, Clone)]
enum RegisteredMcpTransport {
    Http {
        url: String,
        headers: HashMap<String, String>,
    },
    Stdio {
        argv: Vec<String>,
        env: HashMap<String, String>,
    },
}

#[derive(Debug, Clone)]
pub struct RegisteredMcpServer {
    alias: String,
    prompt: Option<String>,
    startup_timeout_sec: f32,
    tool_timeout_sec: f32,
    transport: RegisteredMcpTransport,
}

impl RegisteredMcpServer {
    pub fn from_config(entry: &McpServerConfig) -> Result<Self, ToolError> {
        match entry {
            McpServerConfig::Http(cfg) | McpServerConfig::StreamableHttp(cfg) => {
                Self::from_http_config(cfg)
            }
            McpServerConfig::Stdio(cfg) => Self::from_stdio_config(cfg),
        }
    }

    fn from_http_config(cfg: &McpHttpServerConfig) -> Result<Self, ToolError> {
        let alias = sanitize_id(&cfg.name, "server");
        let url = cfg.url.trim().to_string();
        if url.is_empty() {
            return Err(ToolError::InvalidArguments(format!(
                "MCP server '{}' has empty url",
                cfg.name
            )));
        }

        Ok(Self {
            alias,
            prompt: cfg.prompt.clone(),
            startup_timeout_sec: cfg.startup_timeout_sec,
            tool_timeout_sec: cfg.tool_timeout_sec,
            transport: RegisteredMcpTransport::Http {
                url,
                headers: cfg.resolved_headers(),
            },
        })
    }

    fn from_stdio_config(cfg: &McpStdioServerConfig) -> Result<Self, ToolError> {
        let alias = sanitize_id(&cfg.name, "server");
        let argv = cfg.argv();
        if argv.is_empty() {
            return Err(ToolError::InvalidArguments(format!(
                "MCP stdio server '{}' has empty command",
                cfg.name
            )));
        }

        Ok(Self {
            alias,
            prompt: cfg.prompt.clone(),
            startup_timeout_sec: cfg.startup_timeout_sec,
            tool_timeout_sec: cfg.tool_timeout_sec,
            transport: RegisteredMcpTransport::Stdio {
                argv,
                env: cfg.env.clone(),
            },
        })
    }

    pub fn alias(&self) -> &str {
        self.alias.as_str()
    }

    pub fn build_tool_name(&self, remote_name: &str) -> String {
        format!(
            "mcp_{}_{}",
            self.alias(),
            sanitize_id(remote_name, "tool")
        )
    }

    fn startup_timeout(&self) -> Duration {
        timeout_from_secs(self.startup_timeout_sec, 10.0)
    }

    fn tool_timeout(&self) -> Duration {
        timeout_from_secs(self.tool_timeout_sec, 60.0)
    }

    fn target_label(&self) -> String {
        match &self.transport {
            RegisteredMcpTransport::Http { url, .. } => url.clone(),
            RegisteredMcpTransport::Stdio { argv, .. } => argv.join(" "),
        }
    }
}

#[derive(Debug, Clone)]
pub struct McpProxyTool {
    name: String,
    description: String,
    input_schema: Value,
    remote_name: String,
    server: Arc<RegisteredMcpServer>,
}

impl McpProxyTool {
    pub fn from_remote(
        server: Arc<RegisteredMcpServer>,
        remote: RemoteToolSpec,
        published_name: String,
    ) -> Self {
        let mut description = remote
            .description
            .clone()
            .unwrap_or_else(|| format!("MCP tool '{}' from {}", remote.name, server.target_label()));
        if let Some(prompt) = server.prompt.clone() {
            let hint = prompt.trim();
            if !hint.is_empty() {
                description.push_str("\nHint: ");
                description.push_str(hint);
            }
        }

        Self {
            name: published_name,
            description,
            input_schema: normalize_input_schema(remote.input_schema),
            remote_name: remote.name,
            server,
        }
    }
}

#[async_trait]
impl Tool for McpProxyTool {
    fn name(&self) -> &str {
        self.name.as_str()
    }

    fn description(&self) -> &str {
        self.description.as_str()
    }

    fn parameters_schema(&self) -> Value {
        self.input_schema.clone()
    }

    async fn invoke(&self, args: Value, _ctx: &InvokeContext) -> Result<ToolOutput, ToolError> {
        if !args.is_object() {
            return Err(ToolError::InvalidArguments(
                "MCP tool arguments must be an object".to_string(),
            ));
        }

        let call_result = call_tool(&self.server, &self.remote_name, args).await?;
        Ok(parse_call_result(&self.server.alias, &self.remote_name, &call_result))
    }

    fn permission(&self) -> ToolPermission {
        ToolPermission::Ask
    }
}

pub async fn discover_tools(config: &NcConfig) -> Vec<(Arc<RegisteredMcpServer>, RemoteToolSpec)> {
    let mut discovered = Vec::new();

    for entry in &config.mcp_servers {
        let Ok(server) = RegisteredMcpServer::from_config(entry) else {
            continue;
        };
        let server = Arc::new(server);
        if let Ok(remote_tools) = list_tools(&server).await {
            for remote in remote_tools {
                discovered.push((server.clone(), remote));
            }
        }
    }

    discovered
}

async fn list_tools(server: &RegisteredMcpServer) -> Result<Vec<RemoteToolSpec>, ToolError> {
    match &server.transport {
        RegisteredMcpTransport::Http { url, headers } => {
            list_tools_http(url, headers, server.startup_timeout()).await
        }
        RegisteredMcpTransport::Stdio { argv, env } => {
            list_tools_stdio(argv, env, server.startup_timeout()).await
        }
    }
}

async fn call_tool(
    server: &RegisteredMcpServer,
    tool_name: &str,
    arguments: Value,
) -> Result<Value, ToolError> {
    match &server.transport {
        RegisteredMcpTransport::Http { url, headers } => {
            call_tool_http(
                url,
                headers,
                server.startup_timeout(),
                server.tool_timeout(),
                tool_name,
                arguments,
            )
            .await
        }
        RegisteredMcpTransport::Stdio { argv, env } => {
            call_tool_stdio(
                argv,
                env,
                server.startup_timeout(),
                server.tool_timeout(),
                tool_name,
                arguments,
            )
            .await
        }
    }
}

async fn list_tools_http(
    url: &str,
    headers: &HashMap<String, String>,
    startup_timeout: Duration,
) -> Result<Vec<RemoteToolSpec>, ToolError> {
    let client = reqwest::Client::new();
    initialize_http(&client, url, headers, startup_timeout).await?;
    let result = jsonrpc_http_request(
        &client,
        url,
        headers,
        2,
        "tools/list",
        Some(json!({})),
        startup_timeout,
    )
    .await?;
    parse_tools_list_result(&result)
}

async fn call_tool_http(
    url: &str,
    headers: &HashMap<String, String>,
    startup_timeout: Duration,
    tool_timeout: Duration,
    tool_name: &str,
    arguments: Value,
) -> Result<Value, ToolError> {
    let client = reqwest::Client::new();
    initialize_http(&client, url, headers, startup_timeout).await?;
    jsonrpc_http_request(
        &client,
        url,
        headers,
        3,
        "tools/call",
        Some(json!({
            "name": tool_name,
            "arguments": arguments
        })),
        tool_timeout,
    )
    .await
}

async fn initialize_http(
    client: &reqwest::Client,
    url: &str,
    headers: &HashMap<String, String>,
    startup_timeout: Duration,
) -> Result<(), ToolError> {
    let _ = jsonrpc_http_request(
        client,
        url,
        headers,
        1,
        "initialize",
        Some(initialize_params()),
        startup_timeout,
    )
    .await?;
    let _ = jsonrpc_http_notify(
        client,
        url,
        headers,
        "notifications/initialized",
        Some(json!({})),
        startup_timeout,
    )
    .await;
    Ok(())
}

async fn jsonrpc_http_request(
    client: &reqwest::Client,
    url: &str,
    headers: &HashMap<String, String>,
    id: u64,
    method: &str,
    params: Option<Value>,
    timeout_dur: Duration,
) -> Result<Value, ToolError> {
    let payload = json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
        "params": params.unwrap_or_else(|| json!({})),
    });

    let mut req = client.post(url).timeout(timeout_dur).json(&payload);
    for (k, v) in headers {
        req = req.header(k, v);
    }
    let response = req.send().await.map_err(|e| {
        ToolError::ExecutionFailed(format!("MCP HTTP request failed ({method}): {e}"))
    })?;
    let status = response.status();
    if !status.is_success() {
        return Err(ToolError::ExecutionFailed(format!(
            "MCP HTTP {method} failed with status {status}"
        )));
    }

    let value: Value = response.json().await.map_err(|e| {
        ToolError::ExecutionFailed(format!("Invalid MCP HTTP response ({method}): {e}"))
    })?;
    parse_jsonrpc_response(method, value)
}

async fn jsonrpc_http_notify(
    client: &reqwest::Client,
    url: &str,
    headers: &HashMap<String, String>,
    method: &str,
    params: Option<Value>,
    timeout_dur: Duration,
) -> Result<(), ToolError> {
    let payload = json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params.unwrap_or_else(|| json!({})),
    });

    let mut req = client.post(url).timeout(timeout_dur).json(&payload);
    for (k, v) in headers {
        req = req.header(k, v);
    }
    let response = req.send().await.map_err(|e| {
        ToolError::ExecutionFailed(format!("MCP HTTP notify failed ({method}): {e}"))
    })?;
    if !response.status().is_success() {
        return Err(ToolError::ExecutionFailed(format!(
            "MCP HTTP notify failed ({method}) with status {}",
            response.status()
        )));
    }
    Ok(())
}

async fn list_tools_stdio(
    argv: &[String],
    env: &HashMap<String, String>,
    startup_timeout: Duration,
) -> Result<Vec<RemoteToolSpec>, ToolError> {
    let mut session = StdioJsonRpcClient::spawn(argv, env)?;
    let result = async {
        session.initialize(startup_timeout).await?;
        let list = session
            .request("tools/list", json!({}), startup_timeout)
            .await?;
        parse_tools_list_result(&list)
    }
    .await;
    session.close().await;
    result
}

async fn call_tool_stdio(
    argv: &[String],
    env: &HashMap<String, String>,
    startup_timeout: Duration,
    tool_timeout: Duration,
    tool_name: &str,
    arguments: Value,
) -> Result<Value, ToolError> {
    let mut session = StdioJsonRpcClient::spawn(argv, env)?;
    let result = async {
        session.initialize(startup_timeout).await?;
        session
            .request(
                "tools/call",
                json!({
                    "name": tool_name,
                    "arguments": arguments
                }),
                tool_timeout,
            )
            .await
    }
    .await;
    session.close().await;
    result
}

struct StdioJsonRpcClient {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl StdioJsonRpcClient {
    fn spawn(argv: &[String], env: &HashMap<String, String>) -> Result<Self, ToolError> {
        if argv.is_empty() {
            return Err(ToolError::InvalidArguments(
                "MCP stdio argv is empty".to_string(),
            ));
        }

        let mut cmd = Command::new(&argv[0]);
        cmd.args(&argv[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        for (k, v) in env {
            cmd.env(k, v);
        }

        let mut child = cmd.spawn().map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to start MCP stdio server {:?}: {e}", argv))
        })?;
        let stdin = child.stdin.take().ok_or_else(|| {
            ToolError::ExecutionFailed("Failed to open MCP stdin pipe".to_string())
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            ToolError::ExecutionFailed("Failed to open MCP stdout pipe".to_string())
        })?;

        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 1,
        })
    }

    async fn initialize(&mut self, startup_timeout: Duration) -> Result<(), ToolError> {
        let _ = self
            .request("initialize", initialize_params(), startup_timeout)
            .await?;
        let _ = self
            .notify("notifications/initialized", json!({}))
            .await
            .ok();
        Ok(())
    }

    async fn request(
        &mut self,
        method: &str,
        params: Value,
        timeout_dur: Duration,
    ) -> Result<Value, ToolError> {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);

        self.write_message(&json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        }))
        .await?;

        let wait_response = async {
            loop {
                let message = self.read_message().await?;
                let Some(response_id) = message.get("id") else {
                    continue;
                };
                if !response_id_matches(response_id, id) {
                    continue;
                }
                if let Some(err) = message.get("error") {
                    return Err(ToolError::ExecutionFailed(format!(
                        "MCP stdio error ({method}): {}",
                        err
                    )));
                }
                return Ok(message.get("result").cloned().unwrap_or(Value::Null));
            }
        };

        timeout(timeout_dur, wait_response)
            .await
            .map_err(|_| ToolError::Timeout)?
    }

    async fn notify(&mut self, method: &str, params: Value) -> Result<(), ToolError> {
        self.write_message(&json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }))
        .await
    }

    async fn write_message(&mut self, payload: &Value) -> Result<(), ToolError> {
        let bytes = serde_json::to_vec(payload).map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to encode MCP message: {e}"))
        })?;
        let header = format!("Content-Length: {}\r\n\r\n", bytes.len());
        self.stdin.write_all(header.as_bytes()).await?;
        self.stdin.write_all(&bytes).await?;
        self.stdin.flush().await?;
        Ok(())
    }

    async fn read_message(&mut self) -> Result<Value, ToolError> {
        let mut content_length: Option<usize> = None;
        loop {
            let mut line = String::new();
            let read = self.stdout.read_line(&mut line).await?;
            if read == 0 {
                return Err(ToolError::ExecutionFailed(
                    "MCP stdio closed while waiting response".to_string(),
                ));
            }
            let trimmed = line.trim_end_matches(['\r', '\n']);
            if trimmed.is_empty() {
                break;
            }
            if let Some(raw) = trimmed.strip_prefix("Content-Length:") {
                let parsed = raw.trim().parse::<usize>().map_err(|_| {
                    ToolError::ExecutionFailed(format!(
                        "Invalid MCP Content-Length header: {}",
                        raw.trim()
                    ))
                })?;
                content_length = Some(parsed);
            }
        }

        let length = content_length.ok_or_else(|| {
            ToolError::ExecutionFailed("Missing MCP Content-Length header".to_string())
        })?;
        let mut body = vec![0u8; length];
        self.stdout.read_exact(&mut body).await?;
        serde_json::from_slice::<Value>(&body).map_err(|e| {
            ToolError::ExecutionFailed(format!("Invalid MCP JSON payload: {e}"))
        })
    }

    async fn close(mut self) {
        let _ = self.child.start_kill();
        let _ = timeout(Duration::from_millis(800), self.child.wait()).await;
    }
}

fn parse_tools_list_result(value: &Value) -> Result<Vec<RemoteToolSpec>, ToolError> {
    let tools = value
        .get("tools")
        .and_then(Value::as_array)
        .ok_or_else(|| ToolError::ExecutionFailed("MCP tools/list missing `tools`".to_string()))?;

    let mut out = Vec::new();
    for tool in tools {
        let Some(name) = tool.get("name").and_then(Value::as_str) else {
            continue;
        };
        let description = tool
            .get("description")
            .and_then(Value::as_str)
            .map(ToString::to_string);
        let input_schema = tool
            .get("inputSchema")
            .cloned()
            .or_else(|| tool.get("input_schema").cloned())
            .unwrap_or_else(|| json!({"type":"object","properties":{}}));

        out.push(RemoteToolSpec {
            name: name.to_string(),
            description,
            input_schema: normalize_input_schema(input_schema),
        });
    }

    Ok(out)
}

fn parse_call_result(server: &str, tool: &str, result: &Value) -> ToolOutput {
    let structured = result
        .get("structuredContent")
        .cloned()
        .or_else(|| result.get("structured_content").cloned());

    let text = result
        .get("content")
        .and_then(Value::as_array)
        .map(|blocks| {
            blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .filter(|joined| !joined.trim().is_empty())
        .or_else(|| {
            result
                .get("text")
                .and_then(Value::as_str)
                .map(ToString::to_string)
        });

    let is_error = result
        .get("isError")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    ToolOutput::Structured(json!({
        "ok": !is_error,
        "server": server,
        "tool": tool,
        "text": text,
        "structured": structured
    }))
}

fn normalize_input_schema(schema: Value) -> Value {
    if let Some(obj) = schema.as_object() {
        let mut normalized = obj.clone();
        if !normalized.contains_key("type") {
            normalized.insert("type".to_string(), Value::String("object".to_string()));
        }
        if !normalized.contains_key("properties") {
            normalized.insert("properties".to_string(), json!({}));
        }
        Value::Object(normalized)
    } else {
        json!({
            "type": "object",
            "properties": {}
        })
    }
}

fn parse_jsonrpc_response(method: &str, value: Value) -> Result<Value, ToolError> {
    if let Some(error) = value.get("error") {
        return Err(ToolError::ExecutionFailed(format!(
            "MCP JSON-RPC error ({method}): {}",
            error
        )));
    }
    if let Some(result) = value.get("result") {
        return Ok(result.clone());
    }
    Ok(value)
}

fn initialize_params() -> Value {
    json!({
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "capabilities": {},
        "clientInfo": {
            "name": MCP_CLIENT_NAME,
            "version": env!("CARGO_PKG_VERSION"),
        }
    })
}

fn response_id_matches(id_value: &Value, expected: u64) -> bool {
    id_value
        .as_u64()
        .map(|id| id == expected)
        .or_else(|| id_value.as_str().map(|id| id == expected.to_string()))
        .unwrap_or(false)
}

fn timeout_from_secs(raw: f32, default_secs: f32) -> Duration {
    if !raw.is_finite() || raw <= 0.0 {
        return Duration::from_secs_f32(default_secs);
    }
    Duration::from_secs_f32(raw)
}

fn sanitize_id(value: &str, fallback: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push('_');
        }
    }
    let trimmed = out.trim_matches('_').trim_matches('-').to_string();
    if trimmed.is_empty() {
        fallback.to_string()
    } else {
        trimmed
    }
}

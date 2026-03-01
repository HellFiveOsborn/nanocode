//! Tool call parsing and formatting

use crate::types::{AvailableTool, LlmMessage, ToolCall};
use serde_json::Value;

/// Parse tool calls from assistant response
pub fn parse_tool_calls(content: &str) -> Vec<ToolCall> {
    let Ok(json) = serde_json::from_str::<Value>(content) else {
        return Vec::new();
    };
    let Some(tool_calls) = json.get("tool_calls").and_then(|v| v.as_array()) else {
        return Vec::new();
    };

    let mut calls = Vec::with_capacity(tool_calls.len());
    for tc in tool_calls {
        let Some(id) = tc.get("id").and_then(|v| v.as_str()) else {
            return Vec::new();
        };
        let Some(name) = tc
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
        else {
            return Vec::new();
        };
        let Some(args_text) = tc
            .get("function")
            .and_then(|f| f.get("arguments"))
            .and_then(|a| a.as_str())
        else {
            return Vec::new();
        };
        let Ok(arguments) = serde_json::from_str::<Value>(args_text) else {
            return Vec::new();
        };
        if !arguments.is_object() {
            return Vec::new();
        }

        calls.push(ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments,
        });
    }

    calls
}

fn escape_gbnf_literal(input: &str) -> String {
    input
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Build strict GBNF grammar for OpenAI-like tool call contract.
pub fn tool_call_contract_gbnf(tools: &[AvailableTool]) -> String {
    let names = tools
        .iter()
        .map(|t| format!("\"{}\"", escape_gbnf_literal(&t.name)))
        .collect::<Vec<_>>();
    let name_rule = if names.is_empty() {
        "\"\"".to_string()
    } else {
        names.join(" | ")
    };

    format!(
        r#"root ::= ws "{{" ws "\"tool_calls\"" ws ":" ws "[" ws tool-call ( ws "," ws tool-call )* ws "]" ws "}}" ws

tool-call ::= "{{" ws "\"id\"" ws ":" ws string ws "," ws "\"type\"" ws ":" ws "\"function\"" ws "," ws "\"function\"" ws ":" ws function-object ws "}}"
function-object ::= "{{" ws "\"name\"" ws ":" ws tool-name ws "," ws "\"arguments\"" ws ":" ws json-string-object ws "}}"
tool-name ::= {name_rule}

json-string-object ::= string
string ::= "\"" char* "\""
char ::= [^"\\] | "\\" (["\\/bfnrt] | "u" hex hex hex hex)
hex ::= [0-9a-fA-F]
ws ::= [ \t\n\r]*"#,
        name_rule = name_rule
    )
}

/// Build a tool result message
pub fn build_tool_result_message(tool_call_id: &str, result: &str) -> LlmMessage {
    LlmMessage::tool(result, tool_call_id)
}

/// Build available tools schema for the LLM
pub fn build_available_tools_schema(tools: &[&dyn crate::tools::Tool]) -> Vec<AvailableTool> {
    tools
        .iter()
        .map(|t| AvailableTool {
            name: t.name().to_string(),
            description: t.description().to_string(),
            parameters: t.parameters_schema(),
        })
        .collect()
}

/// Convert available tools to JSON for chat template
pub fn tools_to_json(tools: &[AvailableTool]) -> String {
    let mut tool_schemas = Vec::new();
    for tool in tools {
        let schema = serde_json::json!({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
        });
        tool_schemas.push(schema);
    }
    serde_json::to_string(&tool_schemas).unwrap_or_else(|_| "[]".to_string())
}

#[cfg(test)]
mod tests {
    use super::{parse_tool_calls, tool_call_contract_gbnf};
    use crate::types::AvailableTool;
    use serde_json::json;

    #[test]
    fn parse_tool_calls_accepts_contract_only() {
        let raw = r#"{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"Cargo.toml\"}"}}]}"#;
        let calls = parse_tool_calls(raw);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments, json!({ "path": "Cargo.toml" }));
    }

    #[test]
    fn parse_tool_calls_rejects_non_contract_formats() {
        let xml =
            r#"<tool_call>{"name":"read_file","arguments":{"path":"Cargo.toml"}}</tool_call>"#;
        assert!(parse_tool_calls(xml).is_empty());

        let arr = r#"[{"name":"read_file","arguments":{"path":"Cargo.toml"}}]"#;
        assert!(parse_tool_calls(arr).is_empty());
    }

    #[test]
    fn grammar_contains_only_allowed_tool_names() {
        let tools = vec![
            AvailableTool {
                name: "read_file".to_string(),
                description: String::new(),
                parameters: json!({}),
            },
            AvailableTool {
                name: "grep".to_string(),
                description: String::new(),
                parameters: json!({}),
            },
        ];

        let grammar = tool_call_contract_gbnf(&tools);
        assert!(grammar.contains("tool-name ::= \"read_file\" | \"grep\""));
    }
}

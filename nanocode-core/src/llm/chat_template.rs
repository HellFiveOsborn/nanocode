//! Chat template handling for different model families

use crate::types::LlmMessage;
use serde::{Deserialize, Serialize};

/// Prompt family (dialect)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromptFamily {
    Qwen3,
    Llama,
    GptOss,
}

impl Default for PromptFamily {
    fn default() -> Self {
        Self::Qwen3
    }
}

/// Prompt variant
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromptVariant {
    AgentDefault,
    AgentPlan,
    AgentBuild,
    SubagentExplore,
}

impl Default for PromptVariant {
    fn default() -> Self {
        Self::AgentDefault
    }
}

/// Chat template source
#[derive(Debug, Clone)]
pub enum ChatTemplateSource {
    Embedded(&'static str),
    FromTokenizerConfig,
}

/// Prompt profile for a model
#[derive(Debug, Clone)]
pub struct PromptProfile {
    pub family: PromptFamily,
    pub supports_thinking_tags: bool,
    pub chat_template: ChatTemplateSource,
}

impl Default for PromptProfile {
    fn default() -> Self {
        Self {
            family: PromptFamily::Qwen3,
            supports_thinking_tags: true,
            chat_template: ChatTemplateSource::Embedded(QWEN3_CHAT_TEMPLATE),
        }
    }
}

/// Qwen3 chat template (simplified)
const QWEN3_CHAT_TEMPLATE: &str = r#"<|im_start|>system
{{ system_message }}
<|im_end|>
{% for message in messages %}
<|im_start|>{{ message.role }}
{{ message.content }}
{% if message.tool_calls %}
{% for tool_call in message.tool_calls %}
<|tool_call|>{"name": "{{ tool_call.name }}", "arguments": {{ tool_call.arguments }}}<|tool_end|>
{% endfor %}
{% endif %}
<|im_end|>
{% endfor %}
{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"#;

/// Chat template renderer - simplified version
pub struct ChatTemplateRenderer;

impl ChatTemplateRenderer {
    /// Create a new renderer with the given template
    pub fn new(_template: &str) -> Self {
        // Simplified: just use a basic template for now
        Self
    }

    /// Render messages to a prompt (simplified)
    pub fn render(
        &self,
        messages: &[LlmMessage],
        _tools: Option<&str>,
        add_generation_prompt: bool,
    ) -> String {
        // Simplified rendering - just concatenate messages
        let mut output = String::new();

        // System message
        if let Some(msg) = messages.first() {
            if msg.role == crate::types::MessageRole::System {
                output.push_str("<|im_start|>system\n");
                output.push_str(&msg.content);
                output.push_str("\n<|im_end|>\n");
            }
        }

        // Other messages
        for msg in messages.iter().skip(
            if messages
                .first()
                .map(|m| m.role == crate::types::MessageRole::System)
                .unwrap_or(false)
            {
                1
            } else {
                0
            },
        ) {
            output.push_str(&format!("<|im_start|>{}\n", msg.role));
            output.push_str(&msg.content);

            if let Some(tool_calls) = &msg.tool_calls {
                for tc in tool_calls {
                    let args = serde_json::to_string(&tc.arguments).unwrap_or_default();
                    output.push_str("\n<|tool_call|>");
                    output.push_str(&format!(
                        "{{\"name\": \"{}\", \"arguments\": {}}}",
                        tc.name, args
                    ));
                    output.push_str("<|tool_end|>");
                }
            }

            output.push_str("\n<|im_end|>\n");
        }

        if add_generation_prompt {
            output.push_str("<|im_start|>assistant\n");
        }

        output
    }
}

impl Default for ChatTemplateRenderer {
    fn default() -> Self {
        Self::new(QWEN3_CHAT_TEMPLATE)
    }
}

/// Get the appropriate chat template for a family
pub fn get_chat_template(family: PromptFamily) -> &'static str {
    match family {
        PromptFamily::Qwen3 => QWEN3_CHAT_TEMPLATE,
        PromptFamily::Llama => QWEN3_CHAT_TEMPLATE,
        PromptFamily::GptOss => QWEN3_CHAT_TEMPLATE,
    }
}

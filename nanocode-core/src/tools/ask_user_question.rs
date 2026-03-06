//! Ask user question tool

use super::base::Tool;
use crate::types::{
    InvokeContext, QuestionAnswerSource, ToolError, ToolOutput, ToolPermission, UserQuestionRequest,
};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

const MAX_CHOICES: usize = 20;

#[derive(Debug, Deserialize)]
struct AskUserQuestionArgs {
    question: String,
    #[serde(default)]
    choices: Vec<String>,
    #[serde(default = "default_allow_free_text")]
    allow_free_text: bool,
    #[serde(default)]
    placeholder: Option<String>,
}

fn default_allow_free_text() -> bool {
    true
}

pub struct AskUserQuestionTool;

impl AskUserQuestionTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for AskUserQuestionTool {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_and_validate_args(args: serde_json::Value) -> Result<AskUserQuestionArgs, ToolError> {
    let mut parsed: AskUserQuestionArgs =
        serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

    parsed.question = parsed.question.trim().to_string();
    if parsed.question.is_empty() {
        return Err(ToolError::InvalidArguments(
            "question must not be empty".to_string(),
        ));
    }

    if parsed.choices.len() > MAX_CHOICES {
        return Err(ToolError::InvalidArguments(format!(
            "choices supports at most {} entries",
            MAX_CHOICES
        )));
    }

    for choice in &mut parsed.choices {
        *choice = choice.trim().to_string();
        if choice.is_empty() {
            return Err(ToolError::InvalidArguments(
                "choices must not contain empty values".to_string(),
            ));
        }
    }

    if !parsed.allow_free_text && parsed.choices.is_empty() {
        return Err(ToolError::InvalidArguments(
            "choices is required when allow_free_text is false".to_string(),
        ));
    }

    parsed.placeholder = parsed
        .placeholder
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    Ok(parsed)
}

fn normalize_choice_response(
    answer: &str,
    choice_index: Option<usize>,
    choices: &[String],
) -> (String, Option<usize>, QuestionAnswerSource, bool) {
    if let Some(index) = choice_index.filter(|idx| *idx < choices.len()) {
        return (
            choices[index].clone(),
            Some(index),
            QuestionAnswerSource::Choice,
            false,
        );
    }

    if !answer.is_empty() {
        if let Some(index) = choices.iter().position(|choice| choice == answer) {
            return (
                choices[index].clone(),
                Some(index),
                QuestionAnswerSource::Choice,
                false,
            );
        }
    }

    (String::new(), None, QuestionAnswerSource::Cancelled, true)
}

#[async_trait]
impl Tool for AskUserQuestionTool {
    fn name(&self) -> &str {
        "ask_user_question"
    }

    fn description(&self) -> &str {
        "Ask the user a question in the interactive UI and wait for the answer."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Question shown to the user."
                },
                "choices": {
                    "type": "array",
                    "description": "Optional quick-select options.",
                    "items": { "type": "string" },
                    "maxItems": MAX_CHOICES
                },
                "allow_free_text": {
                    "type": "boolean",
                    "description": "Allow free-text typing in addition to choices.",
                    "default": true
                },
                "placeholder": {
                    "type": "string",
                    "description": "Optional placeholder shown in free-text input."
                }
            },
            "required": ["question"]
        })
    }

    async fn invoke(
        &self,
        args: serde_json::Value,
        ctx: &InvokeContext,
    ) -> Result<ToolOutput, ToolError> {
        let args = parse_and_validate_args(args)?;

        let request = UserQuestionRequest {
            tool_call_id: ctx.tool_call_id.clone(),
            question: args.question.clone(),
            choices: args.choices.clone(),
            allow_free_text: args.allow_free_text,
            placeholder: args.placeholder.clone(),
        };

        let response = if let Some(handler) = &ctx.question_handler {
            handler(request)
        } else {
            crate::types::UserQuestionResponse::cancelled()
        };

        let answer = response.answer.trim().to_string();
        let (answer, choice_index, source, cancelled) = match response.source {
            QuestionAnswerSource::Cancelled => {
                (String::new(), None, QuestionAnswerSource::Cancelled, true)
            }
            QuestionAnswerSource::Choice => {
                normalize_choice_response(&answer, response.choice_index, &args.choices)
            }
            QuestionAnswerSource::Text => {
                if args.allow_free_text {
                    (answer, None, QuestionAnswerSource::Text, false)
                } else {
                    normalize_choice_response(&answer, response.choice_index, &args.choices)
                }
            }
        };

        Ok(ToolOutput::Structured(json!({
            "answer": answer,
            "choice_index": choice_index,
            "source": source,
            "cancelled": cancelled
        })))
    }

    fn permission(&self) -> ToolPermission {
        ToolPermission::Always
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_and_validate_args, AskUserQuestionTool, MAX_CHOICES};
    use crate::tools::Tool;
    use crate::types::{InvokeContext, QuestionAnswerSource, ToolOutput, UserQuestionResponse};
    use serde_json::json;
    use std::sync::Arc;

    #[test]
    fn args_validation_rejects_empty_question() {
        let result = parse_and_validate_args(json!({
            "question": "   "
        }));
        assert!(result.is_err());
    }

    #[test]
    fn args_validation_rejects_too_many_choices() {
        let choices = (0..=MAX_CHOICES)
            .map(|idx| format!("option-{idx}"))
            .collect::<Vec<_>>();

        let result = parse_and_validate_args(json!({
            "question": "Pick one",
            "choices": choices
        }));

        assert!(result.is_err());
    }

    #[test]
    fn args_validation_requires_choices_when_free_text_is_disabled() {
        let result = parse_and_validate_args(json!({
            "question": "Pick one",
            "allow_free_text": false
        }));

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn invoke_returns_structured_choice_answer() {
        let tool = AskUserQuestionTool::new();
        let ctx = InvokeContext {
            tool_call_id: "call_1".to_string(),
            approval_tx: None,
            question_handler: Some(Arc::new(|_| UserQuestionResponse {
                answer: String::new(),
                choice_index: Some(1),
                source: QuestionAnswerSource::Choice,
                cancelled: false,
            })),
            subagent_progress_tx: None,
            runtime_config: None,
            runtime_model_path: None,
            llm_engine: None,
            bash_kill_signal: None,
            thinking_control: nanocode_hf::ThinkingControl::None,
        };

        let output = tool
            .invoke(
                json!({
                    "question": "Continue?",
                    "choices": ["yes", "no"]
                }),
                &ctx,
            )
            .await
            .expect("tool should succeed");

        let ToolOutput::Structured(v) = output else {
            panic!("expected structured output");
        };

        assert_eq!(
            v,
            json!({
                "answer": "no",
                "choice_index": 1,
                "source": "choice",
                "cancelled": false
            })
        );
    }

    #[tokio::test]
    async fn invoke_without_handler_returns_cancelled() {
        let tool = AskUserQuestionTool::new();
        let ctx = InvokeContext {
            tool_call_id: "call_2".to_string(),
            approval_tx: None,
            question_handler: None,
            subagent_progress_tx: None,
            runtime_config: None,
            runtime_model_path: None,
            llm_engine: None,
            bash_kill_signal: None,
            thinking_control: nanocode_hf::ThinkingControl::None,
        };

        let output = tool
            .invoke(
                json!({
                    "question": "Need input?"
                }),
                &ctx,
            )
            .await
            .expect("tool should succeed");

        let ToolOutput::Structured(v) = output else {
            panic!("expected structured output");
        };

        assert_eq!(
            v,
            json!({
                "answer": "",
                "choice_index": serde_json::Value::Null,
                "source": "cancelled",
                "cancelled": true
            })
        );
    }
}

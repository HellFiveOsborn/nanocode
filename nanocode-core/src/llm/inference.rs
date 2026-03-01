//! LLM inference module using embedded llama.cpp (llama-cpp-2)

use crate::config::NcConfig;
use crate::types::LlmMessage;
use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, ChatTemplateResult, LlamaModel};
use llama_cpp_2::openai::OpenAIChatTemplateParams;
use llama_cpp_2::sampling::LlamaSampler;
use serde_json::{json, Value};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

/// LLM inference engine using embedded llama.cpp runtime
pub struct LlmEngine {
    model_path: PathBuf,
    config: NcConfig,
}

impl LlmEngine {
    /// Create a new engine
    pub fn new(model_path: &Path, config: &NcConfig) -> Result<Self, String> {
        Ok(Self {
            model_path: model_path.to_path_buf(),
            config: config.clone(),
        })
    }

    /// Generate completion
    pub fn generate(
        &self,
        messages: &[LlmMessage],
        max_tokens: u32,
        tools: Option<serde_json::Value>,
        tool_choice: Option<serde_json::Value>,
    ) -> Result<String, String> {
        self.generate_with_chunk_callback(messages, max_tokens, tools, tool_choice, None)
    }

    pub fn generate_with_chunk_callback(
        &self,
        messages: &[LlmMessage],
        max_tokens: u32,
        tools: Option<serde_json::Value>,
        tool_choice: Option<serde_json::Value>,
        mut on_chunk: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<String, String> {
        let mut backend =
            LlamaBackend::init().map_err(|e| format!("llama backend init failed: {e}"))?;
        backend.void_logs();

        let wants_gpu_layers = if self.config.model.n_gpu_layers < 0 {
            u32::MAX
        } else {
            self.config.model.n_gpu_layers as u32
        };

        if wants_gpu_layers > 0 && !backend.supports_gpu_offload() {
            return Err("GPU offload indisponível. Verifique se as libs do sistema (libllama/libggml) foram compiladas com backend CUDA e se o driver NVIDIA está ativo.".to_string());
        }

        let n_gpu_layers = if backend.supports_gpu_offload() {
            wants_gpu_layers
        } else {
            0
        };

        let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
        let model = LlamaModel::load_from_file(&backend, &self.model_path, &model_params)
            .map_err(|e| format!("failed to load model '{}': {e}", self.model_path.display()))?;

        let ctx_size = self
            .config
            .model
            .context_size
            .unwrap_or(32_768)
            .clamp(512, 262_144);

        let tools_json_owned = tools.as_ref().map(|v| v.to_string());
        let tool_choice_json_owned = tool_choice.as_ref().map(|v| {
            v.as_str()
                .map(|s| s.to_string())
                .unwrap_or_else(|| v.to_string())
        });
        let tool_choice_json = tool_choice_json_owned.as_deref();
        let tool_choice_json = match tool_choice_json {
            Some("auto") | Some("required") | Some("none") => tool_choice_json,
            Some(_) => tool_choice_json,
            None => None,
        };
        let template_result = self.build_prompt(
            &model,
            messages,
            tools_json_owned.as_deref(),
            tool_choice_json,
            None,
        )?;
        let prompt = &template_result.prompt;
        let tokens = model
            .str_to_token(prompt, AddBos::Never)
            .map_err(|e| format!("failed to tokenize prompt: {e}"))?;

        if tokens.is_empty() {
            return Err("prompt produced no tokens".to_string());
        }

        if tokens.len() as u32 > ctx_size {
            return Err(format!(
                "Prompt too long for context: {} tokens > n_ctx {}. Reduza prompt/system prompt ou aumente context_size.",
                tokens.len(),
                ctx_size
            ));
        }

        let n_batch = (tokens.len() as u32).max(1);
        let ctx_params = LlamaContextParams::default()
            .with_n_threads(num_cpus::get() as i32)
            .with_n_threads_batch((num_cpus::get() as i32).max(1))
            .with_n_batch(n_batch)
            .with_n_ctx(Some(NonZeroU32::new(ctx_size).expect("ctx_size > 0")));
        let ctx_params = if let Some(k_type) = self
            .config
            .model
            .kv_cache_type_k
            .as_deref()
            .and_then(parse_kv_cache_type)
        {
            ctx_params.with_type_k(k_type)
        } else {
            ctx_params
        };
        let ctx_params = if let Some(v_type) = self
            .config
            .model
            .kv_cache_type_v
            .as_deref()
            .and_then(parse_kv_cache_type)
        {
            ctx_params.with_type_v(v_type)
        } else {
            ctx_params
        };

        let mut ctx = model
            .new_context(&backend, ctx_params)
            .map_err(|e| format!("failed to create llama context: {e}"))?;

        let mut batch = LlamaBatch::new(tokens.len(), 1);
        for (i, token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch
                .add(*token, i as i32, &[0], is_last)
                .map_err(|e| format!("failed to add token to prompt batch: {e}"))?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| format!("failed to decode prompt: {e}"))?;

        let top_k = self.config.model.top_k.clamp(1, 200) as i32;
        let top_p = self.config.model.top_p.clamp(0.1, 1.0);
        let min_p = self.config.model.min_p.clamp(0.0, 1.0);
        let temperature = self.config.model.temperature.clamp(0.0, 2.0);
        let repeat_penalty = self.config.model.repeat_penalty.clamp(1.0, 2.0);
        let repeat_last_n = self.config.model.repeat_last_n.clamp(-1, 4096);
        let frequency_penalty = self.config.model.frequency_penalty.clamp(0.0, 2.0);
        let presence_penalty = self.config.model.presence_penalty.clamp(0.0, 2.0);

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::penalties(
                repeat_last_n,
                repeat_penalty,
                frequency_penalty,
                presence_penalty,
            ),
            LlamaSampler::top_k(top_k),
            LlamaSampler::top_p(top_p, 1),
            LlamaSampler::min_p(min_p, 1),
            LlamaSampler::temp(temperature),
            LlamaSampler::dist(42),
        ]);

        let mut result = String::new();
        let mut pos = tokens.len() as i32;
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        for _ in 0..max_tokens {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if model.is_eog_token(token) {
                break;
            }

            let piece = model
                .token_to_piece(token, &mut decoder, true, None)
                .map_err(|e| format!("failed to decode token piece: {e}"))?;
            result.push_str(&piece);
            if let Some(callback) = on_chunk.as_deref_mut() {
                if !callback(&piece) {
                    return Err("Generation interrupted by user".to_string());
                }
            }

            batch.clear();
            batch
                .add(token, pos, &[0], true)
                .map_err(|e| format!("failed to add generated token to batch: {e}"))?;
            pos += 1;

            ctx.decode(&mut batch)
                .map_err(|e| format!("failed to decode generated token: {e}"))?;
        }

        if let Ok(parsed_json) = template_result.parse_response_oaicompat(&result, false) {
            if let Ok(value) = serde_json::from_str::<Value>(&parsed_json) {
                if let Some(tool_calls) = value.get("tool_calls").and_then(|v| v.as_array()) {
                    if !tool_calls.is_empty() {
                        return Ok(json!({ "tool_calls": tool_calls }).to_string());
                    }
                }

                if let Some(content) = value.get("content").and_then(|c| c.as_str()) {
                    let cleaned = extract_final_answer(content);
                    let content = cleaned.trim();
                    if !content.is_empty() {
                        return Ok(content.to_string());
                    }
                }
            }
        }

        let result = extract_final_answer(&result);
        if result.is_empty() {
            return Err("No output generated".to_string());
        }

        Ok(result)
    }

    /// Build prompt from messages using model chat template
    fn build_prompt(
        &self,
        model: &LlamaModel,
        messages: &[LlmMessage],
        tools_json: Option<&str>,
        tool_choice_json: Option<&str>,
        tool_grammar: Option<&str>,
    ) -> Result<ChatTemplateResult, String> {
        let template = model
            .chat_template(None)
            .map_err(|e| format!("failed to read model chat template: {e}"))?;

        let messages_json = serde_json::to_string(
            &messages
                .iter()
                .map(|msg| {
                    let mut obj = serde_json::Map::new();
                    obj.insert("role".to_string(), json!(msg.role.to_string()));
                    obj.insert("content".to_string(), json!(msg.content));

                    if let Some(name) = &msg.name {
                        obj.insert("name".to_string(), json!(name));
                    }
                    if let Some(tool_call_id) = &msg.tool_call_id {
                        obj.insert("tool_call_id".to_string(), json!(tool_call_id));
                    }
                    if let Some(tool_calls) = &msg.tool_calls {
                        let oa_tool_calls = tool_calls
                            .iter()
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
                        obj.insert("tool_calls".to_string(), json!(oa_tool_calls));
                    }

                    serde_json::Value::Object(obj)
                })
                .collect::<Vec<_>>(),
        )
        .map_err(|e| format!("failed to serialize messages to OpenAI JSON: {e}"))?;

        let params = OpenAIChatTemplateParams {
            messages_json: &messages_json,
            tools_json,
            tool_choice: tool_choice_json,
            json_schema: None,
            grammar: tool_grammar,
            reasoning_format: Some("none"),
            chat_template_kwargs: Some("{\"enable_thinking\":false}"),
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: false,
            add_bos: false,
            add_eos: false,
            parse_tool_calls: true,
        };

        model
            .apply_chat_template_oaicompat(&template, &params)
            .map_err(|e| format!("failed to apply chat template (oaicompat): {e}"))
    }
}

fn extract_final_answer(text: &str) -> String {
    let mut out = text.to_string();

    while let Some(start) = out.find("<think>") {
        if let Some(end_rel) = out[start..].find("</think>") {
            let end = start + end_rel + "</think>".len();
            out.replace_range(start..end, "");
        } else {
            out.truncate(start);
            break;
        }
    }

    while let Some(start) = out.find("[Start thinking]") {
        if let Some(end_rel) = out[start..].find("[End thinking]") {
            let end = start + end_rel + "[End thinking]".len();
            out.replace_range(start..end, "");
        } else {
            out.truncate(start);
            break;
        }
    }

    // Alguns modelos podem emitir apenas o fechamento </think>.
    // Nesse caso, mantemos somente o conteúdo após o último fechamento.
    if let Some(end_tag_idx) = out.rfind("</think>") {
        let after = end_tag_idx + "</think>".len();
        out = out[after..].to_string();
    }

    out.replace("<|im_start|>assistant", "")
        .replace("<|im_end|>", "")
        .trim()
        .to_string()
}

/// Thread-safe wrapper for LLM engine
pub struct LlmEngineHandle {
    engine: std::sync::Mutex<LlmEngine>,
}

impl LlmEngineHandle {
    pub fn new(engine: LlmEngine) -> Self {
        Self {
            engine: std::sync::Mutex::new(engine),
        }
    }

    pub fn generate(
        &self,
        messages: &[LlmMessage],
        max_tokens: u32,
        tools: Option<serde_json::Value>,
        tool_choice: Option<serde_json::Value>,
    ) -> Result<String, String> {
        let engine = self.engine.lock().map_err(|e| e.to_string())?;
        engine.generate(messages, max_tokens, tools, tool_choice)
    }

    pub fn generate_with_chunk_callback(
        &self,
        messages: &[LlmMessage],
        max_tokens: u32,
        tools: Option<serde_json::Value>,
        tool_choice: Option<serde_json::Value>,
        on_chunk: &mut dyn FnMut(&str) -> bool,
    ) -> Result<String, String> {
        let engine = self.engine.lock().map_err(|e| e.to_string())?;
        engine.generate_with_chunk_callback(
            messages,
            max_tokens,
            tools,
            tool_choice,
            Some(on_chunk),
        )
    }
}

fn parse_kv_cache_type(raw: &str) -> Option<KvCacheType> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "f32" => Some(KvCacheType::F32),
        "f16" => Some(KvCacheType::F16),
        "q4_0" => Some(KvCacheType::Q4_0),
        "q4_1" => Some(KvCacheType::Q4_1),
        "q5_0" => Some(KvCacheType::Q5_0),
        "q5_1" => Some(KvCacheType::Q5_1),
        "q8_0" => Some(KvCacheType::Q8_0),
        "q8_1" => Some(KvCacheType::Q8_1),
        "q2_k" => Some(KvCacheType::Q2_K),
        "q3_k" => Some(KvCacheType::Q3_K),
        "q4_k" => Some(KvCacheType::Q4_K),
        "q5_k" => Some(KvCacheType::Q5_K),
        "q6_k" => Some(KvCacheType::Q6_K),
        "q8_k" => Some(KvCacheType::Q8_K),
        "bf16" => Some(KvCacheType::BF16),
        _ => None,
    }
}

//! LLM inference module using embedded llama.cpp (llama-cpp-2)

use crate::config::NcConfig;
use crate::interrupt::user_interrupted_error;
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

        let auto_gpu_layers = self.config.model.n_gpu_layers < 0;
        let wants_gpu_layers = if auto_gpu_layers {
            u32::MAX
        } else {
            self.config.model.n_gpu_layers as u32
        };

        let n_gpu_layers = if backend.supports_gpu_offload() {
            wants_gpu_layers
        } else {
            if wants_gpu_layers > 0 {
                eprintln!(
                    "Aviso: backend atual sem GPU offload. Executando em CPU-only (n_gpu_layers=0)."
                );
            }
            0
        };

        let model = load_model_with_fallback(&backend, &self.model_path, n_gpu_layers, auto_gpu_layers)
            .map_err(|err| {
                format!(
                    "failed to load model '{}': {err}",
                    self.model_path.display()
                )
            })?;

        let ctx_size = self
            .config
            .model
            .context_size
            .unwrap_or(8_192)
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

        let requested_k_type = self
            .config
            .model
            .kv_cache_type_k
            .as_deref()
            .and_then(parse_kv_cache_type);
        let requested_v_type = self
            .config
            .model
            .kv_cache_type_v
            .as_deref()
            .and_then(parse_kv_cache_type);

        let n_threads = self.config.model.n_threads.unwrap_or(0);
        let flash_attention = self.config.model.flash_attention.unwrap_or(false);

        let config_n_batch = self.config.model.n_batch.unwrap_or(0);
        let context_candidates =
            build_context_candidates(ctx_size, tokens.len() as u32, requested_k_type, requested_v_type, config_n_batch);
        let mut context_errors = Vec::new();
        let mut used_fallback = false;
        let mut used_candidate = context_candidates[0];
        let mut ctx: Option<_> = None;

        for (idx, candidate) in context_candidates.iter().copied().enumerate() {
            let thread_count = if n_threads > 0 { n_threads as i32 } else { num_cpus::get_physical().max(1) as i32 };
            let batch_threads = if n_threads > 0 { n_threads as i32 } else { num_cpus::get().max(1) as i32 };
            let mut ctx_params = LlamaContextParams::default()
                .with_n_threads(thread_count)
                .with_n_threads_batch(batch_threads)
                .with_n_batch(candidate.n_batch)
                .with_n_ctx(Some(NonZeroU32::new(candidate.ctx_size).expect("ctx_size > 0")));
            if let Some(k_type) = candidate.k_type {
                ctx_params = ctx_params.with_type_k(k_type);
            }
            if let Some(v_type) = candidate.v_type {
                ctx_params = ctx_params.with_type_v(v_type);
            }
            if flash_attention {
                ctx_params = ctx_params.with_flash_attention_policy(
                    llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED,
                );
            }

            match model.new_context(&backend, ctx_params) {
                Ok(created_ctx) => {
                    used_fallback = idx > 0;
                    used_candidate = candidate;
                    ctx = Some(created_ctx);
                    break;
                }
                Err(err) => context_errors.push(format!(
                    "{} -> {}",
                    candidate.describe(),
                    err
                )),
            }
        }

        let mut ctx = ctx.ok_or_else(|| {
            format!(
                "failed to create llama context after {} attempts: {}",
                context_errors.len(),
                context_errors.join(" | ")
            )
        })?;
        if used_fallback {
            eprintln!(
                "Aviso: fallback de contexto aplicado ({})",
                used_candidate.describe()
            );
        }

        let prompt_batch_capacity = used_candidate.n_batch.max(1) as usize;
        let mut batch = LlamaBatch::new(prompt_batch_capacity, 1);
        let mut prompt_pos = 0usize;
        while prompt_pos < tokens.len() {
            // Check for interrupt between prompt batches (allows ESC to cancel quickly).
            if let Some(callback) = on_chunk.as_deref_mut() {
                if !callback("") {
                    return Err(user_interrupted_error());
                }
            }

            batch.clear();
            let chunk_end = (prompt_pos + prompt_batch_capacity).min(tokens.len());
            for (local_idx, token) in tokens[prompt_pos..chunk_end].iter().enumerate() {
                let global_pos = prompt_pos + local_idx;
                let is_last = global_pos + 1 == tokens.len();
                batch
                    .add(*token, global_pos as i32, &[0], is_last)
                    .map_err(|e| format!("failed to add token to prompt batch: {e}"))?;
            }

            ctx.decode(&mut batch)
                .map_err(|e| format!("failed to decode prompt: {e}"))?;
            prompt_pos = chunk_end;
        }

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
            if let Some(callback) = on_chunk.as_deref_mut() {
                if !callback("") {
                    return Err(user_interrupted_error());
                }
            }
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
                    return Err(user_interrupted_error());
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

                if let Some(content) = extract_response_content_text(value.get("content")) {
                    let cleaned = extract_final_answer(&content);
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
            reasoning_format: None,
            chat_template_kwargs: Some("{\"enable_thinking\":true}"),
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: true,
            add_bos: false,
            add_eos: false,
            parse_tool_calls: true,
        };

        model
            .apply_chat_template_oaicompat(&template, &params)
            .map_err(|e| format!("failed to apply chat template (oaicompat): {e}"))
    }
}

fn extract_response_content_text(content: Option<&Value>) -> Option<String> {
    let content = content?;
    match content {
        Value::String(text) => Some(text.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                let is_text = part
                    .get("type")
                    .and_then(Value::as_str)
                    .map(|t| t == "text")
                    .unwrap_or(false);
                if !is_text {
                    continue;
                }
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str(text);
                }
            }
            if out.is_empty() {
                None
            } else {
                Some(out)
            }
        }
        _ => None,
    }
}

#[derive(Clone, Copy)]
struct ContextCandidate {
    ctx_size: u32,
    n_batch: u32,
    k_type: Option<KvCacheType>,
    v_type: Option<KvCacheType>,
}

impl ContextCandidate {
    fn describe(&self) -> String {
        format!(
            "n_ctx={}, n_batch={}, kv_k={:?}, kv_v={:?}",
            self.ctx_size, self.n_batch, self.k_type, self.v_type
        )
    }
}

fn build_context_candidates(
    requested_ctx_size: u32,
    prompt_tokens: u32,
    requested_k_type: Option<KvCacheType>,
    requested_v_type: Option<KvCacheType>,
    config_n_batch: u32,
) -> Vec<ContextCandidate> {
    // Use config n_batch if set, otherwise scale based on prompt size (cap at 2048).
    let prompt_safe_batch = if config_n_batch > 0 {
        config_n_batch.clamp(64, 4096)
    } else {
        prompt_tokens.clamp(256, 2048)
    };
    let min_ctx = prompt_tokens.saturating_add(256).max(512);

    let mut ctx_sizes = vec![requested_ctx_size, requested_ctx_size.saturating_mul(3) / 4];
    if requested_ctx_size > 12_288 {
        ctx_sizes.push(12_288);
    }
    if requested_ctx_size > 8_192 {
        ctx_sizes.push(8_192);
    }
    if requested_ctx_size > 6_144 {
        ctx_sizes.push(6_144);
    }
    ctx_sizes.push(4_096);

    ctx_sizes.sort_unstable_by(|a, b| b.cmp(a));
    ctx_sizes.dedup();

    let mut candidates = Vec::new();
    for ctx_size in ctx_sizes {
        if ctx_size < min_ctx {
            continue;
        }

        let n_batch = prompt_safe_batch.min(ctx_size);
        candidates.push(ContextCandidate {
            ctx_size,
            n_batch,
            k_type: requested_k_type,
            v_type: requested_v_type,
        });
        candidates.push(ContextCandidate {
            ctx_size,
            n_batch,
            k_type: Some(KvCacheType::Q4_0),
            v_type: Some(KvCacheType::Q4_0),
        });
        candidates.push(ContextCandidate {
            ctx_size,
            n_batch,
            k_type: None,
            v_type: None,
        });
    }

    if candidates.is_empty() {
        let fallback_ctx = min_ctx.min(requested_ctx_size).max(512);
        let n_batch = prompt_safe_batch.min(fallback_ctx);
        candidates.push(ContextCandidate {
            ctx_size: fallback_ctx,
            n_batch,
            k_type: None,
            v_type: None,
        });
    }

    candidates
}

fn load_model_with_fallback(
    backend: &LlamaBackend,
    model_path: &Path,
    n_gpu_layers: u32,
    auto_gpu_layers: bool,
) -> Result<LlamaModel, String> {
    let candidates = gpu_layer_candidates(n_gpu_layers, auto_gpu_layers);
    let mut errors = Vec::new();

    for candidate in candidates {
        let model_params = LlamaModelParams::default().with_n_gpu_layers(candidate);
        match LlamaModel::load_from_file(backend, model_path, &model_params) {
            Ok(model) => {
                if candidate != n_gpu_layers {
                    eprintln!(
                        "Aviso: fallback de offload aplicado (n_gpu_layers={} -> {}).",
                        n_gpu_layers, candidate
                    );
                }
                return Ok(model);
            }
            Err(err) => {
                errors.push((candidate, err.to_string()));
            }
        }
    }

    if errors.is_empty() {
        return Err("no model load candidates available".to_string());
    }

    let attempts = errors
        .iter()
        .map(|(layers, err)| format!("n_gpu_layers={layers}: {err}"))
        .collect::<Vec<_>>()
        .join(" | ");

    Err(format!(
        "{attempts}. Se estiver usando quantizações extremas (ex: IQ3_XXS/IQ2), tente Q4_K_M ou Q5_K_M."
    ))
}

fn gpu_layer_candidates(initial: u32, auto_gpu_layers: bool) -> Vec<u32> {
    if initial == 0 {
        return vec![0];
    }

    let mut candidates = Vec::new();
    candidates.push(initial);

    if auto_gpu_layers {
        if initial == u32::MAX {
            candidates.extend([96, 64, 48, 32, 24, 16, 8, 0]);
        } else {
            let mut current = initial;
            while current > 16 {
                current = (current.saturating_mul(3)).saturating_div(4);
                if current > 0 {
                    candidates.push(current);
                } else {
                    break;
                }
            }
            candidates.extend([16, 8, 0]);
        }
    } else {
        let mut current = initial;
        while current > 16 {
            current /= 2;
            if current > 0 {
                candidates.push(current);
            } else {
                break;
            }
        }
        candidates.push(0);
    }

    candidates.sort_unstable();
    candidates.dedup();
    candidates.sort_unstable_by(|a, b| b.cmp(a));
    candidates
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

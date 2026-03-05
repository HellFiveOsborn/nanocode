use nanocode_core::NcConfig;
use nanocode_hf::{
    default_model, find_any_installed_model_quant, find_model, find_quant_by_name,
    is_compatible_quant_size, recommend_inference_tuning, recommend_runtime_limits,
    select_cached_quant_for_hardware, HardwareInfo, ModelSpec, QuantizationVariant,
};

const MIN_CONTEXT_SIZE: u32 = 8_192;

#[derive(Clone)]
pub struct RuntimeEnv {
    pub config: NcConfig,
    pub model: &'static ModelSpec,
    pub quant: &'static QuantizationVariant,
    pub model_file: std::path::PathBuf,
    pub max_tokens: u32,
    pub model_label: String,
    pub hardware: HardwareInfo,
}

#[derive(Clone)]
pub struct RuntimeSnapshot {
    pub model_label: String,
    pub context_size: u32,
    pub max_tokens: u32,
}

pub fn build_runtime(
    config: &NcConfig,
    ctk_override: Option<String>,
    ctv_override: Option<String>,
) -> Result<RuntimeEnv, String> {
    let hw = HardwareInfo::detect();
    let configured_model = config
        .active_model
        .as_deref()
        .and_then(find_model)
        .unwrap_or_else(default_model);

    let model_path = NcConfig::models_dir();
    let no_model_error = || {
        if has_any_gguf_file(&model_path) {
            "Nenhum modelo compatível está pronto. Foram encontrados arquivos .gguf, mas a quantização selecionada pode não ser suportada pela sua build local do llama.cpp (ex: IQ3_*). Execute 'nanocode setup' e escolha Q2_K, Q3_K_S ou Q4_K_M.".to_string()
        } else {
            "Nenhum modelo instalado".to_string()
        }
    };
    let (model, quant) = if let Some(active_name) = config.active_quant.as_deref() {
        if let Some(active_quant) = find_quant_by_name(configured_model, active_name) {
            let active_path = model_path.join(active_quant.filename);
            if is_valid_quant_file(&active_path, active_quant.size_bytes) {
                (configured_model, active_quant)
            } else {
                select_cached_quant_for_hardware(&model_path, configured_model, &hw)
                    .map(|q| (configured_model, q))
                    .or_else(|| find_any_installed_model_quant(&model_path))
                    .ok_or_else(no_model_error)?
            }
        } else {
            select_cached_quant_for_hardware(&model_path, configured_model, &hw)
                .map(|q| (configured_model, q))
                .or_else(|| find_any_installed_model_quant(&model_path))
                .ok_or_else(no_model_error)?
        }
    } else {
        select_cached_quant_for_hardware(&model_path, configured_model, &hw)
            .map(|q| (configured_model, q))
            .or_else(|| find_any_installed_model_quant(&model_path))
            .ok_or_else(no_model_error)?
    };

    let model_file = model_path.join(quant.filename);
    let memory_mb = hw.vram_mb.map(|vram| vram.min(hw.ram_mb)).unwrap_or(hw.ram_mb);
    let runtime_limits = recommend_runtime_limits(memory_mb, model, true);
    let inference_tuning = recommend_inference_tuning(&hw, quant);

    let mut runtime_config = config.clone();
    runtime_config.model.context_size = Some(
        runtime_config
            .model
            .context_size
            .unwrap_or(runtime_limits.context_size)
            .min(model.max_context_size)
            .max(MIN_CONTEXT_SIZE),
    );
    if runtime_config.model.n_gpu_layers < 0 {
        runtime_config.model.n_gpu_layers = inference_tuning.n_gpu_layers;
    }
    if runtime_config.model.kv_cache_type_k.is_none() {
        runtime_config.model.kv_cache_type_k = Some(inference_tuning.kv_cache_type_k.to_string());
    }
    if runtime_config.model.kv_cache_type_v.is_none() {
        runtime_config.model.kv_cache_type_v = Some(inference_tuning.kv_cache_type_v.to_string());
    }
    if runtime_config.model.n_batch.is_none() {
        runtime_config.model.n_batch = Some(inference_tuning.n_batch);
    }
    if runtime_config.model.flash_attention.is_none() {
        runtime_config.model.flash_attention = Some(inference_tuning.flash_attention);
    }
    if runtime_config.model.n_threads.is_none() && inference_tuning.n_threads > 0 {
        runtime_config.model.n_threads = Some(inference_tuning.n_threads);
    }

    // Cap context_size to what actually fits in VRAM/RAM with chosen KV cache type.
    let tuned_ctx = runtime_config.model.context_size.unwrap_or(MIN_CONTEXT_SIZE);
    if tuned_ctx > inference_tuning.context_size_cap && inference_tuning.context_size_cap >= 2_048 {
        runtime_config.model.context_size = Some(inference_tuning.context_size_cap);
    }

    if let Some(ctk) = ctk_override {
        runtime_config.model.kv_cache_type_k = Some(ctk);
    }
    if let Some(ctv) = ctv_override {
        runtime_config.model.kv_cache_type_v = Some(ctv);
    }

    let max_tokens = runtime_config
        .model
        .max_tokens
        .min(runtime_limits.max_tokens)
        .clamp(512, 8192);

    Ok(RuntimeEnv {
        config: runtime_config,
        model,
        quant,
        model_file,
        max_tokens,
        model_label: format!("{} ({})", model.display_name, quant.name),
        hardware: hw,
    })
}

pub fn runtime_snapshot(
    config: &NcConfig,
    ctk_override: Option<String>,
    ctv_override: Option<String>,
) -> Option<RuntimeSnapshot> {
    let env = build_runtime(config, ctk_override, ctv_override).ok()?;
    Some(RuntimeSnapshot {
        model_label: env.model_label,
        context_size: env.config.model.context_size.unwrap_or(MIN_CONTEXT_SIZE),
        max_tokens: env.max_tokens,
    })
}

pub fn is_thinking_model(display_name: &str, quant_name: &str) -> bool {
    let combined = format!(
        "{} {}",
        display_name.to_ascii_lowercase(),
        quant_name.to_ascii_lowercase()
    );
    combined.contains("thinking") || combined.contains("r1")
}

fn is_valid_quant_file(path: &std::path::Path, expected_size: u64) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    let actual_size = meta.len();
    is_compatible_quant_size(actual_size, expected_size)
}

fn has_any_gguf_file(models_dir: &std::path::Path) -> bool {
    let Ok(entries) = std::fs::read_dir(models_dir) else {
        return false;
    };
    entries.flatten().any(|entry| {
        entry
            .path()
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
    })
}

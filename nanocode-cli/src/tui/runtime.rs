use nanocode_core::NcConfig;
use nanocode_hf::{
    default_model, find_any_installed_model_quant, find_installed_quant, find_model,
    find_quant_by_name, is_compatible_quant_size, recommend_runtime_limits, HardwareInfo,
    ModelSpec, QuantizationVariant,
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
    let configured_model = config
        .active_model
        .as_deref()
        .and_then(find_model)
        .unwrap_or_else(default_model);

    let model_path = NcConfig::models_dir();
    let (model, quant) = if let Some(active_name) = config.active_quant.as_deref() {
        if let Some(active_quant) = find_quant_by_name(configured_model, active_name) {
            let active_path = model_path.join(active_quant.filename);
            if is_valid_quant_file(&active_path, active_quant.size_bytes) {
                (configured_model, active_quant)
            } else {
                find_installed_quant(&model_path, configured_model)
                    .map(|q| (configured_model, q))
                    .or_else(|| find_any_installed_model_quant(&model_path))
                    .ok_or_else(|| "No model installed".to_string())?
            }
        } else {
            find_installed_quant(&model_path, configured_model)
                .map(|q| (configured_model, q))
                .or_else(|| find_any_installed_model_quant(&model_path))
                .ok_or_else(|| "No model installed".to_string())?
        }
    } else {
        find_installed_quant(&model_path, configured_model)
            .map(|q| (configured_model, q))
            .or_else(|| find_any_installed_model_quant(&model_path))
            .ok_or_else(|| "No model installed".to_string())?
    };

    let model_file = model_path.join(quant.filename);
    let hw = HardwareInfo::detect();
    let memory_mb = hw.vram_mb.unwrap_or(hw.ram_mb);
    let runtime_limits = recommend_runtime_limits(memory_mb, model, true);

    let mut runtime_config = config.clone();
    runtime_config.model.context_size = Some(
        runtime_config
            .model
            .context_size
            .unwrap_or(runtime_limits.context_size)
            .min(model.max_context_size)
            .max(MIN_CONTEXT_SIZE),
    );

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

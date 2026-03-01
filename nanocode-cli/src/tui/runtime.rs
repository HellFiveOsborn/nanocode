use nanocode_core::NcConfig;
use nanocode_hf::{
    find_installed_quant, find_quant_by_name, recommend_runtime_limits, HardwareInfo, THE_MODEL,
};

const MIN_CONTEXT_SIZE: u32 = 8_192;

#[derive(Clone)]
pub struct RuntimeEnv {
    pub config: NcConfig,
    pub model_file: std::path::PathBuf,
    pub max_tokens: u32,
    pub model_label: String,
    pub hardware: HardwareInfo,
}

pub fn build_runtime(
    config: &NcConfig,
    ctk_override: Option<String>,
    ctv_override: Option<String>,
) -> RuntimeEnv {
    let model_path = NcConfig::models_dir();
    let quant = if let Some(active_name) = config.active_quant.as_deref() {
        if let Some(active_quant) = find_quant_by_name(active_name) {
            let active_path = model_path.join(active_quant.filename);
            if active_path.exists() {
                active_quant
            } else {
                find_installed_quant(&model_path).expect("No model installed")
            }
        } else {
            find_installed_quant(&model_path).expect("No model installed")
        }
    } else {
        find_installed_quant(&model_path).expect("No model installed")
    };

    let model_file = model_path.join(quant.filename);
    let hw = HardwareInfo::detect();
    let memory_mb = hw.vram_mb.unwrap_or(hw.ram_mb);
    let runtime_limits = recommend_runtime_limits(memory_mb, &THE_MODEL, true);

    let mut runtime_config = config.clone();
    runtime_config.model.context_size = Some(
        runtime_config
            .model
            .context_size
            .unwrap_or(runtime_limits.context_size)
            .min(THE_MODEL.max_context_size)
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

    RuntimeEnv {
        config: runtime_config,
        model_file,
        max_tokens,
        model_label: format!("{} ({})", THE_MODEL.display_name, quant.name),
        hardware: hw,
    }
}

pub fn is_thinking_model(display_name: &str, quant_name: &str) -> bool {
    let combined = format!(
        "{} {}",
        display_name.to_ascii_lowercase(),
        quant_name.to_ascii_lowercase()
    );
    combined.contains("thinking") || combined.contains("r1")
}

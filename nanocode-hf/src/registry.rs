//! Model registry

use crate::quantization::{QuantizationVariant, QWEN3_4B_QUANTIZATIONS};
use std::path::Path;

/// Model specification
pub struct ModelSpec {
    pub display_name: &'static str,
    pub hf_repo: &'static str,
    /// Maximum context supported by the model architecture
    pub max_context_size: u32,
    /// Recommended context for general usage
    pub recommended_context_general: u32,
    /// Recommended context for coding / harder reasoning tasks
    pub recommended_context_coding: u32,
    pub quantizations: &'static [QuantizationVariant],
}

/// Runtime limits computed from model specs + hardware
#[derive(Debug, Clone, Copy)]
pub struct RuntimeLimits {
    pub context_size: u32,
    pub max_tokens: u32,
}

/// The model: Qwen3 4B Thinking
pub const THE_MODEL: ModelSpec = ModelSpec {
    display_name: "Qwen3 4B Thinking",
    hf_repo: "unsloth/Qwen3-4B-Thinking-2507-GGUF",
    max_context_size: 262_144,
    recommended_context_general: 32_768,
    recommended_context_coding: 81_920,
    quantizations: QWEN3_4B_QUANTIZATIONS,
};

/// Compute runtime context/token limits from model and hardware memory budget.
///
/// Memory policy (VRAM preferred, RAM fallback):
/// - >= 24 GB: up to 81_920
/// - >= 16 GB: up to 65_536
/// - >= 12 GB: up to 49_152
/// - >= 8  GB: up to 32_768
/// - >= 6  GB: up to 24_576
/// - >= 4  GB: up to 16_384
/// - < 4  GB: up to 8_192
pub fn recommend_runtime_limits(
    memory_mb: u64,
    model: &ModelSpec,
    coding_mode: bool,
) -> RuntimeLimits {
    let target_context = if coding_mode {
        model.recommended_context_coding
    } else {
        model.recommended_context_general
    };

    let memory_cap = if memory_mb >= 24_000 {
        81_920
    } else if memory_mb >= 16_000 {
        65_536
    } else if memory_mb >= 12_000 {
        49_152
    } else if memory_mb >= 8_000 {
        32_768
    } else if memory_mb >= 6_000 {
        24_576
    } else if memory_mb >= 4_000 {
        16_384
    } else {
        8_192
    };

    let context_size = target_context
        .min(memory_cap)
        .min(model.max_context_size)
        .max(8_192);

    // Keep completion budget proporcional ao contexto com folga para respostas de coding.
    let max_tokens = (context_size / 4).clamp(2_048, 12_288);

    RuntimeLimits {
        context_size,
        max_tokens,
    }
}

/// Find installed quantization in models directory
pub fn find_installed_quant(models_dir: &Path) -> Option<&'static QuantizationVariant> {
    if !models_dir.exists() {
        return None;
    }

    // Check each quantization variant
    for quant in THE_MODEL.quantizations {
        let path = models_dir.join(quant.filename);
        if path.exists() {
            return Some(quant);
        }
    }

    None
}

/// Find a quantization by name
pub fn find_quant_by_name(name: &str) -> Option<&'static QuantizationVariant> {
    THE_MODEL.quantizations.iter().find(|q| q.name == name)
}

/// Get download URL for a quantization
pub fn get_download_url(quant: &QuantizationVariant) -> String {
    format!(
        "https://huggingface.co/{}/resolve/main/{}",
        THE_MODEL.hf_repo, quant.filename
    )
}

//! Model registry

use crate::catalog::{models, ModelSpec};
use crate::quantization::is_compatible_quant_size;
use crate::quantization::QuantizationVariant;
use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};

/// Runtime limits computed from model specs + hardware
#[derive(Debug, Clone, Copy)]
pub struct RuntimeLimits {
    pub context_size: u32,
    pub max_tokens: u32,
}

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
pub fn find_installed_quant(
    models_dir: &Path,
    model: &ModelSpec,
) -> Option<&'static QuantizationVariant> {
    if !models_dir.exists() {
        return None;
    }

    // Check each quantization variant
    for quant in model.quantizations {
        let path = models_dir.join(quant.filename);
        if is_cached_quant_valid(&path, quant.size_bytes) {
            return Some(quant);
        }
    }

    None
}

/// List all cached quantizations for a model.
pub fn list_cached_quants(
    models_dir: &Path,
    model: &ModelSpec,
) -> Vec<&'static QuantizationVariant> {
    if !models_dir.exists() {
        return Vec::new();
    }

    model
        .quantizations
        .iter()
        .filter(|quant| {
            let path = models_dir.join(quant.filename);
            is_cached_quant_valid(&path, quant.size_bytes)
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct CacheCleanupReport {
    pub removed_paths: Vec<PathBuf>,
}

/// Enforce a single cached quantization for the given model.
///
/// Keeps `keep_quant_name` and removes all other `.gguf` and `.part` artifacts
/// for that model.
pub fn enforce_single_quant_cache(
    models_dir: &Path,
    model: &ModelSpec,
    keep_quant_name: &str,
) -> Result<CacheCleanupReport> {
    let keep_quant = find_quant_by_name(model, keep_quant_name)
        .ok_or_else(|| anyhow!("Unknown quantization: {}", keep_quant_name))?;
    let keep_path = models_dir.join(keep_quant.filename);
    if !is_cached_quant_valid(&keep_path, keep_quant.size_bytes) {
        return Err(anyhow!(
            "Selected quantization is not cached with a valid size on disk: {}",
            keep_quant_name
        ));
    }

    let mut removed_paths = Vec::new();
    for quant in model.quantizations {
        if quant.name == keep_quant_name {
            continue;
        }
        let gguf_path = models_dir.join(quant.filename);
        if gguf_path.exists() {
            std::fs::remove_file(&gguf_path)?;
            removed_paths.push(gguf_path);
        }
        let part_path = models_dir.join(format!("{}.part", quant.filename));
        if part_path.exists() {
            std::fs::remove_file(&part_path)?;
            removed_paths.push(part_path);
        }
    }

    Ok(CacheCleanupReport { removed_paths })
}

/// Find a quantization by name
pub fn find_quant_by_name(model: &ModelSpec, name: &str) -> Option<&'static QuantizationVariant> {
    model.quantizations.iter().find(|q| q.name == name)
}

/// Get download URL for a quantization
pub fn get_download_url(model: &ModelSpec, quant: &QuantizationVariant) -> String {
    format!(
        "https://huggingface.co/{}/resolve/main/{}",
        model.hf_repo, quant.filename
    )
}

/// Find any installed model + quantization pair from the catalog.
pub fn find_any_installed_model_quant(
    models_dir: &Path,
) -> Option<(&'static ModelSpec, &'static QuantizationVariant)> {
    for model in models() {
        if let Some(quant) = find_installed_quant(models_dir, model) {
            return Some((model, quant));
        }
    }
    None
}

fn is_cached_quant_valid(path: &Path, expected_size: u64) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    let actual_size = meta.len();
    is_compatible_quant_size(actual_size, expected_size)
}

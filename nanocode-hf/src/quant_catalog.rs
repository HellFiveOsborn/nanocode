//! Dynamic quantization catalog with on-disk cache.

use crate::catalog::{models, ModelSpec};
use crate::quantization::{QualityTier, QuantizationVariant};
use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

const USER_AGENT: &str = "nanocode-hf/1.0";

static DYNAMIC_QUANTS: OnceLock<Mutex<HashMap<&'static str, &'static [QuantizationVariant]>>> =
    OnceLock::new();

#[derive(Debug, Deserialize)]
struct HfTreeEntry {
    #[serde(rename = "type")]
    entry_type: String,
    path: String,
    size: Option<u64>,
    lfs: Option<HfLfsMeta>,
}

#[derive(Debug, Deserialize)]
struct HfLfsMeta {
    size: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedQuant {
    name: String,
    filename: String,
    size_bytes: u64,
    quality: QualityTier,
    notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuantCacheFile {
    model_id: String,
    hf_repo: String,
    fetched_at_unix: u64,
    variants: Vec<CachedQuant>,
}

fn dynamic_quants_map() -> &'static Mutex<HashMap<&'static str, &'static [QuantizationVariant]>> {
    DYNAMIC_QUANTS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Returns quantizations for a model, loading dynamic variants from cache when needed.
pub fn model_quantizations(model: &ModelSpec) -> &'static [QuantizationVariant] {
    if !model.quantizations.is_empty() {
        return model.quantizations;
    }

    if let Some(variants) = {
        let map = dynamic_quants_map()
            .lock()
            .expect("dynamic quant map poisoned");
        map.get(model.id).copied()
    } {
        return variants;
    }

    let variants = load_cached_variants(model)
        .ok()
        .and_then(|cache| (!cache.variants.is_empty()).then_some(cache))
        .map(leak_cached_variants);

    if let Some(variants) = variants {
        let mut map = dynamic_quants_map()
            .lock()
            .expect("dynamic quant map poisoned");
        map.insert(model.id, variants);
        return variants;
    }

    &[]
}

/// Preload dynamic model quantizations from local cache or Hugging Face API.
pub async fn preload_dynamic_quantizations() {
    let client = Client::builder()
        .user_agent(USER_AGENT)
        .timeout(std::time::Duration::from_secs(8))
        .build()
        .unwrap_or_else(|_| Client::new());

    for model in models()
        .iter()
        .copied()
        .filter(|m| m.quantizations.is_empty())
    {
        let cached = model_quantizations(&model);
        let needs_refresh = cached.is_empty() || cached.iter().all(|q| q.size_bytes == 0);
        if needs_refresh {
            let _ = fetch_and_cache_model_quantizations(&client, &model).await;
        }
    }
}

async fn fetch_and_cache_model_quantizations(client: &Client, model: &ModelSpec) -> Result<()> {
    let url = format!(
        "https://huggingface.co/api/models/{}/tree/main?recursive=1&limit=1000",
        model.hf_repo
    );
    let response = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("failed to query Hugging Face tree API: {}", model.hf_repo))?
        .error_for_status()
        .with_context(|| format!("Hugging Face tree API returned error for {}", model.hf_repo))?;

    let payload: Vec<HfTreeEntry> = response.json().await.with_context(|| {
        format!(
            "failed to parse Hugging Face tree response for {}",
            model.hf_repo
        )
    })?;

    let mut variants = Vec::<CachedQuant>::new();
    for entry in payload {
        if entry.entry_type != "file" {
            continue;
        }

        let filename = entry
            .path
            .rsplit('/')
            .next()
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned);
        let Some(filename) = filename else {
            continue;
        };
        if !filename.to_ascii_lowercase().ends_with(".gguf") {
            continue;
        }

        let Some(name) = infer_quant_name(&filename) else {
            continue;
        };
        if !is_runtime_supported_quant_name(&name) {
            continue;
        }

        variants.push(CachedQuant {
            quality: infer_quality(&name),
            notes: None,
            size_bytes: entry
                .size
                .or(entry.lfs.and_then(|lfs| lfs.size))
                .unwrap_or(0),
            name,
            filename,
        });
    }

    variants.sort_by(|a, b| {
        quality_rank(b.quality)
            .cmp(&quality_rank(a.quality))
            .then_with(|| b.size_bytes.cmp(&a.size_bytes))
            .then_with(|| a.name.cmp(&b.name))
    });
    variants.dedup_by(|a, b| a.name.eq_ignore_ascii_case(&b.name));

    if variants.is_empty() {
        return Ok(());
    }

    let cache_file = QuantCacheFile {
        model_id: model.id.to_string(),
        hf_repo: model.hf_repo.to_string(),
        fetched_at_unix: now_unix_epoch(),
        variants,
    };
    save_cached_variants(model, &cache_file)?;

    let leaked = leak_cached_variants(cache_file);
    let mut map = dynamic_quants_map()
        .lock()
        .expect("dynamic quant map poisoned");
    map.insert(model.id, leaked);
    Ok(())
}

fn infer_quant_name(filename: &str) -> Option<String> {
    let stem = filename.strip_suffix(".gguf").unwrap_or(filename);
    let upper = stem.to_ascii_uppercase();

    const TOKENS: &[&str] = &[
        "Q8_K_XL", "Q6_K_XL", "Q5_K_XL", "Q3_K_XL", "Q3_K_L", "Q2_K_L", "Q5_K_M", "Q5_K_S",
        "Q4_K_M", "Q4_K_S", "Q3_K_M", "Q3_K_S", "IQ4_NL", "IQ4_XS", "IQ3_XS", "IQ2_XXS", "IQ2_XS",
        "IQ1_S", "Q8_0", "Q6_K", "Q4_1", "Q4_0", "Q2_K", "BF16", "F16",
    ];

    for token in TOKENS {
        if upper.contains(token) {
            return Some((*token).to_string());
        }
    }

    stem.rsplit('-').next().map(|v| v.to_string())
}

fn infer_quality(name: &str) -> QualityTier {
    if name.eq_ignore_ascii_case("BF16") || name.eq_ignore_ascii_case("F16") {
        QualityTier::Maximum
    } else if name.starts_with("Q8") {
        QualityTier::Excellent
    } else if name.starts_with("Q6") || name.starts_with("Q5") {
        QualityTier::VeryGood
    } else if name.starts_with("Q4") || name.starts_with("IQ4") {
        QualityTier::Good
    } else if name.starts_with("Q3") || name.starts_with("IQ3") {
        QualityTier::Acceptable
    } else if name.starts_with("Q2") {
        QualityTier::Low
    } else if name.starts_with("IQ2") || name.starts_with("IQ1") {
        QualityTier::VeryLow
    } else {
        QualityTier::Good
    }
}

fn quality_rank(quality: QualityTier) -> u8 {
    match quality {
        QualityTier::Maximum => 7,
        QualityTier::Excellent => 6,
        QualityTier::VeryGood => 5,
        QualityTier::Good => 4,
        QualityTier::Acceptable => 3,
        QualityTier::Low => 2,
        QualityTier::VeryLow => 1,
    }
}

fn leak_cached_variants(cache: QuantCacheFile) -> &'static [QuantizationVariant] {
    let mut variants = Vec::with_capacity(cache.variants.len());
    for entry in cache.variants {
        if !is_runtime_supported_quant_name(&entry.name) {
            continue;
        }
        let name: &'static str = Box::leak(entry.name.into_boxed_str());
        let filename: &'static str = Box::leak(entry.filename.into_boxed_str());
        let notes = entry
            .notes
            .map(|note| Box::leak(note.into_boxed_str()) as &'static str);
        variants.push(QuantizationVariant {
            name,
            filename,
            size_bytes: entry.size_bytes,
            quality: entry.quality,
            notes,
        });
    }
    Box::leak(variants.into_boxed_slice())
}

fn is_runtime_supported_quant_name(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    !upper.starts_with("IQ3_") && upper != "F32"
}

fn save_cached_variants(model: &ModelSpec, cache: &QuantCacheFile) -> Result<()> {
    let path = cache_path_for(model);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec_pretty(cache)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

fn load_cached_variants(model: &ModelSpec) -> Result<QuantCacheFile> {
    let path = cache_path_for(model);
    let bytes = std::fs::read(&path)
        .with_context(|| format!("failed to read quantization cache: {}", path.display()))?;
    let cache: QuantCacheFile = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse quantization cache: {}", path.display()))?;

    if cache.model_id != model.id || cache.hf_repo != model.hf_repo {
        anyhow::bail!("cache metadata mismatch for {}", model.id);
    }
    Ok(cache)
}

fn cache_path_for(model: &ModelSpec) -> PathBuf {
    let base = dirs::cache_dir().unwrap_or_else(|| PathBuf::from("."));
    base.join("nanocode")
        .join("hf-catalog")
        .join(format!("{}.json", sanitize_file_component(model.id)))
}

fn sanitize_file_component(input: &str) -> String {
    input
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn now_unix_epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

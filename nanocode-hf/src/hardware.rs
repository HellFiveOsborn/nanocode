//! Hardware detection

use crate::catalog::ModelSpec;
use crate::quant_catalog::model_quantizations;
use crate::quantization::QuantizationVariant;
use serde::{Deserialize, Serialize};
use sysinfo::System;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeMode {
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeTelemetry {
    pub mode: ComputeMode,
    pub gpu_vendor: Option<GpuVendor>,
    pub cpu_usage_pct: f32,
    pub ram_used_mb: u64,
    pub ram_total_mb: u64,
    pub cpu_temp_c: Option<f32>,
    pub gpu_usage_pct: Option<f32>,
    pub gpu_temp_c: Option<f32>,
    pub vram_used_mb: Option<u64>,
    pub vram_total_mb: Option<u64>,
}

/// GPU vendor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuVendor {
    Nvidia,
    Apple,
    Amd,
    Unknown,
}

/// Hardware information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub vram_mb: Option<u64>,
    pub ram_mb: u64,
    pub gpu_vendor: Option<GpuVendor>,
    pub has_cuda: bool,
    pub has_metal: bool,
}

/// Runtime tuning recommendations derived from detected hardware + selected quantization.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct InferenceTuning {
    pub n_gpu_layers: i32,
    pub kv_cache_type_k: &'static str,
    pub kv_cache_type_v: &'static str,
    /// Recommended n_batch for prompt processing (higher = faster on GPU).
    pub n_batch: u32,
    /// Maximum context size that fits in available memory with this config.
    pub context_size_cap: u32,
    /// Whether flash attention should be enabled.
    pub flash_attention: bool,
    /// Number of threads for inference (0 = auto/default).
    pub n_threads: u32,
}

impl HardwareInfo {
    /// Detect hardware on the system
    pub fn detect() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();

        // Get RAM
        let ram_mb = sys.total_memory() / (1024 * 1024);

        // Try to detect GPU
        let (vram_mb, gpu_vendor, has_cuda, has_metal) = Self::detect_gpu();

        Self {
            vram_mb,
            ram_mb,
            gpu_vendor,
            has_cuda,
            has_metal,
        }
    }

    fn detect_gpu() -> (Option<u64>, Option<GpuVendor>, bool, bool) {
        #[cfg(target_os = "linux")]
        {
            // Try nvidia-smi first
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .arg("--query-gpu=memory.total")
                .arg("--format=csv,noheader,nounits")
                .output()
            {
                if output.status.success() {
                    let vram = String::from_utf8_lossy(&output.stdout);
                    if let Some(mb) = vram.trim().parse::<u64>().ok() {
                        return (Some(mb), Some(GpuVendor::Nvidia), true, false);
                    }
                }
            }

            // Try rocm-smi for AMD
            if let Ok(output) = std::process::Command::new("rocm-smi")
                .arg("--showmeminfo")
                .output()
            {
                if output.status.success() {
                    return (None, Some(GpuVendor::Amd), false, false);
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            // On macOS, unified memory is available
            let sys = System::new_all();
            let total_mem = sys.total_memory() / (1024 * 1024);
            return (Some(total_mem / 2), Some(GpuVendor::Apple), false, true);
        }

        (None, None, false, false)
    }

    /// Get display string
    pub fn display(&self) -> String {
        let mut parts = Vec::new();

        // RAM
        if self.ram_mb >= 1024 {
            parts.push(format!("{} GB RAM", self.ram_mb / 1024));
        } else {
            parts.push(format!("{} MB RAM", self.ram_mb));
        }

        // GPU
        if let Some(vendor) = &self.gpu_vendor {
            let gpu_name = match vendor {
                GpuVendor::Nvidia => "NVIDIA",
                GpuVendor::Apple => "Apple Silicon",
                GpuVendor::Amd => "AMD",
                GpuVendor::Unknown => "GPU",
            };

            if let Some(vram) = self.vram_mb {
                if vram >= 1024 {
                    parts.push(format!("{} {} GB VRAM", gpu_name, vram / 1024));
                } else {
                    parts.push(format!("{} {} MB VRAM", gpu_name, vram));
                }
            } else {
                parts.push(gpu_name.to_string());
            }
        }

        parts.join(" · ")
    }

    /// Collect runtime telemetry for TUI.
    pub fn sample_runtime_telemetry(&self) -> RuntimeTelemetry {
        let mut sys = System::new_all();
        sys.refresh_all();

        let cpu_usage_pct = sys.global_cpu_usage();
        let ram_total_mb = sys.total_memory() / (1024 * 1024);
        let ram_used_mb = sys.used_memory() / (1024 * 1024);

        let cpu_temp_c = detect_cpu_temperature();
        let gpu_metrics = match self.gpu_vendor {
            Some(GpuVendor::Nvidia) => detect_nvidia_gpu_metrics(),
            _ => None,
        };

        let (gpu_usage_pct, gpu_temp_c, vram_used_mb, vram_total_mb) = gpu_metrics
            .map(|m| {
                (
                    m.gpu_usage_pct,
                    m.gpu_temp_c,
                    m.vram_used_mb,
                    Some(m.vram_total_mb),
                )
            })
            .unwrap_or((None, None, None, self.vram_mb));

        RuntimeTelemetry {
            mode: if self.gpu_vendor.is_some() {
                ComputeMode::Gpu
            } else {
                ComputeMode::Cpu
            },
            gpu_vendor: self.gpu_vendor,
            cpu_usage_pct,
            ram_used_mb,
            ram_total_mb,
            cpu_temp_c,
            gpu_usage_pct,
            gpu_temp_c,
            vram_used_mb,
            vram_total_mb,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct NvidiaGpuMetrics {
    gpu_usage_pct: Option<f32>,
    gpu_temp_c: Option<f32>,
    vram_used_mb: Option<u64>,
    vram_total_mb: u64,
}

#[cfg(target_os = "linux")]
fn detect_cpu_temperature() -> Option<f32> {
    let thermal_base = std::path::Path::new("/sys/class/thermal");
    let entries = std::fs::read_dir(thermal_base).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path
            .file_name()
            .and_then(|v| v.to_str())
            .unwrap_or("")
            .starts_with("thermal_zone")
        {
            continue;
        }

        let Ok(temp_raw) = std::fs::read_to_string(path.join("temp")) else {
            continue;
        };
        if let Ok(millicelsius) = temp_raw.trim().parse::<f32>() {
            let celsius = millicelsius / 1000.0;
            if celsius.is_finite() && celsius > 0.0 {
                return Some(celsius);
            }
        }
    }

    None
}

#[cfg(not(target_os = "linux"))]
fn detect_cpu_temperature() -> Option<f32> {
    None
}

#[cfg(target_os = "linux")]
fn detect_nvidia_gpu_metrics() -> Option<NvidiaGpuMetrics> {
    let output = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total")
        .arg("--format=csv,noheader,nounits")
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let first_line = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()?
        .trim()
        .to_string();

    let mut fields = first_line.split(',').map(|v| v.trim());
    let gpu_usage_pct = fields.next().and_then(|v| v.parse::<f32>().ok());
    let gpu_temp_c = fields.next().and_then(|v| v.parse::<f32>().ok());
    let vram_used_mb = fields.next().and_then(|v| v.parse::<u64>().ok());
    let vram_total_mb = fields.next().and_then(|v| v.parse::<u64>().ok())?;

    Some(NvidiaGpuMetrics {
        gpu_usage_pct,
        gpu_temp_c,
        vram_used_mb,
        vram_total_mb,
    })
}

#[cfg(not(target_os = "linux"))]
fn detect_nvidia_gpu_metrics() -> Option<NvidiaGpuMetrics> {
    None
}

/// Recommend a quantization based on hardware
pub fn recommend(hw: &HardwareInfo, model: &ModelSpec) -> Option<&'static QuantizationVariant> {
    // Use VRAM if available, otherwise use RAM
    let memory_mb = hw.vram_mb.unwrap_or(hw.ram_mb);

    // Recommendation based on memory
    let recommended = if memory_mb >= 16_000 {
        // 16GB+ - Full quality
        "Q5_K_M"
    } else if memory_mb >= 10_000 {
        // 10-16GB - Good quality
        "Q4_K_M"
    } else if memory_mb >= 6_000 {
        // 6-10GB - Balanced
        "Q3_K_M"
    } else if memory_mb >= 4_000 {
        // 4-6GB - Lower quality
        "Q2_K"
    } else {
        // <4GB - Lowest currently supported in catalog
        "Q2_K"
    };

    model_quantizations(model)
        .iter()
        .find(|q| q.name == recommended)
        .or_else(|| model_quantizations(model).first())
}

/// Recommend inference/runtime tuning parameters for the selected quantization.
///
/// Policy:
/// - Budget-based: compute whether model + KV cache + overhead fit in VRAM.
/// - Full GPU offload whenever possible (dramatically faster than CPU).
/// - Flash attention enabled when CUDA/Metal available (saves KV cache VRAM).
/// - KV cache type chosen to maximize context that fits in remaining VRAM.
/// - n_batch tuned for GPU throughput (larger = faster prompt processing).
/// - Thread count optimized for physical cores.
pub fn recommend_inference_tuning(
    hw: &HardwareInfo,
    quant: &QuantizationVariant,
) -> InferenceTuning {
    const MB: u64 = 1024 * 1024;
    let quant_mb = (quant.size_bytes / MB).max(1);
    let has_accelerator = hw.has_cuda || hw.has_metal || hw.gpu_vendor.is_some();

    let n_threads = optimal_thread_count();

    // --- CPU-only path ---
    if !has_accelerator {
        let free_ram = hw.ram_mb.saturating_sub(quant_mb).saturating_sub(512);
        let (kv_k, kv_v, ctx_cap) = if free_ram >= 8_000 {
            ("q8_0", "q8_0", 16_384u32)
        } else if free_ram >= 4_000 {
            ("q4_0", "q4_0", 8_192)
        } else if free_ram >= 2_000 {
            ("q4_0", "q4_0", 4_096)
        } else {
            ("q4_0", "q4_0", 2_048)
        };
        return InferenceTuning {
            n_gpu_layers: 0,
            kv_cache_type_k: kv_k,
            kv_cache_type_v: kv_v,
            n_batch: 512,
            context_size_cap: ctx_cap,
            flash_attention: false,
            n_threads,
        };
    }

    // --- GPU path ---
    let is_metal = hw.has_metal || matches!(hw.gpu_vendor, Some(GpuVendor::Apple));

    let effective_vram_mb = hw.vram_mb.unwrap_or_else(|| {
        if is_metal {
            // Unified memory: ~60% usable for ML after OS/UI.
            hw.ram_mb.saturating_mul(60) / 100
        } else {
            // Fallback when VRAM unknown but GPU detected: conservative estimate.
            hw.ram_mb.saturating_mul(30) / 100
        }
    });

    // CUDA/Metal runtime overhead: ~200-400 MB depending on driver/framework.
    let runtime_overhead_mb: u64 = if is_metal { 256 } else { 350 };

    // Available VRAM after loading model weights entirely on GPU.
    let vram_after_model = effective_vram_mb
        .saturating_sub(quant_mb)
        .saturating_sub(runtime_overhead_mb);

    // --- Estimate KV cache VRAM per 1K context tokens ---
    // Rough formula for a 4B-class model (28-36 layers, d=2560):
    //   Per token per KV pair (F16): ~2 * n_layers * d_model * 2 bytes
    //   For 32 layers, d=2560: ~327,680 bytes/token ≈ 0.31 MB/token
    //   Q8_0 ≈ 50% of F16, Q4_0 ≈ 25% of F16
    // Per 1K tokens:
    //   F16: ~312 MB, Q8_0: ~156 MB, Q4_0: ~78 MB
    // With flash attention: KV cache is ~30% smaller in practice.
    const KV_PER_1K_Q8_MB: u64 = 156;
    const KV_PER_1K_Q4_MB: u64 = 78;

    // Determine if full offload is feasible.
    // Full offload = model fits in VRAM with room for at least minimal KV cache.
    let min_kv_for_2k_q4 = KV_PER_1K_Q4_MB * 2; // ~156 MB for 2K context
    let can_full_offload = vram_after_model >= min_kv_for_2k_q4;

    if !can_full_offload {
        // Model barely fits or doesn't fit — partial offload.
        let spare_total = effective_vram_mb.saturating_sub(runtime_overhead_mb);
        let offload_fraction = if spare_total > quant_mb {
            // Some VRAM left for partial layers
            let model_fraction = spare_total.saturating_sub(min_kv_for_2k_q4)
                .saturating_mul(100)
                / quant_mb.max(1);
            model_fraction.min(90) as i32 // cap at 90% to leave room for KV
        } else {
            0
        };

        // Convert fraction to approximate layer count (assume ~32 layers for 4B).
        let estimated_layers = (offload_fraction.max(0) * 32) / 100;

        let ctx_cap = estimate_context_cap(
            hw.ram_mb.saturating_sub(quant_mb / 2),
            KV_PER_1K_Q4_MB,
            false,
        );

        return InferenceTuning {
            n_gpu_layers: estimated_layers,
            kv_cache_type_k: "q4_0",
            kv_cache_type_v: "q4_0",
            n_batch: if estimated_layers > 0 { 1024 } else { 512 },
            context_size_cap: ctx_cap,
            flash_attention: estimated_layers > 0,
            n_threads,
        };
    }

    // Full offload is feasible. Now optimize KV cache type and context cap.
    let flash_attn = true; // Always enable when GPU available.
    let flash_savings = 70; // flash attn effective KV = ~70% of normal

    // Try Q8_0 KV first (better quality), then Q4_0 if needed.
    let kv_per_1k_q8_effective = KV_PER_1K_Q8_MB * flash_savings / 100;
    let kv_per_1k_q4_effective = KV_PER_1K_Q4_MB * flash_savings / 100;

    // Mixed Q8_K+Q4_V has effective per-1K ≈ average of q8 and q4.
    let kv_per_1k_mixed_effective = (kv_per_1k_q8_effective + kv_per_1k_q4_effective) / 2;

    let (kv_k, kv_v, ctx_cap) = if vram_after_model >= KV_PER_1K_Q8_MB * 16 {
        // Lots of room: Q8_0 KV, large context.
        let cap = estimate_context_cap(vram_after_model, kv_per_1k_q8_effective, true);
        ("q8_0", "q8_0", cap)
    } else if vram_after_model >= KV_PER_1K_Q8_MB * 4 {
        // Moderate room: mixed Q8_K/Q4_V for better quality with reasonable context.
        let cap = estimate_context_cap(vram_after_model, kv_per_1k_mixed_effective, true);
        ("q8_0", "q4_0", cap)
    } else {
        // Tight: Q4_0 KV to maximize context.
        let cap = estimate_context_cap(vram_after_model, kv_per_1k_q4_effective, true);
        ("q4_0", "q4_0", cap)
    };

    // GPU batch: larger = faster prompt ingestion. 2048 is good for modern GPUs.
    let n_batch = if effective_vram_mb >= 8_000 { 2048 } else { 1024 };

    InferenceTuning {
        n_gpu_layers: -1,
        kv_cache_type_k: kv_k,
        kv_cache_type_v: kv_v,
        n_batch,
        context_size_cap: ctx_cap,
        flash_attention: flash_attn,
        n_threads,
    }
}

/// Estimate the maximum context tokens that fit in available memory.
fn estimate_context_cap(available_mb: u64, kv_per_1k_mb: u64, is_gpu: bool) -> u32 {
    if kv_per_1k_mb == 0 {
        return 8_192;
    }
    // Leave 10% margin for fragmentation / allocator overhead.
    let usable = available_mb * 90 / 100;
    let max_1k_blocks = usable / kv_per_1k_mb.max(1);
    let raw = (max_1k_blocks * 1024) as u32;
    // Round down to nearest 1024 and clamp to sensible range.
    let rounded = (raw / 1024) * 1024;
    if is_gpu {
        rounded.clamp(2_048, 131_072)
    } else {
        rounded.clamp(2_048, 65_536)
    }
}

/// Determine optimal thread count (physical cores, not hyperthreads).
fn optimal_thread_count() -> u32 {
    // num_cpus::get_physical() returns physical core count when available.
    let physical = num_cpus::get_physical();
    let logical = num_cpus::get();

    // Use physical cores for inference (hyperthreads hurt llama.cpp perf).
    // If physical == logical, we're probably on a non-HT system or detection failed.
    let cores = if physical > 0 && physical <= logical {
        physical
    } else {
        // Fallback: use half of logical as a reasonable guess.
        (logical / 2).max(1)
    };

    // Cap at reasonable max to avoid contention on many-core systems.
    (cores as u32).min(16)
}

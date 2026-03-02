//! Hardware detection

use crate::catalog::ModelSpec;
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

    model
        .quantizations
        .iter()
        .find(|q| q.name == recommended)
        .or_else(|| model.quantizations.first())
}

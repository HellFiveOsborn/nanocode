//! Quantization variants

use serde::{Deserialize, Serialize};

/// Allowed variance between catalog metadata and on-disk GGUF size.
///
/// Model hosts occasionally update a quant file while keeping the same filename.
/// We keep cache validation tolerant enough to avoid false negatives.
pub const QUANT_SIZE_TOLERANCE_PCT: u64 = 20;

/// Quality tier for quantization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityTier {
    Maximum,    // F16
    Excellent,  // Q8_0, UD-Q8_K_XL
    VeryGood,   // Q6_K, Q5_K_M, Q5_K_S
    Good,       // Q4_K_M, Q4_K_S, IQ4_NL, IQ4_XS
    Acceptable, // Q3_K_M, Q3_K_S
    Low,        // Q2_K
    VeryLow,    // IQ2_*, IQ1_*
}

impl QualityTier {
    pub fn label(&self) -> &'static str {
        match self {
            QualityTier::Maximum => "Máxima",
            QualityTier::Excellent => "Excelente",
            QualityTier::VeryGood => "Muito boa",
            QualityTier::Good => "Boa",
            QualityTier::Acceptable => "Aceitável",
            QualityTier::Low => "Baixa",
            QualityTier::VeryLow => "Muito baixa",
        }
    }
}

/// Quantization variant
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct QuantizationVariant {
    pub name: &'static str,
    pub filename: &'static str,
    pub size_bytes: u64,
    pub quality: QualityTier,
    pub notes: Option<&'static str>,
}

impl QuantizationVariant {
    /// Get human-readable size
    pub fn size_human(&self) -> String {
        const KB: u64 = 1024;
        const MB: u64 = KB * 1024;
        const GB: u64 = MB * 1024;

        if self.size_bytes >= GB {
            format!("{:.2} GB", self.size_bytes as f64 / GB as f64)
        } else if self.size_bytes >= MB {
            format!("{:.2} MB", self.size_bytes as f64 / MB as f64)
        } else {
            format!("{} KB", self.size_bytes / KB)
        }
    }
}

/// Check whether an on-disk size is compatible with the expected catalog size.
pub fn is_compatible_quant_size(actual: u64, expected: u64) -> bool {
    if expected == 0 {
        return actual > 0;
    }

    let min_size = expected.saturating_mul(100 - QUANT_SIZE_TOLERANCE_PCT) / 100;
    let max_size = expected.saturating_mul(100 + QUANT_SIZE_TOLERANCE_PCT) / 100;
    actual >= min_size && actual <= max_size
}

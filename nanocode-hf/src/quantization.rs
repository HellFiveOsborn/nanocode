//! Quantization variants

use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// All available quantizations for Qwen3 4B
pub const QWEN3_4B_QUANTIZATIONS: &[QuantizationVariant] = &[
    // F16 - Maximum
    QuantizationVariant {
        name: "F16",
        filename: "Qwen3-4B-Thinking-2507-F16.gguf",
        size_bytes: 8_500_000_000,
        quality: QualityTier::Maximum,
        notes: Some("Full precision - best quality"),
    },
    // Q8 - Excellent
    QuantizationVariant {
        name: "Q8_0",
        filename: "Qwen3-4B-Thinking-2507-Q8_0.gguf",
        size_bytes: 4_500_000_000,
        quality: QualityTier::Excellent,
        notes: Some("Unsloth Q8"),
    },
    QuantizationVariant {
        name: "IQ8_XL",
        filename: "Qwen3-4B-Thinking-2507-IQ8_XL.gguf",
        size_bytes: 5_300_000_000,
        quality: QualityTier::Excellent,
        notes: Some("Improved Q8"),
    },
    // Q6 - Very Good
    QuantizationVariant {
        name: "Q6_K",
        filename: "Qwen3-4B-Thinking-2507-Q6_K.gguf",
        size_bytes: 3_400_000_000,
        quality: QualityTier::VeryGood,
        notes: Some("Unsloth Q6"),
    },
    QuantizationVariant {
        name: "IQ6_S",
        filename: "Qwen3-4B-Thinking-2507-IQ6_S.gguf",
        size_bytes: 3_100_000_000,
        quality: QualityTier::VeryGood,
        notes: Some("Improved Q6"),
    },
    // Q5 - Very Good
    QuantizationVariant {
        name: "Q5_K_M",
        filename: "Qwen3-4B-Thinking-2507-Q5_K_M.gguf",
        size_bytes: 2_900_000_000,
        quality: QualityTier::VeryGood,
        notes: Some("Unsloth Q5 Medium - Recommended"),
    },
    QuantizationVariant {
        name: "Q5_K_S",
        filename: "Qwen3-4B-Thinking-2507-Q5_K_S.gguf",
        size_bytes: 2_800_000_000,
        quality: QualityTier::VeryGood,
        notes: Some("Unsloth Q5 Small"),
    },
    QuantizationVariant {
        name: "IQ5_XL",
        filename: "Qwen3-4B-Thinking-2507-IQ5_XL.gguf",
        size_bytes: 2_750_000_000,
        quality: QualityTier::VeryGood,
        notes: Some("Improved Q5"),
    },
    // Q4 - Good
    QuantizationVariant {
        name: "Q4_K_M",
        filename: "Qwen3-4B-Thinking-2507-Q4_K_M.gguf",
        size_bytes: 2_500_000_000,
        quality: QualityTier::Good,
        notes: Some("Unsloth Q4 Medium"),
    },
    QuantizationVariant {
        name: "Q4_K_S",
        filename: "Qwen3-4B-Thinking-2507-Q4_K_S.gguf",
        size_bytes: 2_400_000_000,
        quality: QualityTier::Good,
        notes: Some("Unsloth Q4 Small"),
    },
    QuantizationVariant {
        name: "IQ4_NL",
        filename: "Qwen3-4B-Thinking-2507-IQ4_NL.gguf",
        size_bytes: 2_350_000_000,
        quality: QualityTier::Good,
        notes: Some("Improved Q4"),
    },
    QuantizationVariant {
        name: "IQ4_XS",
        filename: "Qwen3-4B-Thinking-2507-IQ4_XS.gguf",
        size_bytes: 2_270_000_000,
        quality: QualityTier::Good,
        notes: Some("Improved Q4 Extra Small"),
    },
    // Q3 - Acceptable
    QuantizationVariant {
        name: "Q3_K_M",
        filename: "Qwen3-4B-Thinking-2507-Q3_K_M.gguf",
        size_bytes: 2_100_000_000,
        quality: QualityTier::Acceptable,
        notes: Some("Unsloth Q3 Medium"),
    },
    QuantizationVariant {
        name: "IQ3_M",
        filename: "Qwen3-4B-Thinking-2507-IQ3_M.gguf",
        size_bytes: 2_000_000_000,
        quality: QualityTier::Acceptable,
        notes: Some("Improved Q3"),
    },
    // Q2 - Low
    QuantizationVariant {
        name: "Q2_K",
        filename: "Qwen3-4B-Thinking-2507-Q2_K.gguf",
        size_bytes: 1_670_000_000,
        quality: QualityTier::Low,
        notes: Some("Unsloth Q2 - May lose reasoning capability"),
    },
    // IQ2 - Very Low
    QuantizationVariant {
        name: "IQ2_M",
        filename: "Qwen3-4B-Thinking-2507-IQ2_M.gguf",
        size_bytes: 1_400_000_000,
        quality: QualityTier::VeryLow,
        notes: Some("Improved Q2"),
    },
    QuantizationVariant {
        name: "IQ2_S",
        filename: "Qwen3-4B-Thinking-2507-IQ2_S.gguf",
        size_bytes: 1_300_000_000,
        quality: QualityTier::VeryLow,
        notes: Some("Improved Q2 Small"),
    },
];

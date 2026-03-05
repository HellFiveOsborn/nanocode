//! Static catalog of supported models.

use crate::quantization::{QualityTier, QuantizationVariant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelCategory {
    Thinking,
    Instruct,
}

impl ModelCategory {
    pub fn label(&self) -> &'static str {
        match self {
            ModelCategory::Thinking => "Thinking",
            ModelCategory::Instruct => "Instruct",
        }
    }
}

/// Model specification used by setup/runtime selection.
#[derive(Debug, Clone, Copy)]
pub struct ModelSpec {
    pub id: &'static str,
    pub display_name: &'static str,
    pub hf_repo: &'static str,
    pub category: ModelCategory,
    pub supports_thinking: bool,
    pub supports_vision: bool,
    /// Maximum context supported by the model architecture
    pub max_context_size: u32,
    /// Recommended context for general usage
    pub recommended_context_general: u32,
    /// Recommended context for coding / harder reasoning tasks
    pub recommended_context_coding: u32,
    /// Static quantizations. Leave empty when variants are discovered dynamically.
    pub quantizations: &'static [QuantizationVariant],
}

/// All available quantizations for Qwen3 4B Thinking.
pub const QWEN3_4B_QUANTIZATIONS: &[QuantizationVariant] = &[
    // F16 - Maximum
    QuantizationVariant {
        name: "F16",
        filename: "Qwen3-4B-Thinking-2507-F16.gguf",
        size_bytes: 7_700_000_000,
        quality: QualityTier::Maximum,
        notes: Some("Full precision - best quality"),
    },
    // Q8 - Excellent
    QuantizationVariant {
        name: "Q8_0",
        filename: "Qwen3-4B-Thinking-2507-Q8_0.gguf",
        size_bytes: 4_430_000_000,
        quality: QualityTier::Excellent,
        notes: Some("Unsloth Q8"),
    },
    QuantizationVariant {
        name: "Q8_K_XL",
        filename: "Qwen3-4B-Thinking-2507-UD-Q8_K_XL.gguf",
        size_bytes: 4_630_000_000,
        quality: QualityTier::Excellent,
        notes: Some("Unsloth Dynamic 2.0 Q8"),
    },
    // Q6 - Very Good
    QuantizationVariant {
        name: "Q6_K",
        filename: "Qwen3-4B-Thinking-2507-Q6_K.gguf",
        size_bytes: 3_410_000_000,
        quality: QualityTier::VeryGood,
        notes: Some("Unsloth Q6"),
    },
    QuantizationVariant {
        name: "Q6_K_XL",
        filename: "Qwen3-4B-Thinking-2507-UD-Q6_K_XL.gguf",
        size_bytes: 3_580_000_000,
        quality: QualityTier::VeryGood,
        notes: Some("Unsloth Dynamic 2.0 Q6"),
    },
    // Q5 - Very Good
    QuantizationVariant {
        name: "Q5_K_M",
        filename: "Qwen3-4B-Thinking-2507-Q5_K_M.gguf",
        size_bytes: 2_910_000_000,
        quality: QualityTier::VeryGood,
        notes: Some("Unsloth Q5 Medium - Recommended"),
    },
    QuantizationVariant {
        name: "Q5_K_S",
        filename: "Qwen3-4B-Thinking-2507-Q5_K_S.gguf",
        size_bytes: 2_830_000_000,
        quality: QualityTier::VeryGood,
        notes: Some("Unsloth Q5 Small"),
    },
    QuantizationVariant {
        name: "Q5_K_XL",
        filename: "Qwen3-4B-Thinking-2507-UD-Q5_K_XL.gguf",
        size_bytes: 3_070_000_000,
        quality: QualityTier::VeryGood,
        notes: Some("Unsloth Dynamic 2.0 Q5"),
    },
    // Q4 - Good
    QuantizationVariant {
        name: "Q4_K_M",
        filename: "Qwen3-4B-Thinking-2507-Q4_K_M.gguf",
        size_bytes: 2_620_000_000,
        quality: QualityTier::Good,
        notes: Some("Unsloth Q4 Medium"),
    },
    QuantizationVariant {
        name: "Q4_K_S",
        filename: "Qwen3-4B-Thinking-2507-Q4_K_S.gguf",
        size_bytes: 2_510_000_000,
        quality: QualityTier::Good,
        notes: Some("Unsloth Q4 Small"),
    },
    QuantizationVariant {
        name: "Q4_1",
        filename: "Qwen3-4B-Thinking-2507-Q4_1.gguf",
        size_bytes: 2_510_000_000,
        quality: QualityTier::Good,
        notes: Some("Classic Q4_1"),
    },
    QuantizationVariant {
        name: "Q4_0",
        filename: "Qwen3-4B-Thinking-2507-Q4_0.gguf",
        size_bytes: 2_270_000_000,
        quality: QualityTier::Good,
        notes: Some("Classic Q4_0"),
    },
    QuantizationVariant {
        name: "IQ4_NL",
        filename: "Qwen3-4B-Thinking-2507-UD-IQ4_NL.gguf",
        size_bytes: 2_310_000_000,
        quality: QualityTier::Good,
        notes: Some("Unsloth Dynamic 2.0 IQ4"),
    },
    QuantizationVariant {
        name: "IQ4_XS",
        filename: "Qwen3-4B-Thinking-2507-UD-IQ4_XS.gguf",
        size_bytes: 2_350_000_000,
        quality: QualityTier::Good,
        notes: Some("Unsloth Dynamic 2.0 IQ4 XS"),
    },
    // Q3 - Acceptable
    QuantizationVariant {
        name: "Q3_K_L",
        filename: "Qwen3-4B-Thinking-2507-Q3_K_L.gguf",
        size_bytes: 2_630_000_000,
        quality: QualityTier::Acceptable,
        notes: Some("Unsloth Q3 Large"),
    },
    QuantizationVariant {
        name: "Q3_K_M",
        filename: "Qwen3-4B-Thinking-2507-Q3_K_M.gguf",
        size_bytes: 2_400_000_000,
        quality: QualityTier::Acceptable,
        notes: Some("Unsloth Q3 Medium"),
    },
    QuantizationVariant {
        name: "Q3_K_S",
        filename: "Qwen3-4B-Thinking-2507-Q3_K_S.gguf",
        size_bytes: 2_190_000_000,
        quality: QualityTier::Acceptable,
        notes: Some("Unsloth Q3 Small"),
    },
    // Q2 - Low
    QuantizationVariant {
        name: "Q2_K_L",
        filename: "Qwen3-4B-Thinking-2507-Q2_K_L.gguf",
        size_bytes: 2_340_000_000,
        quality: QualityTier::Low,
        notes: Some("Unsloth Q2 Large"),
    },
    QuantizationVariant {
        name: "Q2_K",
        filename: "Qwen3-4B-Thinking-2507-Q2_K.gguf",
        size_bytes: 1_669_500_032,
        quality: QualityTier::Low,
        notes: Some("Unsloth Q2 - May lose reasoning capability"),
    },
];

/// Qwen3 4B Thinking model family.
pub const QWEN3_4B_THINKING: ModelSpec = ModelSpec {
    id: "qwen3-4b-thinking",
    display_name: "Qwen3 4B Thinking",
    hf_repo: "unsloth/Qwen3-4B-Thinking-2507-GGUF",
    category: ModelCategory::Thinking,
    supports_thinking: true,
    supports_vision: false,
    max_context_size: 262_144,
    recommended_context_general: 32_768,
    recommended_context_coding: 81_920,
    quantizations: QWEN3_4B_QUANTIZATIONS,
};

/// Qwen3.5 4B model family.
pub const QWEN3_5_4B: ModelSpec = ModelSpec {
    id: "qwen3.5-4b",
    display_name: "Qwen3.5 4B",
    hf_repo: "unsloth/Qwen3.5-4B-GGUF",
    category: ModelCategory::Instruct,
    supports_thinking: true,
    supports_vision: true,
    max_context_size: 131_072,
    recommended_context_general: 32_768,
    recommended_context_coding: 131_072,
    quantizations: &[],
};

/// All models currently supported by Nano Code.
pub const MODELS: &[ModelSpec] = &[QWEN3_4B_THINKING, QWEN3_5_4B];

/// Default model used when config doesn't explicitly select one.
pub fn default_model() -> &'static ModelSpec {
    &MODELS[0]
}

/// List all supported models.
pub fn models() -> &'static [ModelSpec] {
    MODELS
}

/// Find model by stable id.
pub fn find_model(id: &str) -> Option<&'static ModelSpec> {
    MODELS.iter().find(|m| m.id.eq_ignore_ascii_case(id))
}

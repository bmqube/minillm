//! Compute precision selection.
//!
//! GPT-2 benchmarks in this repo were all taken at fp32 and stay there by
//! default, so existing numbers remain reproducible. Modern checkpoints ship
//! bf16 weights and are far too large to hold at fp32 — Qwen3-8B is ~16 GiB at
//! bf16 and ~32 GiB at fp32 — so loading those needs an explicit precision.

use candle_core::DType;

/// Weight / activation precision for a loaded model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    /// 32-bit float. The default: every benchmark in `benchmarks/` uses it, and
    /// it is the only precision the HuggingFace parity thresholds are set for.
    #[default]
    F32,
    /// 16-bit float (IEEE half). Narrower exponent range than bf16; prefer bf16
    /// for checkpoints that were trained in it.
    F16,
    /// bfloat16 — the storage dtype of essentially every modern checkpoint.
    BF16,
}

impl Precision {
    /// Parse `"f32"` / `"f16"` / `"bf16"` and common aliases (case-insensitive).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f32" | "fp32" | "float32" | "full" => Some(Self::F32),
            "f16" | "fp16" | "float16" | "half" => Some(Self::F16),
            "bf16" | "bfloat16" => Some(Self::BF16),
            _ => None,
        }
    }

    /// The candle dtype this maps to.
    pub fn dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::BF16 => DType::BF16,
        }
    }

    /// Bytes per stored element — the multiplier in KV-cache size accounting.
    pub fn size_in_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }

    /// Lower-case name used in CSV output and CLI arguments.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
        }
    }

    /// Best-effort classification of a candle dtype. Integer dtypes have no
    /// `Precision` and return `None`.
    pub fn from_dtype(dtype: DType) -> Option<Self> {
        match dtype {
            DType::F32 => Some(Self::F32),
            DType::F16 => Some(Self::F16),
            DType::BF16 => Some(Self::BF16),
            _ => None,
        }
    }
}

impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_aliases() {
        assert_eq!(Precision::parse("fp32"), Some(Precision::F32));
        assert_eq!(Precision::parse("F32"), Some(Precision::F32));
        assert_eq!(Precision::parse(" bf16 "), Some(Precision::BF16));
        assert_eq!(Precision::parse("half"), Some(Precision::F16));
        assert_eq!(Precision::parse("int8"), None);
    }

    #[test]
    fn byte_widths_are_right() {
        assert_eq!(Precision::F32.size_in_bytes(), 4);
        assert_eq!(Precision::F16.size_in_bytes(), 2);
        assert_eq!(Precision::BF16.size_in_bytes(), 2);
    }

    #[test]
    fn dtype_round_trips() {
        for p in [Precision::F32, Precision::F16, Precision::BF16] {
            assert_eq!(Precision::from_dtype(p.dtype()), Some(p));
        }
        assert_eq!(Precision::from_dtype(DType::U8), None);
    }

    #[test]
    fn default_is_f32_so_existing_benchmarks_reproduce() {
        assert_eq!(Precision::default(), Precision::F32);
    }
}

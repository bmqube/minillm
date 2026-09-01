//! Compute precision selection.
//!
//! GPT-2 benchmarks in this repo were all taken at fp32 and stay there by
//! default, so existing numbers remain reproducible. Modern checkpoints ship
//! bf16 weights and are far too large to hold at fp32 — Qwen3-8B is ~16 GiB at
//! bf16 and ~32 GiB at fp32 — so loading those needs an explicit precision.

use candle_core::{DType, Device};

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

    /// Whether candle can run a matmul at this precision on `device`.
    ///
    /// candle's CPU gemm has no bf16 kernel — a bf16 model loads fine and then
    /// fails on the first attention matmul with "unsupported dtype BF16 for op
    /// matmul", after the weights are already resident. CUDA does bf16 through
    /// cuBLAS, so the same checkpoint is fine on a GPU.
    pub fn is_supported_on(self, device: &Device) -> bool {
        match self {
            Self::F32 | Self::F16 => true,
            Self::BF16 => !device.is_cpu(),
        }
    }

    /// This precision if `device` supports it, otherwise the widest one that
    /// always works.
    ///
    /// Falls back to fp32 rather than fp16: fp16 has bf16's mantissa but a much
    /// narrower exponent range, so silently substituting it for a bf16-trained
    /// checkpoint risks overflowing activations. fp32 is the safe superset, and
    /// it is what every existing benchmark in this repo already uses.
    pub fn resolve_for(self, device: &Device) -> Self {
        if self.is_supported_on(device) {
            self
        } else {
            Self::F32
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

    #[test]
    fn bf16_is_not_runnable_on_cpu() {
        // candle has no CPU bf16 gemm; this is what stops a bf16 checkpoint from
        // loading fine and then dying inside the first attention matmul.
        let cpu = Device::Cpu;
        assert!(!Precision::BF16.is_supported_on(&cpu));
        assert!(Precision::F32.is_supported_on(&cpu));
        assert!(Precision::F16.is_supported_on(&cpu));
    }

    #[test]
    fn cpu_fallback_is_f32_not_f16() {
        let cpu = Device::Cpu;
        assert_eq!(Precision::BF16.resolve_for(&cpu), Precision::F32);
        // Supported precisions are left alone.
        assert_eq!(Precision::F16.resolve_for(&cpu), Precision::F16);
        assert_eq!(Precision::F32.resolve_for(&cpu), Precision::F32);
    }
}

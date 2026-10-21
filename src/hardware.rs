//! What the machine running this binary can do — and what the binary was built
//! to use, which is not the same thing.
//!
//! Two sets of CPU features matter, and they can disagree:
//!
//! - **runtime**: what the CPU actually supports, detected when the program runs.
//! - **compiled**: what this binary was built to emit. candle gates its AVX2 /
//!   F16C kernels (bf16 and f16 dot products, the quantized matmuls) on
//!   `cfg!(target_feature = ...)`, so a binary built for baseline x86-64 never
//!   uses them, however capable the CPU is. The `gemm` crate behind fp32 / f16
//!   matmul picks its SIMD at runtime and is unaffected either way.
//!
//! [`Hardware::warnings`] reports the gap, because nothing else will: a binary
//! built without AVX2 on an AVX2 machine runs correctly, just slower on every
//! path that is not plain fp32.
//!
//! Which dtypes a device can run is probed rather than looked up: a tiny matmul +
//! softmax at that dtype either works or it doesn't. That catches the cases a
//! static table gets wrong — a pre-Ampere GPU without bf16 kernels, or a future
//! candle that adds a CPU bf16 gemm.

use candle_core::{DType, Device, Tensor};

/// CPU SIMD extensions relevant to inference. All `false` off x86-64.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SimdFeatures {
    pub avx2: bool,
    pub fma: bool,
    /// Hardware f16 <-> f32 conversion.
    pub f16c: bool,
    pub avx512f: bool,
    /// Native bf16 dot products (Zen 4+, Cooper Lake+). Nothing in candle uses
    /// it yet; reported so a benchmark row says what the CPU could have done.
    pub avx512bf16: bool,
}

impl SimdFeatures {
    /// What this binary was compiled to use.
    pub fn compiled() -> Self {
        Self {
            avx2: cfg!(target_feature = "avx2"),
            fma: cfg!(target_feature = "fma"),
            f16c: cfg!(target_feature = "f16c"),
            avx512f: cfg!(target_feature = "avx512f"),
            avx512bf16: cfg!(target_feature = "avx512bf16"),
        }
    }

    /// What the CPU running this binary supports.
    pub fn runtime() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx2: std::arch::is_x86_feature_detected!("avx2"),
                fma: std::arch::is_x86_feature_detected!("fma"),
                f16c: std::arch::is_x86_feature_detected!("f16c"),
                avx512f: std::arch::is_x86_feature_detected!("avx512f"),
                avx512bf16: std::arch::is_x86_feature_detected!("avx512bf16"),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self::default()
        }
    }

    /// Compact label for logs and CSV cells, e.g. `"avx2+fma+f16c"`, or
    /// `"baseline"` when none are present.
    pub fn label(&self) -> String {
        let names: Vec<&str> = [
            (self.avx2, "avx2"),
            (self.fma, "fma"),
            (self.f16c, "f16c"),
            (self.avx512f, "avx512f"),
            (self.avx512bf16, "avx512bf16"),
        ]
        .iter()
        .filter(|(on, _)| *on)
        .map(|(_, name)| *name)
        .collect();
        if names.is_empty() {
            "baseline".to_string()
        } else {
            names.join("+")
        }
    }
}

/// A snapshot of the device a model will run on and the host CPU.
#[derive(Debug, Clone)]
pub struct Hardware {
    pub device: Device,
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub simd_runtime: SimdFeatures,
    pub simd_compiled: SimdFeatures,
    /// Whether `device` ran the bf16 probe.
    pub bf16: bool,
    /// Whether `device` ran the f16 probe.
    pub f16: bool,
}

impl Hardware {
    /// Inspect the host CPU and probe `device` for half-precision support.
    pub fn probe(device: &Device) -> Self {
        Self {
            device: device.clone(),
            physical_cores: num_cpus::get_physical().max(1),
            logical_cores: num_cpus::get().max(1),
            simd_runtime: SimdFeatures::runtime(),
            simd_compiled: SimdFeatures::compiled(),
            bf16: supports(device, DType::BF16),
            f16: supports(device, DType::F16),
        }
    }

    /// Ways this binary is leaving the hardware's performance unused.
    pub fn warnings(&self) -> Vec<String> {
        let mut out = Vec::new();
        let (rt, built) = (self.simd_runtime, self.simd_compiled);
        if (rt.avx2 && !built.avx2) || (rt.f16c && !built.f16c) {
            out.push(format!(
                "this CPU supports {} but the binary was built for {}; candle's \
                 f16/bf16/quantized CPU kernels are compiled out. Build without \
                 RUSTFLAGS set so .cargo/config.toml applies -C target-cpu=native",
                rt.label(),
                built.label()
            ));
        }
        out
    }
}

/// Whether `device` can run a matmul and softmax at `dtype`.
///
/// Runs a 4x4 problem and discards the result; fp32 is always reported as
/// supported without probing.
pub fn supports(device: &Device, dtype: DType) -> bool {
    if dtype == DType::F32 {
        return true;
    }
    let probe = || -> candle_core::Result<f32> {
        let x = Tensor::ones((4, 4), dtype, device)?;
        let y = candle_nn::ops::softmax_last_dim(&x.matmul(&x)?)?;
        y.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()
    };
    // Each row of the softmax sums to 1.
    matches!(probe(), Ok(s) if (s - 4.0).abs() < 1e-2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_probe_matches_candle_cpu_support() {
        // candle 0.11 has an f16 CPU gemm but no bf16 one.
        assert!(supports(&Device::Cpu, DType::F32));
        assert!(supports(&Device::Cpu, DType::F16));
        assert!(!supports(&Device::Cpu, DType::BF16));
    }

    #[test]
    fn labels_are_compact() {
        assert_eq!(SimdFeatures::default().label(), "baseline");
        let f = SimdFeatures {
            avx2: true,
            fma: true,
            ..Default::default()
        };
        assert_eq!(f.label(), "avx2+fma");
    }

    #[test]
    fn a_binary_never_compiles_in_more_than_the_cpu_has() {
        // If it did, it would have died with SIGILL before reaching this test.
        let (rt, built) = (SimdFeatures::runtime(), SimdFeatures::compiled());
        if cfg!(target_arch = "x86_64") {
            assert!(!built.avx2 || rt.avx2);
            assert!(!built.f16c || rt.f16c);
        }
    }

    #[test]
    fn probe_reports_sane_core_counts() {
        let hw = Hardware::probe(&Device::Cpu);
        assert!(hw.physical_cores >= 1);
        assert!(hw.logical_cores >= hw.physical_cores);
    }
}

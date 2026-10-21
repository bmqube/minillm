//! Turning [`Hardware`] facts into run-time decisions.
//!
//! An [`ExecutionPlan`] is built once, at the top of `main`, and records every
//! hardware-dependent choice the engine makes: which device, how many CPU
//! threads, and how greedy tokens are picked. Binaries print
//! [`summary`](ExecutionPlan::summary) at startup and write the same facts into
//! their CSV rows, so a benchmark number always says what produced it.
//!
//! What the plan cannot choose is what the binary was *compiled* with: the
//! `cuda` feature, and the CPU instruction set (`-C target-cpu`). It picks the
//! best option among those, and [`Hardware::warnings`] says when the build is
//! leaving the hardware unused.
//!
//! Weight precision is resolved per checkpoint by [`crate::loader`], using
//! [`Precision::is_supported_on`](crate::dtype::Precision::is_supported_on),
//! which consults the same probe as [`Hardware`].

use candle_core::Device;

use crate::device;
use crate::hardware::Hardware;

/// Settings the caller wants pinned rather than chosen automatically.
#[derive(Debug, Clone, Copy, Default)]
pub struct PlanOverrides {
    /// CPU worker threads for candle's matmuls and element-wise ops.
    pub threads: Option<usize>,
}

/// The hardware-dependent choices for one process.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub hardware: Hardware,
    /// CPU worker threads. Applies to CPU compute; a GPU run still uses the CPU
    /// for tokenization and sampling, which this barely touches.
    pub threads: usize,
    /// Whether greedy decoding takes its argmax on the device, copying back one
    /// id instead of the whole vocabulary row every step. True off-CPU; this
    /// records the choice [`crate::generation::sample_with`] makes from the
    /// logits' device.
    pub device_argmax: bool,
}

impl ExecutionPlan {
    /// Detect everything; override nothing.
    pub fn detect() -> Self {
        Self::with(PlanOverrides::default())
    }

    /// Detect, then apply `overrides`.
    ///
    /// Call this **before** anything else touches candle or rayon: the thread
    /// count is handed to them through `RAYON_NUM_THREADS` /
    /// `CANDLE_NUM_THREADS`, which each reads once when its pool is first
    /// built. An explicitly set `RAYON_NUM_THREADS` is respected when no
    /// override is given.
    pub fn with(overrides: PlanOverrides) -> Self {
        let from_env = std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0);
        // Physical cores, not logical: decode is memory-bandwidth-bound, and SMT
        // siblings compete for the same bandwidth instead of adding to it.
        let threads = overrides
            .threads
            .or(from_env)
            .unwrap_or_else(crate::hardware::usable_physical_cores)
            .max(1);
        // Single-threaded here: no other thread can be reading the environment.
        std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
        std::env::set_var("CANDLE_NUM_THREADS", threads.to_string());

        let device = device::best();
        let hardware = Hardware::probe(&device);
        Self {
            device_argmax: !device.is_cpu(),
            hardware,
            threads,
        }
    }

    /// The device to load models onto.
    pub fn device(&self) -> &Device {
        &self.hardware.device
    }

    /// Short device label for CSV cells: `"cpu"` or `"cuda"`.
    pub fn device_label(&self) -> &'static str {
        match self.device() {
            Device::Cpu => "cpu",
            Device::Cuda(_) => "cuda",
            Device::Metal(_) => "metal",
        }
    }

    /// One line describing the plan, for startup logs.
    pub fn summary(&self) -> String {
        let hw = &self.hardware;
        let half = match (hw.bf16, hw.f16) {
            (true, true) => "bf16+f16",
            (false, true) => "f16",
            (true, false) => "bf16",
            (false, false) => "none",
        };
        format!(
            "device={} threads={} ({}C/{}T) cpu_simd=built:{} available:{} half={} argmax={}",
            self.device_label(),
            self.threads,
            hw.physical_cores,
            hw.logical_cores,
            hw.simd_compiled.label(),
            hw.simd_runtime.label(),
            half,
            if self.device_argmax { "device" } else { "host" },
        )
    }

    /// Print the summary and any warnings to stderr.
    pub fn log(&self) {
        eprintln!("plan   : {}", self.summary());
        for w in self.hardware.warnings() {
            eprintln!("warning: {w}");
        }
    }
}

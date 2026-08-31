use candle_core::Device;

/// Pick the best available compute device.
///
/// Returns CUDA GPU 0 when this crate was built with the `cuda` feature and a
/// usable device is present; otherwise falls back to the CPU. This never fails:
/// a machine with no GPU (or a CPU-only build) simply gets [`Device::Cpu`].
pub fn best() -> Device {
    Device::cuda_if_available(0).unwrap_or(Device::Cpu)
}

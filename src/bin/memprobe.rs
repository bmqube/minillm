//! Peak resident-set-size probe for the KV cache.
//!
//! ```text
//! cargo run --release --bin memprobe -- <off|on> [PREFILL] [DECODE] [MODEL] [KVQUANT]
//! ```
//!
//! Defaults: `on 512 128 benchmarks/gpt2 none`. Loads the model, runs a greedy
//! decode loop in the chosen path (`off` = full recompute, `on` = KV cache with
//! `KVQUANT` storage `none`|`int8`), then prints this process's peak RSS. Run it
//! once per mode with the **same** `PREFILL` / `DECODE` and diff the two peaks to
//! get the cache's real memory cost — which includes the transient dequant to
//! fp32 each step, so it runs above the steady-state analytic figure.
//!
//! One process = one mode on purpose: `bench` runs both paths, so its peak would
//! just be the larger of the two.

use candle_core::{IndexOp, Tensor};
use minillm::kv_cache::{KvCache, KvQuant};
use minillm::{device, loader};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenvy::dotenv().ok();

    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "on".into());
    let prefill: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(512);
    let decode: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(128);
    let model_id = args.next().unwrap_or_else(|| "benchmarks/gpt2".into());
    let kvq_arg = args.next().unwrap_or_else(|| "none".into());
    let kv_quant = KvQuant::parse(&kvq_arg)
        .unwrap_or_else(|| panic!("KVQUANT expects none|int8|int4, got {kvq_arg:?}"));

    let dev = device::best();
    let (model, _tok) = loader::load(&model_id, &dev)?;
    let ids: Vec<u32> = (0..prefill as u32).collect();

    match mode.as_str() {
        "off" => {
            let mut seq = ids.clone();
            for _ in 0..decode {
                let input = Tensor::from_vec(seq.clone(), (1, seq.len()), &dev)?;
                let row = model
                    .forward(&input)?
                    .i((0, seq.len() - 1))?
                    .to_vec1::<f32>()?;
                seq.push(argmax(&row));
            }
        }
        "on" => {
            let mut cache = KvCache::with_quant(model.config().n_layer, kv_quant);
            let input = Tensor::from_vec(ids.clone(), (1, ids.len()), &dev)?;
            let prime = model.forward_with_cache(&input, &mut cache)?;
            let mut next = argmax(&prime.i((0, ids.len() - 1))?.to_vec1::<f32>()?);
            for _ in 0..decode {
                let step = Tensor::from_vec(vec![next], (1, 1), &dev)?;
                let row = model
                    .forward_with_cache(&step, &mut cache)?
                    .i((0, 0))?
                    .to_vec1::<f32>()?;
                next = argmax(&row);
            }
        }
        other => return Err(format!("mode must be 'off' or 'on', got {other:?}").into()),
    }

    let peak = peak_rss_bytes();
    let seq_end = prefill + decode;
    let cfg = model.config();
    let head_dim = cfg.n_embd / cfg.n_head;
    let bpt = if mode == "on" {
        kv_quant.bytes_per_token(cfg.n_layer, cfg.n_head, head_dim)
    } else {
        cfg.kv_cache_bytes_per_token(4)
    };
    let analytic_kv = bpt * seq_end;
    let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
    println!(
        "mode={mode} kv_quant={kv_quant:?} prefill={prefill} decode={decode} seq_end={seq_end} \
         peak_rss_bytes={peak} peak_rss_mib={:.1} analytic_kv_mib={:.1}",
        mib(peak),
        mib(analytic_kv as u64),
    );
    Ok(())
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// Peak working set of the current process, in bytes (0 if unavailable).
#[cfg(windows)]
fn peak_rss_bytes() -> u64 {
    #[repr(C)]
    struct Pmc {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }
    // K32-prefixed PSAPI entry points live in kernel32 (linked by default).
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(process: isize, counters: *mut Pmc, cb: u32) -> i32;
    }
    let mut pmc: Pmc = unsafe { std::mem::zeroed() };
    pmc.cb = std::mem::size_of::<Pmc>() as u32;
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb) };
    if ok == 0 {
        0
    } else {
        pmc.peak_working_set_size as u64
    }
}

/// `VmHWM` from `/proc/self/status` is the peak RSS, reported in kB.
#[cfg(unix)]
fn peak_rss_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<u64>().ok())
        })
        .map(|kb| kb * 1024)
        .unwrap_or(0)
}

#[cfg(not(any(windows, unix)))]
fn peak_rss_bytes() -> u64 {
    0
}

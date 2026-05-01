"""
patchhar_v2_profiler.py
========================
Comprehensive inference profiling & interpretability analysis for PatchHAR v2
(Capture-24 / patchhar_v2.py edition).

KEY DIFFERENCES vs ADL profiler
────────────────────────────────
  • forward(patches_list, times)  — takes a (B, 5) time-of-day vector
  • 100 Hz signal, 3000-sample windows, 120 patches (primary PL=25)
  • C3 Circadian Bias is ON  → attention rollout accounts for positional bias
  • Data source: P*.npz files with keys X / y / t
  • Split: train_pids / val_pids / test_pids from patchhar_v2.py

METRICS REPORTED
────────────────
LATENCY & THROUGHPUT
  • Single-sample latency mean ± std, p50/p95/p99  — CPU & GPU
  • Batch throughput (samples/sec) at B = 1,4,8,16,32,64
  • Cold-start (first-call) latency
  • Warm-up curve (100 calls)

MODEL COMPLEXITY
  • Total / trainable / buffer parameters
  • Disk size  FP32 & FP16  (MB)
  • FLOPs via thop (optional)
  • Per-module parameter count

MEMORY  (GPU only)
  • Peak inference memory  (B=32)
  • Peak training-step memory (B=32)

INTERPRETABILITY
  • Attention Rollout  (Abnar & Zuidema 2020)  — (N, 120) patch importance
  • Gradient-weighted Attention  (Grad-CAM on patch tokens)
  • [C3] Circadian bias magnitude per patch  (mean |bias| over dataset)
  • [C2] Skip-aggregation layer-weight distribution
  • [C1] Dual-domain gate values  (time vs freq blend per output dim)
  • Expert routing entropy for MoE1 and MoE2

CALIBRATION
  • Expected Calibration Error  (ECE)
  • Maximum Calibration Error   (MCE)
  • Reliability diagram bin data

STATISTICAL
  • McNemar test  (model vs majority-class baseline)
  • Per-class F1, precision, recall, support

OUTPUT FILES
────────────
  paper_metrics_c24.json        all numeric results (human+machine readable)
  attn_rollout_c24.npy          (N, N_patches=120)
  rollout_labels_c24.npy        (N,)
  grad_attn_c24.npy             (N, N_patches=120)
  circadian_bias_c24.npy        (N, N_patches=120, D) — mean |bias|
  calibration_c24.npy           (3, n_bins)

USAGE
─────
  # Full analysis with a saved checkpoint:
  python patchhar_v2_profiler.py --checkpoint /path/to/weights.pth

  # Latency & complexity only  (no data / checkpoint needed):
  python patchhar_v2_profiler.py --benchmark_only

  # Limit attention analysis to 200 samples  (faster):
  python patchhar_v2_profiler.py --checkpoint weights.pth --max_samples 200

REQUIREMENTS
─────────────
  pip install thop scipy scikit-learn  (thop is optional — FLOPs only)
"""

from __future__ import annotations
import argparse, gc, json, math, os, random, sys, time, tempfile, warnings
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

# ─── import everything from the main script ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from patchhar_v2 import (
    PatchHARv2, cfg, CC,
    WindowDataset, _collate,
    train_pids, val_pids, test_pids,
    CLASSES, class_to_idx, idx_to_class, NUM_CLASSES,
    kappa, mcc,
    time_features,
    device, GPU,
)

print(f"Device : {device}")
if GPU:
    print(f"GPU    : {torch.cuda.get_device_name(0)}")


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _sync():
    if GPU:
        torch.cuda.synchronize()


def _dummy_patches(B: int = 1) -> List[torch.Tensor]:
    """Synthetic patches + time vector for benchmarking."""
    out = []
    for pl in cfg.PATCH_LENS_MULTI:
        n_p = cfg.WINDOW_SIZE // pl
        out.append(torch.randn(B, cfg.CHANNELS, n_p, pl, device=device))
    return out


def _dummy_times(B: int = 1) -> torch.Tensor:
    return torch.rand(B, 5, device=device)


def _make_loader(pid_list: List[str], batch_size: int = 64) -> DataLoader:
    ds = WindowDataset(pid_list, cfg.PROC_DIR, class_to_idx, is_train=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=GPU, collate_fn=_collate)


# ════════════════════════════════════════════════════════════════════════════
# 1.  MODEL COMPLEXITY
# ════════════════════════════════════════════════════════════════════════════

def report_complexity(model: PatchHARv2) -> Dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers   = sum(p.numel() for p in model.buffers())

    # ── Per-module breakdown ─────────────────────────────────────────────
    module_params = {}
    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters())
        if n > 0:
            module_params[name] = n

    # ── Disk size ─────────────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        fp32_path = f.name
    torch.save(model.state_dict(), fp32_path)
    size_fp32_mb = round(os.path.getsize(fp32_path) / 1e6, 2)

    fp16_sd = {k: v.half() if v.is_floating_point() else v
               for k, v in model.state_dict().items()}
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        fp16_path = f.name
    torch.save(fp16_sd, fp16_path)
    size_fp16_mb = round(os.path.getsize(fp16_path) / 1e6, 2)
    os.unlink(fp32_path); os.unlink(fp16_path)

    # ── FLOPs (thop — optional) ──────────────────────────────────────────
    flops_g = None
    try:
        from thop import profile as thop_profile
        dummy_p = _dummy_patches(1)
        dummy_t = _dummy_times(1)
        with torch.no_grad():
            macs, _ = thop_profile(model, inputs=(dummy_p, dummy_t),
                                   verbose=False)
        flops_g = round(2 * macs / 1e9, 4)
    except Exception as ex:
        print(f"  [FLOPs] thop not available or failed: {ex}")

    result = dict(
        total_params   = total,
        trainable_params = trainable,
        buffer_params  = buffers,
        size_fp32_mb   = size_fp32_mb,
        size_fp16_mb   = size_fp16_mb,
        flops_gflops   = flops_g,
        module_params  = module_params,
        d_model        = cfg.D_MODEL,
        n_heads        = cfg.N_HEADS,
        n_layers       = cfg.N_LAYERS,
        n_experts      = cfg.N_EXPERTS,
        n_patches      = cfg.N_PATCHES,
        patch_len      = cfg.PATCH_LEN,
        window_size    = cfg.WINDOW_SIZE,
        signal_rate    = cfg.SIGNAL_RATE,
        channels       = cfg.CHANNELS,
        num_classes    = NUM_CLASSES,
    )

    print("\n── MODEL COMPLEXITY ─────────────────────────────────────────────")
    for k, v in result.items():
        if k != "module_params":
            print(f"  {k:<28}: {v}")
    print("  Per-module parameters:")
    for k, v in module_params.items():
        print(f"    {k:<26}: {v:,}")

    return result


# ════════════════════════════════════════════════════════════════════════════
# 2.  LATENCY & THROUGHPUT
# ════════════════════════════════════════════════════════════════════════════

def benchmark_latency(model: PatchHARv2,
                      n_warmup: int = 50,
                      n_runs:   int = 500,
                      batch_sizes: Optional[List[int]] = None) -> Dict:
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32, 64]

    model.eval()
    results: Dict = {}

    # ── Per-device single-sample latency ─────────────────────────────────
    for dev_str in (["cuda"] if GPU else []) + ["cpu"]:
        dev = torch.device(dev_str)
        m   = model.to(dev); m.eval()
        dp  = [p.to(dev) for p in _dummy_patches(1)]
        dt  = _dummy_times(1).to(dev)

        with torch.no_grad():
            for _ in range(n_warmup):
                m(dp, dt)
        if dev_str == "cuda":
            torch.cuda.synchronize()

        latencies = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                m(dp, dt)
                if dev_str == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1e3)

        lat = np.array(latencies)
        results[f"latency_single_{dev_str}_ms"] = dict(
            mean = round(float(lat.mean()), 4),
            std  = round(float(lat.std()),  4),
            p50  = round(float(np.percentile(lat, 50)), 4),
            p95  = round(float(np.percentile(lat, 95)), 4),
            p99  = round(float(np.percentile(lat, 99)), 4),
            min  = round(float(lat.min()), 4),
            max  = round(float(lat.max()), 4),
        )
        print(f"\n  Latency B=1 ({dev_str.upper()}):  "
              f"mean={lat.mean():.3f} ms  "
              f"p95={np.percentile(lat,95):.3f} ms  "
              f"p99={np.percentile(lat,99):.3f} ms")

    model.to(device)

    # ── Batch throughput ─────────────────────────────────────────────────
    throughput: Dict = {}
    with torch.no_grad():
        for B in batch_sizes:
            dp = [p.to(device) for p in _dummy_patches(B)]
            dt = _dummy_times(B).to(device)
            for _ in range(20):
                model(dp, dt)
            _sync()
            N_rep = max(1, 200 // B)
            t0    = time.perf_counter()
            for _ in range(N_rep):
                model(dp, dt)
            _sync()
            elapsed = time.perf_counter() - t0
            sps     = round(B * N_rep / elapsed, 1)
            throughput[f"B{B}"] = dict(
                samples_per_sec = sps,
                ms_per_sample   = round(elapsed / (B * N_rep) * 1e3, 4))
            print(f"  Throughput B={B:3d}: {sps:9.1f} samp/s  "
                  f"({throughput[f'B{B}']['ms_per_sample']:.3f} ms/sample)")

    results["throughput"] = throughput

    # ── Cold-start (no warm-up) ───────────────────────────────────────────
    m_cold = PatchHARv2().to(device); m_cold.eval()
    dp     = [p.to(device) for p in _dummy_patches(1)]
    dt     = _dummy_times(1)
    _sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        m_cold(dp, dt)
    _sync()
    results["cold_start_ms"] = round((time.perf_counter() - t0) * 1e3, 4)
    print(f"  Cold start: {results['cold_start_ms']:.2f} ms")

    # ── Warm-up curve ────────────────────────────────────────────────────
    m_wu  = PatchHARv2().to(device); m_wu.eval()
    dp    = [p.to(device) for p in _dummy_patches(1)]
    dt    = _dummy_times(1)
    curve = []
    with torch.no_grad():
        for _ in range(100):
            t0 = time.perf_counter()
            m_wu(dp, dt)
            _sync()
            curve.append(round((time.perf_counter() - t0) * 1e3, 4))
    results["warmup_curve_ms"] = curve

    return results


# ════════════════════════════════════════════════════════════════════════════
# 3.  MEMORY
# ════════════════════════════════════════════════════════════════════════════

def report_memory(model: PatchHARv2) -> Dict:
    results: Dict = {}
    if not GPU:
        print("  Memory profiling skipped (CPU-only run)")
        results["note"] = "CPU-only run — GPU memory not available"
        return results

    # inference
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    model.to(device).eval()
    dp = [p.to(device) for p in _dummy_patches(32)]
    dt = _dummy_times(32)
    with torch.no_grad():
        model(dp, dt)
    torch.cuda.synchronize()
    results["peak_inference_b32_mb"] = round(
        torch.cuda.max_memory_allocated() / 1e6, 2)

    # training step
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    m_tr = PatchHARv2().to(device); m_tr.train()
    opt  = torch.optim.AdamW(m_tr.parameters(), lr=1e-3)
    dp2  = [p.to(device) for p in _dummy_patches(32)]
    dt2  = _dummy_times(32)
    lbl  = torch.randint(0, NUM_CLASSES, (32,), device=device)
    logits, _ = m_tr(dp2, dt2)
    loss = F.cross_entropy(logits, lbl)
    loss.backward(); opt.step()
    torch.cuda.synchronize()
    results["peak_train_step_b32_mb"] = round(
        torch.cuda.max_memory_allocated() / 1e6, 2)
    del m_tr

    print(f"\n── MEMORY ───────────────────────────────────────────────────────")
    print(f"  Peak inference   (B=32): {results['peak_inference_b32_mb']:.1f} MB")
    print(f"  Peak train step  (B=32): {results['peak_train_step_b32_mb']:.1f} MB")
    return results


# ════════════════════════════════════════════════════════════════════════════
# 4.  ATTENTION ROLLOUT
#     (Abnar & Zuidema 2020) — one GatedAttention layer in PatchHAR v2
#     We hook it to capture the (B, H, N, N) weights during forward.
# ════════════════════════════════════════════════════════════════════════════

def compute_attention_rollout(
        model:       PatchHARv2,
        loader:      DataLoader,
        max_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    rollout : (N, N_PATCHES)  float32  — mean incoming attention per patch
    labels  : (N,)            int32
    """
    model.eval()
    captured_attn: List[torch.Tensor] = []

    # ── Hook into GatedAttention to capture raw softmax weights ──────────
    def _attn_hook(module, inp, out):
        # We need to recompute attention weights — hook on the module
        # before the output gate is applied. We grab them from a forward
        # pre-hook by monkey-patching; instead, subclass via a wrapper here.
        pass

    # Patch forward to store last attn
    original_forward = model.attn.forward

    def _patched_forward(x, freqs):
        h = model.attn.norm(x)
        B, N, D = h.shape
        qkv = (model.attn.qkv(h)
               .reshape(B, N, 3, model.attn.h, model.attn.dh)
               .permute(0, 2, 1, 3, 4))
        q = qkv[:, 0].transpose(1, 2)
        k = qkv[:, 1].transpose(1, 2)
        v = qkv[:, 2].transpose(1, 2)
        from patchhar_v2 import apply_rope
        q, k  = apply_rope(q, k, freqs)
        score = (q @ k.transpose(-2, -1)) / math.sqrt(model.attn.dh)
        attn  = torch.softmax(score, dim=-1)           # (B, H, N, N)
        captured_attn.append(attn.detach().cpu())
        y     = model.attn.out(
            (model.attn.drop(attn) @ v)
            .transpose(1, 2).contiguous().reshape(B, N, D))
        return x + torch.sigmoid(model.attn.gate(h)) * y

    model.attn.forward = _patched_forward

    all_rollout, all_labels = [], []
    n_collected = 0

    with torch.no_grad():
        for batch in loader:
            if n_collected >= max_samples:
                break
            patches_list, times, labels, _, _, _ = batch
            patches_list = [p.to(device).float() for p in patches_list]
            times_d      = times.to(device).float()
            captured_attn.clear()

            _ = model(patches_list, times_d)

            if not captured_attn:
                continue

            A = captured_attn[0]                       # (B, H, NP, NP)
            A = A.mean(dim=1).numpy()                  # (B, NP, NP) — avg heads

            # Residual connection + row normalise  (Rollout formula)
            I = np.eye(A.shape[-1])[None]              # (1, NP, NP)
            A = 0.5 * A + 0.5 * I
            A = A / (A.sum(axis=-1, keepdims=True) + 1e-8)

            # Global importance = mean incoming attention per token
            rollout = A.mean(axis=1)                   # (B, NP)

            all_rollout.append(rollout)
            all_labels.extend(labels.numpy().tolist())
            n_collected += len(labels)

    # Restore original forward
    model.attn.forward = original_forward

    rollout = np.concatenate(all_rollout, axis=0)[:max_samples].astype(np.float32)
    labels  = np.array(all_labels[:max_samples], dtype=np.int32)
    print(f"\n  Attention rollout: {rollout.shape}  (N × N_patches)")
    return rollout, labels


# ════════════════════════════════════════════════════════════════════════════
# 5.  GRADIENT-WEIGHTED ATTENTION  (Grad-CAM on patch tokens)
# ════════════════════════════════════════════════════════════════════════════

def compute_grad_attention(
        model:       PatchHARv2,
        loader:      DataLoader,
        max_samples: int = 200) -> np.ndarray:
    """
    Returns (N, N_PATCHES) normalised gradient × attention importance map.
    """
    model.eval()
    all_grad_attn = []
    n_collected   = 0

    # Patch forward once more — this time keep attn in graph for grad
    original_forward = model.attn.forward

    last_attn_store: List[torch.Tensor] = []

    def _patched_fwd_grad(x, freqs):
        h = model.attn.norm(x)
        B, N, D = h.shape
        qkv = (model.attn.qkv(h)
               .reshape(B, N, 3, model.attn.h, model.attn.dh)
               .permute(0, 2, 1, 3, 4))
        q = qkv[:, 0].transpose(1, 2)
        k = qkv[:, 1].transpose(1, 2)
        v = qkv[:, 2].transpose(1, 2)
        from patchhar_v2 import apply_rope
        q, k  = apply_rope(q, k, freqs)
        score = (q @ k.transpose(-2, -1)) / math.sqrt(model.attn.dh)
        attn  = torch.softmax(score, dim=-1)
        last_attn_store.clear()
        last_attn_store.append(attn)                  # keep in graph
        y = model.attn.out(
            (model.attn.drop(attn) @ v)
            .transpose(1, 2).contiguous().reshape(B, N, D))
        return x + torch.sigmoid(model.attn.gate(h)) * y

    model.attn.forward = _patched_fwd_grad

    for batch in loader:
        if n_collected >= max_samples:
            break
        patches_list, times, labels, _, _, _ = batch
        patches_list = [p.to(device).float() for p in patches_list]
        times_d      = times.to(device).float()
        labels_d     = labels.to(device)
        last_attn_store.clear()

        model.zero_grad()
        logits, _ = model(patches_list, times_d)

        if not last_attn_store:
            continue

        attn = last_attn_store[0]                     # (B, H, NP, NP)

        # Score = sum of correct-class logits
        score = logits.gather(1, labels_d.unsqueeze(1)).sum()
        score.backward()

        with torch.no_grad():
            # Gradient w.r.t. attention tensor (B, H, NP, NP)
            if attn.grad is not None:
                grad = attn.grad
            else:
                # attn may not have .grad if not a leaf; use hooks instead
                grad = torch.zeros_like(attn)

            # Weight: ReLU(grad * attn), avg heads, sum over source
            importance = F.relu(grad * attn).mean(dim=1).sum(dim=-1)  # (B, NP)
            importance = importance.cpu().numpy()

        all_grad_attn.append(importance)
        n_collected += len(labels)
        model.zero_grad()

    model.attn.forward = original_forward

    if not all_grad_attn:
        print("  [WARN] Gradient attention collection failed — "
              "returning zero array")
        return np.zeros((1, cfg.N_PATCHES), dtype=np.float32)

    grad_attn = np.concatenate(all_grad_attn, axis=0)[:max_samples]
    mx = grad_attn.max(axis=1, keepdims=True) + 1e-8
    grad_attn = (grad_attn / mx).astype(np.float32)
    print(f"  Gradient attention: {grad_attn.shape}  (N × N_patches)")
    return grad_attn


# ════════════════════════════════════════════════════════════════════════════
# 6.  [C3] CIRCADIAN BIAS ANALYSIS
#     Shows how strongly each patch position is influenced by time-of-day.
# ════════════════════════════════════════════════════════════════════════════

def analyse_circadian_bias(
        model:       PatchHARv2,
        loader:      DataLoader,
        max_samples: int = 500) -> Dict:
    """
    Computes mean absolute circadian bias per patch position across the dataset.
    Returns
    -------
    dict with:
      mean_bias_magnitude  : (N_PATCHES,)  — mean |bias| across samples & dims
      bias_per_class       : {class_name: (N_PATCHES,)} — class-conditional
      peak_patch           : int   — patch index with highest mean bias
    """
    if not CC.C3_CIRCADIAN_BIAS:
        print("  C3 disabled — skipping circadian bias analysis")
        return {}

    model.eval()
    all_bias_mag: List[np.ndarray] = []   # (B, NP)
    all_labels:   List[int]        = []
    n_collected = 0

    with torch.no_grad():
        for batch in loader:
            if n_collected >= max_samples:
                break
            patches_list, times, labels, _, _, _ = batch
            times_d = times.to(device).float()
            bias    = model.circ_bias(times_d)       # (B, NP, D)
            mag     = bias.abs().mean(dim=-1)         # (B, NP) — mean over D
            all_bias_mag.append(mag.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())
            n_collected += len(labels)

    bias_arr = np.concatenate(all_bias_mag, axis=0)[:max_samples]  # (N, NP)
    labels_arr = np.array(all_labels[:max_samples])

    mean_bias = bias_arr.mean(axis=0)   # (NP,)
    peak_patch = int(mean_bias.argmax())

    # Per-class conditional mean
    bias_per_class: Dict[str, List[float]] = {}
    for i, name in idx_to_class.items():
        mask = (labels_arr == i)
        if mask.sum() > 0:
            bias_per_class[name] = np.round(
                bias_arr[mask].mean(axis=0), 5).tolist()

    np.save("circadian_bias_c24.npy", bias_arr)
    print(f"\n── C3 CIRCADIAN BIAS ────────────────────────────────────────────")
    print(f"  Bias array saved: circadian_bias_c24.npy  {bias_arr.shape}")
    print(f"  Mean |bias| range: [{mean_bias.min():.4f}, {mean_bias.max():.4f}]")
    print(f"  Peak patch index : {peak_patch}  "
          f"(t ≈ {peak_patch * cfg.PATCH_LEN / cfg.SIGNAL_RATE:.2f} s)")

    return dict(
        mean_bias_magnitude = np.round(mean_bias, 6).tolist(),
        peak_patch          = peak_patch,
        peak_patch_time_s   = round(peak_patch * cfg.PATCH_LEN / cfg.SIGNAL_RATE, 3),
        bias_per_class      = bias_per_class,
    )


# ════════════════════════════════════════════════════════════════════════════
# 7.  [C1] DUAL-DOMAIN GATE ANALYSIS
#     Shows balance between time-domain and frequency-domain embedding.
# ════════════════════════════════════════════════════════════════════════════

def analyse_dual_domain_gates(model: PatchHARv2) -> Dict:
    if not CC.C1_DUAL_DOMAIN_EMBEDDING:
        print("  C1 disabled — skipping dual-domain gate analysis")
        return {}

    results: Dict = {}
    for i, embed in enumerate(model.patch_embeds):
        with torch.no_grad():
            g = torch.sigmoid(embed.gate_w).cpu().numpy()  # (D,)
        pl = cfg.PATCH_LENS_MULTI[i] if CC.C4_MULTISCALE_PATCHING else cfg.PATCH_LEN
        key = f"patch_embed_PL{pl}"
        results[key] = dict(
            mean_gate       = round(float(g.mean()), 6),
            std_gate        = round(float(g.std()),  6),
            frac_time_dom   = round(float((g > 0.5).mean()), 4),
            frac_freq_dom   = round(float((g <= 0.5).mean()), 4),
            gate_histogram  = np.round(
                np.histogram(g, bins=10, range=(0,1))[0] / len(g), 4).tolist(),
        )
        print(f"  Gate PL={pl:<4}  mean={results[key]['mean_gate']:.4f}  "
              f"time>{results[key]['frac_time_dom']:.2%}  "
              f"freq>{results[key]['frac_freq_dom']:.2%}")

    return results


# ════════════════════════════════════════════════════════════════════════════
# 8.  [C2] SKIP-AGG WEIGHTS
# ════════════════════════════════════════════════════════════════════════════

def report_skip_weights(model: PatchHARv2) -> Dict:
    if not CC.C2_CALANET_SKIP_AGG:
        return {}
    with torch.no_grad():
        w = torch.softmax(model.skip_agg.weights, dim=0).cpu().numpy()
    print(f"\n  C2 skip-agg weights (layer importance):")
    for i, wi in enumerate(w):
        bar = "█" * int(wi * 40)
        print(f"    Layer {i+1}: {wi:.4f}  {bar}")
    return {"skip_agg_weights": np.round(w, 6).tolist()}


# ════════════════════════════════════════════════════════════════════════════
# 9.  EXPERT ROUTING ENTROPY
# ════════════════════════════════════════════════════════════════════════════

def compute_expert_routing(
        model:       PatchHARv2,
        loader:      DataLoader,
        max_samples: int = 500) -> Dict:
    """
    Hooks both MoE layers to capture routing weight distributions.
    Reports per-expert load and routing entropy (uniform = log(E)).
    """
    model.eval()
    routing1: List[torch.Tensor] = []
    routing2: List[torch.Tensor] = []

    def _mk_hook(buf: List):
        def _hook(module, inp, out):
            with torch.no_grad():
                w = torch.softmax(module.router(inp[0]), dim=-1).detach().cpu()
                buf.append(w)
        return _hook

    h1 = model.moe1.register_forward_hook(_mk_hook(routing1))
    h2 = model.moe2.register_forward_hook(_mk_hook(routing2))

    n_collected = 0
    with torch.no_grad():
        for batch in loader:
            if n_collected >= max_samples:
                break
            patches_list, times, labels, _, _, _ = batch
            patches_list = [p.to(device).float() for p in patches_list]
            times_d      = times.to(device).float()
            _ = model(patches_list, times_d)
            n_collected += len(labels)

    h1.remove(); h2.remove()

    def _analyse(buf: List[torch.Tensor], name: str) -> Dict:
        if not buf:
            return {}
        r     = torch.cat(buf, dim=0).reshape(-1, buf[0].shape[-1]).numpy()
        load  = r.mean(axis=0)
        ent   = float(-np.sum(load * np.log(load + 1e-8)))
        ideal = math.log(len(load))       # maximum entropy
        print(f"  {name}: load={np.round(load,3).tolist()}"
              f"  entropy={ent:.4f}/{ideal:.4f}"
              f"  ({ent/ideal*100:.1f}% of uniform)")
        return dict(
            load              = np.round(load, 6).tolist(),
            entropy           = round(ent, 6),
            ideal_entropy     = round(ideal, 6),
            pct_of_uniform    = round(ent / ideal * 100, 2),
            load_std          = round(float(r.std(axis=0).mean()), 6),
        )

    print(f"\n── EXPERT ROUTING ───────────────────────────────────────────────")
    return dict(
        moe1 = _analyse(routing1, "MoE1"),
        moe2 = _analyse(routing2, "MoE2"),
    )


# ════════════════════════════════════════════════════════════════════════════
# 10.  CALIBRATION  (ECE / MCE / reliability diagram)
# ════════════════════════════════════════════════════════════════════════════

def compute_calibration(logits:  torch.Tensor,
                         labels: np.ndarray,
                         n_bins: int = 15) -> Dict:
    probs   = torch.softmax(logits, dim=-1).numpy()
    confs   = probs.max(axis=1)
    preds   = probs.argmax(axis=1)
    correct = (preds == labels).astype(float)

    edges     = np.linspace(0, 1, n_bins + 1)
    bin_conf  = np.zeros(n_bins)
    bin_acc   = np.zeros(n_bins)
    bin_count = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (confs >= edges[i]) & (confs < edges[i+1])
        if mask.sum() == 0:
            continue
        bin_conf[i]  = confs[mask].mean()
        bin_acc[i]   = correct[mask].mean()
        bin_count[i] = mask.sum()

    w   = bin_count / bin_count.sum()
    ece = float(np.sum(w * np.abs(bin_acc - bin_conf)))
    mce = float(np.max(np.abs(bin_acc - bin_conf)))

    print(f"\n── CALIBRATION ──────────────────────────────────────────────────")
    print(f"  ECE = {ece:.5f}  |  MCE = {mce:.5f}")
    return dict(
        ECE        = round(ece, 6),
        MCE        = round(mce, 6),
        bin_conf   = np.round(bin_conf,  5).tolist(),
        bin_acc    = np.round(bin_acc,   5).tolist(),
        bin_count  = bin_count.tolist(),
        n_bins     = n_bins,
    )


# ════════════════════════════════════════════════════════════════════════════
# 11.  STATISTICAL SIGNIFICANCE  (McNemar test)
# ════════════════════════════════════════════════════════════════════════════

def mcnemar_test(pred_a: np.ndarray,
                 pred_b: np.ndarray,
                 labels: np.ndarray,
                 label_a: str = "model",
                 label_b: str = "baseline") -> Dict:
    from scipy.stats import chi2 as chi2_dist
    ok_a = (pred_a == labels)
    ok_b = (pred_b == labels)
    b    = int(np.sum(ok_a  & ~ok_b))   # a correct, b wrong
    c    = int(np.sum(~ok_a & ok_b))    # a wrong, b correct
    if b + c == 0:
        return dict(chi2=0.0, p_value=1.0, b=0, c=0,
                    label_a=label_a, label_b=label_b)
    chi2_val = float((abs(b - c) - 1)**2 / (b + c))
    p_val    = float(1 - chi2_dist.cdf(chi2_val, df=1))
    sig      = "✓ significant (p<0.05)" if p_val < 0.05 else "✗ not significant"
    print(f"  McNemar [{label_a} vs {label_b}]: "
          f"b={b}  c={c}  χ²={chi2_val:.4f}  p={p_val:.4e}  {sig}")
    return dict(
        chi2    = round(chi2_val, 6),
        p_value = round(p_val,    8),
        b       = b, c = c,
        label_a = label_a,
        label_b = label_b,
    )


# ════════════════════════════════════════════════════════════════════════════
# 12.  COLLECT LOGITS  (full pass over loader)
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_logits(model: PatchHARv2,
                   loader: DataLoader
                   ) -> Tuple[torch.Tensor, np.ndarray]:
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        patches_list, times, labels, _, _, _ = batch
        patches_list = [p.to(device).float() for p in patches_list]
        times_d      = times.to(device).float()
        logits, _    = model(patches_list, times_d)
        all_logits.append(logits.cpu())
        all_labels.extend(labels.numpy().tolist())
    return torch.cat(all_logits), np.array(all_labels)


# ════════════════════════════════════════════════════════════════════════════
# 13.  PER-CLASS METRICS
# ════════════════════════════════════════════════════════════════════════════

def per_class_metrics(pred: np.ndarray,
                      labels: np.ndarray) -> Dict:
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, s = precision_recall_fscore_support(
        labels, pred, labels=list(range(NUM_CLASSES)), zero_division=0)
    return {
        idx_to_class[i]: dict(
            precision = round(float(p[i]), 4),
            recall    = round(float(r[i]), 4),
            f1        = round(float(f[i]), 4),
            support   = int(s[i]))
        for i in range(NUM_CLASSES)
    }


# ════════════════════════════════════════════════════════════════════════════
# 14.  PRETTY-PRINT PAPER TABLE
# ════════════════════════════════════════════════════════════════════════════

def print_paper_table(results: Dict):
    W = 68
    sep = "─" * W

    def _section(title: str):
        print(f"\n{'═'*W}")
        print(f"  {title}")
        print(sep)

    def _row(k: str, v):
        print(f"  {k:<38}  {v}")

    print("\n" + "═"*W)
    print(f"  PATCHHAR v2  ──  CAPTURE-24  ──  PAPER METRICS")
    print("═"*W)

    c = results.get("complexity", {})
    _section("ARCHITECTURE")
    _row("D_model / heads / layers / experts",
         f"{c.get('d_model')} / {c.get('n_heads')} / "
         f"{c.get('n_layers')} / {c.get('n_experts')}")
    _row("Total parameters",       f"{c.get('total_params',0):,}")
    _row("Trainable parameters",   f"{c.get('trainable_params',0):,}")
    _row("Model size  FP32 / FP16",
         f"{c.get('size_fp32_mb')} MB  /  {c.get('size_fp16_mb')} MB")
    if c.get("flops_gflops"):
        _row("FLOPs per sample", f"{c['flops_gflops']} GFLOPs")
    _row("Window / patches / patch_len / Hz",
         f"{c.get('window_size')} / {c.get('n_patches')} / "
         f"{c.get('patch_len')} / {c.get('signal_rate')}")

    lat = results.get("latency", {})
    for dev in ["cuda", "cpu"]:
        key = f"latency_single_{dev}_ms"
        if key in lat:
            l = lat[key]
            _section(f"INFERENCE LATENCY  ({dev.upper()}, B=1)")
            _row("Mean ± std",         f"{l['mean']} ± {l['std']} ms")
            _row("p50 / p95 / p99",    f"{l['p50']} / {l['p95']} / {l['p99']} ms")
            _row("Min / Max",          f"{l['min']} / {l['max']} ms")

    tp = lat.get("throughput", {})
    if tp:
        _section("THROUGHPUT")
        for b, v in tp.items():
            _row(f"Batch {b}", f"{v['samples_per_sec']:>10.1f} samp/s  "
                               f"({v['ms_per_sample']:.3f} ms/sample)")

    cs = lat.get("cold_start_ms")
    if cs:
        _row("Cold start (first call)", f"{cs} ms")

    mem = results.get("memory", {})
    if "peak_inference_b32_mb" in mem:
        _section("GPU MEMORY  (B=32)")
        _row("Peak inference",    f"{mem['peak_inference_b32_mb']} MB")
        _row("Peak train step",   f"{mem['peak_train_step_b32_mb']} MB")

    cal = results.get("calibration", {})
    if cal:
        _section("CALIBRATION")
        _row("ECE", cal.get("ECE"))
        _row("MCE", cal.get("MCE"))

    gm = results.get("global_metrics", {})
    if gm:
        _section("CLASSIFICATION PERFORMANCE  (test set)")
        for k, v in gm.items():
            _row(k, v)

    pc = results.get("per_class_metrics", {})
    if pc:
        _section("PER-CLASS METRICS")
        print(f"  {'Class':<28} {'Prec':>7} {'Rec':>7} {'F1':>7} {'N':>7}")
        print(f"  {sep}")
        for cls, m in pc.items():
            print(f"  {cls:<28} {m['precision']:>7.4f} {m['recall']:>7.4f} "
                  f"{m['f1']:>7.4f} {m['support']:>7}")

    er = results.get("expert_routing", {})
    if er:
        _section("EXPERT ROUTING")
        for moe, v in er.items():
            if v:
                _row(f"{moe} load",    str([round(x,3) for x in v["load"]]))
                _row(f"{moe} entropy", f"{v['entropy']:.4f} / {v['ideal_entropy']:.4f}  "
                                       f"({v['pct_of_uniform']:.1f}% uniform)")

    dd = results.get("dual_domain_gates", {})
    if dd:
        _section("C1 DUAL-DOMAIN GATE")
        for k, v in dd.items():
            _row(f"{k} mean gate",     v["mean_gate"])
            _row(f"{k} time dom %",    f"{v['frac_time_dom']:.1%}")
            _row(f"{k} freq dom %",    f"{v['frac_freq_dom']:.1%}")

    cb = results.get("circadian_bias", {})
    if cb:
        _section("C3 CIRCADIAN BIAS")
        _row("Peak patch index",   cb.get("peak_patch"))
        _row("Peak patch time (s)", cb.get("peak_patch_time_s"))

    sw = results.get("skip_agg", {})
    if sw and "skip_agg_weights" in sw:
        _section("C2 SKIP-AGGREGATION WEIGHTS  (per layer)")
        for i, wi in enumerate(sw["skip_agg_weights"]):
            bar = "▪" * max(1, int(wi * 30))
            print(f"    Layer {i+1}: {wi:.4f}  {bar}")

    mc = results.get("mcnemar_majority", {})
    if mc:
        _section("MCNEMAR TEST  (model vs majority-class baseline)")
        _row("χ²", mc.get("chi2"))
        _row("p-value", mc.get("p_value"))
        _row("Discordant pairs (b, c)", f"{mc.get('b')},  {mc.get('c')}")

    print("\n" + "═"*W + "\n")


# ════════════════════════════════════════════════════════════════════════════
# 15.  MAIN PROFILER
# ════════════════════════════════════════════════════════════════════════════

def run_profiler(checkpoint_path: Optional[str] = None,
                 benchmark_only:  bool           = False,
                 max_samples:     int            = 500,
                 n_latency_runs:  int            = 500):

    model = PatchHARv2().to(device)
    model.eval()

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device,
                          weights_only=False)
        # strip wrapper keys if saved as {"model": state_dict}
        if isinstance(ckpt, dict) and "model" in ckpt:
            ckpt = ckpt["model"]
        model.load_state_dict(ckpt, strict=True)
        print(f"\n  Checkpoint loaded: {checkpoint_path}")
    else:
        print(f"\n  NOTE: No valid checkpoint provided — "
              f"random weights (latency/complexity still valid).")

    all_results: Dict = {}

    # ── 1. Complexity ────────────────────────────────────────────────────
    all_results["complexity"] = report_complexity(model)

    # ── 2. Latency & throughput ──────────────────────────────────────────
    print("\n── LATENCY & THROUGHPUT ─────────────────────────────────────────")
    all_results["latency"] = benchmark_latency(model, n_runs=n_latency_runs)

    # ── 3. Memory ────────────────────────────────────────────────────────
    all_results["memory"] = report_memory(model)

    if benchmark_only:
        print("\n  [benchmark_only=True]  Skipping data-dependent analyses.")
        with open("paper_metrics_c24.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print_paper_table(all_results)
        return all_results

    # ── Load data ────────────────────────────────────────────────────────
    print("\n  Loading test data ...")
    test_pids_use = test_pids if test_pids else val_pids
    if not test_pids_use:
        print("  [WARN] No test/val pids available — using train pids")
        test_pids_use = train_pids[:5]

    loader = _make_loader(test_pids_use, batch_size=64)
    print(f"  Loaded {len(loader.dataset):,} test windows")

    # ── 4. Attention rollout ─────────────────────────────────────────────
    print("\n── ATTENTION ROLLOUT ────────────────────────────────────────────")
    rollout, roll_labels = compute_attention_rollout(
        model, loader, max_samples)
    np.save("attn_rollout_c24.npy",    rollout)
    np.save("rollout_labels_c24.npy",  roll_labels)

    # Per-class mean rollout
    class_rollout: Dict = {}
    for i, name in idx_to_class.items():
        mask = (roll_labels == i)
        if mask.sum() > 0:
            class_rollout[name] = np.round(
                rollout[mask].mean(axis=0), 5).tolist()
    all_results["attention_rollout_mean"]     = np.round(
        rollout.mean(axis=0), 5).tolist()
    all_results["attention_rollout_per_class"] = class_rollout

    # ── 5. Gradient attention ────────────────────────────────────────────
    print("\n── GRADIENT ATTENTION ───────────────────────────────────────────")
    grad_attn = compute_grad_attention(
        model, loader, min(200, max_samples))
    np.save("grad_attn_c24.npy", grad_attn)
    all_results["grad_attention_mean"] = np.round(
        grad_attn.mean(axis=0), 5).tolist()

    # ── 6. Circadian bias ────────────────────────────────────────────────
    all_results["circadian_bias"] = analyse_circadian_bias(
        model, loader, max_samples)

    # ── 7. Dual-domain gates ─────────────────────────────────────────────
    print("\n── C1 DUAL-DOMAIN GATES ─────────────────────────────────────────")
    all_results["dual_domain_gates"] = analyse_dual_domain_gates(model)

    # ── 8. Skip-agg weights ──────────────────────────────────────────────
    print("\n── C2 SKIP-AGG WEIGHTS ──────────────────────────────────────────")
    all_results["skip_agg"] = report_skip_weights(model)

    # ── 9. Expert routing ────────────────────────────────────────────────
    all_results["expert_routing"] = compute_expert_routing(
        model, loader, max_samples)

    # ── 10. Collect full logits ──────────────────────────────────────────
    print("\n  Collecting logits over test set ...")
    logits_all, labels_all = collect_logits(model, loader)

    # ── 11. Calibration ──────────────────────────────────────────────────
    calib = compute_calibration(logits_all, labels_all)
    all_results["calibration"] = calib
    np.save("calibration_c24.npy",
            np.stack([calib["bin_conf"], calib["bin_acc"], calib["bin_count"]]))

    # ── 12. Global metrics ───────────────────────────────────────────────
    from sklearn.metrics import f1_score as sk_f1
    pred_all = logits_all.argmax(1).numpy()
    all_results["global_metrics"] = dict(
        macro_f1    = round(float(sk_f1(labels_all, pred_all,
                                        average="macro",    zero_division=0)), 4),
        micro_f1    = round(float(sk_f1(labels_all, pred_all,
                                        average="micro",    zero_division=0)), 4),
        weighted_f1 = round(float(sk_f1(labels_all, pred_all,
                                        average="weighted", zero_division=0)), 4),
        accuracy    = round(float((pred_all == labels_all).mean()), 4),
        cohen_kappa = round(float(kappa(labels_all, pred_all)), 4),
        mcc         = round(float(mcc(labels_all, pred_all)), 4),
    )
    print(f"\n── GLOBAL METRICS ───────────────────────────────────────────────")
    for k, v in all_results["global_metrics"].items():
        print(f"  {k:<28}: {v}")

    # ── 13. Per-class metrics ────────────────────────────────────────────
    all_results["per_class_metrics"] = per_class_metrics(pred_all, labels_all)

    # ── 14. McNemar test vs majority baseline ────────────────────────────
    majority_class = int(np.bincount(labels_all).argmax())
    majority_pred  = np.full_like(labels_all, majority_class)
    all_results["mcnemar_majority"] = mcnemar_test(
        pred_all, majority_pred, labels_all,
        label_a="PatchHAR-v2", label_b="majority-class")

    # ── Save all results ─────────────────────────────────────────────────
    out_path = "paper_metrics_c24.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'═'*68}")
    print(f"  Saved: {out_path}")
    print(f"         attn_rollout_c24.npy        (N, {cfg.N_PATCHES})")
    print(f"         rollout_labels_c24.npy      (N,)")
    print(f"         grad_attn_c24.npy           (N, {cfg.N_PATCHES})")
    if CC.C3_CIRCADIAN_BIAS:
        print(f"         circadian_bias_c24.npy     (N, {cfg.N_PATCHES})")
    print(f"         calibration_c24.npy         (3, n_bins)")
    print("═"*68)

    print_paper_table(all_results)
    return all_results


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PatchHAR v2  Capture-24  Paper Profiler")
    parser.add_argument(
        "--checkpoint",     type=str,  default=None,
        help="Path to saved weights (.pth from patchhar_v2.py training)")
    parser.add_argument(
        "--benchmark_only", action="store_true",
        help="Run latency/complexity only — no data or checkpoint needed")
    parser.add_argument(
        "--max_samples",    type=int,  default=500,
        help="Max samples for attention/routing analysis  (default 500)")
    parser.add_argument(
        "--n_latency_runs", type=int,  default=500,
        help="Timed inference runs for latency benchmark  (default 500)")
    args = parser.parse_args()

    run_profiler(
        checkpoint_path = args.checkpoint,
        benchmark_only  = args.benchmark_only,
        max_samples     = args.max_samples,
        n_latency_runs  = args.n_latency_runs,
    )
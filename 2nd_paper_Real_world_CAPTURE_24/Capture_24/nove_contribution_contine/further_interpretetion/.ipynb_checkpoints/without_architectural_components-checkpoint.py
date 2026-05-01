"""
ablation_patchhar.py
====================
Ablation study for PatchHAR v2 — four model variants, full latency profiling.

  baseline              — all contributions OFF: GDN local layers + SoftMoE
  ffn                   — SoftMoE replaced with standard 2-layer FFN
  self_attn             — GDN replaced with standard MHSA
  standard_transformer  — pure Transformer baseline: MHSA local + FFN + standard global MHSA
                          (no RoPE, no gating, no MoE — fairest apples-to-apples reference)

Latency statistics: 500 forward passes at batch size 1 (GPU) /
                    100 forward passes at batch size 1 (CPU).
"""

from __future__ import annotations
import math, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ─────────────────────────── Configuration ────────────────────────────────────
D   = 64          # model dimension
H   = 2           # attention heads
L   = 3           # number of local layers
E   = 4           # SoftMoE experts
P   = 0.25        # dropout

WS  = 3000        # window samples (30 s × 100 Hz)
PL  = 25          # patch length (samples)
C   = 3           # IMU channels
NP  = WS // PL   # 120 patches
K   = 8           # number of activity classes

N_WARMUP   = 50
N_RUNS_GPU = 500
N_RUNS_CPU = 100
THRU_ITERS = 200

GPU    = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print(f"Device : {device}")
if GPU:
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

try:
    from thop import profile as _thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("[warn] thop not installed — install with  pip install thop  for FLOPs.")


# ───────────────────────── Shared building blocks ─────────────────────────────

class ZCRMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.g, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x - x.mean(-1, keepdim=True)
        return x0 / x0.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt() * self.g


def precompute_freqs(dim: int, n: int, theta: float = 10000.0) -> torch.Tensor:
    assert dim % 2 == 0
    f = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    return torch.polar(torch.ones(n, dim // 2),
                       torch.outer(torch.arange(n).float(), f))


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    B, Hd, N, Dh = q.shape
    d2 = Dh // 2
    f  = freqs[:N].to(q.device).view(1, 1, N, d2)
    qc = torch.view_as_complex(q.float().contiguous().view(B, Hd, N, d2, 2))
    kc = torch.view_as_complex(k.float().contiguous().view(B, Hd, N, d2, 2))
    return (torch.view_as_real(qc * f).view(B, Hd, N, Dh).type_as(q),
            torch.view_as_real(kc * f).view(B, Hd, N, Dh).type_as(k))


# ─────────────────── Local layer A: GatedDeltaNet (original) ──────────────────

class GatedDeltaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm      = ZCRMSNorm(D)
        self.q_lin     = nn.Linear(D, D)
        self.k_lin     = nn.Linear(D, D)
        self.v_lin     = nn.Linear(D, D)
        self.q_conv    = nn.Conv1d(D, D, 3, padding=1, groups=D)
        self.k_conv    = nn.Conv1d(D, D, 3, padding=1, groups=D)
        self.v_conv    = nn.Conv1d(D, D, 3, padding=1, groups=D)
        self.act       = nn.Sigmoid()
        self.alpha     = nn.Linear(D, D)
        self.beta      = nn.Linear(D, D)
        self.post_norm = ZCRMSNorm(D)
        self.post      = nn.Linear(D, D)
        self.silu      = nn.SiLU()
        self.gate      = nn.Sigmoid()
        self.drop      = nn.Dropout(P)

    @staticmethod
    def _l2(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return x / x.pow(2).sum(-1, keepdim=True).add(eps).sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q = self.act(self.q_conv(self.q_lin(h).transpose(1, 2)).transpose(1, 2))
        k = self.act(self.k_conv(self.k_lin(h).transpose(1, 2)).transpose(1, 2))
        v = self.act(self.v_conv(self.v_lin(h).transpose(1, 2)).transpose(1, 2))
        q, k  = self._l2(q), self._l2(k)
        delta = q * (k * v)
        delta = torch.tanh(self.alpha(x)) * delta + self.beta(x)
        dhat  = self.post(self.post_norm(delta))
        return x + self.drop(self.gate(self.silu(dhat)) * dhat)


# ─────────────────── Local layer B: Standard MHSA (ablation) ─────────────────

class StandardMHSA(nn.Module):
    """Pre-norm multi-head self-attention — drop-in replacement for GDN."""
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, H, dropout=P, batch_first=True)
        self.drop = nn.Dropout(P)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        y, _ = self.attn(h, h, h, need_weights=False)
        return x + self.drop(y)


# ─────────────────── Feed-forward A: SoftMoE (original) ──────────────────────

class SoftMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.router  = nn.Linear(D, E)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(D, 2 * D), nn.SiLU(),
                          nn.Dropout(P), nn.Linear(2 * D, D))
            for _ in range(E)
        ])
        self.drop = nn.Dropout(P)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.router(x), dim=-1)
        s = torch.stack([e(x) for e in self.experts], dim=-2)
        return self.drop((w.unsqueeze(-1) * s).sum(-2))


# ─────────────────── Feed-forward B: Standard FFN (ablation) ─────────────────

class StandardFFN(nn.Module):
    """Pre-norm 2-layer FFN — drop-in replacement for SoftMoE (no residual)."""
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(D)
        self.net  = nn.Sequential(
            nn.Linear(D, 2 * D), nn.SiLU(),
            nn.Dropout(P), nn.Linear(2 * D, D)
        )
        self.drop = nn.Dropout(P)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.net(self.norm(x)))


# ─────────────────── Global attention: GatedAttention + RoPE ─────────────────

class GatedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert D % H == 0 and (D // H) % 2 == 0
        self.h    = H
        self.dh   = D // H
        self.norm = ZCRMSNorm(D)
        self.qkv  = nn.Linear(D, 3 * D)
        self.out  = nn.Linear(D, D)
        self.gate = nn.Linear(D, D)
        self.drop = nn.Dropout(P)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        B, N, _ = h.shape
        qkv = self.qkv(h).reshape(B, N, 3, self.h, self.dh).permute(0, 2, 1, 3, 4)
        q = qkv[:, 0].transpose(1, 2)
        k = qkv[:, 1].transpose(1, 2)
        v = qkv[:, 2].transpose(1, 2)
        q, k  = apply_rope(q, k, freqs)
        score = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        attn  = self.drop(torch.softmax(score, dim=-1))
        y     = self.out((attn @ v).transpose(1, 2).contiguous().reshape(B, N, D))
        return x + torch.sigmoid(self.gate(h)) * y


# ─────────────── Global attention B: Standard MHSA (standard_transformer) ────

class StandardGlobalAttn(nn.Module):
    """Pre-norm standard MHSA — drop-in replacement for GatedAttention+RoPE."""
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, H, dropout=P, batch_first=True)
        self.drop = nn.Dropout(P)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        y, _ = self.attn(h, h, h, need_weights=False)
        return x + self.drop(y)


# ───────────────────────── Four ablation models ───────────────────────────────

class AblationModel(nn.Module):
    """
    variant               local layer    feed-forward    global attention
    ──────────────────    ───────────    ────────────    ────────────────
    baseline              GDN            SoftMoE         GatedAttn + RoPE
    ffn                   GDN            standard FFN    GatedAttn + RoPE
    self_attn             MHSA           SoftMoE         GatedAttn + RoPE
    standard_transformer  MHSA           standard FFN    standard MHSA
    """

    def __init__(self, variant: str):
        assert variant in ("baseline", "ffn", "self_attn", "standard_transformer"), \
            f"Unknown variant: {variant}"
        super().__init__()
        self.variant = variant

        self.patch_embed = nn.Linear(PL * C, D)
        self.time_emb    = nn.Sequential(nn.Linear(5, D), nn.ReLU(), nn.Dropout(0.1))
        self.input_norm  = nn.LayerNorm(D)

        if variant in ("self_attn", "standard_transformer"):
            self.local_layers = nn.ModuleList([StandardMHSA() for _ in range(L)])
        else:
            self.local_layers = nn.ModuleList([GatedDeltaNet() for _ in range(L)])

        if variant in ("ffn", "standard_transformer"):
            self.ff1 = StandardFFN()
            self.ff2 = StandardFFN()
        else:
            self.ff1 = SoftMoE()
            self.ff2 = SoftMoE()

        if variant == "standard_transformer":
            self.global_attn = StandardGlobalAttn()
        else:
            self.global_attn = GatedAttention()
            freqs = precompute_freqs(D // H, NP)
            self.register_buffer("freqs", freqs)

        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(D, D // 2), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(D // 2, K),
        )

    def forward(self, patches: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        B, Ch, NP_, PL_ = patches.shape
        x = patches.permute(0, 2, 1, 3).reshape(B, NP_, Ch * PL_)
        x = self.patch_embed(x) + self.time_emb(times).unsqueeze(1)
        x = self.input_norm(x)

        for layer in self.local_layers:
            x = layer(x)

        x = x + self.ff1(x)

        if self.variant == "standard_transformer":
            x = self.global_attn(x)
        else:
            x = self.global_attn(x, self.freqs)

        x = x + self.ff2(x)

        return self.head(x.mean(dim=1))


# ───────────────────────── Profiling utilities ────────────────────────────────

def _dummy(B: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.randn(B, C, NP, PL, device=device),
            torch.randn(B, 5, device=device))


def compute_flops(model: nn.Module) -> float:
    if not HAS_THOP:
        return float("nan")
    p, t = _dummy(1)
    try:
        flops, _ = _thop_profile(model, inputs=(p, t), verbose=False)
        return flops / 1e9
    except Exception as exc:
        print(f"  [FLOPs] thop error: {exc}")
        return float("nan")


def measure_gpu_latency(model: nn.Module) -> np.ndarray:
    p, t = _dummy(1)
    for _ in range(N_WARMUP):
        with torch.no_grad():
            model(p, t)
    torch.cuda.synchronize()

    lats = []
    for _ in range(N_RUNS_GPU):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        with torch.no_grad():
            model(p, t)
        e.record()
        torch.cuda.synchronize()
        lats.append(s.elapsed_time(e))
    return np.array(lats)


def measure_cpu_latency(variant: str) -> np.ndarray:
    m = AblationModel(variant)
    m.eval()
    p = torch.randn(1, C, NP, PL)
    t = torch.randn(1, 5)
    for _ in range(20):
        with torch.no_grad():
            m(p, t)
    lats = []
    for _ in range(N_RUNS_CPU):
        t0 = time.perf_counter()
        with torch.no_grad():
            m(p, t)
        lats.append((time.perf_counter() - t0) * 1000)
    return np.array(lats)


def measure_cold_start_gpu(variant: str) -> float:
    m = AblationModel(variant).to(device)
    m.eval()
    p, t = _dummy(1)
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    with torch.no_grad():
        m(p, t)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e)


def measure_throughput(model: nn.Module, B: int) -> int:
    p, t = _dummy(B)
    for _ in range(20):
        with torch.no_grad():
            model(p, t)
    if GPU:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(THRU_ITERS):
        with torch.no_grad():
            model(p, t)
    if GPU:
        torch.cuda.synchronize()
    return int(B * THRU_ITERS / (time.perf_counter() - t0))


def measure_peak_mem_inference(variant: str) -> float:
    m = AblationModel(variant).to(device)
    m.eval()
    p, t = _dummy(32)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        m(p, t)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / 1024 ** 2


def measure_peak_mem_training(variant: str) -> float:
    m = AblationModel(variant).to(device)
    m.train()
    p, t  = _dummy(32)
    labs  = torch.randint(0, K, (32,), device=device)
    torch.cuda.reset_peak_memory_stats(device)
    loss  = F.cross_entropy(m(p, t), labs)
    loss.backward()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / 1024 ** 2


# ────────────────────────── Profile orchestrator ──────────────────────────────

def profile_variant(variant: str) -> dict:
    print(f"\n  ── Profiling  '{variant}' ──")
    model = AblationModel(variant).to(device)
    model.eval()

    n_params  = sum(x.numel() for x in model.parameters())
    trainable = sum(x.numel() for x in model.parameters() if x.requires_grad)
    gflops    = compute_flops(model)

    print(f"     FLOPs … done")
    gpu_lats  = measure_gpu_latency(model) if GPU else None
    if GPU:
        print(f"     GPU latency ({N_RUNS_GPU} runs) … done")
    cold_ms   = measure_cold_start_gpu(variant) if GPU else None
    cpu_lats  = measure_cpu_latency(variant)
    print(f"     CPU latency ({N_RUNS_CPU} runs) … done")

    throughputs = {}
    for B in (1, 8, 32, 64):
        throughputs[B] = measure_throughput(model, B)
    print(f"     Throughput … done")

    mem_inf   = measure_peak_mem_inference(variant) if GPU else None
    mem_train = measure_peak_mem_training(variant)  if GPU else None
    if GPU:
        print(f"     Peak memory … done")

    return dict(
        n_params=n_params, trainable=trainable,
        gflops=gflops,
        gpu_lats=gpu_lats, cold_ms=cold_ms,
        cpu_lats=cpu_lats,
        throughputs=throughputs,
        mem_inf=mem_inf, mem_train=mem_train,
    )


# ──────────────────────── Formatted output ────────────────────────────────────

_VARIANT_LABELS = {
    "baseline":             "Baseline  (all contributions OFF — GDN + SoftMoE)",
    "ffn":                  "FFN variant  (SoftMoE → standard 2-layer FFN)",
    "self_attn":            "SelfAttn variant  (GDN → standard MHSA)",
    "standard_transformer": "Standard Transformer  (MHSA local + FFN + standard global MHSA)",
}

W = 46   # property column width


def _fv(v, fmt=".2f", na="N/A"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return na
    return format(v, fmt)


def _row(prop: str, val: str):
    print(f"  {prop:<{W}} {val}")


def print_variant_stats(variant: str, s: dict):
    gl = s["gpu_lats"]
    cl = s["cpu_lats"]

    print(f"\n{'═' * 72}")
    print(f"  {_VARIANT_LABELS[variant]}")
    print(f"{'═' * 72}")
    print(f"  {'Property':<{W}} Value")
    print(f"  {'-' * 68}")

    _row("Parameters (total / trainable)",
         f"{s['n_params']:,} / {s['trainable']:,}")
    _row("Model size FP32 / FP16",
         f"{s['n_params'] * 4 / 1024**2:.2f} MB / "
         f"{s['n_params'] * 2 / 1024**2:.2f} MB")
    _row("FLOPs per 30-second window",
         f"{_fv(s['gflops'], '.4f')} GFLOPs")

    if gl is not None:
        _row("GPU latency: mean / p50 / p95 / p99",
             f"{np.mean(gl):.2f} / {np.percentile(gl, 50):.2f} / "
             f"{np.percentile(gl, 95):.2f} / {np.percentile(gl, 99):.2f} ms")
    else:
        _row("GPU latency", "N/A (no GPU)")

    if cl is not None:
        _row("CPU latency: mean / p50 / p95",
             f"{np.mean(cl):.1f} / {np.percentile(cl, 50):.1f} / "
             f"{np.percentile(cl, 95):.1f} ms")

    _row("Cold start (first call, GPU)", f"{_fv(s['cold_ms'])} ms")

    for B in (1, 8, 32, 64):
        _row(f"Throughput (B={B})", f"{s['throughputs'][B]:,} windows/s")

    _row("Peak GPU memory, inference (B=32)",
         f"{_fv(s['mem_inf'], '.1f')} MB")
    _row("Peak GPU memory, training step (B=32)",
         f"{_fv(s['mem_train'], '.1f')} MB")

    print(f"  {'Architecture':<{W}} "
          f"D={D}, heads={H}, layers={L}, experts={E}, patches={NP}.E0")
    print()


def print_comparison(results: dict):
    variants = ("baseline", "ffn", "self_attn", "standard_transformer")
    short    = {
        "baseline":             "baseline",
        "ffn":                  "ffn",
        "self_attn":            "self_attn",
        "standard_transformer": "std_transf",
    }

    print(f"\n{'═' * 80}")
    print("  Comparison Summary")
    print(f"{'═' * 80}")

    hdr = f"  {'Metric':<36}"
    for v in variants:
        hdr += f"  {short[v]:>11}"
    print(hdr)
    print(f"  {'-' * 76}")

    def cmp(label: str, fn):
        row = f"  {label:<36}"
        for v in variants:
            row += f"  {fn(results[v]):>11}"
        print(row)

    cmp("Parameters",
        lambda s: f"{s['n_params']/1e6:.3f} M")
    cmp("Model size (FP32 MB)",
        lambda s: f"{s['n_params']*4/1024**2:.2f}")
    cmp("FLOPs (GFLOPs)",
        lambda s: _fv(s['gflops'], '.4f'))

    if GPU:
        cmp("GPU mean latency (ms)",
            lambda s: f"{np.mean(s['gpu_lats']):.2f}")
        cmp("GPU p50 latency (ms)",
            lambda s: f"{np.percentile(s['gpu_lats'], 50):.2f}")
        cmp("GPU p95 latency (ms)",
            lambda s: f"{np.percentile(s['gpu_lats'], 95):.2f}")
        cmp("GPU p99 latency (ms)",
            lambda s: f"{np.percentile(s['gpu_lats'], 99):.2f}")

    cmp("CPU mean latency (ms)",
        lambda s: f"{np.mean(s['cpu_lats']):.1f}")
    cmp("CPU p50 latency (ms)",
        lambda s: f"{np.percentile(s['cpu_lats'], 50):.1f}")
    cmp("CPU p95 latency (ms)",
        lambda s: f"{np.percentile(s['cpu_lats'], 95):.1f}")

    if GPU:
        cmp("Cold start GPU (ms)",
            lambda s: _fv(s['cold_ms']))

    for B in (1, 8, 32, 64):
        cmp(f"Throughput B={B} (win/s)",
            lambda s, b=B: f"{s['throughputs'][b]:,}")

    if GPU:
        cmp("Peak mem inference B=32 (MB)",
            lambda s: _fv(s['mem_inf'], '.1f'))
        cmp("Peak mem training B=32 (MB)",
            lambda s: _fv(s['mem_train'], '.1f'))

    print(f"{'═' * 80}")


# ─────────────────────────────── Main ─────────────────────────────────────────

def main():
    print("=" * 80)
    print("  PatchHAR v2 — Ablation Latency Profiler")
    print(f"  GPU runs: {N_RUNS_GPU} × B=1 | CPU runs: {N_RUNS_CPU} × B=1")
    print(f"  Variants: baseline | ffn | self_attn | standard_transformer")
    print("=" * 80)

    variants = ("baseline", "ffn", "self_attn", "standard_transformer")

    results = {}
    for variant in variants:
        results[variant] = profile_variant(variant)

    for variant in variants:
        print_variant_stats(variant, results[variant])

    print_comparison(results)


if __name__ == "__main__":
    main()
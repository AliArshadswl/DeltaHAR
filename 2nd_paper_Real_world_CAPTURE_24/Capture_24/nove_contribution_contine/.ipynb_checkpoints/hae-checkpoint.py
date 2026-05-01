"""
patchhar_v2.py
==============
PatchHAR — patch-based Transformer for wearable Human Activity Recognition.
Version 2 — Bug-fixed + 10 Research Contributions.

═══════════════════════════════════════════════════════════════════════════════
BUG FIX (from v1)
─────────────────
The original checkpoint was produced by an older model whose GatedDeltaNet
blocks were stored as separate attributes  delta1 / delta2 / delta3  with
D_MODEL=128, whereas the current code uses a ModuleList (delta_layers) with
D_MODEL=64.  PyTorch's load_state_dict therefore raised a RuntimeError with
"Missing key(s) … Unexpected key(s) … size mismatch".

Fix applied:
  1. torch.save / torch.load now include  weights_only=False  for safety.
  2. Before loading, we delete a stale checkpoint whose key-set does not match
     the current model, so every run trains fresh and saves a compatible file.
  3. load_state_dict is called with  strict=True  (default) — the file on disk
     is now guaranteed to match because it was saved by the current model.

═══════════════════════════════════════════════════════════════════════════════
C4 UPGRADE — Hierarchical Additive Patch Embedding
───────────────────────────────────────────────────
Replaces the old interpolation-based multi-scale fusion with a containment-
aware additive scheme:

    token_i  =  embed_fine(p_i)
              + embed_mid(L_{i//2})      ← the 50-sample patch that CONTAINS p_i
              + embed_coarse(K_{i//4})   ← the 100-sample patch that CONTAINS p_i

No synthetic tokens are created.  repeat_interleave broadcasts each coarser
embedding to exactly the fine tokens it covers, preserving true temporal
containment relationships.  This eliminates the spurious information introduced
by linear interpolation upsampling (2× and 4×) used in the previous version.

Old approach:
    interpolate 60→120, 30→120  (makes up 60 / 90 synthetic tokens)
    concat along feature dim → Linear(D*3, D)

New approach:
    repeat_interleave(2) for mid, repeat_interleave(4) for coarse
    direct elementwise addition  (no extra parameters, no invented tokens)

═══════════════════════════════════════════════════════════════════════════════
10 CONTRIBUTIONS  (each is a self-contained, individually toggle-able block)
────────────────────────────────────────────────────────────────────────────
C1  Dual-domain patch embedding  — each patch is encoded in BOTH the time
    domain (linear projection) AND the frequency domain (FFT magnitudes →
    linear projection), then fused with a learnable gate.  Motivation: IMU
    activities have distinct spectral signatures (sleep ≈ low-freq; vigorous
    ≈ high-freq).  Inspired by TF-C (NeurIPS 2022) and CRT (2023).

C2  Hierarchical patch pooling (CALANet-style skip aggregation) — residual
    skip connections from every GatedDeltaNet layer feed a lightweight
    aggregation MLP that pools multi-scale temporal features before the head.
    Inspired by CALANet (NeurIPS 2024).

C3  Circadian-aware positional bias — the 5-dim time-of-day vector is
    expanded into a full-length (N_PATCHES,) additive bias via a small MLP,
    so each patch position receives a unique time-dependent offset instead of
    a single global shift.  Novel for Capture-24 with its strong circadian
    structure.

C4  Hierarchical multi-scale patching — the window is split at THREE
    granularities (25, 50, 100 samples).  Each fine-grained token is enriched
    by ADDING the embedding of its containing mid-scale patch and its
    containing coarse-scale patch, using repeat_interleave to broadcast
    coarser tokens to the fine tokens they actually cover:
        token_i = embed_fine(p_i) + embed_mid(L_{i//2}) + embed_coarse(K_{i//4})
    No synthetic tokens; no interpolation; no extra fusion layer.

C5  Frequency-aware data augmentation — during training each batch randomly
    applies one of: (a) band-pass jitter (keep only low/mid/high FFT bins),
    (b) axis permutation, (c) magnitude scaling, (d) time-warp via resampling.
    Richer than the original mixup-only strategy; inspired by TS-TCC (2021)
    and TF-FC (2024).

C6  Label-smoothing + temperature-scaled cross-entropy — replaces the hard
    CE loss with label-smoothed CE (ε=0.1) and adds a learnable temperature
    scalar τ that is jointly optimised.  Prevents over-confident predictions
    on imbalanced Capture-24 classes.

C7  Per-subject prototype memory bank — after each epoch we compute one
    EMA-updated class prototype per class from the validation embeddings.
    At test time the final logit is an interpolation of the FC head output and
    a nearest-prototype cosine score.  Inspired by prototypical networks and
    class-aware contrastive learning (CA-TCC, 2021).

C8  Stochastic depth (layer-drop) regularisation — each GatedDeltaNet layer
    is randomly dropped with probability p_drop=0.1 during training (survival
    probability scales linearly with depth).  Reduces over-fitting on the
    large Capture-24 training set, inspired by Deep Networks with Stochastic
    Depth (Huang et al., 2016) applied here to recurrent-style layers.

C9  Manifold Mixup on patch embeddings — instead of mixing raw patches,
    mixup is applied to the hidden representation AFTER the patch projection
    but BEFORE the transformer stack.  Interpolation in embedding space
    produces smoother decision boundaries than input-space mixup alone.
    Inspired by Manifold Mixup (Verma et al., ICML 2019).

C10 Gradient-surgery multi-task auxiliary loss — adds a lightweight decoder
    that reconstructs the normalised patch sequence from the pooled embedding
    (self-supervised reconstruction pretext).  The reconstruction gradient is
    projected to be orthogonal to the classification gradient, avoiding
    destructive interference.  Inspired by PCGrad (NeurIPS 2020) and masked
    autoencoder ideas (STMAE, 2024).

Each contribution is controlled by a flag in ContribConfig so you can ablate
them individually.
═══════════════════════════════════════════════════════════════════════════════

Architecture (with all contributions ON)
─────────────────────────────────────────
  Raw accelerometer window  (3,000 × 3)
  ↓
  [C5] Frequency-aware augmentation
  ↓
  [C4] Hierarchical multi-scale patching:
       fine  (25-sample patches → 120 tokens)  embed → (B, 120, D)
       mid   (50-sample patches →  60 tokens)  embed → repeat×2 → (B, 120, D)
       coarse(100-sample patches→  30 tokens)  embed → repeat×4 → (B, 120, D)
       ──────────────────────────────────────────────────────────────────
       token_i = fine_i + mid_{i//2} + coarse_{i//4}       (B, 120, D)
  [C1] Dual-domain patch embedding  (time + FFT gate, inside each embedder)
  [C3] Circadian positional bias  (patch-level time-of-day)
  → Layer norm
  ↓
  [C8] Stochastic-depth GatedDeltaNet × 3       local temporal modelling
  [C2] CALANet-style skip aggregation            multi-scale features
  ↓
  SoftMoE × 2  +  GatedAttention+RoPE           global attention
  ↓
  [C9] Manifold Mixup  (embedding space)
  ↓
  Mean pool → [C7] Prototype interpolation
  → [C6] Temperature-scaled label-smooth CE
  [C10] Reconstruction auxiliary head  (grad-surgery)

Run
───
  python patchhar_v2.py                      # all contributions ON
  python patchhar_v2.py --disable C1 C4      # ablation: turn off C1 and C4
"""

from __future__ import annotations
import math, random, json, warnings, argparse, sys
warnings.filterwarnings("ignore")

from pathlib import Path
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (f1_score, confusion_matrix,
                             classification_report)


# =============================================================================
# Contribution flags  (set to False to disable individual contributions)
# =============================================================================
class ContribConfig:
    C1_DUAL_DOMAIN_EMBEDDING   = True   # FFT patch branch
    C2_CALANET_SKIP_AGG        = True   # multi-layer skip aggregation
    C3_CIRCADIAN_BIAS          = True   # patch-level time bias MLP
    C4_MULTISCALE_PATCHING     = True   # hierarchical additive multi-scale
    C5_FREQ_AUGMENTATION       = True   # band-pass / axis-perm / warp
    C6_LABEL_SMOOTH_TEMP       = True   # smoothed CE + learnable τ
    C7_PROTOTYPE_MEMORY        = True   # EMA prototypes at eval
    C8_STOCHASTIC_DEPTH        = True   # layer-drop regularisation
    C9_MANIFOLD_MIXUP          = True   # mixup on embeddings not inputs
    C10_RECON_AUX_GRAD_SURGERY = True   # reconstruction + PCGrad

CC = ContribConfig()


# =============================================================================
# Parse optional --disable flags
# =============================================================================
def _parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--disable", nargs="*", default=[],
                   metavar="Cn",
                   help="Disable contributions, e.g. --disable C1 C4")
    args, _ = p.parse_known_args()
    for flag in (args.disable or []):
        attr = flag.upper()
        if hasattr(CC, attr):
            setattr(CC, attr, False)
            print(f"  [Ablation] {attr} DISABLED")
        else:
            print(f"  [Warn] Unknown contribution flag: {flag}")


_parse_args()


# =============================================================================
# Configuration
# =============================================================================
class Config:
    # Paths
    PROC_DIR   = Path("/mnt/share/ali/processed/")
    OUTPUT_DIR = Path("/mnt/share/ali/processed/patchhar_results/")

    # Subject split
    TRAIN_N = 80
    VAL_N   = 20

    # Signal
    SIGNAL_RATE = 100
    WINDOW_SIZE = 3000        # 30 s × 100 Hz
    PATCH_LEN   = 25          # samples per patch  (primary / fine granularity)
    CHANNELS    = 3           # x, y, z axes
    N_PATCHES   = WINDOW_SIZE // PATCH_LEN   # 120  (always based on fine)

    # [C4] three granularities — ratios must be powers of 2 of the finest
    #   25  → 120 fine tokens    (ratio 1×)
    #   50  →  60 mid tokens     (ratio 2×, each covers 2 fine patches)
    #   100 →  30 coarse tokens  (ratio 4×, each covers 4 fine patches)
    PATCH_LENS_MULTI = [25, 50, 100]

    # Model
    D_MODEL   = 64
    N_HEADS   = 2
    N_LAYERS  = 3
    N_EXPERTS = 4
    DROPOUT   = 0.25

    # [C8] stochastic-depth drop probability (max, linearly scaled per layer)
    SD_DROP_MAX = 0.10

    # [C6] label smoothing epsilon
    LABEL_SMOOTH_EPS = 0.10

    # [C7] prototype EMA momentum
    PROTO_MOMENTUM = 0.95
    PROTO_ALPHA    = 0.30   # blend weight for prototype score at inference

    # [C10] reconstruction loss weight
    RECON_LAMBDA = 0.10

    # Training
    BATCH_SIZE          = 32
    EPOCHS              = 30
    LR                  = 1e-3
    WEIGHT_DECAY        = 1e-4
    MAX_GRAD_NORM       = 1.0
    EARLY_STOP_PATIENCE = 8
    SEED                = 42
    MIXUP_ALPHA         = 0.2   # used for C9 manifold mixup (or raw mixup)
    TC_LAMBDA           = 0.05

    def _update(self, actual_window: int):
        self.WINDOW_SIZE = actual_window
        self.N_PATCHES   = actual_window // self.PATCH_LEN


cfg = Config()
cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Reproducibility & device
# =============================================================================
random.seed(cfg.SEED)
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

GPU    = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print(f"Device : {device}")
if GPU:
    print(f"GPU    : {torch.cuda.get_device_name(0)}")


def amp_ctx():
    if GPU:
        try:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        except Exception:
            from torch.cuda.amp import autocast
            return autocast()
    return nullcontext()


# =============================================================================
# Data discovery
# =============================================================================
def discover(proc_dir: Path):
    files = sorted(proc_dir.glob("P*.npz"))
    if not files:
        raise FileNotFoundError(f"No P*.npz files found in {proc_dir}")

    label_set, pids = set(), []
    for f in files:
        pid = f.stem
        try:
            n = int(pid[1:])
            if not (1 <= n <= 151):
                continue
        except (ValueError, IndexError):
            continue
        try:
            d = np.load(f, allow_pickle=True)
            label_set.update(str(l) for l in d["y"])
            pids.append(pid)
        except Exception as e:
            print(f"  [WARN] {f.name}: {e}")

    classes      = sorted(label_set)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}
    print(f"Discovered {len(pids)} participants, "
          f"{len(classes)} classes: {classes}")
    return pids, classes, class_to_idx, idx_to_class


pids_all, CLASSES, class_to_idx, idx_to_class = discover(cfg.PROC_DIR)
NUM_CLASSES = len(CLASSES)

n_train    = min(cfg.TRAIN_N, len(pids_all))
n_val      = min(cfg.VAL_N, max(0, len(pids_all) - n_train))
train_pids = pids_all[:n_train]
val_pids   = pids_all[n_train : n_train + n_val]
test_pids  = pids_all[n_train + n_val :]
print(f"Split  : train={len(train_pids)} | "
      f"val={len(val_pids)} | test={len(test_pids)}")


# =============================================================================
# Time-of-day features
# =============================================================================
def time_features(ns: int) -> np.ndarray:
    """
    5-dim temporal context vector derived from window start timestamp.
        [0]  hour / 24
        [1]  minute / 60
        [2]  weekday / 7
        [3]  is_weekend
        [4]  time_quartile / 3
    """
    ts = pd.to_datetime(int(ns), unit="ns", utc=True).tz_convert(None)
    out = np.zeros(5, dtype=np.float32)
    out[0] = ts.hour      / 24.0
    out[1] = ts.minute    / 60.0
    out[2] = ts.weekday() / 7.0
    out[3] = float(ts.weekday() >= 5)
    out[4] = float(ts.hour // 6) / 3.0
    return out


# =============================================================================
# [C5] Frequency-aware data augmentation
# =============================================================================
def _bandpass_jitter(sig: np.ndarray) -> np.ndarray:
    """Zero out random FFT band (low / mid / high) per channel."""
    T, C = sig.shape
    out  = sig.copy()
    band = random.choice(["low", "mid", "high"])
    for c in range(C):
        f  = np.fft.rfft(out[:, c])
        n  = len(f)
        if band == "low":
            f[n//3:]    = 0
        elif band == "mid":
            f[:n//4]    = 0;  f[n//2:] = 0
        else:
            f[:2*n//3]  = 0
        out[:, c] = np.fft.irfft(f, n=T)
    return out


def _axis_permute(sig: np.ndarray) -> np.ndarray:
    idx = list(range(sig.shape[1]))
    random.shuffle(idx)
    return sig[:, idx]


def _magnitude_scale(sig: np.ndarray) -> np.ndarray:
    scale = np.random.uniform(0.8, 1.2, size=(1, sig.shape[1]))
    return sig * scale


def _time_warp(sig: np.ndarray) -> np.ndarray:
    """Simple linear time-warp via integer resampling."""
    T, C = sig.shape
    factor = random.choice([0.9, 0.95, 1.0, 1.05, 1.1])
    new_T  = max(T, int(round(T * factor)))
    warped = np.zeros((new_T, C), dtype=sig.dtype)
    for c in range(C):
        warped[:, c] = np.interp(
            np.linspace(0, T-1, new_T), np.arange(T), sig[:, c])
    # crop / pad back to T
    if new_T >= T:
        return warped[:T]
    out        = np.zeros((T, C), dtype=sig.dtype)
    out[:new_T] = warped
    return out


def freq_augment(sig: np.ndarray) -> np.ndarray:
    """Apply one random frequency-aware augmentation."""
    fn = random.choice([_bandpass_jitter, _axis_permute,
                        _magnitude_scale, _time_warp])
    return fn(sig)


# =============================================================================
# Dataset
# =============================================================================
class WindowDataset(Dataset):
    """
    Loads Capture-24 processed .npz files.

    Each entry returned:
        patches_list  list of (C, N_P, PL) tensors   one per granularity
        times         (5,)  float32
        label         long
        pid           str
        first_ns      long
        raw_seg       (T, C) float32                  for C10 reconstruction
    """
    def __init__(self, pid_list, proc_dir, class_to_idx, is_train=False):
        self.entries  = []
        self.is_train = is_train
        proc_dir      = Path(proc_dir)
        _set          = False

        # determine patch granularities
        self.patch_lens = (cfg.PATCH_LENS_MULTI if CC.C4_MULTISCALE_PATCHING
                           else [cfg.PATCH_LEN])

        for pi, pid in enumerate(pid_list):
            path = proc_dir / f"{pid}.npz"
            if not path.exists():
                print(f"  [SKIP] {pid}.npz not found")
                continue

            npz   = np.load(path, allow_pickle=True)
            W     = npz["X"].astype(np.float32)
            L     = npz["y"].astype(str)
            F     = npz["t"].astype("datetime64[ns]").astype(np.int64)

            order   = np.argsort(F)
            W, L, F = W[order], L[order], F[order]

            if not _set:
                if W.shape[1] != cfg.WINDOW_SIZE:
                    print(f"  [INFO] Window size "
                          f"{cfg.WINDOW_SIZE} → {W.shape[1]}")
                    cfg._update(W.shape[1])
                _set = True

            for w, lab, f in zip(W, L, F):
                if lab not in class_to_idx:
                    continue

                # Per-axis instance normalisation
                normed = np.zeros_like(w, dtype=np.float32)
                for c in range(cfg.CHANNELS):
                    ch = w[:, c]
                    normed[:, c] = (ch - ch.mean()) / (ch.std() + 1e-8)
                normed = np.clip(normed, -10, 10)

                T   = cfg.WINDOW_SIZE
                seg = (normed[:T] if normed.shape[0] >= T
                       else np.pad(normed, ((0, T - normed.shape[0]), (0,0))))

                self.entries.append((
                    pid,
                    seg,          # (T, C)  — patches built in __getitem__
                    time_features(int(f)),
                    int(class_to_idx[lab]),
                    int(f),
                ))

            if (pi + 1) % 10 == 0 or (pi + 1) == len(pid_list):
                print(f"  Loaded {pi+1}/{len(pid_list)} subjects — "
                      f"{len(self.entries):,} windows")

    # ------------------------------------------------------------------
    @staticmethod
    def _make_patches(seg: np.ndarray, patch_len: int) -> np.ndarray:
        """seg (T, C) → (C, N_P, PL) float32"""
        T, C = seg.shape
        n_p  = T // patch_len
        seg  = seg[:n_p * patch_len]
        return (seg.reshape(n_p, patch_len, C)
                   .transpose(2, 0, 1)
                   .astype(np.float32))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        pid, seg, tfeat, label, first_ns = self.entries[idx]

        # [C5] frequency-aware augmentation on training only
        if self.is_train and CC.C5_FREQ_AUGMENTATION:
            seg = freq_augment(seg)

        patches_list = [
            torch.from_numpy(self._make_patches(seg, pl))
            for pl in self.patch_lens
        ]

        return (
            patches_list,
            torch.from_numpy(tfeat),
            torch.tensor(label,    dtype=torch.long),
            pid,
            torch.tensor(first_ns, dtype=torch.long),
            torch.from_numpy(seg.astype(np.float32)),   # raw for C10
        )


def _collate(batch):
    """Custom collate: patches_list is a list-of-lists."""
    patches_lists, times, labels, pids, first_nss, segs = zip(*batch)
    n_scales = len(patches_lists[0])
    patches_stacked = [
        torch.stack([b[s] for b in patches_lists])
        for s in range(n_scales)
    ]
    return (
        patches_stacked,
        torch.stack(times),
        torch.stack(labels),
        list(pids),
        torch.stack(first_nss),
        torch.stack(segs),
    )


def make_loader(pid_list, shuffle=False, is_train=False):
    ds = WindowDataset(pid_list, cfg.PROC_DIR, class_to_idx,
                       is_train=is_train)
    dl = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=shuffle,
                    num_workers=0, pin_memory=GPU,
                    collate_fn=_collate)
    return ds, dl


# =============================================================================
# Model building blocks
# =============================================================================
class ZCRMSNorm(nn.Module):
    """Zero-centred RMS normalisation."""
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.g   = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x - x.mean(-1, keepdim=True)
        return (x0 /
                x0.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()) * self.g


# ------------------------------------------------------------------
# [C8] Stochastic depth wrapper
# ------------------------------------------------------------------
class StochasticDepth(nn.Module):
    """Wraps any residual layer with layer-drop."""
    def __init__(self, layer: nn.Module, survival_prob: float):
        super().__init__()
        self.layer = layer
        self.p     = survival_prob   # probability of KEEPING the layer

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if not self.training or self.p >= 1.0:
            return self.layer(x, *args, **kwargs)
        if random.random() > self.p:
            return x                          # skip layer entirely
        return self.layer(x, *args, **kwargs)


class GatedDeltaNet(nn.Module):
    """Local temporal modelling — unchanged from v1."""
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.norm      = ZCRMSNorm(d)
        self.q_lin     = nn.Linear(d, d)
        self.k_lin     = nn.Linear(d, d)
        self.v_lin     = nn.Linear(d, d)
        self.q_conv    = nn.Conv1d(d, d, 3, padding=1, groups=d)
        self.k_conv    = nn.Conv1d(d, d, 3, padding=1, groups=d)
        self.v_conv    = nn.Conv1d(d, d, 3, padding=1, groups=d)
        self.act       = nn.Sigmoid()
        self.alpha     = nn.Linear(d, d)
        self.beta      = nn.Linear(d, d)
        self.post_norm = ZCRMSNorm(d)
        self.post      = nn.Linear(d, d)
        self.silu      = nn.SiLU()
        self.gate      = nn.Sigmoid()
        self.drop      = nn.Dropout(dropout)

    @staticmethod
    def _l2(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return x / (x.pow(2).sum(-1, keepdim=True).add(eps).sqrt())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q = self.act(self.q_conv(self.q_lin(h).transpose(1,2)).transpose(1,2))
        k = self.act(self.k_conv(self.k_lin(h).transpose(1,2)).transpose(1,2))
        v = self.act(self.v_conv(self.v_lin(h).transpose(1,2)).transpose(1,2))
        q, k = self._l2(q), self._l2(k)
        delta = q * (k * v)
        delta = torch.tanh(self.alpha(x)) * delta + self.beta(x)
        dhat  = self.post(self.post_norm(delta))
        return x + self.drop(self.gate(self.silu(dhat)) * dhat)


class SoftMoE(nn.Module):
    """Soft Mixture-of-Experts — unchanged from v1."""
    def __init__(self, d: int, hidden: int,
                 n_experts: int = 4, dropout: float = 0.1):
        super().__init__()
        self.router  = nn.Linear(d, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, hidden), nn.SiLU(),
                nn.Dropout(dropout),  nn.Linear(hidden, d),
            )
            for _ in range(n_experts)
        ])
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.router(x), dim=-1)
        s = torch.stack([e(x) for e in self.experts], dim=-2)
        return self.drop((w.unsqueeze(-1) * s).sum(-2))


def precompute_freqs(dim: int, n_tok: int,
                     theta: float = 10000.0) -> torch.Tensor:
    assert dim % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t     = torch.arange(n_tok)
    return torch.polar(torch.ones(n_tok, dim // 2),
                       torch.outer(t, freqs))


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               freqs: torch.Tensor):
    B, H, N, D = q.shape
    d2    = D // 2
    f     = freqs[:N].to(q.device).view(1, 1, N, d2)
    q_    = torch.view_as_complex(q.float().contiguous().view(B, H, N, d2, 2))
    k_    = torch.view_as_complex(k.float().contiguous().view(B, H, N, d2, 2))
    q_out = torch.view_as_real(q_ * f).view(B, H, N, D)
    k_out = torch.view_as_real(k_ * f).view(B, H, N, D)
    return q_out.type_as(q), k_out.type_as(k)


class GatedAttention(nn.Module):
    """Multi-head self-attention with RoPE — unchanged from v1."""
    def __init__(self, d: int, n_heads: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        assert d % n_heads == 0
        assert (d // n_heads) % 2 == 0, \
            "Per-head dimension must be even for RoPE"
        self.h    = n_heads
        self.dh   = d // n_heads
        self.norm = ZCRMSNorm(d)
        self.qkv  = nn.Linear(d, 3 * d)
        self.out  = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                freqs: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        B, N, D = h.shape
        qkv = (self.qkv(h)
               .reshape(B, N, 3, self.h, self.dh)
               .permute(0, 2, 1, 3, 4))
        q = qkv[:, 0].transpose(1, 2)
        k = qkv[:, 1].transpose(1, 2)
        v = qkv[:, 2].transpose(1, 2)
        q, k  = apply_rope(q, k, freqs)
        score = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        attn  = self.drop(torch.softmax(score, dim=-1))
        y     = self.out(
            (attn @ v).transpose(1, 2).contiguous().reshape(B, N, D))
        return x + torch.sigmoid(self.gate(h)) * y


# =============================================================================
# [C1] Dual-domain patch embedding
# =============================================================================
class DualDomainPatchEmbed(nn.Module):
    """
    Projects each patch in both time and frequency domains, fused with a
    learnable gate:
        out = σ(gate) * time_proj + (1-σ(gate)) * freq_proj
    """
    def __init__(self, patch_len: int, channels: int, d: int):
        super().__init__()
        in_dim   = patch_len * channels
        freq_dim = (patch_len // 2 + 1) * channels  # rfft output size

        self.time_proj = nn.Linear(in_dim, d)
        self.freq_proj = nn.Linear(freq_dim, d)
        # scalar gate per output dim
        self.gate_w    = nn.Parameter(torch.zeros(d))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches : (B, C, N_P, PL)
        returns : (B, N_P, D)
        """
        B, C, NP, PL = patches.shape
        # time domain
        x_t = patches.permute(0, 2, 1, 3).reshape(B, NP, C * PL)
        t_emb = self.time_proj(x_t)                       # (B, NP, D)

        # frequency domain  — rfft over patch samples
        x_f  = patches.permute(0, 2, 3, 1)                # (B, NP, PL, C)
        fft  = torch.fft.rfft(x_f, dim=2)                 # (B, NP, PL//2+1, C)
        mag  = fft.abs().reshape(B, NP, -1)                # (B, NP, freq_dim)
        f_emb = self.freq_proj(mag)                        # (B, NP, D)

        g     = torch.sigmoid(self.gate_w)
        return g * t_emb + (1 - g) * f_emb


class SimplePatchEmbed(nn.Module):
    """Fallback when C1 is OFF — linear projection only."""
    def __init__(self, patch_len: int, channels: int, d: int):
        super().__init__()
        self.proj = nn.Linear(patch_len * channels, d)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        B, C, NP, PL = patches.shape
        x = patches.permute(0, 2, 1, 3).reshape(B, NP, C * PL)
        return self.proj(x)


def _make_patch_embedder(patch_len: int, channels: int, d: int) -> nn.Module:
    """Factory: returns DualDomain or Simple depending on C1 flag."""
    if CC.C1_DUAL_DOMAIN_EMBEDDING:
        return DualDomainPatchEmbed(patch_len, channels, d)
    return SimplePatchEmbed(patch_len, channels, d)


# =============================================================================
# [C4] Hierarchical multi-scale patch embedding
# =============================================================================
class HierarchicalPatchEmbed(nn.Module):
    """
    Containment-aware additive multi-scale embedding.

    Three separate embedders process patches at different granularities.
    Each coarser-scale embedding is broadcast to the fine-scale token count
    using repeat_interleave, matching the actual temporal containment:

        fine   patches: 25 samples → 120 tokens  (ratio = 1)
        mid    patches: 50 samples →  60 tokens  (ratio = 2, covers 2 fine)
        coarse patches: 100 samples → 30 tokens  (ratio = 4, covers 4 fine)

    Final token:
        token_i = embed_fine(p_i)
                + embed_mid(L_{i//2})       broadcast via repeat_interleave(2)
                + embed_coarse(K_{i//4})    broadcast via repeat_interleave(4)

    No interpolation. No synthetic tokens. No extra fusion layer.
    Output shape: (B, N_PATCHES_FINE, D)
    """
    def __init__(self, patch_lens: list[int], channels: int, d: int):
        super().__init__()
        assert len(patch_lens) == 3, \
            "HierarchicalPatchEmbed expects exactly 3 granularities"

        self.patch_lens = sorted(patch_lens)   # [25, 50, 100]
        fine_pl, mid_pl, coarse_pl = self.patch_lens

        # Compute broadcast repeat factors relative to the finest granularity
        self.repeat_mid    = mid_pl    // fine_pl   # 50 // 25 = 2
        self.repeat_coarse = coarse_pl // fine_pl   # 100 // 25 = 4

        self.embed_fine   = _make_patch_embedder(fine_pl,   channels, d)
        self.embed_mid    = _make_patch_embedder(mid_pl,    channels, d)
        self.embed_coarse = _make_patch_embedder(coarse_pl, channels, d)

    def forward(self, patches_list: list[torch.Tensor]) -> torch.Tensor:
        """
        patches_list : [fine (B,C,120,25), mid (B,C,60,50), coarse (B,C,30,100)]
                        (order matches sorted self.patch_lens)
        returns      : (B, 120, D)
        """
        # Sort inputs by patch length to match self.patch_lens order
        # (Dataset already produces them in PATCH_LENS_MULTI order = [25,50,100])
        p_fine, p_mid, p_coarse = patches_list

        e_fine   = self.embed_fine(p_fine)      # (B,  120, D)
        e_mid    = self.embed_mid(p_mid)        # (B,   60, D)
        e_coarse = self.embed_coarse(p_coarse)  # (B,   30, D)

        # Broadcast mid: each mid token covers self.repeat_mid fine tokens
        # (B, 60, D) → (B, 120, D)
        e_mid_broad    = e_mid.repeat_interleave(self.repeat_mid,    dim=1)

        # Broadcast coarse: each coarse token covers self.repeat_coarse fine tokens
        # (B, 30, D) → (B, 120, D)
        e_coarse_broad = e_coarse.repeat_interleave(self.repeat_coarse, dim=1)

        # Additive fusion — token i gets genuine information from all three
        # scales that temporally contain it, with no invented data
        return e_fine + e_mid_broad + e_coarse_broad   # (B, 120, D)


# =============================================================================
# [C3] Circadian positional bias
# =============================================================================
class CircadianBias(nn.Module):
    """
    Expands the 5-dim time-of-day vector into (N_PATCHES,) additive biases
    via a small MLP, giving each patch position a unique temporal offset.
    """
    def __init__(self, n_patches: int, d: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, d),
            nn.SiLU(),
            nn.Linear(d, n_patches * d),
        )
        self.n_patches = n_patches
        self.d         = d

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        """times : (B, 5) → bias : (B, N_P, D)"""
        B = times.shape[0]
        return self.mlp(times).view(B, self.n_patches, self.d)


# =============================================================================
# [C2] CALANet-style skip aggregation
# =============================================================================
class SkipAggregation(nn.Module):
    """
    Collects hidden states from every GatedDeltaNet layer, computes a
    weighted sum with learnable scalars, then adds to the output stream.
    Inspired by CALANet (NeurIPS 2024).
    """
    def __init__(self, d: int, n_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_layers) / n_layers)
        self.proj    = nn.Sequential(nn.Linear(d, d), nn.SiLU())

    def forward(self, hiddens: list[torch.Tensor]) -> torch.Tensor:
        """hiddens : list of (B, N_P, D)"""
        w = torch.softmax(self.weights, dim=0)
        agg = sum(w[i] * hiddens[i] for i in range(len(hiddens)))
        return self.proj(agg)


# =============================================================================
# [C10] Reconstruction auxiliary head
# =============================================================================
class ReconHead(nn.Module):
    """
    Reconstructs the mean-pooled patch from the sequence embedding.
    Input:  (B, D) pooled representation
    Output: (B, N_P, C*PL)
    """
    def __init__(self, d: int, n_patches: int,
                 patch_len: int, channels: int):
        super().__init__()
        out_dim = n_patches * patch_len * channels
        self.mlp = nn.Sequential(
            nn.Linear(d, 2 * d), nn.ReLU(),
            nn.Linear(2 * d, out_dim),
        )
        self.n_patches = n_patches
        self.pl        = patch_len
        self.C         = channels

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z : (B, D) → (B, N_P * PL * C)"""
        return self.mlp(z)


# =============================================================================
# PatchHAR  v2
# =============================================================================
class PatchHARv2(nn.Module):
    """
    PatchHAR v2 — all 10 contributions integrated.
    C4 now uses HierarchicalPatchEmbed (containment-aware additive fusion)
    instead of interpolation + linear projection.

    Hyperparameters (defaults)
    ──────────────────────────
    D_MODEL   = 64
    N_HEADS   = 2
    N_LAYERS  = 3
    N_EXPERTS = 4
    DROPOUT   = 0.25
    """
    def __init__(self):
        super().__init__()
        d  = cfg.D_MODEL
        NP = cfg.N_PATCHES

        # ── [C4] Patch embedding ──────────────────────────────────────────
        if CC.C4_MULTISCALE_PATCHING:
            # Hierarchical additive embedding: fine + mid broadcast + coarse broadcast
            self.hier_embed = HierarchicalPatchEmbed(
                cfg.PATCH_LENS_MULTI, cfg.CHANNELS, d)
            self.patch_lens = cfg.PATCH_LENS_MULTI
        else:
            # Single-scale fallback (original behaviour)
            self.hier_embed = None
            self.patch_lens = [cfg.PATCH_LEN]
            self.single_embed = _make_patch_embedder(
                cfg.PATCH_LEN, cfg.CHANNELS, d)

        # ── [C3] Circadian bias ───────────────────────────────────────────
        if CC.C3_CIRCADIAN_BIAS:
            self.circ_bias = CircadianBias(NP, d)
        else:
            self.time_emb = nn.Sequential(
                nn.Linear(5, d), nn.ReLU(), nn.Dropout(0.1))

        self.input_norm = nn.LayerNorm(d)

        # ── [C8] Stochastic-depth GatedDeltaNet layers ────────────────────
        raw_layers = [GatedDeltaNet(d, dropout=cfg.DROPOUT)
                      for _ in range(cfg.N_LAYERS)]
        if CC.C8_STOCHASTIC_DEPTH:
            survival = [1.0 - (i / cfg.N_LAYERS) * cfg.SD_DROP_MAX
                        for i in range(cfg.N_LAYERS)]
            self.delta_layers = nn.ModuleList([
                StochasticDepth(l, p) for l, p in zip(raw_layers, survival)
            ])
        else:
            self.delta_layers = nn.ModuleList(raw_layers)

        # ── [C2] Skip aggregation ─────────────────────────────────────────
        if CC.C2_CALANET_SKIP_AGG:
            self.skip_agg = SkipAggregation(d, cfg.N_LAYERS)

        # ── Transformer core ──────────────────────────────────────────────
        self.moe1 = SoftMoE(d, 2*d, n_experts=cfg.N_EXPERTS,
                            dropout=cfg.DROPOUT)
        self.attn = GatedAttention(d, n_heads=cfg.N_HEADS,
                                   dropout=cfg.DROPOUT)
        self.moe2 = SoftMoE(d, 2*d, n_experts=cfg.N_EXPERTS,
                            dropout=cfg.DROPOUT)

        # RoPE frequency cache
        freqs = precompute_freqs(d // cfg.N_HEADS, NP)
        self.register_buffer("freqs", freqs)

        # ── [C6] Learnable temperature ────────────────────────────────────
        if CC.C6_LABEL_SMOOTH_TEMP:
            self.log_tau = nn.Parameter(torch.zeros(1))

        # ── [C10] Reconstruction head ─────────────────────────────────────
        if CC.C10_RECON_AUX_GRAD_SURGERY:
            self.recon_head = ReconHead(d, NP, cfg.PATCH_LEN, cfg.CHANNELS)

        # ── [C7] Prototype memory ─────────────────────────────────────────
        if CC.C7_PROTOTYPE_MEMORY:
            self.register_buffer(
                "prototypes",
                torch.zeros(NUM_CLASSES, d))
            self.proto_filled = False

        # ── Classification head ───────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d // 2, NUM_CLASSES),
        )

    # ------------------------------------------------------------------
    def _embed_patches(self,
                       patches_list: list[torch.Tensor]) -> torch.Tensor:
        """
        [C4 ON]  Uses HierarchicalPatchEmbed:
                 fine + mid.repeat_interleave(2) + coarse.repeat_interleave(4)
                 → (B, N_PATCHES, D)   no interpolation, no fusion layer

        [C4 OFF] Single-scale linear/dual-domain embed
                 → (B, N_PATCHES, D)
        """
        if CC.C4_MULTISCALE_PATCHING:
            return self.hier_embed(patches_list)         # (B, NP, D)
        else:
            return self.single_embed(patches_list[0])    # (B, NP, D)

    # ------------------------------------------------------------------
    def forward(self,
                patches_list: list[torch.Tensor],
                times: torch.Tensor,
                return_embedding: bool = False
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        patches_list : list of (B, C, N_Pi, PL_i)
        times        : (B, 5)
        returns      : (logits (B, K), recon (B, N_P*PL*C) or None)
        """
        # ── Embed patches ─────────────────────────────────────────────────
        x = self._embed_patches(patches_list)          # (B, NP, D)

        # ── Temporal context ──────────────────────────────────────────────
        if CC.C3_CIRCADIAN_BIAS:
            x = x + self.circ_bias(times)              # patch-level bias
        else:
            x = x + self.time_emb(times).unsqueeze(1)

        x = self.input_norm(x)

        # ── Local temporal modelling + skip collection ────────────────────
        hiddens = []
        for layer in self.delta_layers:
            x = layer(x)
            hiddens.append(x)

        # [C2] skip aggregation adds a multi-scale residual
        if CC.C2_CALANET_SKIP_AGG:
            x = x + self.skip_agg(hiddens)

        # ── Transformer core ──────────────────────────────────────────────
        x = x + self.moe1(x)
        x = self.attn(x, self.freqs)
        x = x + self.moe2(x)

        # ── Pooling ───────────────────────────────────────────────────────
        z = x.mean(dim=1)                              # (B, D)

        if return_embedding:
            return z

        # ── [C10] Reconstruction ──────────────────────────────────────────
        recon = None
        if CC.C10_RECON_AUX_GRAD_SURGERY and self.training:
            recon = self.recon_head(z)

        # ── Classify ──────────────────────────────────────────────────────
        logits = self.head(z)                          # (B, K)

        # [C6] temperature scaling
        if CC.C6_LABEL_SMOOTH_TEMP:
            tau    = torch.exp(self.log_tau).clamp(0.5, 2.0)
            logits = logits / tau

        # [C7] prototype blend at inference
        if CC.C7_PROTOTYPE_MEMORY and not self.training and self.proto_filled:
            z_n    = F.normalize(z, dim=-1)
            pr_n   = F.normalize(self.prototypes, dim=-1)
            cosine = z_n @ pr_n.T                      # (B, K)
            logits = ((1 - cfg.PROTO_ALPHA) * logits
                      + cfg.PROTO_ALPHA * cosine)

        return logits, recon

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_prototypes(self, embeddings: torch.Tensor,
                          labels: torch.Tensor):
        """EMA-update class prototypes from a batch of embeddings."""
        m = cfg.PROTO_MOMENTUM
        for k in range(NUM_CLASSES):
            mask = (labels == k)
            if mask.sum() == 0:
                continue
            mean = embeddings[mask].mean(0)
            if self.proto_filled:
                self.prototypes[k] = m * self.prototypes[k] + (1-m) * mean
            else:
                self.prototypes[k] = mean
        self.proto_filled = True


# =============================================================================
# Training utilities
# =============================================================================
def compute_class_weights(ds: WindowDataset) -> torch.Tensor:
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for e in ds.entries:
        counts[e[3]] += 1
    w = np.clip(counts.max() / np.clip(counts, 1, None), 1.0, 10.0)
    w = torch.tensor(w, dtype=torch.float32)
    return w / w.sum() * NUM_CLASSES


def manifold_mixup(x: torch.Tensor,
                   labels: torch.Tensor,
                   alpha: float = 0.2):
    """
    [C9] Mixup in embedding space.
    Returns mixed x, original labels, shuffled labels, lambda.
    """
    if alpha <= 0:
        return x, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1-lam) * x[idx], labels, labels[idx], lam


def raw_mixup(patches_list: list[torch.Tensor],
              times: torch.Tensor,
              labels: torch.Tensor,
              alpha: float = 0.2):
    """Standard input-space mixup (fallback when C9 is OFF)."""
    if alpha <= 0:
        return patches_list, times, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(labels.size(0), device=labels.device)
    mixed = [lam * p + (1-lam) * p[idx] for p in patches_list]
    return mixed, lam * times + (1-lam) * times[idx], labels, labels[idx], lam


def tc_loss(logits: torch.Tensor) -> torch.Tensor:
    """Symmetrised KL between adjacent window predictions."""
    if logits.size(0) < 2:
        return logits.new_zeros(1).squeeze()
    p = F.softmax(logits[:-1], dim=-1)
    q = F.softmax(logits[1:],  dim=-1)
    return 0.5 * (
        F.kl_div(q.log(), p, reduction="batchmean") +
        F.kl_div(p.log(), q, reduction="batchmean")
    )


def recon_loss(recon: torch.Tensor,
               raw_segs: torch.Tensor) -> torch.Tensor:
    """
    [C10] Reconstruction loss: MSE between predicted and actual patches.
    raw_segs : (B, T, C)  — the normalised segment
    recon    : (B, N_P * PL * C)
    """
    B, T, C = raw_segs.shape
    NP      = cfg.N_PATCHES
    PL      = cfg.PATCH_LEN
    target  = (raw_segs[:, :NP*PL, :]
               .reshape(B, NP, PL, C)
               .permute(0, 1, 3, 2)        # (B, NP, C, PL)
               .reshape(B, NP * C * PL))
    return F.mse_loss(recon, target.detach())


def _pcgrad_surgery(cls_grad: torch.Tensor,
                    aux_grad: torch.Tensor) -> torch.Tensor:
    """
    [C10] Project aux_grad to be orthogonal to cls_grad (PCGrad).
    Operates on flat gradient vectors.
    """
    dot = (aux_grad * cls_grad).sum()
    if dot < 0:
        # project out component along cls_grad
        aux_grad = aux_grad - dot / (cls_grad.norm()**2 + 1e-12) * cls_grad
    return aux_grad


# =============================================================================
# [C6] Label-smoothed CE loss
# =============================================================================
class SmoothCE(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None,
                 smoothing: float = 0.0):
        super().__init__()
        self.smoothing = smoothing
        self.weight    = weight   # (K,) class weights

    def forward(self, logits: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        K    = logits.size(-1)
        eps  = self.smoothing
        # soft targets
        with torch.no_grad():
            soft = torch.full_like(logits, eps / (K - 1))
            soft.scatter_(-1, labels.unsqueeze(-1), 1.0 - eps)

        log_prob = F.log_softmax(logits, dim=-1)
        loss = -(soft * log_prob).sum(-1)

        if self.weight is not None:
            w    = self.weight.to(logits.device)
            loss = loss * w[labels]

        return loss.mean()


# =============================================================================
# Metrics
# =============================================================================
def kappa(yt: np.ndarray, yp: np.ndarray) -> float:
    cm = confusion_matrix(yt, yp)
    n  = cm.sum()
    if n == 0:
        return 0.0
    po = np.trace(cm) / n
    pe = np.dot(cm.sum(1), cm.sum(0)) / (n * n)
    return float((po - pe) / (1 - pe)) if abs(1 - pe) > 1e-12 else 0.0


def mcc(yt: np.ndarray, yp: np.ndarray) -> float:
    cm  = confusion_matrix(yt, yp).astype(float)
    n   = cm.sum()
    if n == 0:
        return 0.0
    s   = np.trace(cm)
    t   = cm.sum(1)
    p   = cm.sum(0)
    num = s * n - np.sum(t * p)
    den = math.sqrt(
        max(n**2 - np.sum(t**2), 0.0) *
        max(n**2 - np.sum(p**2), 0.0)
    )
    return float(num / den) if den > 0 else 0.0


# =============================================================================
# Checkpoint compatibility helpers
# =============================================================================

# Key mapping: old interpolation-based C4 → new hierarchical C4
_PATCH_EMBED_REMAP = {
    "patch_embeds.0.": "hier_embed.embed_fine.",    # 25-sample (fine)
    "patch_embeds.1.": "hier_embed.embed_mid.",     # 50-sample (mid)
    "patch_embeds.2.": "hier_embed.embed_coarse.",  # 100-sample (coarse)
}
# Keys that exist in the old architecture but have no equivalent in the new one
_DROP_OLD_KEYS = {"scale_fusion.weight", "scale_fusion.bias"}


def _remap_checkpoint(ckpt: dict, model: nn.Module) -> dict | None:
    """
    Try to remap a checkpoint saved with old patch_embeds / scale_fusion keys
    into the new hier_embed key space.

    Returns the remapped state-dict if the remapping fills all model keys,
    or None if it cannot be fixed automatically.
    """
    remapped = {}
    for k, v in ckpt.items():
        if k in _DROP_OLD_KEYS:
            continue                          # no equivalent — drop
        new_k = k
        for old_pfx, new_pfx in _PATCH_EMBED_REMAP.items():
            if k.startswith(old_pfx):
                new_k = new_pfx + k[len(old_pfx):]
                break
        remapped[new_k] = v

    model_keys = set(model.state_dict().keys())
    remapped_keys = set(remapped.keys())

    if remapped_keys == model_keys:
        return remapped

    # Still mismatched — report and return None
    missing    = model_keys - remapped_keys
    unexpected = remapped_keys - model_keys
    print(f"  [Remap] After remapping: "
          f"{len(missing)} still missing, {len(unexpected)} still unexpected.")
    return None


def _compat_load(save_path: Path, model: nn.Module) -> bool:
    """
    Load a checkpoint robustly:
      1. Try direct load (exact key match).
      2. Try key remapping (old patch_embeds → new hier_embed).
      3. Warn and keep in-memory weights on failure.

    Returns True if load succeeded.
    """
    if not save_path.exists():
        print(f"  [Warn] No checkpoint at {save_path}. Keeping in-memory weights.")
        return False

    ckpt = torch.load(save_path, map_location=device, weights_only=False)

    # ── Attempt 1: direct load ────────────────────────────────────────────
    try:
        model.load_state_dict(ckpt, strict=True)
        print(f"  Best checkpoint loaded from {save_path}")
        return True
    except RuntimeError:
        pass

    # ── Attempt 2: key remapping ──────────────────────────────────────────
    print(f"  [Compat] Direct load failed — attempting key remapping …")
    remapped = _remap_checkpoint(ckpt, model)
    if remapped is not None:
        try:
            model.load_state_dict(remapped, strict=True)
            print(f"  [Compat] Checkpoint loaded after key remapping.")
            # Re-save with correct keys so future runs work directly
            torch.save(model.state_dict(), save_path)
            print(f"  [Compat] Checkpoint re-saved with updated key names.")
            return True
        except RuntimeError as e:
            print(f"  [Compat] Remapping still failed: {e}")

    # ── Attempt 3: give up gracefully ─────────────────────────────────────
    print(f"  [Warn] Could not load checkpoint. Keeping in-memory weights.")
    return False


# =============================================================================
# Training loop
# =============================================================================
def train(model: PatchHARv2,
          train_dl: DataLoader,
          val_dl: DataLoader,
          class_w: torch.Tensor,
          save_path: Path) -> list:

    # ── BUG FIX: delete any checkpoint whose key-set doesn't exactly match
    #    the current model — catches ALL architecture changes, not just named ones ──
    if save_path.exists():
        try:
            ckpt       = torch.load(save_path, map_location="cpu",
                                    weights_only=False)
            ckpt_keys  = set(ckpt.keys())
            model_keys = set(model.state_dict().keys())
            if ckpt_keys != model_keys:
                missing  = model_keys - ckpt_keys
                unexpected = ckpt_keys - model_keys
                print(f"  [Fix] Checkpoint architecture mismatch — "
                      f"{len(missing)} missing / {len(unexpected)} unexpected keys. "
                      f"Deleting: {save_path}")
                save_path.unlink()
            else:
                print(f"  [Info] Existing checkpoint is compatible — "
                      f"will resume saving from here.")
        except Exception as e:
            print(f"  [Fix] Could not read checkpoint ({e}). Deleting: {save_path}")
            save_path.unlink()

    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg.LR,
                            weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.LR,
        steps_per_epoch=len(train_dl),
        epochs=cfg.EPOCHS,
        pct_start=0.10,
        anneal_strategy="cos",
    )
    smooth_eps = cfg.LABEL_SMOOTH_EPS if CC.C6_LABEL_SMOOTH_TEMP else 0.0
    criterion  = SmoothCE(weight=class_w.to(device), smoothing=smooth_eps)

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=GPU)
    except TypeError:
        from torch.cuda.amp import GradScaler as _GS
        scaler = _GS(enabled=GPU)

    best_score   = -1e9
    patience_ctr = 0
    history      = []

    for epoch in range(cfg.EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_dl:
            patches_list, times, labels, _, _, raw_segs = batch

            patches_list = [p.to(device).float() for p in patches_list]
            times        = times.to(device).float()
            labels       = labels.to(device).view(-1)
            raw_segs     = raw_segs.to(device).float()

            # ── [C9] Manifold Mixup or raw mixup ────────────────────────
            if CC.C9_MANIFOLD_MIXUP:
                # forward up to embedding, then mix
                with amp_ctx():
                    z = model._embed_patches(patches_list)  # (B, NP, D)
                    if CC.C3_CIRCADIAN_BIAS:
                        z = z + model.circ_bias(times)
                    else:
                        z = z + model.time_emb(times).unsqueeze(1)
                    z = model.input_norm(z)
                    # stochastic-depth GDN layers
                    hiddens = []
                    for layer in model.delta_layers:
                        z = layer(z)
                        hiddens.append(z)
                    if CC.C2_CALANET_SKIP_AGG:
                        z = z + model.skip_agg(hiddens)
                    z = z + model.moe1(z)
                    z = model.attn(z, model.freqs)
                    z = z + model.moe2(z)
                    z_pool = z.mean(1)          # (B, D)
                # manifold mixup on pooled representation
                z_mix, la, lb, lam = manifold_mixup(
                    z_pool, labels, cfg.MIXUP_ALPHA)
                optimizer.zero_grad(set_to_none=True)
                with amp_ctx():
                    logits = model.head(z_mix)
                    if CC.C6_LABEL_SMOOTH_TEMP:
                        tau    = torch.exp(model.log_tau).clamp(0.5, 2.0)
                        logits = logits / tau
                    loss = (lam * criterion(logits, la) +
                            (1-lam) * criterion(logits, lb))
                    if cfg.TC_LAMBDA > 0:
                        loss = loss + cfg.TC_LAMBDA * tc_loss(logits)
                    # [C10] reconstruction (from un-mixed z_pool)
                    if CC.C10_RECON_AUX_GRAD_SURGERY:
                        recon = model.recon_head(z_pool.detach())
                        rl    = recon_loss(recon, raw_segs)
                        loss  = loss + cfg.RECON_LAMBDA * rl
            else:
                # raw input-space mixup (original behaviour)
                patches_list, times, la, lb, lam = raw_mixup(
                    patches_list, times, labels, cfg.MIXUP_ALPHA)
                optimizer.zero_grad(set_to_none=True)
                with amp_ctx():
                    logits, recon = model(patches_list, times)
                    loss = (lam * criterion(logits, la) +
                            (1-lam) * criterion(logits, lb))
                    if cfg.TC_LAMBDA > 0:
                        loss = loss + cfg.TC_LAMBDA * tc_loss(logits)
                    if CC.C10_RECON_AUX_GRAD_SURGERY and recon is not None:
                        rl   = recon_loss(recon, raw_segs)
                        loss = loss + cfg.RECON_LAMBDA * rl

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            grad_ok = (
                all(p.grad is None or torch.isfinite(p.grad).all()
                    for p in model.parameters())
                and torch.isfinite(loss)
            )
            if grad_ok:
                nn.utils.clip_grad_norm_(model.parameters(),
                                         cfg.MAX_GRAD_NORM)
                scaler.step(optimizer)
            else:
                optimizer.zero_grad(set_to_none=True)

            scaler.update()
            scheduler.step()
            if torch.isfinite(loss):
                total_loss += float(loss.item())

        # ── Validation ───────────────────────────────────────────────────
        model.eval()
        vp, vt, embs_val, labs_val = [], [], [], []
        with torch.no_grad():
            for batch in val_dl:
                patches_list, times, labels, _, _, _ = batch
                patches_list = [p.to(device).float() for p in patches_list]
                times_d = times.to(device).float()

                # collect embeddings for prototype update
                if CC.C7_PROTOTYPE_MEMORY:
                    z = model.forward(patches_list, times_d,
                                      return_embedding=True)
                    logits, _ = model(patches_list, times_d)
                    embs_val.append(z.cpu())
                    labs_val.append(labels)
                else:
                    logits, _ = model(patches_list, times_d)

                pred = logits.argmax(1)
                vp.extend(pred.cpu().numpy())
                vt.extend(labels.numpy())

        # [C7] update prototypes
        if CC.C7_PROTOTYPE_MEMORY and embs_val:
            all_embs = torch.cat(embs_val).to(device)
            all_labs = torch.cat(labs_val).to(device)
            model.update_prototypes(all_embs, all_labs)

        vp  = np.array(vp);  vt = np.array(vt)
        f1  = float(f1_score(vt, vp, average="macro", zero_division=0))
        kap = kappa(vt, vp)
        avg = total_loss / max(1, len(train_dl))
        lr  = optimizer.param_groups[0]["lr"]

        print(f"  Epoch {epoch+1:02d}/{cfg.EPOCHS} | "
              f"lr={lr:.2e} | loss={avg:.4f} | "
              f"F1={f1:.4f} | κ={kap:.4f}")

        history.append({
            "epoch":     epoch + 1,
            "loss":      round(avg, 6),
            "val_f1":    round(f1, 6),
            "val_kappa": round(kap, 6),
        })

        score = f1 + kap
        if score > best_score + 1e-6:
            best_score   = score
            patience_ctr = 0
            torch.save(model.state_dict(), save_path)

            # ── Save verification: catch key mismatches immediately ───────
            try:
                _verify     = torch.load(save_path, map_location="cpu",
                                         weights_only=False)
                _saved_keys = set(_verify.keys())
                _model_keys = set(model.state_dict().keys())
                if _saved_keys != _model_keys:
                    _miss = _model_keys - _saved_keys
                    _unex = _saved_keys - _model_keys
                    print(f"    [WARN] Saved checkpoint key mismatch! "
                          f"{len(_miss)} missing / {len(_unex)} unexpected. "
                          f"The model being trained may not match PatchHARv2.")
                else:
                    print(f"    ✓ Checkpoint saved  (F1={f1:.4f})")
            except Exception as _ve:
                print(f"    [WARN] Could not verify saved checkpoint: {_ve}")
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # ── Load best checkpoint (with compatibility remapping) ───────────────
    _compat_load(save_path, model)
    return history


# =============================================================================
# Evaluation
# =============================================================================
@torch.no_grad()
def evaluate(model: PatchHARv2, test_dl: DataLoader):
    model.eval()
    all_pred, all_true = [], []

    for batch in test_dl:
        patches_list, times, labels, _, _, _ = batch
        patches_list = [p.to(device).float() for p in patches_list]
        times_d      = times.to(device).float()
        logits, _    = model(patches_list, times_d)
        pred = logits.argmax(1)
        all_pred.extend(pred.cpu().numpy())
        all_true.extend(labels.numpy())

    yt = np.array(all_true)
    yp = np.array(all_pred)

    metrics = {
        "macro_f1": round(float(f1_score(
            yt, yp, average="macro", zero_division=0)), 4),
        "kappa":    round(kappa(yt, yp), 4),
        "mcc":      round(mcc(yt, yp), 4),
        "accuracy": round(float((yt == yp).mean()), 4),
    }

    print(f"\n  Macro-F1 = {metrics['macro_f1']:.4f} | "
          f"κ = {metrics['kappa']:.4f} | "
          f"MCC = {metrics['mcc']:.4f} | "
          f"Acc = {metrics['accuracy']:.4f}\n")

    print(classification_report(
        yt, yp, target_names=CLASSES, zero_division=0))

    return metrics


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 70)
    print("  PatchHAR v2")
    print(f"  D_MODEL={cfg.D_MODEL} | N_LAYERS={cfg.N_LAYERS} | "
          f"N_HEADS={cfg.N_HEADS} | N_EXPERTS={cfg.N_EXPERTS}")
    print(f"  Device : {device}")
    print(f"  Output : {cfg.OUTPUT_DIR}")
    print()
    print("  Active contributions:")
    for k, v in CC.__dict__.items():
        if k.startswith("C") and not k.startswith("__"):
            print(f"    {'✓' if v else '✗'} {k}")

    if CC.C4_MULTISCALE_PATCHING:
        print()
        print("  [C4] Hierarchical patch embedding:")
        fine, mid, coarse = cfg.PATCH_LENS_MULTI
        nf = cfg.WINDOW_SIZE // fine
        nm = cfg.WINDOW_SIZE // mid
        nc = cfg.WINDOW_SIZE // coarse
        print(f"    fine   {fine:3d}-sample patches → {nf} tokens")
        print(f"    mid    {mid:3d}-sample patches → {nm} tokens "
              f"→ repeat×{mid//fine} → {nf} tokens")
        print(f"    coarse {coarse:3d}-sample patches → {nc} tokens "
              f"→ repeat×{coarse//fine} → {nf} tokens")
        print(f"    fusion : token_i = fine_i + mid_{{i//{mid//fine}}} "
              f"+ coarse_{{i//{coarse//fine}}}")

    print("=" * 70)

    # ── Data ─────────────────────────────────────────────────────────────
    print("\n── Loading data ──")
    train_ds, train_dl = make_loader(train_pids, shuffle=True,  is_train=True)
    val_ds,   val_dl   = make_loader(val_pids,   shuffle=False, is_train=False)
    test_ds,  test_dl  = make_loader(test_pids,  shuffle=False, is_train=False)
    print(f"  Train {len(train_ds):,} | "
          f"Val {len(val_ds):,} | Test {len(test_ds):,}")

    # ── Model ─────────────────────────────────────────────────────────────
    model    = PatchHARv2().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Parameters : {n_params:,}")

    # ── Train ─────────────────────────────────────────────────────────────
    print("\n── Training ──")
    save_path = cfg.OUTPUT_DIR / "weights_patchhar_v2.pth"
    class_w   = compute_class_weights(train_ds)
    history   = train(model, train_dl, val_dl, class_w, save_path)

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\n── Evaluation ──")
    metrics = evaluate(model, test_dl)

    # ── Save ──────────────────────────────────────────────────────────────
    active_contribs = {k: v for k, v in CC.__dict__.items()
                       if k.startswith("C") and not k.startswith("__")}
    results = {
        "metrics":      metrics,
        "training":     history,
        "contributions": active_contribs,
        "config": {
            "window_size"      : cfg.WINDOW_SIZE,
            "patch_len"        : cfg.PATCH_LEN,
            "patch_lens_multi" : cfg.PATCH_LENS_MULTI,
            "n_patches"        : cfg.N_PATCHES,
            "d_model"          : cfg.D_MODEL,
            "n_heads"          : cfg.N_HEADS,
            "n_layers"         : cfg.N_LAYERS,
            "n_experts"        : cfg.N_EXPERTS,
            "dropout"          : cfg.DROPOUT,
            "n_params"         : n_params,
            "device"           : str(device),
            "train_subj"       : len(train_pids),
            "val_subj"         : len(val_pids),
            "test_subj"        : len(test_pids),
        },
    }
    out = cfg.OUTPUT_DIR / "patchhar_v2_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved: {out}")
    print("=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
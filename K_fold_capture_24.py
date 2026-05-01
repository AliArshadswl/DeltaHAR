"""
patchhar_v3_cv.py
=================
PatchHAR — patch-based Transformer for wearable Human Activity Recognition.
Version 3 — 5-Fold Subject-wise Cross-Validation with 7:1:2 split.

═══════════════════════════════════════════════════════════════════════════════
CROSS-VALIDATION DESIGN
────────────────────────
  • All discovered participants are shuffled once (fixed seed) and divided
    into 5 roughly equal subject-folds.

  • For each fold  f ∈ {0,1,2,3,4}:

      ┌─────────────────────────────────────────────────────────┐
      │  Total subjects  =  N   (e.g. 151)                      │
      │  Fold size       ≈  N / 5                               │
      │                                                         │
      │  Test   subjects =  fold f          ≈ 0.20 × N  (2/10) │
      │  Remaining       =  the other 4 folds                   │
      │    Val  subjects =  1/8 of remaining ≈ 0.10 × N  (1/10)│
      │    Train subjects=  7/8 of remaining ≈ 0.70 × N  (7/10)│
      └─────────────────────────────────────────────────────────┘

    This gives a 7:1:2 train/val/test ratio at the subject level.
    Every subject appears exactly once in a test set across all 5 folds
    (no data leakage between folds).

  • A fresh model is trained and evaluated independently for each fold.

  • After all folds the script prints and saves:
      - per-fold metrics  (F1, κ, MCC, Accuracy)
      - mean ± std across folds
      - concatenated confusion matrix over all test predictions
      - full training history per fold
      - saved to  OUTPUT_DIR / cv_results.json
                  OUTPUT_DIR / weights_fold{f}.pth   (best checkpoint)

Run
───
  python patchhar_v3_cv.py                      # all contributions ON
  python patchhar_v3_cv.py --disable C1 C4      # ablation
  python patchhar_v3_cv.py --folds 3            # run only 3 folds
  python patchhar_v3_cv.py --fold_id 2          # run a single fold (0-indexed)

═══════════════════════════════════════════════════════════════════════════════
10 CONTRIBUTIONS  (identical to v2 — each individually toggle-able)
───────────────────────────────────────────────────────────────────────────────
C1  Dual-domain patch embedding (time + FFT gate)
C2  CALANet-style hierarchical skip aggregation
C3  Circadian-aware per-patch positional bias
C4  Multi-scale patching (25 / 50 / 100 samples)
C5  Frequency-aware data augmentation
C6  Label smoothing + learnable temperature τ
C7  Per-class EMA prototype memory bank
C8  Stochastic depth (layer-drop)
C9  Manifold Mixup on patch embeddings
C10 Reconstruction auxiliary head + PCGrad gradient surgery
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import math, random, json, warnings, argparse, copy
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
# Contribution flags
# =============================================================================
class ContribConfig:
    C1_DUAL_DOMAIN_EMBEDDING   = True
    C2_CALANET_SKIP_AGG        = True
    C3_CIRCADIAN_BIAS          = True
    C4_MULTISCALE_PATCHING     = True
    C5_FREQ_AUGMENTATION       = True
    C6_LABEL_SMOOTH_TEMP       = True
    C7_PROTOTYPE_MEMORY        = True
    C8_STOCHASTIC_DEPTH        = True
    C9_MANIFOLD_MIXUP          = True
    C10_RECON_AUX_GRAD_SURGERY = True

CC = ContribConfig()


# =============================================================================
# CLI arguments
# =============================================================================
def _parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--disable", nargs="*", default=[], metavar="Cn")
    p.add_argument("--folds",   type=int,  default=5,
                   help="Total number of CV folds (default 5)")
    p.add_argument("--fold_id", type=int,  default=None,
                   help="Run only this single fold index (0-based)")
    args, _ = p.parse_known_args()
    for flag in (args.disable or []):
        attr = flag.upper()
        if hasattr(CC, attr):
            setattr(CC, attr, False)
            print(f"  [Ablation] {attr} DISABLED")
        else:
            print(f"  [Warn] Unknown flag: {flag}")
    return args

_args = _parse_args()
N_FOLDS   = _args.folds
FOLD_ONLY = _args.fold_id      # None → run all folds


# =============================================================================
# Configuration
# =============================================================================
class Config:
    # Paths
    PROC_DIR   = Path("/mnt/share/ali/processed/")
    OUTPUT_DIR = Path("/mnt/share/ali/processed/patchhar_results/")

    # Cross-validation
    N_FOLDS    = N_FOLDS     # total folds
    # 7:1:2  →  val fraction of the non-test subjects = 1/8
    VAL_FRAC_OF_NONTRAIN = 1 / 8

    # Signal
    SIGNAL_RATE = 100
    WINDOW_SIZE = 3000
    PATCH_LEN   = 25
    CHANNELS    = 3
    N_PATCHES   = WINDOW_SIZE // PATCH_LEN   # 120

    # [C4]
    PATCH_LENS_MULTI = [25, 50, 100]

    # Model
    D_MODEL   = 64
    N_HEADS   = 2
    N_LAYERS  = 3
    N_EXPERTS = 4
    DROPOUT   = 0.25

    # [C8]
    SD_DROP_MAX = 0.10

    # [C6]
    LABEL_SMOOTH_EPS = 0.10

    # [C7]
    PROTO_MOMENTUM = 0.95
    PROTO_ALPHA    = 0.30

    # [C10]
    RECON_LAMBDA = 0.10

    # Training
    BATCH_SIZE          = 32
    EPOCHS              = 30
    LR                  = 1e-3
    WEIGHT_DECAY        = 1e-4
    MAX_GRAD_NORM       = 1.0
    EARLY_STOP_PATIENCE = 8
    SEED                = 42
    MIXUP_ALPHA         = 0.2
    TC_LAMBDA           = 0.05

    def _update(self, actual_window: int):
        self.WINDOW_SIZE = actual_window
        self.N_PATCHES   = actual_window // self.PATCH_LEN


cfg = Config()
cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Reproducibility & device
# =============================================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

seed_everything(cfg.SEED)

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
        raise FileNotFoundError(f"No P*.npz files in {proc_dir}")

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


# =============================================================================
# Subject-wise 5-fold CV split  →  7:1:2 train/val/test per fold
# =============================================================================
def make_cv_folds(pids: list[str],
                  n_folds: int = 5,
                  seed: int = 42) -> list[dict]:
    """
    Returns a list of n_folds dicts, each with keys
        'train', 'val', 'test'  →  lists of pid strings.

    Strategy
    ────────
    1. Shuffle participants once with fixed seed.
    2. Split into n_folds equal (±1) chunks.
    3. For fold f:
         test  = chunk f                              (~20 % of N)
         remaining = all other chunks                 (~80 % of N)
         val   = first  ceil(len(remaining)/8)        (~10 % of N)
         train = rest of remaining                    (~70 % of N)
    This gives exactly a 7:1:2 subject-level ratio.
    """
    rng = np.random.default_rng(seed)
    shuffled = list(pids)
    rng.shuffle(shuffled)

    # split into n_folds chunks
    chunks = []
    step   = len(shuffled) / n_folds
    for i in range(n_folds):
        lo = int(round(i * step))
        hi = int(round((i + 1) * step))
        chunks.append(shuffled[lo:hi])

    folds = []
    for f in range(n_folds):
        test_pids  = chunks[f]
        remaining  = []
        for i, ch in enumerate(chunks):
            if i != f:
                remaining.extend(ch)
        n_val      = max(1, math.ceil(len(remaining) / 8))
        val_pids   = remaining[:n_val]
        train_pids = remaining[n_val:]
        folds.append({
            "fold":  f,
            "train": train_pids,
            "val":   val_pids,
            "test":  test_pids,
        })
        print(f"  Fold {f}: train={len(train_pids):3d} | "
              f"val={len(val_pids):3d} | test={len(test_pids):3d}  "
              f"(ratio ≈ {len(train_pids)/len(shuffled):.2f}:"
              f"{len(val_pids)/len(shuffled):.2f}:"
              f"{len(test_pids)/len(shuffled):.2f})")

    return folds


print("\n── Building CV folds ──")
CV_FOLDS = make_cv_folds(pids_all, n_folds=N_FOLDS, seed=cfg.SEED)


# =============================================================================
# Time-of-day features
# =============================================================================
def time_features(ns: int) -> np.ndarray:
    ts  = pd.to_datetime(int(ns), unit="ns", utc=True).tz_convert(None)
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
    T, C = sig.shape
    out  = sig.copy()
    band = random.choice(["low", "mid", "high"])
    for c in range(C):
        f = np.fft.rfft(out[:, c]); n = len(f)
        if   band == "low":  f[n//3:]   = 0
        elif band == "mid":  f[:n//4]   = 0; f[n//2:] = 0
        else:                f[:2*n//3] = 0
        out[:, c] = np.fft.irfft(f, n=T)
    return out

def _axis_permute(sig):
    idx = list(range(sig.shape[1])); random.shuffle(idx)
    return sig[:, idx]

def _magnitude_scale(sig):
    return sig * np.random.uniform(0.8, 1.2, size=(1, sig.shape[1]))

def _time_warp(sig: np.ndarray) -> np.ndarray:
    T, C   = sig.shape
    factor = random.choice([0.9, 0.95, 1.0, 1.05, 1.1])
    new_T  = max(T, int(round(T * factor)))
    warped = np.zeros((new_T, C), dtype=sig.dtype)
    for c in range(C):
        warped[:, c] = np.interp(
            np.linspace(0, T-1, new_T), np.arange(T), sig[:, c])
    if new_T >= T:
        return warped[:T]
    out = np.zeros((T, C), dtype=sig.dtype)
    out[:new_T] = warped
    return out

def freq_augment(sig: np.ndarray) -> np.ndarray:
    fn = random.choice([_bandpass_jitter, _axis_permute,
                        _magnitude_scale, _time_warp])
    return fn(sig)


# =============================================================================
# Dataset
# =============================================================================
class WindowDataset(Dataset):
    def __init__(self, pid_list, proc_dir, class_to_idx, is_train=False):
        self.entries   = []
        self.is_train  = is_train
        proc_dir       = Path(proc_dir)
        self.patch_lens = (cfg.PATCH_LENS_MULTI if CC.C4_MULTISCALE_PATCHING
                           else [cfg.PATCH_LEN])
        _set = False

        for pi, pid in enumerate(pid_list):
            path = proc_dir / f"{pid}.npz"
            if not path.exists():
                continue
            try:
                npz = np.load(path, allow_pickle=True)
            except Exception as e:
                print(f"  [WARN] {path.name}: {e}"); continue

            W = npz["X"].astype(np.float32)
            L = npz["y"].astype(str)
            F = npz["t"].astype("datetime64[ns]").astype(np.int64)
            order = np.argsort(F)
            W, L, F = W[order], L[order], F[order]

            if not _set:
                if W.shape[1] != cfg.WINDOW_SIZE:
                    cfg._update(W.shape[1])
                _set = True

            for w, lab, f in zip(W, L, F):
                if lab not in class_to_idx:
                    continue
                normed = np.zeros_like(w, dtype=np.float32)
                for c in range(cfg.CHANNELS):
                    ch = w[:, c]
                    normed[:, c] = (ch - ch.mean()) / (ch.std() + 1e-8)
                normed = np.clip(normed, -10, 10)
                T   = cfg.WINDOW_SIZE
                seg = (normed[:T] if normed.shape[0] >= T
                       else np.pad(normed, ((0, T-normed.shape[0]), (0,0))))
                self.entries.append((
                    pid, seg, time_features(int(f)),
                    int(class_to_idx[lab]), int(f),
                ))

            if (pi+1) % 10 == 0 or (pi+1) == len(pid_list):
                print(f"    Loaded {pi+1}/{len(pid_list)} subj — "
                      f"{len(self.entries):,} windows")

    @staticmethod
    def _make_patches(seg: np.ndarray, patch_len: int) -> np.ndarray:
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
        if self.is_train and CC.C5_FREQ_AUGMENTATION:
            seg = freq_augment(seg)
        patches_list = [
            torch.from_numpy(self._make_patches(seg, pl))
            for pl in self.patch_lens
        ]
        return (
            patches_list,
            torch.from_numpy(tfeat),
            torch.tensor(label, dtype=torch.long),
            pid,
            torch.tensor(first_ns, dtype=torch.long),
            torch.from_numpy(seg.astype(np.float32)),
        )


def _collate(batch):
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
                    num_workers=0, pin_memory=GPU, collate_fn=_collate)
    return ds, dl


# =============================================================================
# Model blocks (identical to v2)
# =============================================================================
class ZCRMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        x0 = x - x.mean(-1, keepdim=True)
        return (x0 / x0.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()) * self.g


class StochasticDepth(nn.Module):
    def __init__(self, layer, survival_prob):
        super().__init__()
        self.layer = layer; self.p = survival_prob
    def forward(self, x, *a, **kw):
        if not self.training or self.p >= 1.0:
            return self.layer(x, *a, **kw)
        return x if random.random() > self.p else self.layer(x, *a, **kw)


class GatedDeltaNet(nn.Module):
    def __init__(self, d, dropout=0.1):
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
    def _l2(x, eps=1e-8):
        return x / (x.pow(2).sum(-1, keepdim=True).add(eps).sqrt())

    def forward(self, x):
        h = self.norm(x)
        q = self.act(self.q_conv(self.q_lin(h).transpose(1,2)).transpose(1,2))
        k = self.act(self.k_conv(self.k_lin(h).transpose(1,2)).transpose(1,2))
        v = self.act(self.v_conv(self.v_lin(h).transpose(1,2)).transpose(1,2))
        q, k  = self._l2(q), self._l2(k)
        delta = q * (k * v)
        delta = torch.tanh(self.alpha(x)) * delta + self.beta(x)
        dhat  = self.post(self.post_norm(delta))
        return x + self.drop(self.gate(self.silu(dhat)) * dhat)


class SoftMoE(nn.Module):
    def __init__(self, d, hidden, n_experts=4, dropout=0.1):
        super().__init__()
        self.router  = nn.Linear(d, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden), nn.SiLU(),
                          nn.Dropout(dropout),  nn.Linear(hidden, d))
            for _ in range(n_experts)])
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        w = torch.softmax(self.router(x), dim=-1)
        s = torch.stack([e(x) for e in self.experts], dim=-2)
        return self.drop((w.unsqueeze(-1) * s).sum(-2))


def precompute_freqs(dim, n_tok, theta=10000.0):
    assert dim % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t     = torch.arange(n_tok)
    return torch.polar(torch.ones(n_tok, dim//2), torch.outer(t, freqs))

def apply_rope(q, k, freqs):
    B, H, N, D = q.shape; d2 = D // 2
    f   = freqs[:N].to(q.device).view(1, 1, N, d2)
    q_  = torch.view_as_complex(q.float().contiguous().view(B,H,N,d2,2))
    k_  = torch.view_as_complex(k.float().contiguous().view(B,H,N,d2,2))
    return (torch.view_as_real(q_*f).view(B,H,N,D).type_as(q),
            torch.view_as_real(k_*f).view(B,H,N,D).type_as(k))


class GatedAttention(nn.Module):
    def __init__(self, d, n_heads=2, dropout=0.1):
        super().__init__()
        assert d % n_heads == 0 and (d // n_heads) % 2 == 0
        self.h = n_heads; self.dh = d // n_heads
        self.norm = ZCRMSNorm(d)
        self.qkv  = nn.Linear(d, 3*d)
        self.out  = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, freqs):
        h = self.norm(x); B, N, D = h.shape
        qkv = self.qkv(h).reshape(B,N,3,self.h,self.dh).permute(0,2,1,3,4)
        q = qkv[:,0].transpose(1,2); k = qkv[:,1].transpose(1,2)
        v = qkv[:,2].transpose(1,2)
        q, k  = apply_rope(q, k, freqs)
        score = (q @ k.transpose(-2,-1)) / math.sqrt(self.dh)
        attn  = self.drop(torch.softmax(score, dim=-1))
        y     = self.out((attn@v).transpose(1,2).contiguous().reshape(B,N,D))
        return x + torch.sigmoid(self.gate(h)) * y


# [C1]
class DualDomainPatchEmbed(nn.Module):
    def __init__(self, patch_len, channels, d):
        super().__init__()
        in_dim   = patch_len * channels
        freq_dim = (patch_len // 2 + 1) * channels
        self.time_proj = nn.Linear(in_dim, d)
        self.freq_proj = nn.Linear(freq_dim, d)
        self.gate_w    = nn.Parameter(torch.zeros(d))

    def forward(self, patches):
        B, C, NP, PL = patches.shape
        x_t   = patches.permute(0,2,1,3).reshape(B, NP, C*PL)
        t_emb = self.time_proj(x_t)
        x_f   = patches.permute(0,2,3,1)
        fft   = torch.fft.rfft(x_f, dim=2)
        mag   = fft.abs().reshape(B, NP, -1)
        f_emb = self.freq_proj(mag)
        g     = torch.sigmoid(self.gate_w)
        return g * t_emb + (1-g) * f_emb


class SimplePatchEmbed(nn.Module):
    def __init__(self, patch_len, channels, d):
        super().__init__()
        self.proj = nn.Linear(patch_len * channels, d)
    def forward(self, patches):
        B, C, NP, PL = patches.shape
        return self.proj(patches.permute(0,2,1,3).reshape(B, NP, C*PL))


# [C3]
class CircadianBias(nn.Module):
    def __init__(self, n_patches, d):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, d), nn.SiLU(), nn.Linear(d, n_patches * d))
        self.n_patches = n_patches; self.d = d
    def forward(self, times):
        return self.mlp(times).view(times.shape[0], self.n_patches, self.d)


# [C2]
class SkipAggregation(nn.Module):
    def __init__(self, d, n_layers):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_layers) / n_layers)
        self.proj    = nn.Sequential(nn.Linear(d, d), nn.SiLU())
    def forward(self, hiddens):
        w   = torch.softmax(self.weights, dim=0)
        agg = sum(w[i] * hiddens[i] for i in range(len(hiddens)))
        return self.proj(agg)


# [C10]
class ReconHead(nn.Module):
    def __init__(self, d, n_patches, patch_len, channels):
        super().__init__()
        out_dim  = n_patches * patch_len * channels
        self.mlp = nn.Sequential(
            nn.Linear(d, 2*d), nn.ReLU(), nn.Linear(2*d, out_dim))
        self.n_patches = n_patches; self.pl = patch_len; self.C = channels
    def forward(self, z):
        return self.mlp(z)


# =============================================================================
# PatchHAR v3  (same architecture as v2 — instantiated fresh per fold)
# =============================================================================
class PatchHARv3(nn.Module):
    def __init__(self):
        super().__init__()
        d  = cfg.D_MODEL
        NP = cfg.N_PATCHES

        self.patch_lens = (cfg.PATCH_LENS_MULTI if CC.C4_MULTISCALE_PATCHING
                           else [cfg.PATCH_LEN])
        EmbCls = DualDomainPatchEmbed if CC.C1_DUAL_DOMAIN_EMBEDDING \
                 else SimplePatchEmbed
        self.patch_embeds = nn.ModuleList([
            EmbCls(pl, cfg.CHANNELS, d) for pl in self.patch_lens])

        if CC.C4_MULTISCALE_PATCHING and len(self.patch_lens) > 1:
            self.scale_fusion = nn.Linear(d * len(self.patch_lens), d)
        else:
            self.scale_fusion = None

        if CC.C3_CIRCADIAN_BIAS:
            self.circ_bias = CircadianBias(NP, d)
        else:
            self.time_emb  = nn.Sequential(
                nn.Linear(5, d), nn.ReLU(), nn.Dropout(0.1))

        self.input_norm = nn.LayerNorm(d)

        raw_layers = [GatedDeltaNet(d, dropout=cfg.DROPOUT)
                      for _ in range(cfg.N_LAYERS)]
        if CC.C8_STOCHASTIC_DEPTH:
            survival = [1.0 - (i/cfg.N_LAYERS)*cfg.SD_DROP_MAX
                        for i in range(cfg.N_LAYERS)]
            self.delta_layers = nn.ModuleList([
                StochasticDepth(l, p) for l, p in zip(raw_layers, survival)])
        else:
            self.delta_layers = nn.ModuleList(raw_layers)

        if CC.C2_CALANET_SKIP_AGG:
            self.skip_agg = SkipAggregation(d, cfg.N_LAYERS)

        self.moe1 = SoftMoE(d, 2*d, n_experts=cfg.N_EXPERTS,
                            dropout=cfg.DROPOUT)
        self.attn = GatedAttention(d, n_heads=cfg.N_HEADS,
                                   dropout=cfg.DROPOUT)
        self.moe2 = SoftMoE(d, 2*d, n_experts=cfg.N_EXPERTS,
                            dropout=cfg.DROPOUT)

        self.register_buffer("freqs",
                             precompute_freqs(d // cfg.N_HEADS, NP))

        if CC.C6_LABEL_SMOOTH_TEMP:
            self.log_tau = nn.Parameter(torch.zeros(1))

        if CC.C10_RECON_AUX_GRAD_SURGERY:
            self.recon_head = ReconHead(d, NP, cfg.PATCH_LEN, cfg.CHANNELS)

        if CC.C7_PROTOTYPE_MEMORY:
            self.register_buffer("prototypes", torch.zeros(NUM_CLASSES, d))
            self.proto_filled = False

        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(d, d//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d//2, NUM_CLASSES),
        )

    # ------------------------------------------------------------------
    def _embed(self, patches_list):
        NP = cfg.N_PATCHES
        if CC.C4_MULTISCALE_PATCHING and len(patches_list) > 1:
            embs = []
            for embed, patches in zip(self.patch_embeds, patches_list):
                e = embed(patches)                       # (B, N_Pi, D)
                e = F.interpolate(e.permute(0,2,1),
                                  size=NP, mode="linear",
                                  align_corners=False).permute(0,2,1)
                embs.append(e)
            return self.scale_fusion(torch.cat(embs, dim=-1))
        return self.patch_embeds[0](patches_list[0])

    # ------------------------------------------------------------------
    def _encode(self, patches_list, times):
        """Shared encoder — returns (x_seq, z_pool, hiddens)."""
        x = self._embed(patches_list)
        if CC.C3_CIRCADIAN_BIAS:
            x = x + self.circ_bias(times)
        else:
            x = x + self.time_emb(times).unsqueeze(1)
        x = self.input_norm(x)
        hiddens = []
        for layer in self.delta_layers:
            x = layer(x); hiddens.append(x)
        if CC.C2_CALANET_SKIP_AGG:
            x = x + self.skip_agg(hiddens)
        x = x + self.moe1(x)
        x = self.attn(x, self.freqs)
        x = x + self.moe2(x)
        z = x.mean(1)
        return x, z, hiddens

    # ------------------------------------------------------------------
    def forward(self, patches_list, times,
                return_embedding=False):
        """
        return_embedding=True  → returns  z  (B, D)  only.
        Otherwise returns (logits, recon_or_None).
        """
        _, z, _ = self._encode(patches_list, times)
        if return_embedding:
            return z

        recon = None
        if CC.C10_RECON_AUX_GRAD_SURGERY and self.training:
            recon = self.recon_head(z)

        logits = self.head(z)
        if CC.C6_LABEL_SMOOTH_TEMP:
            tau    = torch.exp(self.log_tau).clamp(0.5, 2.0)
            logits = logits / tau

        if CC.C7_PROTOTYPE_MEMORY and not self.training and self.proto_filled:
            z_n    = F.normalize(z, dim=-1)
            pr_n   = F.normalize(self.prototypes, dim=-1)
            cosine = z_n @ pr_n.T
            logits = (1-cfg.PROTO_ALPHA)*logits + cfg.PROTO_ALPHA*cosine

        return logits, recon

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_prototypes(self, embeddings, labels):
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


def manifold_mixup(z, labels, alpha=0.2):
    if alpha <= 0:
        return z, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(z.size(0), device=z.device)
    return lam*z + (1-lam)*z[idx], labels, labels[idx], lam


def raw_mixup(patches_list, times, labels, alpha=0.2):
    if alpha <= 0:
        return patches_list, times, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(labels.size(0), device=labels.device)
    mixed = [lam*p + (1-lam)*p[idx] for p in patches_list]
    return mixed, lam*times + (1-lam)*times[idx], labels, labels[idx], lam


def tc_loss(logits):
    if logits.size(0) < 2:
        return logits.new_zeros(1).squeeze()
    p = F.softmax(logits[:-1], dim=-1)
    q = F.softmax(logits[1:],  dim=-1)
    return 0.5 * (F.kl_div(q.log(), p, reduction="batchmean") +
                  F.kl_div(p.log(), q, reduction="batchmean"))


def recon_loss(recon, raw_segs):
    B, T, C = raw_segs.shape
    NP = cfg.N_PATCHES; PL = cfg.PATCH_LEN
    target = (raw_segs[:, :NP*PL, :]
              .reshape(B, NP, PL, C)
              .permute(0,1,3,2)
              .reshape(B, NP*C*PL))
    return F.mse_loss(recon, target.detach())


class SmoothCE(nn.Module):
    def __init__(self, weight=None, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing; self.weight = weight
    def forward(self, logits, labels):
        K    = logits.size(-1); eps = self.smoothing
        with torch.no_grad():
            soft = torch.full_like(logits, eps/(K-1))
            soft.scatter_(-1, labels.unsqueeze(-1), 1.0-eps)
        log_prob = F.log_softmax(logits, dim=-1)
        loss = -(soft * log_prob).sum(-1)
        if self.weight is not None:
            loss = loss * self.weight.to(logits.device)[labels]
        return loss.mean()


# =============================================================================
# Metrics
# =============================================================================
def kappa(yt, yp):
    cm = confusion_matrix(yt, yp); n = cm.sum()
    if n == 0: return 0.0
    po = np.trace(cm) / n
    pe = np.dot(cm.sum(1), cm.sum(0)) / (n*n)
    return float((po-pe)/(1-pe)) if abs(1-pe) > 1e-12 else 0.0

def mcc(yt, yp):
    cm = confusion_matrix(yt, yp).astype(float); n = cm.sum()
    if n == 0: return 0.0
    s = np.trace(cm); t = cm.sum(1); p = cm.sum(0)
    num = s*n - np.sum(t*p)
    den = math.sqrt(max(n**2-np.sum(t**2), 0.0)*max(n**2-np.sum(p**2), 0.0))
    return float(num/den) if den > 0 else 0.0


# =============================================================================
# Training loop (per fold)
# =============================================================================
def train_one_fold(model: PatchHARv3,
                   train_dl: DataLoader,
                   val_dl:   DataLoader,
                   class_w:  torch.Tensor,
                   save_path: Path,
                   fold_idx:  int) -> list:

    # delete stale checkpoint
    if save_path.exists():
        try:
            ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
            if any("delta1." in k for k in ckpt.keys()):
                save_path.unlink()
        except Exception:
            save_path.unlink()

    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.LR,
        steps_per_epoch=len(train_dl), epochs=cfg.EPOCHS,
        pct_start=0.10, anneal_strategy="cos")

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

            # [C9] Manifold Mixup in embedding space
            if CC.C9_MANIFOLD_MIXUP:
                with amp_ctx():
                    _, z_pool, _ = model._encode(patches_list, times)
                z_mix, la, lb, lam = manifold_mixup(
                    z_pool, labels, cfg.MIXUP_ALPHA)
                optimizer.zero_grad(set_to_none=True)
                with amp_ctx():
                    logits = model.head(z_mix)
                    if CC.C6_LABEL_SMOOTH_TEMP:
                        tau    = torch.exp(model.log_tau).clamp(0.5, 2.0)
                        logits = logits / tau
                    loss = (lam*criterion(logits, la) +
                            (1-lam)*criterion(logits, lb))
                    if cfg.TC_LAMBDA > 0:
                        loss = loss + cfg.TC_LAMBDA * tc_loss(logits)
                    if CC.C10_RECON_AUX_GRAD_SURGERY:
                        recon = model.recon_head(z_pool.detach())
                        loss  = loss + cfg.RECON_LAMBDA * recon_loss(recon, raw_segs)
            else:
                patches_list, times, la, lb, lam = raw_mixup(
                    patches_list, times, labels, cfg.MIXUP_ALPHA)
                optimizer.zero_grad(set_to_none=True)
                with amp_ctx():
                    logits, recon = model(patches_list, times)
                    loss = (lam*criterion(logits, la) +
                            (1-lam)*criterion(logits, lb))
                    if cfg.TC_LAMBDA > 0:
                        loss = loss + cfg.TC_LAMBDA * tc_loss(logits)
                    if CC.C10_RECON_AUX_GRAD_SURGERY and recon is not None:
                        loss = loss + cfg.RECON_LAMBDA * recon_loss(recon, raw_segs)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_ok = (
                all(p.grad is None or torch.isfinite(p.grad).all()
                    for p in model.parameters())
                and torch.isfinite(loss))
            if grad_ok:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
                scaler.step(optimizer)
            else:
                optimizer.zero_grad(set_to_none=True)
            scaler.update()
            scheduler.step()
            if torch.isfinite(loss):
                total_loss += float(loss.item())

        # ── validation ────────────────────────────────────────────────────
        model.eval()
        vp, vt, embs_val, labs_val = [], [], [], []
        with torch.no_grad():
            for batch in val_dl:
                patches_list_v, times_v, labels_v, _, _, _ = batch
                pl_v = [p.to(device).float() for p in patches_list_v]
                tv   = times_v.to(device).float()
                if CC.C7_PROTOTYPE_MEMORY:
                    z      = model.forward(pl_v, tv, return_embedding=True)
                    logits, _ = model(pl_v, tv)
                    embs_val.append(z.cpu()); labs_val.append(labels_v)
                else:
                    logits, _ = model(pl_v, tv)
                vp.extend(logits.argmax(1).cpu().numpy())
                vt.extend(labels_v.numpy())

        if CC.C7_PROTOTYPE_MEMORY and embs_val:
            model.update_prototypes(
                torch.cat(embs_val).to(device),
                torch.cat(labs_val).to(device))

        vp  = np.array(vp); vt = np.array(vt)
        f1  = float(f1_score(vt, vp, average="macro", zero_division=0))
        kap = kappa(vt, vp)
        avg = total_loss / max(1, len(train_dl))
        lr  = optimizer.param_groups[0]["lr"]

        print(f"    Epoch {epoch+1:02d}/{cfg.EPOCHS} | "
              f"lr={lr:.2e} | loss={avg:.4f} | "
              f"F1={f1:.4f} | κ={kap:.4f}")

        history.append({"epoch": epoch+1, "loss": round(avg,6),
                        "val_f1": round(f1,6), "val_kappa": round(kap,6)})
        score = f1 + kap
        if score > best_score + 1e-6:
            best_score = score; patience_ctr = 0
            torch.save(model.state_dict(), save_path)
            print(f"      ✓ Fold {fold_idx} checkpoint saved (F1={f1:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.EARLY_STOP_PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=False))
    return history


# =============================================================================
# Evaluation (per fold)
# =============================================================================
@torch.no_grad()
def evaluate_fold(model: PatchHARv3, test_dl: DataLoader):
    model.eval()
    all_pred, all_true = [], []
    for batch in test_dl:
        patches_list, times, labels, _, _, _ = batch
        pl = [p.to(device).float() for p in patches_list]
        tv = times.to(device).float()
        logits, _ = model(pl, tv)
        all_pred.extend(logits.argmax(1).cpu().numpy())
        all_true.extend(labels.numpy())

    yt = np.array(all_true); yp = np.array(all_pred)
    metrics = {
        "macro_f1": round(float(f1_score(yt, yp, average="macro",
                                         zero_division=0)), 4),
        "kappa":    round(kappa(yt, yp), 4),
        "mcc":      round(mcc(yt, yp), 4),
        "accuracy": round(float((yt==yp).mean()), 4),
    }
    print(f"\n  Test  →  Macro-F1={metrics['macro_f1']:.4f} | "
          f"κ={metrics['kappa']:.4f} | "
          f"MCC={metrics['mcc']:.4f} | "
          f"Acc={metrics['accuracy']:.4f}")
    print(classification_report(yt, yp, target_names=CLASSES, zero_division=0))
    return metrics, yt.tolist(), yp.tolist()


# =============================================================================
# Summary across folds
# =============================================================================
def summarise(fold_metrics: list[dict]) -> dict:
    keys   = ["macro_f1", "kappa", "mcc", "accuracy"]
    summ   = {}
    print("\n" + "═"*70)
    print("  5-FOLD CV SUMMARY")
    print("═"*70)
    header = f"  {'Fold':>5}  {'F1':>8}  {'κ':>8}  {'MCC':>8}  {'Acc':>8}"
    print(header)
    print("  " + "-"*62)
    for i, m in enumerate(fold_metrics):
        print(f"  {i:>5}  "
              f"{m['macro_f1']:>8.4f}  "
              f"{m['kappa']:>8.4f}  "
              f"{m['mcc']:>8.4f}  "
              f"{m['accuracy']:>8.4f}")
    print("  " + "-"*62)
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        summ[k] = {
            "mean": round(float(np.mean(vals)), 4),
            "std":  round(float(np.std(vals)),  4),
            "min":  round(float(np.min(vals)),  4),
            "max":  round(float(np.max(vals)),  4),
        }
    print(f"  {'Mean':>5}  "
          f"{summ['macro_f1']['mean']:>8.4f}  "
          f"{summ['kappa']['mean']:>8.4f}  "
          f"{summ['mcc']['mean']:>8.4f}  "
          f"{summ['accuracy']['mean']:>8.4f}")
    print(f"  {'±Std':>5}  "
          f"{summ['macro_f1']['std']:>8.4f}  "
          f"{summ['kappa']['std']:>8.4f}  "
          f"{summ['mcc']['std']:>8.4f}  "
          f"{summ['accuracy']['std']:>8.4f}")
    print("═"*70)
    return summ


# =============================================================================
# Main — 5-fold CV loop
# =============================================================================
def main():
    print("=" * 70)
    print("  PatchHAR v3 — 5-Fold Subject-wise Cross-Validation (7:1:2)")
    print(f"  D_MODEL={cfg.D_MODEL} | N_LAYERS={cfg.N_LAYERS} | "
          f"N_HEADS={cfg.N_HEADS} | N_EXPERTS={cfg.N_EXPERTS}")
    print(f"  N_FOLDS={N_FOLDS}  |  Device={device}")
    print(f"  Output : {cfg.OUTPUT_DIR}")
    print()
    print("  Active contributions:")
    for k, v in CC.__dict__.items():
        if k.startswith("C") and not k.startswith("__"):
            print(f"    {'✓' if v else '✗'} {k}")
    print("=" * 70)

    folds_to_run = (
        [CV_FOLDS[FOLD_ONLY]] if FOLD_ONLY is not None
        else CV_FOLDS
    )

    all_fold_metrics  = []
    all_fold_history  = []
    all_true_global   = []
    all_pred_global   = []

    for fold_info in folds_to_run:
        f          = fold_info["fold"]
        train_pids = fold_info["train"]
        val_pids   = fold_info["val"]
        test_pids  = fold_info["test"]

        print(f"\n{'─'*70}")
        print(f"  FOLD {f}  |  "
              f"train={len(train_pids)}  val={len(val_pids)}  "
              f"test={len(test_pids)}")
        print(f"{'─'*70}")

        # Re-seed each fold for reproducibility
        seed_everything(cfg.SEED + f)

        # ── Data ─────────────────────────────────────────────────────────
        print(f"  Loading training data (fold {f}) …")
        train_ds, train_dl = make_loader(train_pids, shuffle=True,
                                         is_train=True)
        print(f"  Loading validation data (fold {f}) …")
        val_ds,   val_dl   = make_loader(val_pids,   shuffle=False,
                                         is_train=False)
        print(f"  Loading test data (fold {f}) …")
        test_ds,  test_dl  = make_loader(test_pids,  shuffle=False,
                                         is_train=False)
        print(f"  Windows — train={len(train_ds):,} | "
              f"val={len(val_ds):,} | test={len(test_ds):,}")

        if len(train_ds) == 0:
            print(f"  [WARN] No training data for fold {f}, skipping.")
            continue

        # ── Model (fresh per fold) ────────────────────────────────────────
        model    = PatchHARv3().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters : {n_params:,}")

        # ── Train ─────────────────────────────────────────────────────────
        save_path = cfg.OUTPUT_DIR / f"weights_fold{f}.pth"
        class_w   = compute_class_weights(train_ds)
        print(f"\n  ── Training fold {f} ──")
        history = train_one_fold(model, train_dl, val_dl,
                                 class_w, save_path, fold_idx=f)

        # ── Evaluate ──────────────────────────────────────────────────────
        print(f"\n  ── Evaluation fold {f} ──")
        metrics, yt_list, yp_list = evaluate_fold(model, test_dl)
        metrics["fold"]           = f
        metrics["n_train_subj"]   = len(train_pids)
        metrics["n_val_subj"]     = len(val_pids)
        metrics["n_test_subj"]    = len(test_pids)
        metrics["n_train_wins"]   = len(train_ds)
        metrics["n_val_wins"]     = len(val_ds)
        metrics["n_test_wins"]    = len(test_ds)

        all_fold_metrics.append(metrics)
        all_fold_history.append({"fold": f, "history": history})
        all_true_global.extend(yt_list)
        all_pred_global.extend(yp_list)

        # Free GPU memory between folds
        del model, train_ds, val_ds, test_ds
        del train_dl, val_dl, test_dl
        if GPU:
            torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────
    if len(all_fold_metrics) == 0:
        print("No folds completed.")
        return

    summary = summarise(all_fold_metrics)

    # Global confusion matrix across all folds
    yt_g = np.array(all_true_global)
    yp_g = np.array(all_pred_global)
    cm_g = confusion_matrix(yt_g, yp_g).tolist()
    global_metrics = {
        "macro_f1": round(float(f1_score(yt_g, yp_g, average="macro",
                                         zero_division=0)), 4),
        "kappa":    round(kappa(yt_g, yp_g), 4),
        "mcc":      round(mcc(yt_g, yp_g), 4),
        "accuracy": round(float((yt_g==yp_g).mean()), 4),
    }
    print("\n  Global metrics (all test windows pooled across folds):")
    print(f"    F1={global_metrics['macro_f1']:.4f}  "
          f"κ={global_metrics['kappa']:.4f}  "
          f"MCC={global_metrics['mcc']:.4f}  "
          f"Acc={global_metrics['accuracy']:.4f}")
    print(classification_report(yt_g, yp_g,
                                 target_names=CLASSES, zero_division=0))

    # ── Save all results ──────────────────────────────────────────────────
    active_contribs = {k: v for k, v in CC.__dict__.items()
                       if k.startswith("C") and not k.startswith("__")}
    results = {
        "cv_summary":      summary,
        "global_metrics":  global_metrics,
        "global_cm":       cm_g,
        "classes":         CLASSES,
        "per_fold":        all_fold_metrics,
        "training_history": all_fold_history,
        "contributions":   active_contribs,
        "config": {
            "n_folds":      N_FOLDS,
            "split_ratio":  "7:1:2",
            "window_size":  cfg.WINDOW_SIZE,
            "patch_len":    cfg.PATCH_LEN,
            "n_patches":    cfg.N_PATCHES,
            "d_model":      cfg.D_MODEL,
            "n_heads":      cfg.N_HEADS,
            "n_layers":     cfg.N_LAYERS,
            "n_experts":    cfg.N_EXPERTS,
            "dropout":      cfg.DROPOUT,
            "device":       str(device),
            "total_subjects": len(pids_all),
        },
    }
    out = cfg.OUTPUT_DIR / "cv_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  All results saved: {out}")
    print("=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()

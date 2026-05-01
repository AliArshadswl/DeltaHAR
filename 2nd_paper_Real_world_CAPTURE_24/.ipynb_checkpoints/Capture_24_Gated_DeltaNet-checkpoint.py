#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
har_directions_study.py
========================
Five architectural / training directions tested against the
CGA-HybridHAR baseline.  Each direction is isolated — only one thing
changes at a time so causal attribution is clean.

  D0  Baseline          CGA-HybridHAR (21-token interleaved, symmetric bias)
  D4  Transition-Aware  D0 + boundary-window label smoothing  [loss only]
  D1  Cross-axis Attn   Per-patch 3×3 inter-axis attention replaces flat proj
  D2  Hierarchical      Dual-scale tokens (fine 10 + coarse 5 = 15 tokens)
  D5  Subject-stat CGA  Subject-level personalisation token replaces local stat
  D3  Asymmetric CGA    Patch→stat attention gated by patch uncertainty

Combinations also tested:
  D1+D4   cross-axis + transition loss
  D2+D4   hierarchical + transition loss

Identical hyperparameters, data splits and seeds across all variants.
Results saved to OUTPUT_DIR:
  directions_results.json      full metrics per direction
  directions_summary.csv       headline table for the paper
  directions_fold_detail.csv   per-run details
  cga_bias_D0.png / D3.png     learned bias heatmaps
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import gc, json, math, random, time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

try:
    from scipy.signal import find_peaks, peak_prominences
    from scipy.signal import welch as scipy_welch
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

# ══════════════════════════════════════════════════════════════════════════════
# 1.  GLOBAL CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    PROC_DIR   = Path("/mnt/share/ali/processed_minimal/")
    OUTPUT_DIR = Path("/mnt/share/ali/processed_minimal/study_results")

    TRAIN_N = 80
    VAL_N   = 20

    SIGNAL_RATE  = 100
    WINDOW_SIZE  = 1000
    PATCH_LEN    = 100
    CHANNELS     = 3
    N_PATCHES    = WINDOW_SIZE // PATCH_LEN       # 10

    D_MODEL  = 128
    N_HEADS  = 2
    DROPOUT  = 0.25

    # ── CGA feature dims ──────────────────────────────────────────────────
    LOCAL_STAT_DIM  = 18    # per-patch: [mean,var,skew,kurt,zcr,peak_amp]×C
    GLOBAL_STAT_DIM = 8     # per-window: spectral entropy×C + dom_freq/power + corr×3
    SUBJ_STAT_DIM   = 6     # D5: [channel_mean×C, channel_std×C]

    # CGA bias init
    CGA_COUPLED_INIT   = 2.0
    CGA_DECOUPLED_INIT = 0.0

    # Training
    BATCH_SIZE           = 32
    EPOCHS               = 30
    LR                   = 1e-3
    WEIGHT_DECAY         = 1e-4
    MAX_GRAD_NORM        = 1.0
    EARLY_STOP_PATIENCE  = 8

    # Transition-aware loss
    TRANSITION_SMOOTH_LOW  = 0.05   # standard windows
    TRANSITION_SMOOTH_HIGH = 0.25   # boundary windows

    # Mixup
    MIXUP_ALPHA = 0.2
    TC_LAMBDA   = 0.05

    # HMM
    HMM_SMOOTH   = 1.0
    HMM_MIN_PROB = 1e-6

    # Benchmarking
    N_WARMUP  = 10
    N_BENCH   = 100

    SEED = 42

    def _update(self, actual_window: int):
        self.WINDOW_SIZE = actual_window
        self.N_PATCHES   = actual_window // self.PATCH_LEN


cfg = Config()
cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(cfg.SEED)
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

GPU    = torch.cuda.is_available()
DEVICE = torch.device("cuda" if GPU else "cpu")
print(f"Device : {DEVICE}")
if GPU:
    props = torch.cuda.get_device_properties(0)
    print(f"   GPU : {props.name}  {props.total_memory/1e9:.1f} GB")


def _amp():
    if GPU:
        try:
            return torch.autocast("cuda", dtype=torch.float16)
        except TypeError:
            from torch.cuda.amp import autocast
            return autocast()
    return nullcontext()


# ══════════════════════════════════════════════════════════════════════════════
# 2.  METADATA
# ══════════════════════════════════════════════════════════════════════════════

classes       = json.loads((cfg.PROC_DIR / "classes.json").read_text())
label_encoder = json.loads((cfg.PROC_DIR / "label_encoder.json").read_text())
class_to_idx  = {c: int(i) for c, i in label_encoder.items()}
idx_to_class  = {int(i): c  for c, i in label_encoder.items()}
num_classes   = len(classes)
print(f"Classes ({num_classes}): {classes}")

manifest = pd.read_csv(cfg.PROC_DIR / "manifest.csv")
manifest = manifest[(manifest["status"] == "ok") &
                    (manifest["outfile"].astype(str).str.len() > 0)]

def _valid_pid(pid: str) -> bool:
    s = str(pid).strip()
    if not s.upper().startswith("P"):
        return False
    try:
        return 1 <= int(s[1:]) <= 151
    except ValueError:
        return False

pids_all   = (manifest[manifest["participant"].astype(str).apply(_valid_pid)]
              ["participant"].astype(str).sort_values().tolist())
n_train    = min(cfg.TRAIN_N, len(pids_all))
n_val      = min(cfg.VAL_N,  max(0, len(pids_all) - n_train))
train_pids = pids_all[:n_train]
val_pids   = pids_all[n_train:n_train + n_val]
test_pids  = pids_all[n_train + n_val:]
print(f"Participants — train={len(train_pids)}, val={len(val_pids)}, test={len(test_pids)}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FEATURE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _safe_corr(a, b) -> float:
    sa, sb = float(np.std(a)), float(np.std(b))
    if sa < 1e-8 or sb < 1e-8: return 0.
    c = float(np.corrcoef(a, b)[0, 1])
    return 0. if not np.isfinite(c) else c

def _skew(x) -> float:
    mu, sd = float(np.mean(x)), float(np.std(x))
    if sd < 1e-12: return 0.
    return float(np.mean((x - mu) ** 3)) / (sd ** 3)

def _kurt(x) -> float:
    mu, sd = float(np.mean(x)), float(np.std(x))
    if sd < 1e-12: return 0.
    return float(np.mean((x - mu) ** 4)) / (sd ** 4) - 3.

def _zcr(x: np.ndarray) -> float:
    return float(np.sum(np.diff(np.sign(x)) != 0)) / max(1, len(x) - 1)

def _norm_psd(x, sr):
    ps = (np.abs(np.fft.rfft(x)) ** 2).astype(np.float64)
    s  = ps.sum(); ps = ps / s if s > 0 else ps
    return ps, np.fft.rfftfreq(len(x), 1. / sr)

def _dom2(ps, fq):
    if ps.size <= 2: return 0., 0., 0., 0.
    p = ps.copy(); p[0] = 0.
    i1 = int(np.argmax(p)); p1, f1 = float(p[i1]), float(fq[i1]); p[i1] = -1.
    i2 = int(np.argmax(p)); p2, f2 = max(float(p[i2]), 0.), float(fq[i2])
    return f1, p1, f2, p2

def _spec_entropy(ps) -> float:
    p = ps[np.isfinite(ps) & (ps > 0)]
    if p.size == 0: return 0.
    H = -float(np.sum(p * np.log(p)))
    return H / math.log(len(p)) if len(p) > 1 else 0.

def time_features_from_ns(first_ns: int) -> np.ndarray:
    ts = pd.to_datetime(int(first_ns), unit="ns", utc=True).tz_convert(None)
    return np.array([ts.hour / 24., ts.minute / 60., ts.weekday() / 7.,
                     float(ts.weekday() >= 5), float(ts.hour // 6)],
                    dtype=np.float32)


# ── Local stat features (per-patch, 18-dim) ───────────────────────────────────
def compute_local_stats(patch: np.ndarray) -> np.ndarray:
    """(PATCH_LEN, C) → (18,)  [mean,var,skew,kurt,zcr,peak_amp] × C"""
    feats = []
    for c in range(cfg.CHANNELS):
        ch = patch[:, c]
        feats += [float(ch.mean()), float(ch.var()), _skew(ch), _kurt(ch),
                  _zcr(ch), float(np.abs(ch).max())]
    return np.array(feats, dtype=np.float32)


# ── Global stat features (per-window, 8-dim) ──────────────────────────────────
def compute_global_stats(window: np.ndarray) -> np.ndarray:
    """(WINDOW_SIZE, C) → (8,)"""
    feats = []
    for c in range(cfg.CHANNELS):
        ch = window[:, c]
        if _HAVE_SCIPY:
            _, psd = scipy_welch(ch, fs=cfg.SIGNAL_RATE,
                                  nperseg=min(256, len(ch)))
        else:
            psd, _ = _norm_psd(ch, cfg.SIGNAL_RATE)
        pn = psd / (psd.sum() + 1e-12)
        feats.append(-float(np.sum(pn * np.log(pn + 1e-12))))
    mag         = np.linalg.norm(window, axis=1)
    ps, fq      = _norm_psd(mag, cfg.SIGNAL_RATE)
    f1, p1, *_  = _dom2(ps, fq)
    feats      += [float(f1) / (cfg.SIGNAL_RATE / 2 + 1e-8), float(p1)]
    feats      += [_safe_corr(window[:, 0], window[:, 1]),
                   _safe_corr(window[:, 0], window[:, 2]),
                   _safe_corr(window[:, 1], window[:, 2])]
    arr = np.array(feats, dtype=np.float32)
    return np.where(np.isfinite(arr), arr, 0.)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  DATASET
#     Returns everything every model might need; each model picks what it uses.
# ══════════════════════════════════════════════════════════════════════════════

class HARDataset(Dataset):
    """
    Entry tuple:
      patches      (C, NP, PL)           raw patches
      times        (5,)                  time-of-day features
      local_stats  (NP, LOCAL_STAT_DIM)  per-patch stats
      global_stats (GLOBAL_STAT_DIM,)    per-window stats
      label        long scalar
      pid          str
      first_ns     long scalar
      is_transition bool scalar          True if adjacent window has diff label
    """
    def __init__(self, pid_list, proc_dir=cfg.PROC_DIR, c2i=class_to_idx):
        self.entries = []
        proc_dir     = Path(proc_dir)
        _win_set     = False

        for p_idx, pid in enumerate(pid_list):
            path = proc_dir / f"{pid}.npz"
            if not path.exists():
                continue
            npz   = np.load(path, allow_pickle=True)
            W     = npz["windows"].astype(np.float32)
            L     = npz["labels_str"].astype(str)
            F     = npz["first_ts_epoch_ns"].astype(np.int64)
            order = np.argsort(F)
            W, L, F = W[order], L[order], F[order]

            if not _win_set:
                if W.shape[1] != cfg.WINDOW_SIZE:
                    cfg._update(W.shape[1])
                _win_set = True

            # detect transition windows (label differs from neighbours)
            is_trans = np.zeros(len(L), dtype=bool)
            for i in range(len(L)):
                if i > 0          and L[i] != L[i - 1]: is_trans[i] = True
                if i < len(L) - 1 and L[i] != L[i + 1]: is_trans[i] = True

            for i, (w, lab, f, it) in enumerate(zip(W, L, F, is_trans)):
                if lab not in c2i:
                    continue
                normed = np.zeros_like(w)
                for c in range(cfg.CHANNELS):
                    ch = w[:, c]
                    normed[:, c] = np.clip((ch - ch.mean()) / (ch.std() + 1e-8),
                                           -10, 10)
                T   = cfg.WINDOW_SIZE
                seg = normed[:T] if normed.shape[0] >= T \
                      else np.vstack([normed,
                                      np.zeros((T - normed.shape[0],
                                                cfg.CHANNELS),
                                               dtype=np.float32)])
                seg = seg.astype(np.float32)

                patches = (seg.reshape(cfg.N_PATCHES, cfg.PATCH_LEN, cfg.CHANNELS)
                              .transpose(2, 0, 1).astype(np.float32))

                patches_3d = seg.reshape(cfg.N_PATCHES, cfg.PATCH_LEN, cfg.CHANNELS)
                local_stats  = np.stack([compute_local_stats(patches_3d[p])
                                         for p in range(cfg.N_PATCHES)])
                global_stats = compute_global_stats(seg)
                tfeat        = time_features_from_ns(int(f))

                self.entries.append((
                    pid,
                    patches,        # (C, NP, PL)
                    tfeat,          # (5,)
                    local_stats,    # (NP, 18)
                    global_stats,   # (8,)
                    int(c2i[lab]),
                    int(f),
                    bool(it),
                ))

            if (p_idx + 1) % 20 == 0 or p_idx + 1 == len(pid_list):
                print(f"  Loaded {p_idx+1}/{len(pid_list)} subjects | "
                      f"{len(self.entries)} windows", end="\r")
        print()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        pid, patches, tfeat, lstats, gstats, lab, fns, it = self.entries[idx]
        return (torch.from_numpy(patches),
                torch.from_numpy(tfeat),
                torch.from_numpy(lstats),
                torch.from_numpy(gstats),
                torch.tensor(lab, dtype=torch.long),
                pid,
                torch.tensor(fns, dtype=torch.long),
                torch.tensor(it,  dtype=torch.bool))


def make_loader(pids, shuffle=False) -> Tuple[HARDataset, DataLoader]:
    ds = HARDataset(pids)
    dl = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=shuffle,
                    num_workers=0, pin_memory=GPU)
    return ds, dl


# ── Subject-level statistics (for D5) ────────────────────────────────────────
def build_subject_stats(ds: HARDataset) -> Dict[str, np.ndarray]:
    """
    Returns {pid: (6,)} — per-subject channel means + stds computed
    from all windows that belong to that subject in the dataset.
    """
    acc: Dict[str, List[np.ndarray]] = defaultdict(list)
    for pid, patches, *_ in ds.entries:
        # patches: (C, NP, PL) → flatten to (C, NP*PL) → per-channel stats
        p = patches.reshape(cfg.CHANNELS, -1)
        acc[pid].append(np.concatenate([p.mean(1), p.std(1)]))   # (6,)
    return {pid: np.mean(np.stack(v, 0), 0).astype(np.float32)
            for pid, v in acc.items()}


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SHARED BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

class ZCRMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__(); self.g = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        x0 = x - x.mean(-1, keepdim=True)
        return x0 / x0.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt() * self.g


class SoftMoE(nn.Module):
    def __init__(self, d, hidden, n_experts=4, dropout=0.1):
        super().__init__()
        self.router  = nn.Linear(d, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden), nn.SiLU(),
                          nn.Dropout(dropout), nn.Linear(hidden, d))
            for _ in range(n_experts)])
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        w = torch.softmax(self.router(x), dim=-1)
        s = torch.stack([e(x) for e in self.experts], dim=-2)
        return self.drop((w.unsqueeze(-1) * s).sum(-2))


class GatedDeltaNet(nn.Module):
    def __init__(self, d, conv_kernel=3, dropout=0.1):
        super().__init__()
        pad = conv_kernel // 2
        self.norm   = ZCRMSNorm(d)
        self.q_lin  = nn.Linear(d, d); self.k_lin = nn.Linear(d, d)
        self.v_lin  = nn.Linear(d, d)
        self.q_conv = nn.Conv1d(d, d, conv_kernel, padding=pad, groups=d)
        self.k_conv = nn.Conv1d(d, d, conv_kernel, padding=pad, groups=d)
        self.v_conv = nn.Conv1d(d, d, conv_kernel, padding=pad, groups=d)
        self.act    = nn.Sigmoid()
        self.alpha  = nn.Linear(d, d); self.beta  = nn.Linear(d, d)
        self.pnorm  = ZCRMSNorm(d); self.post = nn.Linear(d, d)
        self.silu   = nn.SiLU();  self.gate = nn.Sigmoid()
        self.drop   = nn.Dropout(dropout)
    @staticmethod
    def _l2(x, eps=1e-8):
        return x / x.pow(2).sum(-1, keepdim=True).add(eps).sqrt()
    def forward(self, x):
        h = self.norm(x); B, N, D = h.shape
        def _cv(lin, conv, inp):
            return self.act(conv(lin(inp).transpose(1,2)).transpose(1,2))
        q = self._l2(_cv(self.q_lin, self.q_conv, h))
        k = self._l2(_cv(self.k_lin, self.k_conv, h))
        v =          _cv(self.v_lin, self.v_conv, h)
        delta = self.alpha(x) * (q * k * v) + self.beta(x)
        dhat  = self.post(self.pnorm(delta))
        return x + self.drop(self.gate(self.silu(dhat)) * dhat)


# ── RoPE ─────────────────────────────────────────────────────────────────────
def _freqs_cis(head_dim: int, n_tokens: int, theta: float = 10_000.) -> torch.Tensor:
    assert head_dim % 2 == 0
    freqs = 1. / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    return torch.polar(torch.ones(n_tokens, head_dim // 2),
                       torch.outer(torch.arange(n_tokens), freqs))

def _apply_rope(q, k, fc):
    B, H, N, D = q.shape; d2 = D // 2
    fc  = fc[:N].to(q.device).view(1, 1, N, d2)
    q_c = torch.view_as_complex(q.float().contiguous().view(B, H, N, d2, 2))
    k_c = torch.view_as_complex(k.float().contiguous().view(B, H, N, d2, 2))
    return (torch.view_as_real(q_c * fc).view(B, H, N, D).type_as(q),
            torch.view_as_real(k_c * fc).view(B, H, N, D).type_as(k))


# ── Symmetric CGA attention (D0 / D3 share the base, D3 adds gate) ───────────
class SymmetricCGAAttention(nn.Module):
    """Standard CGA attention — symmetric learned bias, used by D0."""
    def __init__(self, d, n_heads, dropout, n_tokens, n_patches,
                 coupled_init=2., decoupled_init=0.):
        super().__init__()
        assert d % n_heads == 0 and (d // n_heads) % 2 == 0
        self.H = n_heads; self.Dh = d // n_heads
        self.qkv  = nn.Linear(d, 3 * d); self.out  = nn.Linear(d, d)
        self.adrop = nn.Dropout(dropout);  self.odrop = nn.Dropout(dropout)
        self.norm  = ZCRMSNorm(d);         self.gate  = nn.Linear(d, d)

        B_init = torch.full((n_tokens, n_tokens), float(decoupled_init))
        for k in range(1, n_patches + 1):
            pi, si = 2 * k - 1, 2 * k
            B_init[pi, si] = coupled_init
            B_init[si, pi] = coupled_init
        self.B = nn.Parameter(B_init)

    def forward(self, x, fc):
        h = self.norm(x); B, N, D = h.shape; H = self.H; Dh = self.Dh
        qkv = self.qkv(h).reshape(B, N, 3, H, Dh).permute(0, 2, 1, 3, 4)
        q, k, v = (qkv[:, i].transpose(1, 2) for i in range(3))
        q, k    = _apply_rope(q, k, fc)
        scores  = (q @ k.transpose(-2, -1)) / (Dh ** 0.5)
        scores  = scores + self.B.unsqueeze(0).unsqueeze(0)
        attn    = self.adrop(torch.softmax(scores, -1))
        y       = self.odrop(self.out(
            (attn @ v).transpose(1, 2).reshape(B, N, D)))
        return x + torch.sigmoid(self.gate(h)) * y


# ══════════════════════════════════════════════════════════════════════════════
# 6.  D0 — BASELINE (CGA-HybridHAR)
# ══════════════════════════════════════════════════════════════════════════════

class D0_Embedding(nn.Module):
    """Interleaved [s_global, p1, s1, ..., pNP, sNP]."""
    def __init__(self):
        super().__init__()
        self.patch_proj  = nn.Linear(cfg.CHANNELS * cfg.PATCH_LEN, cfg.D_MODEL)
        self.time_emb    = nn.Sequential(nn.Linear(5, cfg.D_MODEL), nn.ReLU(),
                                         nn.Dropout(0.1))
        self.local_emb   = nn.Sequential(
            nn.Linear(cfg.LOCAL_STAT_DIM, cfg.D_MODEL),
            nn.LayerNorm(cfg.D_MODEL), nn.ReLU())
        self.global_emb  = nn.Sequential(
            nn.Linear(cfg.GLOBAL_STAT_DIM, cfg.D_MODEL),
            nn.LayerNorm(cfg.D_MODEL), nn.ReLU())
        self.norm = nn.LayerNorm(cfg.D_MODEL)
        self.n_tokens = 1 + 2 * cfg.N_PATCHES

    def forward(self, patches, times, local_stats, global_stats):
        B, C, NP, PL = patches.shape
        x   = patches.permute(0, 2, 1, 3).reshape(B, NP, C * PL)
        t   = self.time_emb(times)
        ptok = self.patch_proj(x) + t.unsqueeze(1)         # (B, NP, D)
        ltok = self.local_emb(local_stats)                  # (B, NP, D)
        gtok = self.global_emb(global_stats).unsqueeze(1)   # (B, 1,  D)
        paired = torch.stack([ptok, ltok], 2).reshape(B, 2*NP, cfg.D_MODEL)
        return self.norm(torch.cat([gtok, paired], 1))      # (B, 1+2NP, D)


class D0_Classifier(nn.Module):
    """Baseline CGA-HybridHAR."""
    name = "D0_Baseline"

    def __init__(self):
        super().__init__()
        NP    = cfg.N_PATCHES
        n_tok = 1 + 2 * NP
        D     = cfg.D_MODEL

        self.embed  = D0_Embedding()
        self.delta1 = GatedDeltaNet(D, dropout=cfg.DROPOUT)
        self.delta2 = GatedDeltaNet(D, dropout=cfg.DROPOUT)
        self.delta3 = GatedDeltaNet(D, dropout=cfg.DROPOUT)
        self.moe1   = SoftMoE(D, 2*D, dropout=cfg.DROPOUT)
        self.attn   = SymmetricCGAAttention(
            D, cfg.N_HEADS, cfg.DROPOUT, n_tok, NP,
            cfg.CGA_COUPLED_INIT, cfg.CGA_DECOUPLED_INIT)
        self.moe2   = SoftMoE(D, 2*D, dropout=cfg.DROPOUT)
        self.cls    = nn.Sequential(nn.Dropout(0.2), nn.Linear(D, D//2),
                                    nn.ReLU(), nn.Dropout(0.1),
                                    nn.Linear(D//2, num_classes))
        self.register_buffer("freqs_cis", _freqs_cis(D // cfg.N_HEADS, n_tok))
        # patch token positions in interleaved sequence
        patch_pos = [2*k-1 for k in range(1, NP+1)]
        self.register_buffer("patch_pos",
                             torch.tensor(patch_pos, dtype=torch.long))

    def forward(self, patches, times, local_stats, global_stats,
                subj_stats=None):
        x  = self.embed(patches, times, local_stats, global_stats)
        x  = self.delta1(x); x = self.delta2(x); x = self.delta3(x)
        x  = x + self.moe1(x)
        x  = self.attn(x, self.freqs_cis)
        x  = x + self.moe2(x)
        out = x[:, self.patch_pos, :].mean(1)
        return self.cls(out)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  D4 — TRANSITION-AWARE LOSS  (model = D0, only loss changes)
# ══════════════════════════════════════════════════════════════════════════════

class TransitionAwareLoss(nn.Module):
    """
    CrossEntropyLoss with per-sample label smoothing.
    Transition windows (adjacent label change) get higher smoothing.
    """
    def __init__(self, K: int, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.K = K
        self.register_buffer("weight", weight if weight is not None
                             else torch.ones(K))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                is_transition: torch.Tensor) -> torch.Tensor:
        lp  = F.log_softmax(logits, -1)                         # (B, K)
        nll = F.nll_loss(lp, targets, weight=self.weight)
        # uniform label smoothing target
        smooth = torch.where(
            is_transition.to(logits.device),
            torch.full_like(is_transition.float(),
                            cfg.TRANSITION_SMOOTH_HIGH),
            torch.full_like(is_transition.float(),
                            cfg.TRANSITION_SMOOTH_LOW))          # (B,)
        unif = (-lp.mean(-1) * smooth).mean()                   # uniform term
        return (1. - smooth.mean()) * nll + unif


class D4_Classifier(D0_Classifier):
    """Baseline model — flag indicates transition-aware loss should be used."""
    name = "D4_TransitionLoss"


# ══════════════════════════════════════════════════════════════════════════════
# 8.  D1 — CROSS-AXIS ATTENTION
#     Within each patch, project each axis independently then apply 3×3 attn
#     before pooling to a single D-dim patch token.
# ══════════════════════════════════════════════════════════════════════════════

class CrossAxisAttention(nn.Module):
    """
    For a single patch position: (B, C, PL) → (B, D)
    1. Project each axis separately: (B, C, PL) → (B, C, D)
    2. Apply C×C self-attention (C=3, tiny 3-head attn)
    3. Mean-pool over C → (B, D)
    The 3×3 attention is interpretable: which axis-pair the model attends to
    reveals the geometric structure of the activity.
    """
    def __init__(self, patch_len: int = cfg.PATCH_LEN,
                 channels: int = cfg.CHANNELS, d_model: int = cfg.D_MODEL):
        super().__init__()
        self.C    = channels
        self.proj = nn.Linear(patch_len, d_model)    # shared across axes
        # tiny C-head attention over axes (C=3 → each head dim = d//C)
        self.norm = nn.LayerNorm(d_model)
        self.qkv  = nn.Linear(d_model, 3 * d_model)
        self.out  = nn.Linear(d_model, d_model)
        self.n_heads = channels                      # 1 head per axis
        assert d_model % channels == 0
        self.head_dim = d_model // channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, PL) → (B, D)"""
        B, C, PL = x.shape
        # Per-axis projection → (B, C, D)
        tok = self.proj(x)               # (B, C, D)
        h   = self.norm(tok)

        # Cross-axis attention
        H, Dh = self.n_heads, self.head_dim
        qkv   = self.qkv(h).reshape(B, C, 3, H, Dh).permute(0, 3, 2, 1, 4)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # (B,H,C,Dh)
        attn  = torch.softmax((q @ k.transpose(-2,-1)) / (Dh**0.5), -1)
        y     = (attn @ v).transpose(1, 2).reshape(B, C, -1)  # (B,C,D)
        tok   = tok + self.out(y)
        return tok.mean(1)                           # (B, D)


class D1_Embedding(nn.Module):
    """
    Replaces flat patch projection with CrossAxisAttention.
    Keeps interleaved CGA structure: [s_global, p1, s1, ..., pNP, sNP].
    """
    def __init__(self):
        super().__init__()
        self.cross_axis = CrossAxisAttention()
        self.time_emb   = nn.Sequential(nn.Linear(5, cfg.D_MODEL), nn.ReLU(),
                                         nn.Dropout(0.1))
        self.local_emb  = nn.Sequential(
            nn.Linear(cfg.LOCAL_STAT_DIM, cfg.D_MODEL),
            nn.LayerNorm(cfg.D_MODEL), nn.ReLU())
        self.global_emb = nn.Sequential(
            nn.Linear(cfg.GLOBAL_STAT_DIM, cfg.D_MODEL),
            nn.LayerNorm(cfg.D_MODEL), nn.ReLU())
        self.norm = nn.LayerNorm(cfg.D_MODEL)
        self.n_tokens = 1 + 2 * cfg.N_PATCHES

    def forward(self, patches, times, local_stats, global_stats):
        B, C, NP, PL = patches.shape
        # Cross-axis attention per patch
        x_in  = patches.permute(0, 2, 1, 3)           # (B, NP, C, PL)
        x_in  = x_in.reshape(B * NP, C, PL)
        ptok  = self.cross_axis(x_in).reshape(B, NP, cfg.D_MODEL)
        t     = self.time_emb(times)
        ptok  = ptok + t.unsqueeze(1)

        ltok  = self.local_emb(local_stats)
        gtok  = self.global_emb(global_stats).unsqueeze(1)
        paired = torch.stack([ptok, ltok], 2).reshape(B, 2*NP, cfg.D_MODEL)
        return self.norm(torch.cat([gtok, paired], 1))


class D1_Classifier(nn.Module):
    """Cross-axis patch attention, same CGA backbone as D0."""
    name = "D1_CrossAxis"

    def __init__(self):
        super().__init__()
        NP    = cfg.N_PATCHES
        n_tok = 1 + 2 * NP
        D     = cfg.D_MODEL
        self.embed  = D1_Embedding()
        self.delta1 = GatedDeltaNet(D, dropout=cfg.DROPOUT)
        self.delta2 = GatedDeltaNet(D, dropout=cfg.DROPOUT)
        self.delta3 = GatedDeltaNet(D, dropout=cfg.DROPOUT)
        self.moe1   = SoftMoE(D, 2*D, dropout=cfg.DROPOUT)
        self.attn   = SymmetricCGAAttention(
            D, cfg.N_HEADS, cfg.DROPOUT, n_tok, NP,
            cfg.CGA_COUPLED_INIT, cfg.CGA_DECOUPLED_INIT)
        self.moe2   = SoftMoE(D, 2*D, dropout=cfg.DROPOUT)
        self.cls    = nn.Sequential(nn.Dropout(0.2), nn.Linear(D, D//2),
                                    nn.ReLU(), nn.Dropout(0.1),
                                    nn.Linear(D//2, num_classes))
        self.register_buffer("freqs_cis", _freqs_cis(D // cfg.N_HEADS, n_tok))
        patch_pos = [2*k-1 for k in range(1, NP+1)]
        self.register_buffer("patch_pos",
                             torch.tensor(patch_pos, dtype=torch.long))

    def forward(self, patches, times, local_stats, global_stats,
                subj_stats=None):
        x   = self.embed(patches, times, local_stats, global_stats)
        x   = self.delta1(x); x = self.delta2(x); x = self.delta3(x)
        x   = x + self.moe1(x)
        x   = self.attn(x, self.freqs_cis)
        x   = x + self.moe2(x)
        out = x[:, self.patch_pos, :].mean(1)
        return self.cls(out)

    def count_params(self): return sum(p.numel() for p in self.parameters()
                                       if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  D2 — HIERARCHICAL PATCH TOKENISATION
#     Fine (NP=10) + coarse (NP//2=5) tokens concatenated before backbone.
#     No CGA interleaving — clean multi-scale comparison.
# ══════════════════════════════════════════════════════════════════════════════

class HierarchicalEmbedding(nn.Module):
    """
    Two-scale tokenisation:
      Fine   : 10 patches of 100 samples  → 10 tokens
      Coarse :  5 groups  of 2 fine patches (pooled) → 5 tokens
      + 1 global stat token
    Total : 16 tokens
    """
    COARSE_FACTOR = 2   # pool every 2 fine patches into 1 coarse

    def __init__(self):
        super().__init__()
        D    = cfg.D_MODEL
        NP   = cfg.N_PATCHES
        CF   = self.COARSE_FACTOR
        assert NP % CF == 0, f"N_PATCHES ({NP}) must be divisible by {CF}"
        self.n_coarse = NP // CF
        self.n_tokens = 1 + NP + self.n_coarse   # global + fine + coarse

        # Fine patch projection
        self.fine_proj   = nn.Linear(cfg.CHANNELS * cfg.PATCH_LEN, D)
        # Coarse patch projection (input: CF consecutive fine patches flattened)
        self.coarse_proj = nn.Linear(cfg.CHANNELS * cfg.PATCH_LEN * CF, D)
        self.time_emb    = nn.Sequential(nn.Linear(5, D), nn.ReLU(),
                                          nn.Dropout(0.1))
        self.global_emb  = nn.Sequential(
            nn.Linear(cfg.GLOBAL_STAT_DIM, D),
            nn.LayerNorm(D), nn.ReLU())
        self.scale_emb   = nn.Embedding(2, D)   # 0=fine, 1=coarse
        self.norm        = nn.LayerNorm(D)

    def forward(self, patches, times, local_stats, global_stats):
        B, C, NP, PL = patches.shape
        D   = cfg.D_MODEL
        CF  = self.COARSE_FACTOR
        NC  = self.n_coarse
        t   = self.time_emb(times)   # (B, D)

        # Fine tokens
        x_fine  = patches.permute(0, 2, 1, 3).reshape(B, NP, C * PL)
        fine_tok = self.fine_proj(x_fine) + t.unsqueeze(1)   # (B, NP, D)
        fine_tok = fine_tok + self.scale_emb(
            torch.zeros(NP, dtype=torch.long, device=patches.device))

        # Coarse tokens: reshape (B, NP, C*PL) → (B, NC, CF*C*PL)
        x_coarse  = x_fine.reshape(B, NC, CF * C * PL)
        coarse_tok = self.coarse_proj(x_coarse) + t.unsqueeze(1)
        coarse_tok = coarse_tok + self.scale_emb(
            torch.ones(NC, dtype=torch.long, device=patches.device))

        # Global stat token
        gtok = self.global_emb(global_stats).unsqueeze(1)

        z = torch.cat([gtok, fine_tok, coarse_tok], 1)   # (B, 1+NP+NC, D)
        return self.norm(z)


class D2_Classifier(nn.Module):
    """Hierarchical dual-scale tokenisation, standard GatedDeltaNet backbone."""
    name = "D2_Hierarchical"

    def __init__(self):
        super().__init__()
        D     = cfg.D_MODEL
        emb   = HierarchicalEmbedding()
        n_tok = emb.n_tokens    # 16
        self.embed  = emb
        self.delta1 = GatedDeltaNet(D, dropout=cfg.DROPOUT)
        self.delta2 = GatedDeltaNet(D, dropout=cfg.DROPOUT)
        self.delta3 = GatedDeltaNet(D, dropout=cfg.DROPOUT)
        self.moe1   = SoftMoE(D, 2*D, dropout=cfg.DROPOUT)
        # Standard attention (no CGA bias — not using interleaved structure)
        head_dim    = D // cfg.N_HEADS
        assert head_dim % 2 == 0
        self.attn_qkv = nn.Linear(D, 3*D)
        self.attn_out = nn.Linear(D, D)
        self.attn_norm = ZCRMSNorm(D)
        self.attn_adrop= nn.Dropout(cfg.DROPOUT)
        self.attn_gate = nn.Linear(D, D)
        self.moe2   = SoftMoE(D, 2*D, dropout=cfg.DROPOUT)
        self.cls    = nn.Sequential(nn.Dropout(0.2), nn.Linear(D, D//2),
                                    nn.ReLU(), nn.Dropout(0.1),
                                    nn.Linear(D//2, num_classes))
        self.register_buffer("freqs_cis", _freqs_cis(head_dim, n_tok))
        # pool over fine patch positions (1..NP)
        fine_pos = list(range(1, cfg.N_PATCHES + 1))
        self.register_buffer("fine_pos",
                             torch.tensor(fine_pos, dtype=torch.long))

    def _attn(self, x):
        h = self.attn_norm(x); B, N, D = h.shape
        H = cfg.N_HEADS; Dh = D // H
        qkv = self.attn_qkv(h).reshape(B, N, 3, H, Dh).permute(0, 2, 1, 3, 4)
        q, k, v = (qkv[:, i].transpose(1,2) for i in range(3))
        q, k = _apply_rope(q, k, self.freqs_cis)
        a = self.attn_adrop(
            torch.softmax((q @ k.transpose(-2,-1)) / (Dh**0.5), -1))
        y = self.attn_out((a @ v).transpose(1,2).reshape(B, N, D))
        return x + torch.sigmoid(self.attn_gate(h)) * y

    def forward(self, patches, times, local_stats, global_stats,
                subj_stats=None):
        x   = self.embed(patches, times, local_stats, global_stats)
        x   = self.delta1(x); x = self.delta2(x); x = self.delta3(x)
        x   = x + self.moe1(x)
        x   = self._attn(x)
        x   = x + self.moe2(x)
        out = x[:, self.fine_pos, :].mean(1)
        return self.cls(out)

    def count_params(self): return sum(p.numel() for p in self.parameters()
                                       if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# 10. D5 — SUBJECT-STAT CGA
#     Replaces local stat tokens with a single subject-level personalisation
#     token.  This gives CGA a real purpose: accessing information no single
#     window can see — the subject's personal movement baseline.
#     Token layout: [s_global, p1, ..., pNP, s_subject]
#     Total tokens = 1 + NP + 1 = 12
# ══════════════════════════════════════════════════════════════════════════════

class D5_Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        D = cfg.D_MODEL
        self.patch_proj = nn.Linear(cfg.CHANNELS * cfg.PATCH_LEN, D)
        self.time_emb   = nn.Sequential(nn.Linear(5, D), nn.ReLU(),
                                         nn.Dropout(0.1))
        self.global_emb = nn.Sequential(
            nn.Linear(cfg.GLOBAL_STAT_DIM, D), nn.LayerNorm(D), nn.ReLU())
        self.subj_emb   = nn.Sequential(
            nn.Linear(cfg.SUBJ_STAT_DIM, D), nn.LayerNorm(D), nn.ReLU())
        self.norm = nn.LayerNorm(D)
        self.n_tokens = 1 + cfg.N_PATCHES + 1   # global + patches + subject

    def forward(self, patches, times, global_stats, subj_stats):
        B, C, NP, PL = patches.shape
        x      = patches.permute(0, 2, 1, 3).reshape(B, NP, C*PL)
        t      = self.time_emb(times)
        ptok   = self.patch_proj(x) + t.unsqueeze(1)
        gtok   = self.global_emb(global_stats).unsqueeze(1)
        stok   = self.subj_emb(subj_stats).unsqueeze(1)
        z      = torch.cat([gtok, ptok, stok], 1)    # (B, 1+NP+1, D)
        return self.norm(z)


class D5_SubjectCGAAttention(nn.Module):
    """
    CGA attention where the bias encodes prior that each patch token should
    attend to the subject stat token (and vice-versa).
    Positions: 0=s_global, 1..NP=patches, NP+1=s_subject
    High bias on all (pi, s_subject) and (s_subject, pi) pairs.
    """
    def __init__(self):
        super().__init__()
        D   = cfg.D_MODEL; NP = cfg.N_PATCHES
        n_tok = 1 + NP + 1   # 12
        assert D % cfg.N_HEADS == 0 and (D // cfg.N_HEADS) % 2 == 0
        H = cfg.N_HEADS; Dh = D // H
        self.H = H; self.Dh = Dh
        self.qkv  = nn.Linear(D, 3*D); self.out  = nn.Linear(D, D)
        self.adrop = nn.Dropout(cfg.DROPOUT); self.norm = ZCRMSNorm(D)
        self.gate  = nn.Linear(D, D)
        # Bias: high coupling between every patch and the subject token
        B_init = torch.zeros(n_tok, n_tok)
        subj_pos = NP + 1
        for k in range(1, NP + 1):
            B_init[k, subj_pos] = cfg.CGA_COUPLED_INIT
            B_init[subj_pos, k] = cfg.CGA_COUPLED_INIT
        self.B = nn.Parameter(B_init)

    def forward(self, x, fc):
        h = self.norm(x); B, N, D = h.shape; H = self.H; Dh = self.Dh
        qkv = self.qkv(h).reshape(B, N, 3, H, Dh).permute(0, 2, 1, 3, 4)
        q, k, v = (qkv[:, i].transpose(1,2) for i in range(3))
        q, k    = _apply_rope(q, k, fc)
        scores  = (q @ k.transpose(-2,-1)) / (Dh**0.5)
        scores  = scores + self.B.unsqueeze(0).unsqueeze(0)
        a       = self.adrop(torch.softmax(scores, -1))
        y       = self.out((a @ v).transpose(1,2).reshape(B, N, D))
        return x + torch.sigmoid(self.gate(h)) * y


class D5_Classifier(nn.Module):
    """Subject-personalisation CGA."""
    name = "D5_SubjectStat"

    def __init__(self):
        super().__init__()
        D     = cfg.D_MODEL; NP = cfg.N_PATCHES
        n_tok = 1 + NP + 1
        self.embed  = D5_Embedding()
        self.delta1 = GatedDeltaNet(D, dropout=cfg.DROPOUT)
        self.delta2 = GatedDeltaNet(D, dropout=cfg.DROPOUT)
        self.delta3 = GatedDeltaNet(D, dropout=cfg.DROPOUT)
        self.moe1   = SoftMoE(D, 2*D, dropout=cfg.DROPOUT)
        self.attn   = D5_SubjectCGAAttention()
        self.moe2   = SoftMoE(D, 2*D, dropout=cfg.DROPOUT)
        self.cls    = nn.Sequential(nn.Dropout(0.2), nn.Linear(D, D//2),
                                    nn.ReLU(), nn.Dropout(0.1),
                                    nn.Linear(D//2, num_classes))
        self.register_buffer("freqs_cis",
                             _freqs_cis(D // cfg.N_HEADS, n_tok))
        patch_pos = list(range(1, NP + 1))
        self.register_buffer("patch_pos",
                             torch.tensor(patch_pos, dtype=torch.long))

    def forward(self, patches, times, local_stats, global_stats,
                subj_stats=None):
        if subj_stats is None:
            subj_stats = torch.zeros(patches.size(0), cfg.SUBJ_STAT_DIM,
                                     device=patches.device)
        x   = self.embed(patches, times, global_stats, subj_stats)
        x   = self.delta1(x); x = self.delta2(x); x = self.delta3(x)
        x   = x + self.moe1(x)
        x   = self.attn(x, self.freqs_cis)
        x   = x + self.moe2(x)
        out = x[:, self.patch_pos, :].mean(1)
        return self.cls(out)

    def count_params(self): return sum(p.numel() for p in self.parameters()
                                       if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# 11. D3 — ASYMMETRIC GATED CGA
#     Patch→stat attention is gated by patch uncertainty.
#     Gate opens (≈1) when patch representation has low L2 norm (ambiguous).
#     Stat→patch and patch→patch attention are unrestricted.
# ══════════════════════════════════════════════════════════════════════════════

class AsymmetricCGAAttention(nn.Module):
    """
    Same interleaved structure as D0 (21 tokens).
    Modification: when computing patch-token output, the contribution from
    stat tokens is multiplied by a learned gate derived from that patch's
    representation.  Stat tokens attend to everything freely.

    Mechanically:
      1. Compute full attention  y_full = softmax(QK^T/√d + B) V
      2. For each patch position pi: gate_i = σ(g_proj(h_pi))  ∈ (0,1)^D
      3. For patch positions, additionally compute y_patch2stat via a masked
         softmax (patch queries → stat keys only)
      4. out[pi] = y_full[pi] + gate_i ⊙ y_patch2stat[pi]
    """
    def __init__(self, n_patches: int = cfg.N_PATCHES):
        super().__init__()
        D   = cfg.D_MODEL; H = cfg.N_HEADS
        NP  = n_patches; n_tok = 1 + 2 * NP
        assert D % H == 0 and (D // H) % 2 == 0
        self.H = H; self.Dh = D // H; self.NP = NP; self.n_tok = n_tok

        self.qkv    = nn.Linear(D, 3*D); self.out    = nn.Linear(D, D)
        self.adrop  = nn.Dropout(cfg.DROPOUT); self.norm   = ZCRMSNorm(D)
        self.g_proj = nn.Linear(D, D)           # uncertainty gate
        self.r_proj = nn.Linear(D, D)           # stat residual proj
        self.gate_drop = nn.Dropout(0.1)
        self.g_out  = nn.Linear(D, D)

        # CGA bias (same as D0)
        B_init = torch.full((n_tok, n_tok), float(cfg.CGA_DECOUPLED_INIT))
        for k in range(1, NP + 1):
            pi, si = 2*k-1, 2*k
            B_init[pi, si] = cfg.CGA_COUPLED_INIT
            B_init[si, pi] = cfg.CGA_COUPLED_INIT
        self.B = nn.Parameter(B_init)

        # Stat token positions (si = 2k for k=1..NP, plus position 0 = global)
        stat_pos = [0] + [2*k for k in range(1, NP+1)]
        self.register_buffer("stat_pos",
                             torch.tensor(stat_pos, dtype=torch.long))
        patch_pos = [2*k-1 for k in range(1, NP+1)]
        self.register_buffer("patch_pos",
                             torch.tensor(patch_pos, dtype=torch.long))

    def forward(self, x, fc):
        h = self.norm(x); B, N, D = h.shape; H = self.H; Dh = self.Dh
        qkv = self.qkv(h).reshape(B, N, 3, H, Dh).permute(0, 2, 1, 3, 4)
        q, k, v = (qkv[:, i].transpose(1,2) for i in range(3))
        q, k    = _apply_rope(q, k, fc)

        # Full attention with CGA bias
        scores = (q @ k.transpose(-2,-1)) / (Dh**0.5) \
                 + self.B.unsqueeze(0).unsqueeze(0)
        a_full = self.adrop(torch.softmax(scores, -1))
        y_full = (a_full @ v).transpose(1,2).reshape(B, N, D)  # (B,N,D)
        y_full = self.out(y_full)

        # Gated additional patch→stat cross-attention
        # Queries: patch positions only  (B, NP, D)
        q_p = q[:, :, self.patch_pos, :]                 # (B, H, NP, Dh)
        k_s = k[:, :, self.stat_pos,  :]                 # (B, H, NS, Dh)
        v_s = v[:, :, self.stat_pos,  :]                 # (B, H, NS, Dh)
        NS  = len(self.stat_pos)
        sc2 = (q_p @ k_s.transpose(-2,-1)) / (Dh**0.5)  # (B,H,NP,NS)
        a2  = torch.softmax(sc2, -1)
        y2  = (a2 @ v_s).transpose(1,2).reshape(B, self.NP, D)
        y2  = self.r_proj(y2)                            # (B, NP, D)

        # Gate: small norm → uncertain → gate opens
        h_patch = h[:, self.patch_pos, :]                # (B, NP, D)
        # norm-based uncertainty: low norm = ambiguous
        uncertainty = 1. - torch.sigmoid(
            h_patch.norm(dim=-1, keepdim=True) - h_patch.norm(dim=-1).mean())
        gate = torch.sigmoid(self.g_proj(h_patch)) * uncertainty
        gate = self.gate_drop(gate)

        # Build output: start from full attention, add gated stat residual
        y = y_full.clone()
        y[:, self.patch_pos, :] = y[:, self.patch_pos, :] + gate * y2

        return x + y


class D3_Classifier(nn.Module):
    """Asymmetric gated CGA — patch→stat attention controlled by uncertainty."""
    name = "D3_AsymmetricCGA"

    def __init__(self):
        super().__init__()
        NP    = cfg.N_PATCHES; D = cfg.D_MODEL; n_tok = 1 + 2 * NP
        self.embed  = D0_Embedding()
        self.delta1 = GatedDeltaNet(D, dropout=cfg.DROPOUT)
        self.delta2 = GatedDeltaNet(D, dropout=cfg.DROPOUT)
        self.delta3 = GatedDeltaNet(D, dropout=cfg.DROPOUT)
        self.moe1   = SoftMoE(D, 2*D, dropout=cfg.DROPOUT)
        self.attn   = AsymmetricCGAAttention(n_patches=NP)
        self.moe2   = SoftMoE(D, 2*D, dropout=cfg.DROPOUT)
        self.cls    = nn.Sequential(nn.Dropout(0.2), nn.Linear(D, D//2),
                                    nn.ReLU(), nn.Dropout(0.1),
                                    nn.Linear(D//2, num_classes))
        self.register_buffer("freqs_cis",
                             _freqs_cis(D // cfg.N_HEADS, n_tok))
        patch_pos = [2*k-1 for k in range(1, NP+1)]
        self.register_buffer("patch_pos",
                             torch.tensor(patch_pos, dtype=torch.long))

    def forward(self, patches, times, local_stats, global_stats,
                subj_stats=None):
        x   = self.embed(patches, times, local_stats, global_stats)
        x   = self.delta1(x); x = self.delta2(x); x = self.delta3(x)
        x   = x + self.moe1(x)
        x   = self.attn(x, self.freqs_cis)
        x   = x + self.moe2(x)
        out = x[:, self.patch_pos, :].mean(1)
        return self.cls(out)

    def count_params(self): return sum(p.numel() for p in self.parameters()
                                       if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# 12. METRICS / HMM
# ══════════════════════════════════════════════════════════════════════════════

def cohen_kappa(yt, yp) -> float:
    cm = confusion_matrix(yt, yp); n = cm.sum()
    if n == 0: return 0.
    po = np.trace(cm) / n; pe = np.dot(cm.sum(1), cm.sum(0)) / (n*n)
    return (po - pe) / (1-pe) if abs(1-pe) > 1e-12 else 0.

def mcc_score(yt, yp) -> float:
    cm = confusion_matrix(yt, yp).astype(float); n = cm.sum()
    if n == 0: return 0.
    s = np.trace(cm); t = cm.sum(1); p = cm.sum(0)
    num = s*n - np.dot(t,p)
    den = math.sqrt(max(n**2 - np.sum(t**2), 0.) * max(n**2 - np.sum(p**2), 0.))
    return num/den if den > 0 else 0.

def compute_metrics(yt, yp) -> Dict:
    return {
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "kappa":    float(cohen_kappa(yt, yp)),
        "mcc":      float(mcc_score(yt, yp)),
        "accuracy": float((np.array(yt) == np.array(yp)).mean()),
    }

def estimate_hmm(ds: HARDataset):
    by_pid = defaultdict(list)
    for e in ds.entries:
        pid, lab, fn = e[0], e[5], e[6]
        by_pid[pid].append((fn, lab))
    A  = np.full((num_classes, num_classes), cfg.HMM_SMOOTH)
    pi = np.full(num_classes, cfg.HMM_SMOOTH)
    for pid, seq in by_pid.items():
        seq.sort(key=lambda x: x[0])
        if not seq: continue
        pi[seq[0][1]] += 1
        for (_, a), (_, b) in zip(seq[:-1], seq[1:]):
            A[a, b] += 1
    A  = np.clip(A / A.sum(1, keepdims=True), cfg.HMM_MIN_PROB, 1.)
    pi = np.clip(pi / pi.sum(), cfg.HMM_MIN_PROB, 1.)
    return pi, A

def viterbi(E, log_pi, log_A) -> np.ndarray:
    T, K = E.shape
    dp   = np.full((T, K), -np.inf); bp = np.full((T, K), -1, dtype=np.int32)
    dp[0] = log_pi + E[0]
    for t in range(1, T):
        prev  = dp[t-1][:, None] + log_A
        bp[t] = np.argmax(prev, 0)
        dp[t] = prev[bp[t], np.arange(K)] + E[t]
    path = np.zeros(T, dtype=np.int32); path[-1] = int(np.argmax(dp[-1]))
    for t in range(T-2, -1, -1): path[t] = bp[t+1, path[t+1]]
    return path

def class_weights(ds: HARDataset) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.int64)
    for e in ds.entries: counts[e[5]] += 1
    w = np.clip(counts.max() / np.clip(counts, 1, None), 1., 10.)
    w = torch.tensor(w / w.sum() * num_classes, dtype=torch.float32)
    return w


# ══════════════════════════════════════════════════════════════════════════════
# 13. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def mixup(patches, times, lstats, gstats, labels, alpha=0.2):
    if alpha <= 0.:
        return patches, times, lstats, gstats, labels, labels, 1.
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(patches.size(0), device=patches.device)
    return (lam*patches  + (1-lam)*patches[idx],
            lam*times    + (1-lam)*times[idx],
            lam*lstats   + (1-lam)*lstats[idx],
            lam*gstats   + (1-lam)*gstats[idx],
            labels, labels[idx], lam)

def tc_loss(logits):
    if logits.size(0) < 2: return logits.new_zeros(1).squeeze()
    p = F.softmax(logits[:-1], -1); q = F.softmax(logits[1:], -1)
    return .5 * (F.kl_div(q.log(), p, reduction="batchmean") +
                 F.kl_div(p.log(), q, reduction="batchmean"))


def train_one_run(model: nn.Module,
                  train_dl: DataLoader, val_dl: DataLoader,
                  train_ds: HARDataset,
                  use_transition_loss: bool = False,
                  subj_stats_map: Optional[Dict[str, np.ndarray]] = None,
                  save_path: Optional[Path] = None) -> float:
    """
    Returns training wall-clock time in seconds.
    Saves best checkpoint to save_path if provided.
    """
    cw        = class_weights(train_ds).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR,
                            weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.LR,
        steps_per_epoch=len(train_dl), epochs=cfg.EPOCHS,
        pct_start=0.10, anneal_strategy="cos")

    if use_transition_loss:
        criterion = TransitionAwareLoss(num_classes, weight=cw)
    else:
        criterion = nn.CrossEntropyLoss(weight=cw)

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=GPU)
    except TypeError:
        from torch.cuda.amp import GradScaler as _GS
        scaler = _GS(enabled=GPU)

    best_score, patience = -1e9, 0
    t0 = time.perf_counter()

    for epoch in range(cfg.EPOCHS):
        model.train()
        total_loss = 0.

        for batch in train_dl:
            patches, times, lstats, gstats, labels, pids, _, is_trans = batch
            patches = patches.to(DEVICE).float()
            times   = times.to(DEVICE).float()
            lstats  = lstats.to(DEVICE).float()
            gstats  = gstats.to(DEVICE).float()
            labels  = labels.to(DEVICE).view(-1)
            is_trans= is_trans.to(DEVICE)

            # Build subject stats tensor for D5
            ss = None
            if subj_stats_map is not None:
                ss = torch.from_numpy(
                    np.stack([subj_stats_map.get(pid,
                              np.zeros(cfg.SUBJ_STAT_DIM, dtype=np.float32))
                              for pid in pids])).to(DEVICE).float()

            patches, times, lstats, gstats, la, lb, lam = mixup(
                patches, times, lstats, gstats, labels, cfg.MIXUP_ALPHA)

            optimizer.zero_grad(set_to_none=True)
            with _amp():
                logits = model(patches, times, lstats, gstats, subj_stats=ss)
                if use_transition_loss:
                    loss = (lam * criterion(logits, la, is_trans) +
                            (1-lam) * criterion(logits, lb, is_trans))
                else:
                    loss = (lam * criterion(logits, la) +
                            (1-lam) * criterion(logits, lb))
                if cfg.TC_LAMBDA > 0:
                    loss = loss + cfg.TC_LAMBDA * tc_loss(logits)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            ok = all(p.grad is None or torch.isfinite(p.grad).all()
                     for p in model.parameters()) and torch.isfinite(loss)
            if ok:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
                scaler.step(optimizer)
            else:
                optimizer.zero_grad(set_to_none=True)
            scaler.update(); scheduler.step()
            if torch.isfinite(loss): total_loss += float(loss.item())

        # ── Validation ───────────────────────────────────────────────────
        model.eval(); vp, vt = [], []
        with torch.no_grad():
            for batch in val_dl:
                patches, times, lstats, gstats, labels, pids, _, _ = batch
                patches = patches.to(DEVICE).float()
                times   = times.to(DEVICE).float()
                lstats  = lstats.to(DEVICE).float()
                gstats  = gstats.to(DEVICE).float()
                ss = None
                if subj_stats_map is not None:
                    ss = torch.from_numpy(
                        np.stack([subj_stats_map.get(
                            pid, np.zeros(cfg.SUBJ_STAT_DIM, np.float32))
                            for pid in pids])).to(DEVICE).float()
                p = model(patches, times, lstats, gstats,
                          subj_stats=ss).argmax(1)
                vp.extend(p.cpu().numpy().tolist())
                vt.extend(labels.numpy().tolist())

        vp, vt = np.array(vp), np.array(vt)
        f1  = float(f1_score(vt, vp, average="macro", zero_division=0))
        kap = float(cohen_kappa(vt, vp))
        lr  = optimizer.param_groups[0]["lr"]
        print(f"  Ep {epoch+1:02d}/{cfg.EPOCHS}  "
              f"LR {lr:.1e}  loss {total_loss/max(1,len(train_dl)):.4f}  "
              f"F1 {f1:.4f}  κ {kap:.4f}")
        score = f1 + kap
        if score > best_score + 1e-6:
            best_score, patience = score, 0
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"    Saved ✓")
        else:
            patience += 1
            if patience >= cfg.EARLY_STOP_PATIENCE:
                print(f"  Early stop epoch {epoch+1}")
                break

    return time.perf_counter() - t0


# ══════════════════════════════════════════════════════════════════════════════
# 14. EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_run(model: nn.Module, test_dl: DataLoader, train_ds: HARDataset,
                 subj_stats_map: Optional[Dict] = None) -> Dict:
    model.eval()
    raw_pred, raw_true = [], []
    by_pid_probs  = defaultdict(list)
    by_pid_truth  = defaultdict(list)
    by_pid_time   = defaultdict(list)

    for batch in test_dl:
        patches, times, lstats, gstats, labels, pids, firsts, _ = batch
        patches = patches.to(DEVICE).float()
        times   = times.to(DEVICE).float()
        lstats  = lstats.to(DEVICE).float()
        gstats  = gstats.to(DEVICE).float()
        ss = None
        if subj_stats_map is not None:
            ss = torch.from_numpy(
                np.stack([subj_stats_map.get(pid,
                          np.zeros(cfg.SUBJ_STAT_DIM, np.float32))
                          for pid in pids])).to(DEVICE).float()
        probs = torch.softmax(
            model(patches, times, lstats, gstats, subj_stats=ss), -1
        ).cpu().numpy()
        for pr, lb, pid, fn in zip(probs, labels.numpy(), pids, firsts.numpy()):
            raw_pred.append(int(np.argmax(pr)))
            raw_true.append(int(lb))
            by_pid_probs[pid].append(pr)
            by_pid_truth[pid].append(int(lb))
            by_pid_time[pid].append(int(fn))

    m_raw = compute_metrics(raw_true, raw_pred)

    pi, A = estimate_hmm(train_ds)
    lpi   = np.log(np.clip(pi, cfg.HMM_MIN_PROB, 1.))
    lA    = np.log(np.clip(A,  cfg.HMM_MIN_PROB, 1.))
    hmm_p, hmm_t = [], []
    for pid in by_pid_probs:
        order = np.argsort(by_pid_time[pid])
        E     = np.log(np.clip(
            np.vstack([by_pid_probs[pid][i] for i in order]),
            cfg.HMM_MIN_PROB, 1.))
        hmm_p.extend(viterbi(E, lpi, lA).tolist())
        hmm_t.extend([by_pid_truth[pid][i] for i in order])

    m_hmm = compute_metrics(hmm_t, hmm_p)
    cm    = confusion_matrix(hmm_t, hmm_p, labels=np.arange(num_classes))
    return {"raw": m_raw, "hmm": m_hmm,
            "confusion_matrix": cm.tolist(),
            "n_test_windows": len(raw_true)}


# ══════════════════════════════════════════════════════════════════════════════
# 15. INFERENCE BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def benchmark_model(model: nn.Module, test_dl: DataLoader,
                    subj_stats_map: Optional[Dict] = None) -> Dict:
    model.eval()
    if GPU: torch.cuda.reset_peak_memory_stats(DEVICE)

    # Collect enough samples
    all_p, all_t, all_l, all_g, all_s = [], [], [], [], []
    N_need = cfg.N_WARMUP + cfg.N_BENCH
    for batch in test_dl:
        patches, times, lstats, gstats, _, pids, _, _ = batch
        all_p.append(patches); all_t.append(times)
        all_l.append(lstats);  all_g.append(gstats)
        if subj_stats_map is not None:
            ss = np.stack([subj_stats_map.get(pid,
                           np.zeros(cfg.SUBJ_STAT_DIM, np.float32))
                           for pid in pids])
            all_s.append(torch.from_numpy(ss))
        if sum(x.shape[0] for x in all_p) >= N_need: break

    AP = torch.cat(all_p)[:N_need].to(DEVICE).float()
    AT = torch.cat(all_t)[:N_need].to(DEVICE).float()
    AL = torch.cat(all_l)[:N_need].to(DEVICE).float()
    AG = torch.cat(all_g)[:N_need].to(DEVICE).float()
    AS = (torch.cat(all_s)[:N_need].to(DEVICE).float()
          if all_s else None)

    # Warm-up
    for i in range(min(cfg.N_WARMUP, AP.shape[0])):
        model(AP[i:i+1], AT[i:i+1], AL[i:i+1], AG[i:i+1],
              subj_stats=AS[i:i+1] if AS is not None else None)
    if GPU: torch.cuda.synchronize()

    # Single-sample latency
    n_meas = min(cfg.N_BENCH, AP.shape[0] - cfg.N_WARMUP)
    lat_ms = []
    for i in range(cfg.N_WARMUP, cfg.N_WARMUP + n_meas):
        ss_i = AS[i:i+1] if AS is not None else None
        if GPU:
            ev_s = torch.cuda.Event(enable_timing=True)
            ev_e = torch.cuda.Event(enable_timing=True)
            ev_s.record()
            model(AP[i:i+1], AT[i:i+1], AL[i:i+1], AG[i:i+1], subj_stats=ss_i)
            ev_e.record(); torch.cuda.synchronize()
            lat_ms.append(ev_s.elapsed_time(ev_e))
        else:
            t0 = time.perf_counter()
            model(AP[i:i+1], AT[i:i+1], AL[i:i+1], AG[i:i+1], subj_stats=ss_i)
            lat_ms.append((time.perf_counter()-t0)*1000)
    lat_ms = np.array(lat_ms)

    # Batch throughput
    BS  = min(cfg.BATCH_SIZE, AP.shape[0])
    bp  = AP[:BS]; bt = AT[:BS]; bl = AL[:BS]; bg = AG[:BS]
    bs2 = AS[:BS] if AS is not None else None
    for _ in range(5):
        model(bp, bt, bl, bg, subj_stats=bs2)
    if GPU: torch.cuda.synchronize()
    t0 = time.perf_counter()
    N_REPS = 50
    for _ in range(N_REPS):
        model(bp, bt, bl, bg, subj_stats=bs2)
    if GPU: torch.cuda.synchronize()
    throughput = N_REPS * BS / (time.perf_counter() - t0)
    peak_mem_mb = torch.cuda.max_memory_allocated(DEVICE)/1e6 if GPU else 0.

    return {
        "lat_mean_ms":  float(lat_ms.mean()),
        "lat_std_ms":   float(lat_ms.std()),
        "lat_p50_ms":   float(np.percentile(lat_ms, 50)),
        "lat_p95_ms":   float(np.percentile(lat_ms, 95)),
        "lat_p99_ms":   float(np.percentile(lat_ms, 99)),
        "throughput_wps": float(throughput),
        "peak_mem_mb":  float(peak_mem_mb),
        "n_params":     sum(p.numel() for p in model.parameters()
                            if p.requires_grad),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 16. CGA BIAS VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def save_cga_heatmap(model: nn.Module, name: str):
    """Save learned CGA bias matrix as a heatmap."""
    # Find the attention module
    attn = getattr(model, "attn", None)
    if attn is None or not hasattr(attn, "B"):
        return
    B_mat = attn.B.cpu().numpy()
    N     = B_mat.shape[0]
    show  = min(N, 21)
    B_s   = B_mat[:show, :show]

    tl = ["s_g"]
    for k in range(1, cfg.N_PATCHES + 1):
        tl += [f"p{k}", f"s{k}"]
    tl_s = tl[:show]

    fig, ax = plt.subplots(figsize=(9, 7))
    lim = max(abs(B_s).max(), 0.1)
    im  = ax.imshow(B_s, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(show)); ax.set_xticklabels(tl_s, rotation=90, fontsize=7)
    ax.set_yticks(range(show)); ax.set_yticklabels(tl_s, fontsize=7)
    ax.set_title(f"Learned CGA Bias B — {name} (first {show} tokens)", fontsize=10)
    plt.tight_layout()
    p = cfg.OUTPUT_DIR / f"cga_bias_{name}.png"
    plt.savefig(p, dpi=180); plt.close()
    print(f"  CGA bias heatmap → {p.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 17. EXPERIMENT DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENTS = [
    {
        "name":               "D0_Baseline",
        "description":        "CGA-HybridHAR baseline (21-token interleaved, symmetric CGA bias)",
        "model_cls":          D0_Classifier,
        "use_transition_loss":False,
        "needs_subj_stats":   False,
    },
    {
        "name":               "D4_TransitionLoss",
        "description":        "D0 + transition-aware label smoothing (loss modification only)",
        "model_cls":          D4_Classifier,
        "use_transition_loss":True,
        "needs_subj_stats":   False,
    },
    {
        "name":               "D1_CrossAxis",
        "description":        "Per-patch 3×3 cross-axis attention replaces flat projection",
        "model_cls":          D1_Classifier,
        "use_transition_loss":False,
        "needs_subj_stats":   False,
    },
    {
        "name":               "D1D4_CrossAxis_TransLoss",
        "description":        "D1 cross-axis + D4 transition loss (best-of-both)",
        "model_cls":          D1_Classifier,
        "use_transition_loss":True,
        "needs_subj_stats":   False,
    },
    {
        "name":               "D2_Hierarchical",
        "description":        "Dual-scale tokens: 10 fine + 5 coarse patches",
        "model_cls":          D2_Classifier,
        "use_transition_loss":False,
        "needs_subj_stats":   False,
    },
    {
        "name":               "D2D4_Hierarchical_TransLoss",
        "description":        "D2 hierarchical + D4 transition loss",
        "model_cls":          D2_Classifier,
        "use_transition_loss":True,
        "needs_subj_stats":   False,
    },
    {
        "name":               "D5_SubjectStat",
        "description":        "Subject-level personalisation token replaces local stat tokens",
        "model_cls":          D5_Classifier,
        "use_transition_loss":False,
        "needs_subj_stats":   True,
    },
    {
        "name":               "D3_AsymmetricCGA",
        "description":        "Patch→stat attention gated by patch uncertainty",
        "model_cls":          D3_Classifier,
        "use_transition_loss":False,
        "needs_subj_stats":   False,
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# 18. RESULTS REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def print_comparison_table(all_results: Dict):
    cols = ["macro_f1(hmm)", "kappa(hmm)", "mcc(hmm)", "accuracy(hmm)",
            "macro_f1(raw)", "lat_mean_ms", "throughput_wps", "n_params",
            "train_time_s"]
    W = 20
    print("\n" + "═"*90)
    print("  DIRECTION COMPARISON TABLE")
    print("═"*90)
    header = f"  {'Direction':<28}" + "".join(f"{c[:W]:>{W}}" for c in cols)
    print(header); print("  " + "─"*88)
    for name, res in all_results.items():
        vals = {
            "macro_f1(hmm)":   f"{res['eval']['hmm']['macro_f1']:.4f}",
            "kappa(hmm)":      f"{res['eval']['hmm']['kappa']:.4f}",
            "mcc(hmm)":        f"{res['eval']['hmm']['mcc']:.4f}",
            "accuracy(hmm)":   f"{res['eval']['hmm']['accuracy']:.4f}",
            "macro_f1(raw)":   f"{res['eval']['raw']['macro_f1']:.4f}",
            "lat_mean_ms":     f"{res['bench']['lat_mean_ms']:.3f}",
            "throughput_wps":  f"{res['bench']['throughput_wps']:.0f}",
            "n_params":        f"{res['bench']['n_params']:,}",
            "train_time_s":    f"{res['train_time_s']:.0f}",
        }
        row = f"  {name:<28}" + "".join(f"{vals[c]:>{W}}" for c in cols)
        print(row)
    print("═"*90)


def save_results(all_results: Dict):
    # ── JSON ──────────────────────────────────────────────────────────────
    def _to_py(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):    return obj.tolist()
        if isinstance(obj, dict):  return {k: _to_py(v) for k,v in obj.items()}
        if isinstance(obj, list):  return [_to_py(v) for v in obj]
        return obj

    json_path = cfg.OUTPUT_DIR / "directions_results.json"
    with open(json_path, "w") as f:
        json.dump(_to_py(all_results), f, indent=2)

    # ── Summary CSV ────────────────────────────────────────────────────────
    rows = []
    for name, res in all_results.items():
        row = {"direction": name,
               "description": res.get("description", "")}
        for split in ("raw", "hmm"):
            for metric in ("macro_f1", "kappa", "mcc", "accuracy"):
                row[f"{metric}_{split}"] = res["eval"][split][metric]
        for bk in ("lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms",
                   "throughput_wps", "peak_mem_mb", "n_params"):
            row[bk] = res["bench"][bk]
        row["train_time_s"] = res["train_time_s"]
        rows.append(row)
    csv_path = cfg.OUTPUT_DIR / "directions_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # ── Confusion matrix PNGs ──────────────────────────────────────────────
    for name, res in all_results.items():
        cm = np.array(res["eval"]["confusion_matrix"])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes, square=True)
        f1 = res["eval"]["hmm"]["macro_f1"]
        plt.title(f"{name} +HMM | F1={f1:.3f}"); plt.tight_layout()
        plt.savefig(cfg.OUTPUT_DIR / f"cm_{name}.png", dpi=150)
        plt.close()

    print(f"\n  Saved: {json_path.name}  |  {csv_path.name}")
    print(f"  Confusion matrices + CGA heatmaps → {cfg.OUTPUT_DIR}/")


# ══════════════════════════════════════════════════════════════════════════════
# 19. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    total_t0 = time.perf_counter()
    print("\n" + "═"*72)
    print("  HAR Directions Study  —  5 directions vs CGA-HybridHAR baseline")
    print("═"*72)

    # ── Load data once ────────────────────────────────────────────────────
    print("\n── Loading data ─────────────────────────────────────────────────")
    print("  Train subjects …")
    train_ds, train_dl = make_loader(train_pids, shuffle=True)
    print("  Val subjects …")
    val_ds,   val_dl   = make_loader(val_pids,   shuffle=False)
    print("  Test subjects …")
    test_ds,  test_dl  = make_loader(test_pids,  shuffle=False)
    print(f"  Windows — train={len(train_ds):,}  val={len(val_ds):,}  "
          f"test={len(test_ds):,}")

    # Subject stats for D5 (computed from training set; test subjects use own data)
    print("  Building subject-stat map (for D5) …")
    train_subj_stats = build_subject_stats(train_ds)
    test_subj_stats  = build_subject_stats(test_ds)
    subj_stats_map   = {**train_subj_stats, **test_subj_stats}

    all_results: Dict = {}

    # ── Run each experiment ───────────────────────────────────────────────
    for exp in EXPERIMENTS:
        name = exp["name"]
        print(f"\n{'═'*72}")
        print(f"  EXPERIMENT : {name}")
        print(f"  {exp['description']}")
        print(f"{'═'*72}")

        model = exp["model_cls"]().to(DEVICE)
        print(f"  Params     : {model.count_params():,}")
        NP    = cfg.N_PATCHES
        n_tok = getattr(getattr(model, "embed", None), "n_tokens",
                        1 + 2*NP)
        print(f"  N_tokens   : {n_tok}")

        ssmap = subj_stats_map if exp["needs_subj_stats"] else None
        save_path = cfg.OUTPUT_DIR / f"weights_{name}.pth"

        # Train
        print(f"\n  Training …")
        train_secs = train_one_run(
            model, train_dl, val_dl, train_ds,
            use_transition_loss=exp["use_transition_loss"],
            subj_stats_map=ssmap,
            save_path=save_path)

        # Load best checkpoint
        if save_path.exists():
            model.load_state_dict(torch.load(save_path, map_location=DEVICE))

        # CGA bias heatmap (if applicable)
        save_cga_heatmap(model, name)

        # Evaluate
        print(f"\n  Evaluating …")
        eval_res  = evaluate_run(model, test_dl, train_ds, ssmap)
        bench_res = benchmark_model(model, test_dl, ssmap)

        m = eval_res["hmm"]
        print(f"\n  ── RESULTS ─────────────────────────────────────────────────")
        print(f"     Macro-F1  (HMM) : {m['macro_f1']:.4f}")
        print(f"     Accuracy  (HMM) : {m['accuracy']:.4f}")
        print(f"     Cohen κ   (HMM) : {m['kappa']:.4f}")
        print(f"     MCC       (HMM) : {m['mcc']:.4f}")
        print(f"     Macro-F1  (raw) : {eval_res['raw']['macro_f1']:.4f}")
        print(f"  ── BENCH ───────────────────────────────────────────────────")
        print(f"     Latency   : {bench_res['lat_mean_ms']:.3f} ± "
              f"{bench_res['lat_std_ms']:.3f} ms")
        print(f"     p50/p95/p99: {bench_res['lat_p50_ms']:.3f} / "
              f"{bench_res['lat_p95_ms']:.3f} / {bench_res['lat_p99_ms']:.3f} ms")
        print(f"     Throughput: {bench_res['throughput_wps']:.0f} win/s")
        print(f"     Peak GPU  : {bench_res['peak_mem_mb']:.1f} MB")
        print(f"     Train time: {train_secs:.0f} s")

        all_results[name] = {
            "description":  exp["description"],
            "eval":         eval_res,
            "bench":        bench_res,
            "train_time_s": float(train_secs),
        }

        # Free GPU memory between experiments
        del model
        if GPU: torch.cuda.empty_cache()
        gc.collect()

    # ── Final comparison ──────────────────────────────────────────────────
    print_comparison_table(all_results)
    save_results(all_results)

    total_h = (time.perf_counter() - total_t0) / 3600.
    print(f"\n  Total wall-clock: {total_h:.2f} h")
    print("  Done.")


if __name__ == "__main__":
    main()
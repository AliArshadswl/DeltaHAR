#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REALWORLD2016 Ablation Study  (GPU-optimised)
=============================================
Compares three feature configurations across 5-fold subject-wise CV:

  ┌─────────────┬─────────┬──────────┬──────────────┐
  │  Ablation   │ Patches │ Stat (56)│  Topo (24)   │
  ├─────────────┼─────────┼──────────┼──────────────┤
  │ full        │   ✓     │    ✓     │      ✓       │
  │ no_stat     │   ✓     │    ✗     │      ✓       │
  │ no_topo     │   ✓     │    ✓     │      ✗       │
  └─────────────┴─────────┴──────────┴──────────────┘

Research-paper metrics collected per ablation:
  • Macro-F1, Accuracy, Cohen κ, MCC  (mean ± std, 5 folds)
  • Per-class Precision / Recall / F1
  • Aggregated confusion matrix
  • Trainable parameter count
  • Training wall-clock time (s/fold)
  • Single-sample GPU latency — mean, std, p50, p95, p99 (ms)
  • Batch throughput (windows/second)
  • Peak GPU memory (MB)
  • Dataset statistics (windows/class, total windows)

Outputs saved to cfg.OUTPUT_DIR:
  ablation_summary.json          – full nested results
  ablation_scalar_metrics.csv    – headline numbers for LaTeX tables
  per_class_f1.csv               – per-class F1 by ablation
  fold_level_metrics.csv         – every fold, every ablation
"""

# ══════════════════════════════════════════════════════════════════════════════
# 0. IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import gc, json, math, os, random, time, warnings
warnings.filterwarnings("ignore")

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
)
from torch.utils.data import DataLoader, Dataset


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION  (edit only this block)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    # ── Dataset ────────────────────────────────────────────────────────────
    DATA_PATH:    str   = "/mnt/share/ali/realworld2016_dataset/"
    SUBJECTS:     tuple = tuple(range(1, 16))        # proband1 … proband15
    SENSORS:      tuple = ("acc", "gyr", "mag")
    BODY_PART:    str   = "shin"
    OVERLAP:      float = 0.5
    SIGNAL_RATE:  int   = 50                          # Hz
    WINDOW_SEC:   int   = 10
    N_PATCHES:    int   = 10
    PATCH_LEN:    int   = 50                          # 10 × 50 = 500 samples

    # ── Model ──────────────────────────────────────────────────────────────
    D_MODEL:  int   = 56
    N_HEADS:  int   = 4
    N_LAYERS: int   = 4
    DROPOUT:  float = 0.3
    N_STAT:   int   = 56   # statistical feature dim  — do NOT change
    N_TOPO:   int   = 24   # topological feature dim  — do NOT change

    # ── Training ───────────────────────────────────────────────────────────
    BATCH_SIZE:          int   = 64
    EPOCHS:              int   = 120
    LR:                  float = 1e-3
    WEIGHT_DECAY:        float = 1e-4
    MAX_GRAD_NORM:       float = 1.0
    EARLY_STOP_PATIENCE: int   = 30

    # ── Benchmarking ───────────────────────────────────────────────────────
    N_WARMUP:    int = 100   # single-sample GPU warm-up iterations
    N_LATENCY:   int = 500   # single-sample timing iterations
    BENCH_BS:    int = 64    # batch size for throughput test

    # ── Misc ───────────────────────────────────────────────────────────────
    SEED:        int   = 42
    NUM_WORKERS: int   = 4
    OUTPUT_DIR:  str   = "./ablation_results"
    ABLATIONS:   tuple = ("full", "no_stat", "no_topo")


cfg = Config()

# ── Constants derived from config ─────────────────────────────────────────
ACTIVITY_LABELS: Dict[str, int] = {
    "climbingdown": 0, "climbingup": 1, "jumping": 2, "lying": 3,
    "running": 4,      "sitting": 5,    "standing": 6, "walking": 7,
}
CLASSES     = sorted(ACTIVITY_LABELS, key=ACTIVITY_LABELS.get)
NUM_CLASSES = len(CLASSES)
WIN_SAMPLES = cfg.N_PATCHES * cfg.PATCH_LEN          # 500

SENSOR_PREFIX = {"acc": "acc", "gyr": "Gyroscope", "mag": "MagneticField"}

ABLATION_LABEL = {
    "full":    "Full (patches+stat+topo)",
    "no_stat": "No-Stat (patches+topo)",
    "no_topo": "No-Topo (patches+stat)",
}

# ── Reproducibility ───────────────────────────────────────────────────────
random.seed(cfg.SEED)
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU    = DEVICE.type == "cuda"
if GPU:
    torch.cuda.manual_seed_all(cfg.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False   # set True for fixed-shape speed-up (non-repro)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

print("═" * 72)
print("  REALWORLD2016 Ablation Study")
print("═" * 72)
print(f"  Device     : {DEVICE}", end="")
if GPU:
    props = torch.cuda.get_device_properties(0)
    print(f"  │  {props.name}  │  {props.total_memory/1e9:.1f} GB VRAM", end="")
print()
print(f"  Ablations  : {cfg.ABLATIONS}")
print(f"  Subjects   : {len(cfg.SUBJECTS)}")
print(f"  Window     : {cfg.WINDOW_SEC}s  ({WIN_SAMPLES} samples @ {cfg.SIGNAL_RATE} Hz)")
print(f"  Patches    : {cfg.N_PATCHES} × {cfg.PATCH_LEN}")
print(f"  Epochs     : {cfg.EPOCHS}  (patience={cfg.EARLY_STOP_PATIENCE})")
print(f"  Output dir : {cfg.OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _min_time_len(readme_bytes: bytes) -> int:
    lengths = []
    for line in readme_bytes.splitlines():
        if line.startswith(b"> entries"):
            try:
                lengths.append(int(line.split()[2]))
            except Exception:
                pass
    return min(lengths) if lengths else 1000


def _load_xyz(zf: ZipFile, prefix: str, label: str, body: str,
              length: int) -> np.ndarray:
    name = f"{prefix}_{label}_{body}.csv"
    if name not in zf.namelist():
        return np.full((length, 3), np.nan, dtype=np.float32)
    try:
        df  = pd.read_csv(zf.open(name))
        xyz = df[["attr_x", "attr_y", "attr_z"]].values[:length].astype(np.float32)
        if xyz.shape[0] < length:
            pad = np.full((length - xyz.shape[0], 3), np.nan, dtype=np.float32)
            xyz = np.concatenate([xyz, pad], axis=0)
        return xyz
    except Exception:
        return np.full((length, 3), np.nan, dtype=np.float32)


def _quality_ok(w: np.ndarray, max_nan: float = 0.10,
                max_zero: float = 0.90) -> bool:
    n = w.size
    if np.isnan(w).sum() / n > max_nan:  return False
    if (w == 0).sum()       / n > max_zero: return False
    unique = [len(np.unique(w[:, c][np.isfinite(w[:, c])])) for c in range(w.shape[1])]
    if sum(u <= 1 for u in unique) > 10: return False
    finite = w[np.isfinite(w)]
    if finite.size and (finite.max() > 1e4 or finite.min() < -1e4): return False
    return True


def _slide_windows(data: np.ndarray, win: int,
                   overlap: float) -> List[np.ndarray]:
    stride = max(1, int(win * (1 - overlap)))
    return [
        data[i : i + win]
        for i in range(0, len(data) - win + 1, stride)
        if _quality_ok(data[i : i + win])
    ]


def load_subject_activity(sub_id: int, label: str) -> List[np.ndarray]:
    zl = label.replace("_", "")
    open_zips: Dict[str, Optional[ZipFile]] = {}
    for s in cfg.SENSORS:
        path = os.path.join(
            cfg.DATA_PATH, f"proband{sub_id}", "data", f"{s}_{zl}_csv.zip"
        )
        try:
            open_zips[s] = ZipFile(path) if os.path.exists(path) else None
        except Exception:
            open_zips[s] = None

    # determine shared time length
    length = 1000
    for zf in open_zips.values():
        if zf is None:
            continue
        try:
            length = _min_time_len(zf.open("readMe").read())
            if length > 0:
                break
        except Exception:
            pass

    # load & concatenate sensor channels  →  (T, C)
    parts = []
    for s in cfg.SENSORS:
        zf = open_zips.get(s)
        if zf is None:
            parts.append(np.full((length, 3), np.nan, dtype=np.float32))
        else:
            parts.append(_load_xyz(zf, SENSOR_PREFIX[s], zl, cfg.BODY_PART, length))

    for zf in open_zips.values():
        if zf:
            zf.close()

    min_len = min(p.shape[0] for p in parts)
    data    = np.concatenate([p[:min_len] for p in parts], axis=1)
    return _slide_windows(data, WIN_SAMPLES, cfg.OVERLAP)


def generate_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns X (N,T,C), y (N,), subj (N,)."""
    activities = list(ACTIVITY_LABELS.keys())
    subjects   = list(cfg.SUBJECTS)
    total      = len(subjects) * len(activities)
    done       = 0
    X_list, y_list, s_list = [], [], []

    for sub_id in subjects:
        for act in activities:
            wins = load_subject_activity(sub_id, act)
            if wins:
                X_list.extend(wins)
                y_list.extend([ACTIVITY_LABELS[act]] * len(wins))
                s_list.extend([sub_id]               * len(wins))
            done += 1
            print(f"\r  Loading: {done}/{total}  windows={len(X_list)}", end="", flush=True)
    print()

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.int64),
        np.array(s_list, dtype=np.int64),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

try:
    from scipy.signal import find_peaks, peak_prominences
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

try:
    from ripser import ripser
    _HAVE_RIPSER = True
except ImportError:
    _HAVE_RIPSER = False

print(f"  scipy   : {'✓' if _HAVE_SCIPY else '✗ (peak features degraded)'}")
print(f"  ripser  : {'✓' if _HAVE_RIPSER else '✗ (topological fallback active)'}")


# ── Statistical helpers ──────────────────────────────────────────────────────

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = float(np.std(a)), float(np.std(b))
    if sa < 1e-8 or sb < 1e-8:
        return 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    return 0.0 if not np.isfinite(c) else c

def _mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))

def _skew(x: np.ndarray) -> float:
    mu, sd = float(np.mean(x)), float(np.std(x))
    if sd < 1e-12: return 0.0
    return float(np.mean((x - mu) ** 3)) / (sd ** 3)

def _kurt(x: np.ndarray) -> float:
    mu, sd = float(np.mean(x)), float(np.std(x))
    if sd < 1e-12: return 0.0
    return float(np.mean((x - mu) ** 4)) / (sd ** 4) - 3.0

def _norm_psd(x: np.ndarray, sr: int):
    ps = (np.abs(np.fft.rfft(x)) ** 2).astype(np.float64)
    s  = ps.sum()
    if s > 0: ps /= s
    return ps, np.fft.rfftfreq(len(x), 1.0 / sr)

def _spec_entropy(p: np.ndarray) -> float:
    p = p[np.isfinite(p) & (p > 0)]
    if p.size == 0: return 0.0
    H = -float(np.sum(p * np.log(p)))
    return H / math.log(len(p)) if len(p) > 1 else 0.0

def _dom2(ps: np.ndarray, fq: np.ndarray) -> Tuple:
    if ps.size <= 2: return 0., 0., 0., 0.
    p = ps.copy(); p[0] = 0.
    i1 = int(np.argmax(p)); p1 = p[i1]; f1 = fq[i1]; p[i1] = -1.
    i2 = int(np.argmax(p)); p2 = max(p[i2], 0.); f2 = fq[i2]
    return float(f1), float(p1), float(f2), float(p2)

def _autocorr1s(x: np.ndarray, sr: int) -> float:
    L = min(sr, len(x) - 1)
    if L <= 1: return 0.
    return _safe_corr(x[:-L], x[L:])

def _lowpass_g(a: np.ndarray, sr: int, fc: float = 0.5) -> np.ndarray:
    alpha = math.exp(-2. * math.pi * fc / sr)
    g = np.zeros_like(a); g[0] = a[0]
    for t in range(1, len(a)):
        g[t] = alpha * g[t - 1] + (1 - alpha) * a[t]
    return g


def compute_stat_features(w: np.ndarray, sr: int = 50) -> np.ndarray:
    """
    56 statistical features from normalised window (T, C).
    Uses first 3 channels (accelerometer triad proxy).
    """
    tri = w[:, :3]; x, y, z = tri[:, 0], tri[:, 1], tri[:, 2]
    mag = np.linalg.norm(tri, axis=1)
    f   = []

    # (A) per-axis mean / std / range  →  9 features
    for sig in (x, y, z):
        f += [float(np.mean(sig)), float(np.std(sig)),
              float(np.max(sig) - np.min(sig))]

    # (B) pairwise correlations  →  3 features
    f += [_safe_corr(x, y), _safe_corr(x, z), _safe_corr(y, z)]

    # (C) magnitude descriptors  →  7 features
    f += [float(np.mean(mag)), float(np.std(mag)),
          float(np.max(mag) - np.min(mag)), _mad(mag),
          _kurt(mag), _skew(mag), float(np.median(mag))]

    # (D) quantile pack (min/max/q50/q25/q75)  →  4×5 = 20 features
    for sig in (x, y, z, mag):
        q25, q50, q75 = np.percentile(sig, [25, 50, 75])
        f += [float(np.min(sig)), float(np.max(sig)),
              float(q50), float(q25), float(q75)]

    # (E) 1-second autocorrelation  →  1 feature
    f.append(_autocorr1s(mag, sr))

    # (F) spectral  →  5 features
    ps, fq = _norm_psd(mag, sr)
    f1, p1, f2, p2 = _dom2(ps, fq)
    f += [f1, p1, f2, p2, _spec_entropy(ps)]

    # (G) peak count + median prominence  →  2 features
    if _HAVE_SCIPY:
        pks, _ = find_peaks(mag)
        prom   = peak_prominences(mag, pks)[0] if pks.size else np.array([0.])
        f += [float(len(pks)), float(np.median(prom))]
    else:
        pk = int(((mag[1:-1] > mag[:-2]) & (mag[1:-1] > mag[2:])).sum()) if len(mag) >= 3 else 0
        f += [float(pk), 0.]

    # (H) gravity & dynamic angles  →  9 features
    g = _lowpass_g(tri, sr=sr)
    d = tri - g
    # gravity mean angles (roll, pitch, yaw proxy)  →  3
    rx = np.arctan2(g[:, 1], g[:, 2])
    ry = np.arctan2(-g[:, 0], np.sqrt(g[:, 1] ** 2 + g[:, 2] ** 2) + 1e-8)
    rz = np.arctan2(g[:, 0], g[:, 1])
    f += [float(np.mean(rx)), float(np.mean(ry)), float(np.mean(rz))]
    # dynamic mean + std angles  →  6
    dx = np.arctan2(d[:, 1], d[:, 2])
    dy = np.arctan2(-d[:, 0], np.sqrt(d[:, 1] ** 2 + d[:, 2] ** 2) + 1e-8)
    dz = np.arctan2(d[:, 0], d[:, 1])
    for arr in (dx, dy, dz):
        f += [float(np.mean(arr)), float(np.std(arr))]

    # 9+3+7+20+1+5+2+3+6 = 56
    f = [0. if not np.isfinite(v) else v for v in f]
    assert len(f) == cfg.N_STAT, f"stat feature mismatch: got {len(f)}, expected {cfg.N_STAT}"
    return np.array(f, dtype=np.float32)


# ── Topological helpers ──────────────────────────────────────────────────────

def _pers_entropy(diag: np.ndarray) -> float:
    if diag.size == 0: return 0.
    b, d = diag[:, 0], diag[:, 1]; fin = np.isfinite(d)
    p = np.maximum(d[fin] - b[fin], 0.); s = p.sum()
    if s <= 0: return 0.
    p /= s
    return float(-np.sum(p * np.log(p + 1e-12)))

def _topk_lifetimes(diag: np.ndarray, k: int = 3) -> List[float]:
    if diag.size == 0: return [0.] * k
    b, d = diag[:, 0], diag[:, 1]; fin = np.isfinite(d)
    lt  = np.sort(np.maximum(d[fin] - b[fin], 0.))[::-1]
    out = np.zeros(k, dtype=np.float32)
    out[:min(k, len(lt))] = lt[:k]
    return out.tolist()

def _fake_diag(dist: np.ndarray, thr: float) -> np.ndarray:
    M   = (dist < thr).astype(np.float32)
    runs = []
    hi   = min(10, M.shape[0] - 1)
    for k in range(-hi, hi + 1):
        diag_k = np.diag(M, k=k); run = 0
        for v in diag_k:
            if v > .5: run += 1
            elif run:  runs.append(run); run = 0
        if run: runs.append(run)
    if not runs: return np.empty((0, 2), dtype=np.float32)
    lt = np.array(runs, dtype=np.float32)
    return np.stack([np.zeros_like(lt), lt], axis=1)


def compute_topo_features(w: np.ndarray, sr: int = 50,
                          m: int = 3, tau: int = 5,
                          max_pts: int = 600) -> np.ndarray:
    """
    24 topological features via persistent homology.
    Uses Takens delay-embedding of the accelerometer magnitude.
    Falls back to recurrence proxy if ripser unavailable.
    """
    tri = w[:, :3]
    mag = np.linalg.norm(tri, axis=1).astype(np.float32)
    if len(mag) > max_pts:
        mag = mag[:: int(math.ceil(len(mag) / max_pts))]

    T = len(mag); L = T - (m - 1) * tau
    if L < 5:
        return np.zeros(cfg.N_TOPO, dtype=np.float32)

    Xemb = np.stack([mag[i : i + L] for i in range(0, m * tau, tau)], axis=1).astype(np.float32)

    if _HAVE_RIPSER and L >= 8:
        res = ripser(Xemb, maxdim=1)
        D0, D1 = res["dgms"][0], res["dgms"][1]
    else:
        dist = np.abs(np.subtract.outer(mag, mag))
        fin  = dist[np.isfinite(dist)]
        qs   = np.quantile(fin, [0.2, 0.5]) if fin.size else [0., 0.]
        D0 = _fake_diag(dist, qs[0])
        D1 = _fake_diag(dist, qs[1])

    f = []
    for D in (D0, D1):
        if D.size == 0:
            f += [0.] * 12
            continue
        b, d2 = D[:, 0], D[:, 1]; fin = np.isfinite(d2)
        b, d2 = b[fin], d2[fin]; p = np.maximum(d2 - b, 0.)
        f += [
            float(p.max()  if p.size else 0.),
            float(p.mean() if p.size else 0.),
            float(p.sum()  if p.size else 0.),
            _pers_entropy(D),
        ]
        f += _topk_lifetimes(D, k=3)
        f += [
            float(b.max()  if b.size else 0.),
            float(d2.max() if d2.size else 0.),
        ]
        if p.size >= 5:
            qs = np.quantile(p, [0.5, 0.75, 0.9])
            f += [float((p > qs[i]).sum()) for i in range(3)]
        else:
            f += [0., 0., 0.]

    # 2 × 12 = 24
    assert len(f) == cfg.N_TOPO, f"topo feature mismatch: got {len(f)}, expected {cfg.N_TOPO}"
    return np.array(f, dtype=np.float32)


def precompute_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-compute all stat + topo features in RAM (done once, shared across ablations)."""
    N   = X.shape[0]
    sf  = np.zeros((N, cfg.N_STAT), dtype=np.float32)
    tf  = np.zeros((N, cfg.N_TOPO), dtype=np.float32)
    t0  = time.perf_counter()
    for i in range(N):
        mu     = X[i].mean(0, keepdims=True)
        sd     = X[i].std(0,  keepdims=True)
        normed = np.clip((X[i] - mu) / (sd + 1e-8), -10, 10)
        sf[i]  = compute_stat_features(normed, cfg.SIGNAL_RATE)
        tf[i]  = compute_topo_features(normed, cfg.SIGNAL_RATE)
        if (i + 1) % 200 == 0 or (i + 1) == N:
            elapsed = time.perf_counter() - t0
            eta     = elapsed / (i + 1) * (N - i - 1)
            print(f"\r  Feature pre-computation: {i+1}/{N}  "
                  f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)", end="", flush=True)
    print()
    return sf, tf


# ══════════════════════════════════════════════════════════════════════════════
# 4. DATASET & DATALOADER
# ══════════════════════════════════════════════════════════════════════════════

class HARDataset(Dataset):
    def __init__(self, X, y, subj, sfeat, tfeat, idxs):
        self.X     = X[idxs]
        self.y     = y[idxs]
        self.subj  = subj[idxs]
        self.sfeat = sfeat[idxs]
        self.tfeat = tfeat[idxs]
        assert self.X.shape[1] == WIN_SAMPLES, \
            f"Window length mismatch: {self.X.shape[1]} vs {WIN_SAMPLES}"

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        w      = self.X[i]
        mu     = w.mean(0, keepdims=True)
        sd     = w.std(0,  keepdims=True)
        normed = np.clip((w - mu) / (sd + 1e-8), -10, 10).astype(np.float32)
        C      = normed.shape[1]
        # (C, N_PATCHES, PATCH_LEN)
        patches = normed.reshape(cfg.N_PATCHES, cfg.PATCH_LEN, C).transpose(2, 0, 1)
        return (
            torch.from_numpy(patches),
            torch.from_numpy(self.sfeat[i]),
            torch.from_numpy(self.tfeat[i]),
            torch.tensor(self.y[i], dtype=torch.long),
            int(self.subj[i]),
        )


def make_loader(X, y, subj, sfeat, tfeat, idxs,
                shuffle: bool = False, drop_last: bool = False) -> Tuple[HARDataset, DataLoader]:
    ds = HARDataset(X, y, subj, sfeat, tfeat, idxs)
    dl = DataLoader(
        ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=GPU,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )
    return ds, dl


# ══════════════════════════════════════════════════════════════════════════════
# 5. MODEL  (ablation-aware PatchTST + RoPE)
# ══════════════════════════════════════════════════════════════════════════════

def precompute_freqs_cis(head_dim: int, n_tokens: int,
                         theta: float = 10_000.) -> torch.Tensor:
    assert head_dim % 2 == 0
    freqs = 1. / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t     = torch.arange(n_tokens)
    return torch.polar(torch.ones(n_tokens, head_dim // 2),
                       torch.outer(t, freqs))   # complex64


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B, H, N, D = q.shape
    d2  = D // 2
    fc  = freqs_cis[:N].to(q.device).unsqueeze(0).unsqueeze(0)   # (1,1,N,d2)
    q_c = torch.view_as_complex(q.float().contiguous().view(B, H, N, d2, 2))
    k_c = torch.view_as_complex(k.float().contiguous().view(B, H, N, d2, 2))
    q_r = torch.view_as_real(q_c * fc).view(B, H, N, D)
    k_r = torch.view_as_real(k_c * fc).view(B, H, N, D)
    return q_r.type_as(q), k_r.type_as(k)


class PatchEmbedding(nn.Module):
    """
    Projects raw patches + (optional) stat/topo tokens into d_model space.
    n_tokens = N_PATCHES + int(use_stat) + int(use_topo)
    """
    def __init__(self, channels: int, use_stat: bool, use_topo: bool):
        super().__init__()
        self.use_stat   = use_stat
        self.use_topo   = use_topo
        in_dim          = channels * cfg.PATCH_LEN
        self.patch_proj = nn.Linear(in_dim, cfg.D_MODEL)
        if use_stat:
            self.stat_proj = nn.Linear(cfg.N_STAT, cfg.D_MODEL)
        if use_topo:
            self.topo_proj = nn.Linear(cfg.N_TOPO, cfg.D_MODEL)
        self.norm     = nn.LayerNorm(cfg.D_MODEL)
        self.n_tokens = cfg.N_PATCHES + int(use_stat) + int(use_topo)

    def forward(self, patches, stats, topo):
        # patches: (B, C, NP, PL)
        B, C, NP, PL = patches.shape
        x      = patches.permute(0, 2, 1, 3).reshape(B, NP, C * PL)
        tokens = [self.patch_proj(x)]
        if self.use_stat:
            tokens.append(self.stat_proj(stats).unsqueeze(1))
        if self.use_topo:
            tokens.append(self.topo_proj(topo).unsqueeze(1))
        return self.norm(torch.cat(tokens, dim=1))    # (B, n_tokens, D)


class RoPELayer(nn.Module):
    def __init__(self):
        super().__init__()
        D, H        = cfg.D_MODEL, cfg.N_HEADS
        assert D % H == 0 and (D // H) % 2 == 0
        self.H      = H
        self.Dh     = D // H
        self.qkv    = nn.Linear(D, 3 * D)
        self.proj   = nn.Linear(D, D)
        self.norm1  = nn.LayerNorm(D)
        self.norm2  = nn.LayerNorm(D)
        self.ffn    = nn.Sequential(
            nn.Linear(D, 4 * D), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(4 * D, D),
        )
        self.drop   = nn.Dropout(cfg.DROPOUT)

    def forward(self, x, freqs_cis):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.H, self.Dh).permute(0, 3, 2, 1, 4)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]   # (B, H, N, Dh)
        q, k    = apply_rope(q, k, freqs_cis)
        attn    = (q @ k.transpose(-2, -1)) / (self.Dh ** 0.5)
        out     = (attn.softmax(-1) @ v).transpose(1, 2).reshape(B, N, D)
        x       = self.norm1(x + self.drop(self.proj(out)))
        return    self.norm2(x + self.drop(self.ffn(x)))


class PatchTSTClassifier(nn.Module):
    def __init__(self, channels: int, use_stat: bool, use_topo: bool):
        super().__init__()
        self.embed    = PatchEmbedding(channels, use_stat, use_topo)
        n_tok         = self.embed.n_tokens
        self.backbone = nn.ModuleList([RoPELayer() for _ in range(cfg.N_LAYERS)])
        head_dim      = cfg.D_MODEL // cfg.N_HEADS
        self.register_buffer("freqs_cis", precompute_freqs_cis(head_dim, n_tok))
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(cfg.D_MODEL, cfg.D_MODEL // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.D_MODEL // 2, NUM_CLASSES),
        )

    def forward(self, patches, stats, topo):
        x = self.embed(patches, stats, topo)
        for layer in self.backbone:
            x = layer(x, self.freqs_cis)
        return self.head(x.mean(dim=1))

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# 6. TRAINING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _amp_ctx():
    if GPU:
        try:
            return torch.autocast("cuda", dtype=torch.float16)
        except TypeError:
            from torch.cuda.amp import autocast
            return autocast()
    from contextlib import nullcontext
    return nullcontext()


def class_weights_tensor(y_train: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(float)
    w      = counts.max() / np.clip(counts, 1, None)
    w      = torch.tensor(w, dtype=torch.float32)
    return  w / w.sum() * NUM_CLASSES


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred); n = cm.sum()
    if n == 0: return 0.
    po = np.trace(cm) / n
    pe = np.dot(cm.sum(1), cm.sum(0)) / (n * n)
    return (po - pe) / (1 - pe) if abs(1 - pe) > 1e-12 else 0.


def multiclass_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred).astype(float); n = cm.sum()
    if n == 0: return 0.
    s   = np.trace(cm); t = cm.sum(1); p = cm.sum(0)
    num = s * n - np.dot(t, p)
    den = math.sqrt(max(n**2 - np.sum(t**2), 0.) * max(n**2 - np.sum(p**2), 0.))
    return num / den if den > 0 else 0.


def train_fold(tr_dl: DataLoader, va_dl: DataLoader,
               model: nn.Module) -> Tuple[nn.Module, float]:
    """Train with early stopping; returns (best_model, wall_clock_seconds)."""
    cw        = class_weights_tensor(tr_dl.dataset.y).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR,
                            weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                      T_max=cfg.EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=cw)
    scaler    = torch.cuda.amp.GradScaler(enabled=GPU)

    best_score, best_state, bad = -1e9, None, 0
    t0 = time.perf_counter()

    for epoch in range(cfg.EPOCHS):
        # ── Training pass ─────────────────────────────────────────────────
        model.train()
        running_loss = 0.
        for patches, sfeat, tfeat, labels, _ in tr_dl:
            patches = patches.to(DEVICE, non_blocking=True)
            sfeat   = sfeat.to(DEVICE, non_blocking=True)
            tfeat   = tfeat.to(DEVICE, non_blocking=True)
            labels  = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _amp_ctx():
                loss = criterion(model(patches, sfeat, tfeat), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
            scaler.step(optimizer); scaler.update()
            running_loss += loss.item()
        scheduler.step()

        # ── Validation pass ───────────────────────────────────────────────
        model.eval()
        preds, truth = [], []
        with torch.no_grad():
            for patches, sfeat, tfeat, labels, _ in va_dl:
                patches = patches.to(DEVICE, non_blocking=True)
                sfeat   = sfeat.to(DEVICE, non_blocking=True)
                tfeat   = tfeat.to(DEVICE, non_blocking=True)
                p = model(patches, sfeat, tfeat).argmax(1).cpu().numpy()
                preds.extend(p); truth.extend(labels.numpy())
        preds = np.array(preds); truth = np.array(truth)
        f1    = f1_score(truth, preds, average="macro", zero_division=0)
        kap   = cohen_kappa(truth, preds)
        score = f1 + kap
        avg_l = running_loss / max(1, len(tr_dl))
        print(f"    Ep {epoch+1:03d}/{cfg.EPOCHS}  "
              f"loss={avg_l:.4f}  val_F1={f1:.4f}  val_κ={kap:.4f}")

        if score > best_score + 1e-6:
            best_score = score
            best_state = {key: val.cpu().clone()
                          for key, val in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.EARLY_STOP_PATIENCE:
                print(f"    ↳ Early stop at epoch {epoch+1}")
                break

    train_secs = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_secs


# ══════════════════════════════════════════════════════════════════════════════
# 7. EVALUATION  (test set)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(te_dl: DataLoader, model: nn.Module) -> Dict:
    model.eval()
    preds, truth = [], []
    for patches, sfeat, tfeat, labels, _ in te_dl:
        patches = patches.to(DEVICE, non_blocking=True)
        sfeat   = sfeat.to(DEVICE, non_blocking=True)
        tfeat   = tfeat.to(DEVICE, non_blocking=True)
        p = model(patches, sfeat, tfeat).argmax(1).cpu().numpy()
        preds.extend(p); truth.extend(labels.numpy())
    preds = np.array(preds); truth = np.array(truth)

    acc  = float((preds == truth).mean())
    f1   = f1_score(truth, preds, average="macro", zero_division=0)
    kap  = cohen_kappa(truth, preds)
    mcc  = multiclass_mcc(truth, preds)
    pc_p = precision_score(truth, preds, average=None, zero_division=0,
                           labels=list(range(NUM_CLASSES)))
    pc_r = recall_score(truth, preds, average=None, zero_division=0,
                        labels=list(range(NUM_CLASSES)))
    pc_f = f1_score(truth, preds, average=None, zero_division=0,
                    labels=list(range(NUM_CLASSES)))
    cm   = confusion_matrix(truth, preds, labels=list(range(NUM_CLASSES)))

    return dict(
        acc=acc, f1=f1, kappa=kap, mcc=mcc,
        per_class_precision = pc_p.tolist(),
        per_class_recall    = pc_r.tolist(),
        per_class_f1        = pc_f.tolist(),
        confusion_matrix    = cm.tolist(),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 8. LATENCY & THROUGHPUT BENCHMARKING
# ══════════════════════════════════════════════════════════════════════════════

def benchmark(model: nn.Module, te_dl: DataLoader) -> Dict:
    """
    Single-sample latency (CUDA events or CPU timer) + batch throughput.
    """
    model.eval()
    if GPU:
        torch.cuda.reset_peak_memory_stats(DEVICE)

    # ── Collect samples ────────────────────────────────────────────────────
    N_need  = cfg.N_WARMUP + cfg.N_LATENCY
    all_p, all_s, all_t = [], [], []
    for patches, sfeat, tfeat, _, _ in te_dl:
        all_p.append(patches); all_s.append(sfeat); all_t.append(tfeat)
        if sum(x.shape[0] for x in all_p) >= N_need:
            break
    all_p = torch.cat(all_p)[:N_need].to(DEVICE)
    all_s = torch.cat(all_s)[:N_need].to(DEVICE)
    all_t = torch.cat(all_t)[:N_need].to(DEVICE)

    # ── Warm-up ────────────────────────────────────────────────────────────
    with torch.no_grad():
        for i in range(min(cfg.N_WARMUP, all_p.shape[0])):
            _ = model(all_p[i:i+1], all_s[i:i+1], all_t[i:i+1])
    if GPU:
        torch.cuda.synchronize()

    # ── Single-sample latency ──────────────────────────────────────────────
    lat_ms = []
    n_meas = min(cfg.N_LATENCY, all_p.shape[0] - cfg.N_WARMUP)
    with torch.no_grad():
        for i in range(cfg.N_WARMUP, cfg.N_WARMUP + n_meas):
            if GPU:
                ev_s = torch.cuda.Event(enable_timing=True)
                ev_e = torch.cuda.Event(enable_timing=True)
                ev_s.record()
                _ = model(all_p[i:i+1], all_s[i:i+1], all_t[i:i+1])
                ev_e.record()
                torch.cuda.synchronize()
                lat_ms.append(ev_s.elapsed_time(ev_e))
            else:
                t0 = time.perf_counter()
                _ = model(all_p[i:i+1], all_s[i:i+1], all_t[i:i+1])
                lat_ms.append((time.perf_counter() - t0) * 1000.)
    lat_ms = np.array(lat_ms)

    # ── Batch throughput ───────────────────────────────────────────────────
    BS     = min(cfg.BENCH_BS, all_p.shape[0])
    bp, bs_t, bt = all_p[:BS], all_s[:BS], all_t[:BS]
    N_reps = 50
    with torch.no_grad():                          # batch warm-up
        for _ in range(5):
            _ = model(bp, bs_t, bt)
    if GPU: torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N_reps):
            _ = model(bp, bs_t, bt)
    if GPU: torch.cuda.synchronize()
    elapsed       = time.perf_counter() - t0
    throughput    = (N_reps * BS) / elapsed

    # ── Peak GPU memory ────────────────────────────────────────────────────
    peak_mem_mb = torch.cuda.max_memory_allocated(DEVICE) / 1e6 if GPU else 0.

    return dict(
        lat_mean_ms    = float(lat_ms.mean()),
        lat_std_ms     = float(lat_ms.std()),
        lat_p50_ms     = float(np.percentile(lat_ms, 50)),
        lat_p95_ms     = float(np.percentile(lat_ms, 95)),
        lat_p99_ms     = float(np.percentile(lat_ms, 99)),
        throughput_wps = float(throughput),
        peak_mem_mb    = float(peak_mem_mb),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 9. CROSS-VALIDATION SPLITS
# ══════════════════════════════════════════════════════════════════════════════

def subject_folds(subj_ids: np.ndarray, seed: int = 42,
                  n_folds: int = 5) -> List[Tuple]:
    uniq = np.unique(subj_ids)
    rng  = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n      = len(uniq)
    n_test = max(1, round(0.20 * n))
    n_val  = max(1, round(0.10 * n))
    folds  = []
    for i in range(n_folds):
        start   = (i * n_test) % n
        test_s  = [uniq[(start + j) % n] for j in range(n_test)]
        val_s   = [uniq[(start + n_test + j) % n] for j in range(n_val)]
        train_s = [s for s in uniq if s not in set(test_s) and s not in set(val_s)]
        folds.append((np.array(train_s), np.array(val_s), np.array(test_s)))
    return folds


def subj_mask(all_subj: np.ndarray, keep: np.ndarray) -> np.ndarray:
    keep_set = set(int(s) for s in keep)
    return np.array([i for i, s in enumerate(all_subj)
                     if int(s) in keep_set], dtype=np.int64)


# ══════════════════════════════════════════════════════════════════════════════
# 10. ABLATION RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_ablation(X: np.ndarray, y: np.ndarray, subj: np.ndarray,
                 sfeat: np.ndarray, tfeat: np.ndarray) -> Dict:

    folds    = subject_folds(subj, seed=cfg.SEED)
    channels = X.shape[2]
    results  = {}

    for ablation in cfg.ABLATIONS:
        use_stat = (ablation != "no_stat")
        use_topo = (ablation != "no_topo")

        print(f"\n{'═'*72}")
        print(f"  ABLATION : {ABLATION_LABEL[ablation]}")
        print(f"{'═'*72}")

        fold_metrics = []

        for fold_id, (tr_s, va_s, te_s) in enumerate(folds, 1):
            print(f"\n  ── Fold {fold_id}/5 ─────────────────────────────────────────────────")
            print(f"     train={len(tr_s)} subj  val={len(va_s)} subj  test={len(te_s)} subj")

            idx_tr = subj_mask(subj, tr_s)
            idx_va = subj_mask(subj, va_s)
            idx_te = subj_mask(subj, te_s)

            tr_ds, tr_dl = make_loader(X, y, subj, sfeat, tfeat, idx_tr,
                                       shuffle=True, drop_last=True)
            va_ds, va_dl = make_loader(X, y, subj, sfeat, tfeat, idx_va)
            te_ds, te_dl = make_loader(X, y, subj, sfeat, tfeat, idx_te)

            cc   = Counter(tr_ds.y.tolist())
            hist = "  ".join(f"{CLASSES[k][:4]}:{cc.get(k,0)}" for k in range(NUM_CLASSES))
            print(f"     windows: train={len(tr_ds)}  val={len(va_ds)}  test={len(te_ds)}")
            print(f"     train class dist: {hist}")

            # ── Build model ──────────────────────────────────────────────
            if GPU: torch.cuda.reset_peak_memory_stats(DEVICE)
            model = PatchTSTClassifier(channels=channels,
                                       use_stat=use_stat,
                                       use_topo=use_topo).to(DEVICE)
            print(f"     Trainable params : {model.count_params():,}")
            print(f"     Sequence tokens  : {model.embed.n_tokens}  "
                  f"(patches={cfg.N_PATCHES}"
                  f"{'+stat' if use_stat else ''}"
                  f"{'+topo' if use_topo else ''})")

            # ── Train ────────────────────────────────────────────────────
            model, train_secs = train_fold(tr_dl, va_dl, model)

            # ── Evaluate ─────────────────────────────────────────────────
            eval_res  = evaluate(te_dl, model)
            bench_res = benchmark(model, te_dl)

            fold_res = {
                **eval_res,
                "train_time_s": train_secs,
                **bench_res,
                "n_params": model.count_params(),
            }
            fold_metrics.append(fold_res)

            print(f"\n     ── TEST RESULTS ──────────────────────────────────────────")
            print(f"        Macro-F1  : {eval_res['f1']:.4f}")
            print(f"        Accuracy  : {eval_res['acc']:.4f}")
            print(f"        Cohen κ   : {eval_res['kappa']:.4f}")
            print(f"        MCC       : {eval_res['mcc']:.4f}")
            print(f"     ── BENCHMARKS ────────────────────────────────────────────")
            print(f"        Latency   : {bench_res['lat_mean_ms']:.3f} ± "
                  f"{bench_res['lat_std_ms']:.3f} ms/window")
            print(f"        p50/p95/p99: "
                  f"{bench_res['lat_p50_ms']:.3f} / "
                  f"{bench_res['lat_p95_ms']:.3f} / "
                  f"{bench_res['lat_p99_ms']:.3f} ms")
            print(f"        Throughput : {bench_res['throughput_wps']:.0f} windows/s")
            print(f"        Peak GPU   : {bench_res['peak_mem_mb']:.1f} MB")
            print(f"        Train time : {train_secs:.1f} s")

            # ── Cleanup ──────────────────────────────────────────────────
            del model, tr_ds, va_ds, te_ds, tr_dl, va_dl, te_dl
            if GPU: torch.cuda.empty_cache()
            gc.collect()

        results[ablation] = fold_metrics

        # ── Fold summary ─────────────────────────────────────────────────
        F1s   = [fm["f1"]    for fm in fold_metrics]
        kaps  = [fm["kappa"] for fm in fold_metrics]
        mccs  = [fm["mcc"]   for fm in fold_metrics]
        print(f"\n  ── {ABLATION_LABEL[ablation]} — 5-Fold Summary ──────────────────")
        print(f"     Macro-F1 : {np.mean(F1s):.4f} ± {np.std(F1s):.4f}")
        print(f"     Cohen κ  : {np.mean(kaps):.4f} ± {np.std(kaps):.4f}")
        print(f"     MCC      : {np.mean(mccs):.4f} ± {np.std(mccs):.4f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 11. RESULTS REPORTING  (console + files)
# ══════════════════════════════════════════════════════════════════════════════

def _ms(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std())


def print_and_save(results: Dict, dataset_stats: Dict):
    summary = {}
    SCALAR_KEYS = [
        "f1", "acc", "kappa", "mcc", "train_time_s",
        "lat_mean_ms", "lat_std_ms",
        "lat_p50_ms", "lat_p95_ms", "lat_p99_ms",
        "throughput_wps", "peak_mem_mb", "n_params",
    ]
    for abl, folds in results.items():
        s = {}
        for key in SCALAR_KEYS:
            vals = [f[key] for f in folds if key in f]
            if vals:
                m, sd = _ms(vals)
                s[key] = {"mean": m, "std": sd}
        for pc_key in ("per_class_precision", "per_class_recall", "per_class_f1"):
            arrs = [f[pc_key] for f in folds if pc_key in f]
            if arrs:
                s[pc_key] = np.mean(arrs, axis=0).tolist()
        cms = [np.array(f["confusion_matrix"]) for f in folds if "confusion_matrix" in f]
        if cms:
            s["confusion_matrix"] = np.sum(cms, axis=0).tolist()
        summary[abl] = s

    # ── Dataset statistics table ──────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  DATASET STATISTICS")
    print("═" * 72)
    print(f"  Total windows : {dataset_stats['total_windows']}")
    print(f"  Channels (C)  : {dataset_stats['channels']}")
    print(f"  Window len    : {WIN_SAMPLES} samples  ({cfg.WINDOW_SEC}s @ {cfg.SIGNAL_RATE} Hz)")
    print(f"  Overlap       : {cfg.OVERLAP*100:.0f}%")
    print(f"  Subjects      : {len(cfg.SUBJECTS)}")
    print(f"  Classes       : {NUM_CLASSES}")
    print()
    for ci, cname in enumerate(CLASSES):
        n = dataset_stats['class_counts'].get(ci, 0)
        bar = "█" * (n // max(1, dataset_stats['total_windows'] // 40))
        print(f"  {cname:<14}  {n:>5}  {bar}")

    # ── Headline metrics table ─────────────────────────────────────────────
    COL = 22
    print("\n" + "═" * 72)
    print("  ABLATION STUDY — HEADLINE METRICS  (mean ± std, 5-fold CV)")
    print("═" * 72)
    header = f"  {'Metric':<24}" + "".join(
        f"{ABLATION_LABEL[a][:COL]:>{COL}}" for a in cfg.ABLATIONS
    )
    print(header)
    print("  " + "─" * (24 + COL * len(cfg.ABLATIONS)))

    def metric_row(name, key, fmt=".4f"):
        line = f"  {name:<24}"
        for abl in cfg.ABLATIONS:
            if key in summary[abl]:
                m, sd = summary[abl][key]["mean"], summary[abl][key]["std"]
                val   = f"{m:{fmt}}±{sd:{fmt}}"
            else:
                val = "N/A"
            line += f"{val:>{COL}}"
        print(line)

    metric_row("Macro F1",           "f1")
    metric_row("Accuracy",           "acc")
    metric_row("Cohen κ",            "kappa")
    metric_row("MCC",                "mcc")
    metric_row("Train time (s)",     "train_time_s",   ".1f")
    metric_row("Lat mean (ms)",      "lat_mean_ms",    ".3f")
    metric_row("Lat p50  (ms)",      "lat_p50_ms",     ".3f")
    metric_row("Lat p95  (ms)",      "lat_p95_ms",     ".3f")
    metric_row("Lat p99  (ms)",      "lat_p99_ms",     ".3f")
    metric_row("Throughput (win/s)", "throughput_wps", ".1f")
    metric_row("Peak GPU mem (MB)",  "peak_mem_mb",    ".1f")
    metric_row("Params",             "n_params",       ".0f")
    print("  " + "─" * (24 + COL * len(cfg.ABLATIONS)))

    # ── Per-class F1 table ────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  PER-CLASS F1  (averaged over 5 folds)")
    print("═" * 72)
    header = f"  {'Class':<14}" + "".join(
        f"{ABLATION_LABEL[a][:COL]:>{COL}}" for a in cfg.ABLATIONS
    )
    print(header)
    print("  " + "─" * (14 + COL * len(cfg.ABLATIONS)))
    for ci, cname in enumerate(CLASSES):
        line = f"  {cname:<14}"
        for abl in cfg.ABLATIONS:
            pcf = summary[abl].get("per_class_f1")
            val = f"{pcf[ci]:.4f}" if pcf else "N/A"
            line += f"{val:>{COL}}"
        print(line)

    # ── Confusion matrices ────────────────────────────────────────────────
    for abl in cfg.ABLATIONS:
        cm = summary[abl].get("confusion_matrix")
        if cm is None: continue
        cm = np.array(cm)
        print(f"\n  Confusion Matrix — {ABLATION_LABEL[abl]}"
              f"  (aggregated over 5 folds)")
        col_header = "True \\ Pred"
        hdr = f"  {col_header:<14}" + "".join(f"{c[:6]:>8}" for c in CLASSES)
        print(hdr)
        for ri, rname in enumerate(CLASSES):
            row_l = f"  {rname:<14}" + "".join(f"{int(cm[ri, ci]):>8}" for ci in range(NUM_CLASSES))
            print(row_l)

    # ── Save JSON ─────────────────────────────────────────────────────────
    def _to_py(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, dict): return {k: _to_py(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_to_py(v)    for v in obj]
        return obj

    out = {}
    out["config"]          = {k: (list(v) if isinstance(v, tuple) else v)
                               for k, v in cfg.__dict__.items()}
    out["dataset_stats"]   = dataset_stats
    out["ablation_summary"]= _to_py(summary)
    out["fold_detail"]     = _to_py(results)

    json_path = os.path.join(cfg.OUTPUT_DIR, "ablation_results.json")
    with open(json_path, "w") as fh:
        json.dump(out, fh, indent=2)

    # ── Scalar CSV ────────────────────────────────────────────────────────
    rows = []
    for abl in cfg.ABLATIONS:
        r = {"ablation": abl, "label": ABLATION_LABEL[abl]}
        for key in SCALAR_KEYS:
            if key in summary[abl]:
                r[f"{key}_mean"] = summary[abl][key]["mean"]
                r[f"{key}_std"]  = summary[abl][key]["std"]
        rows.append(r)
    scalar_csv = os.path.join(cfg.OUTPUT_DIR, "ablation_scalar_metrics.csv")
    pd.DataFrame(rows).to_csv(scalar_csv, index=False)

    # ── Per-class F1 CSV ──────────────────────────────────────────────────
    pc_rows = []
    for ci, cname in enumerate(CLASSES):
        r = {"class": cname}
        for abl in cfg.ABLATIONS:
            pcf = summary[abl].get("per_class_f1")
            r[abl] = pcf[ci] if pcf else None
            for pck in ("per_class_precision", "per_class_recall"):
                arr = summary[abl].get(pck)
                r[f"{abl}_{pck.split('_')[2]}"] = arr[ci] if arr else None
        pc_rows.append(r)
    pc_csv = os.path.join(cfg.OUTPUT_DIR, "per_class_metrics.csv")
    pd.DataFrame(pc_rows).to_csv(pc_csv, index=False)

    # ── Fold-level CSV ────────────────────────────────────────────────────
    fold_rows = []
    for abl, folds in results.items():
        for fi, fm in enumerate(folds, 1):
            r = {"ablation": abl, "fold": fi}
            for key, val in fm.items():
                if isinstance(val, (int, float)): r[key] = val
            fold_rows.append(r)
    fold_csv = os.path.join(cfg.OUTPUT_DIR, "fold_level_metrics.csv")
    pd.DataFrame(fold_rows).to_csv(fold_csv, index=False)

    print(f"\n{'═'*72}")
    print(f"  Output files saved to: {cfg.OUTPUT_DIR}/")
    print(f"  ├── ablation_results.json       (full results + config)")
    print(f"  ├── ablation_scalar_metrics.csv (headline numbers for tables)")
    print(f"  ├── per_class_metrics.csv       (P/R/F1 per class per ablation)")
    print(f"  └── fold_level_metrics.csv      (every metric, every fold)")
    print(f"{'═'*72}")


# ══════════════════════════════════════════════════════════════════════════════
# 12. MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    t_total = time.perf_counter()

    # ── Step 1: Load dataset ───────────────────────────────────────────────
    print("\n── Step 1 / 3 : Loading dataset ─────────────────────────────────")
    X, y, subj = generate_dataset()

    cc = Counter(y.tolist())
    dataset_stats = {
        "total_windows": int(X.shape[0]),
        "channels":      int(X.shape[2]),
        "window_samples":WIN_SAMPLES,
        "subjects":      len(np.unique(subj)),
        "class_counts":  {int(k): int(v) for k, v in cc.items()},
    }
    print(f"  Total windows : {X.shape[0]}")
    print(f"  Shape         : X={X.shape}   (N, T={WIN_SAMPLES}, C={X.shape[2]})")
    print(f"  Subjects      : {np.unique(subj).tolist()}")
    print("  Class histogram :")
    for ci, cname in enumerate(CLASSES):
        print(f"    {cname:<14} : {cc.get(ci, 0)}")

    if X.shape[0] == 0:
        raise RuntimeError("No windows loaded. Check DATA_PATH and dataset structure.")

    # ── Step 2: Pre-compute features (once, shared across all ablations) ───
    print("\n── Step 2 / 3 : Pre-computing features ──────────────────────────")
    sfeat, tfeat = precompute_features(X)
    print(f"  sfeat : {sfeat.shape}  (statistical, {cfg.N_STAT} dim)")
    print(f"  tfeat : {tfeat.shape}  (topological, {cfg.N_TOPO} dim)")

    # ── Step 3: Run ablation study ─────────────────────────────────────────
    print("\n── Step 3 / 3 : Ablation cross-validation ───────────────────────")
    results = run_ablation(X, y, subj, sfeat, tfeat)

    # ── Print & save all results ───────────────────────────────────────────
    print_and_save(results, dataset_stats)

    total_h = (time.perf_counter() - t_total) / 3600.
    print(f"\n  Total wall-clock time : {total_h:.2f} h")
    print("  Done.")
"""
pamap2_patchhar_v2_logo.py
==========================
PatchHAR v2 (all 10 contributions) adapted for the PAMAP2 Protocol dataset
with Leave-One-Group-Out (subject-level) cross-validation.

Dataset
-------
  Source : pamap2_protocol_combined_common_labels.parquet
  Channels: acc16 (hand/chest/ankle) + gyro + mag + temps  →  30 channels
  Excluded: acc6, heart_rate, timestamps (as model features)
  Windows : 1 000 samples (10 s @ 100 Hz), non-overlapping

Model
-----
  PatchHAR v2  D_MODEL=256 | N_HEADS=8 | N_LAYERS=6 | N_EXPERTS=4
  Extra tokens: statistics (168-d) + topology (72-d) projected to D_MODEL
  C3 (circadian bias) DISABLED – PAMAP2 excludes timestamps as features

Post-processing per LOGO fold
------------------------------
  1. Temperature scaling  (LBFGS on val logits)
  2. HSMM-lite Viterbi with λ tuned on val set

Ablation
--------
  Set any ContribConfig flag to False to disable that contribution, e.g.:
      CC.C1_DUAL_DOMAIN_EMBEDDING = False
"""

from __future__ import annotations
import copy, math, random, json, warnings
from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")


# =============================================================================
# 1.  Contribution flags
# =============================================================================
class ContribConfig:
    C1_DUAL_DOMAIN_EMBEDDING   = True   # time + FFT patch branch with learnable gate
    C2_CALANET_SKIP_AGG        = True   # CALANet-style weighted skip from every GDN layer
    C3_CIRCADIAN_BIAS          = False  # DISABLED – PAMAP2 excludes timestamps as features
    C4_MULTISCALE_PATCHING     = True   # 3 patch granularities (25, 50, 100 samples)
    C5_FREQ_AUGMENTATION       = True   # band-pass jitter / axis-perm / scale / time-warp
    C6_LABEL_SMOOTH_TEMP       = True   # label-smoothed CE + learnable temperature τ
    C7_PROTOTYPE_MEMORY        = True   # EMA class prototypes blended at inference
    C8_STOCHASTIC_DEPTH        = True   # layer-drop regularisation (linear schedule)
    C9_MANIFOLD_MIXUP          = True   # mixup on pooled embeddings, not raw input
    C10_RECON_AUX_GRAD_SURGERY = True   # lightweight patch-mean reconstruction pretext
    STAT_TOPO_TOKENS           = True   # PAMAP2-specific: inject stat+topo extra tokens

CC = ContribConfig()


# =============================================================================
# 2.  Configuration
# =============================================================================
class Config:
    # ── Data paths ────────────────────────────────────────────────────────
    DATA_PATH   = ("/mnt/share/ali/PaMP2_dataset/Protocol/_clean/pamap2_protocol_combined_common_labels.parquet")
    LABEL_COL   = "activity_label"
    SUBJECT_COL = "subject_id"

    # ── Signal ────────────────────────────────────────────────────────────
    SIGNAL_RATE = 100       # Hz
    WINDOW_SIZE = 1000      # 10 s × 100 Hz
    STRIDE      = 1000      # non-overlapping windows
    MAJ_THR     = 0.60      # minimum majority-label fraction to keep a window
    CHANNELS    = 30        # 9 acc16 + 9 gyro + 9 mag + 3 temps

    # ── Patching (C4 multi-scale) ─────────────────────────────────────────
    PATCH_LEN        = 25                         # primary granularity → 40 tokens
    PATCH_LENS_MULTI = [25, 50, 100]              # three scales
    N_PATCHES        = WINDOW_SIZE // PATCH_LEN   # 40

    # ── Stat / Topo tokens (acc16 only, 3 body locations) ─────────────────
    N_STAT_PER_LOC  = 56
    N_TOPO_PER_LOC  = 24
    N_LOCS          = 3
    N_STAT_FEATURES = N_STAT_PER_LOC * N_LOCS    # 168
    N_TOPO_FEATURES = N_TOPO_PER_LOC * N_LOCS    # 72

    # ── Model ─────────────────────────────────────────────────────────────
    D_MODEL     = 256
    N_HEADS     = 8       # per-head dim = 32 (even → RoPE OK)
    N_LAYERS    = 6
    N_EXPERTS   = 4
    DROPOUT     = 0.20
    SD_DROP_MAX = 0.10    # C8: max stochastic-depth drop probability

    # ── C6 / C7 ───────────────────────────────────────────────────────────
    LABEL_SMOOTH_EPS = 0.05
    PROTO_MOMENTUM   = 0.95
    PROTO_ALPHA      = 0.30

    # ── C10 ───────────────────────────────────────────────────────────────
    RECON_LAMBDA = 0.10   # weight for reconstruction auxiliary loss

    # ── Training ──────────────────────────────────────────────────────────
    BATCH_SIZE          = 32
    EPOCHS              = 60
    LR                  = 3e-4
    WEIGHT_DECAY        = 1e-4
    MAX_GRAD_NORM       = 1.0
    WARMUP_FRAC         = 0.07   # fraction of total steps used for LR warmup
    EARLY_STOP_PATIENCE = 15
    EMA_DECAY           = 0.995
    MIXUP_ALPHA         = 0.20
    TC_LAMBDA           = 0.05   # temporal consistency KL weight

    # ── Temperature scaling (post-hoc, per LOGO fold) ─────────────────────
    TEMP_MAX_ITERS = 500
    TEMP_LR        = 5e-2
    TEMP_CLAMP     = (0.05, 10.0)

    # ── HSMM-lite ─────────────────────────────────────────────────────────
    HMM_SMOOTH   = 1.0
    HMM_MIN_PROB = 1e-6

    # ── Reproducibility ───────────────────────────────────────────────────
    SEED = 42


cfg = Config()


# =============================================================================
# 3.  Reproducibility & device
# =============================================================================
random.seed(cfg.SEED); np.random.seed(cfg.SEED); torch.manual_seed(cfg.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

GPU    = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print(f"🚀 Device: {device}")
if GPU:
    print(f"   GPU: {torch.cuda.get_device_name(0)}")


def amp_ctx():
    if GPU:
        try:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        except Exception:
            from torch.cuda.amp import autocast
            return autocast()
    return nullcontext()


# =============================================================================
# 4.  PAMAP2 channel definitions
# =============================================================================
ALL_COLUMNS = [
    "subject_id", "timestamp", "activity_id", "heart_rate",
    "hand_temp",
    "hand_acc16_x",  "hand_acc16_y",  "hand_acc16_z",
    "hand_gyro_x",   "hand_gyro_y",   "hand_gyro_z",
    "hand_mag_x",    "hand_mag_y",    "hand_mag_z",
    "chest_temp",
    "chest_acc16_x", "chest_acc16_y", "chest_acc16_z",
    "chest_gyro_x",  "chest_gyro_y",  "chest_gyro_z",
    "chest_mag_x",   "chest_mag_y",   "chest_mag_z",
    "ankle_temp",
    "ankle_acc16_x", "ankle_acc16_y", "ankle_acc16_z",
    "ankle_gyro_x",  "ankle_gyro_y",  "ankle_gyro_z",
    "ankle_mag_x",   "ankle_mag_y",   "ankle_mag_z",
    "activity_label",
]

# acc16 triads per location – used for stats / topology features only
ACC16_TRIPLETS = [
    ("hand_acc16_x",  "hand_acc16_y",  "hand_acc16_z"),
    ("chest_acc16_x", "chest_acc16_y", "chest_acc16_z"),
    ("ankle_acc16_x", "ankle_acc16_y", "ankle_acc16_z"),
]
GYRO_TRIPLETS = [
    ("hand_gyro_x",  "hand_gyro_y",  "hand_gyro_z"),
    ("chest_gyro_x", "chest_gyro_y", "chest_gyro_z"),
    ("ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z"),
]
MAG_TRIPLETS = [
    ("hand_mag_x",  "hand_mag_y",  "hand_mag_z"),
    ("chest_mag_x", "chest_mag_y", "chest_mag_z"),
    ("ankle_mag_x", "ankle_mag_y", "ankle_mag_z"),
]
SCALARS = ["hand_temp", "chest_temp", "ankle_temp"]

# 9 acc16 + 9 gyro + 9 mag + 3 temps = 30  (NO acc6, NO heart_rate)
SEQ_CHANNELS = list(sum(ACC16_TRIPLETS + GYRO_TRIPLETS + MAG_TRIPLETS, ())) + SCALARS
assert len(SEQ_CHANNELS) == cfg.CHANNELS, f"Expected 30 channels, got {len(SEQ_CHANNELS)}"


# =============================================================================
# 5.  Stats & topology feature helpers  (computed on acc16 triads only)
# =============================================================================
try:
    from scipy.signal import find_peaks, peak_prominences
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

try:
    from ripser import ripser
    HAVE_RIPSER = True
except ImportError:
    HAVE_RIPSER = False


# ── Scalar helpers ────────────────────────────────────────────────────────────
def _safe_corr(a, b):
    sa, sb = float(np.std(a)), float(np.std(b))
    if sa < 1e-8 or sb < 1e-8: return 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    return 0.0 if not np.isfinite(c) else c

def _range(x):  return float(np.max(x) - np.min(x)) if x.size else 0.0
def _mad(x):    m = float(np.median(x)); return float(np.median(np.abs(x - m)))
def _skew(x):
    mu, sd = float(np.mean(x)), float(np.std(x))
    return 0.0 if sd < 1e-12 else float(np.mean(((x - mu) / sd) ** 3))
def _kurt(x):
    mu, sd = float(np.mean(x)), float(np.std(x))
    return 0.0 if sd < 1e-12 else float(np.mean(((x - mu) / sd) ** 4) - 3.0)
def _one_sec_autocorr(x, sr):
    L = min(sr, x.shape[0] - 1)
    return _safe_corr(x[:-L], x[L:]) if L > 1 else 0.0
def _norm_psd(x):
    ps = (np.abs(np.fft.rfft(x)) ** 2).astype(np.float64)
    s  = ps.sum(); ps = ps / s if s > 0 else ps
    return ps, np.fft.rfftfreq(x.shape[0], d=1.0 / cfg.SIGNAL_RATE)
def _spec_entropy(p):
    p = p[np.isfinite(p) & (p > 0)]
    if p.size == 0: return 0.0
    H = -float(np.sum(p * np.log(p))); Hmax = math.log(len(p))
    return float(H / Hmax) if Hmax > 0 else 0.0
def _dominant_two(ps, fhz):
    if ps.size <= 2: return 0., 0., 0., 0.
    p = ps.copy(); p[0] = 0.
    i1 = int(np.argmax(p)); p1, f1 = float(p[i1]), float(fhz[i1])
    p[i1] = -1.; i2 = int(np.argmax(p))
    return f1, p1, float(fhz[i2]), max(float(p[i2]), 0.)
def _lowpass_gravity(a, fc=0.5):
    alpha = math.exp(-2.0 * math.pi * fc / cfg.SIGNAL_RATE)
    g = np.zeros_like(a, dtype=np.float32); g[0] = a[0]
    for t in range(1, a.shape[0]):
        g[t] = alpha * g[t - 1] + (1.0 - alpha) * a[t]
    return g
def _angles(vx, vy, vz):
    return (np.arctan2(vy, vz),
            np.arctan2(-vx, np.sqrt(vy**2 + vz**2) + 1e-8),
            np.arctan2(vx, vy))


def stats_from_xyz(xyz: np.ndarray) -> np.ndarray:
    """56-dim stat feature vector for one acc16 location. xyz: (T,3)"""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    mag = np.linalg.norm(xyz, axis=1)
    feats = []

    # (A) per-axis mean / std / range  ×3 = 9
    for s in (x, y, z):
        feats.extend([float(np.mean(s)), float(np.std(s)), _range(s)])
    # (A) cross-axis correlations             3
    feats.extend([_safe_corr(x, y), _safe_corr(x, z), _safe_corr(y, z)])
    # (A) magnitude stats (7)
    feats.extend([float(np.mean(mag)), float(np.std(mag)), _range(mag),
                  _mad(mag), _kurt(mag), _skew(mag), float(np.median(mag))])
    # (B) quantiles x,y,z,mag × 5 each       20
    def qpack(s):
        q25, q50, q75 = np.percentile(s, [25, 50, 75])
        return [float(np.min(s)), float(np.max(s)), float(q50),
                float(q25), float(q75)]
    for s in (x, y, z, mag): feats.extend(qpack(s))
    # (C) 1-second autocorr on magnitude      1
    feats.append(_one_sec_autocorr(mag, cfg.SIGNAL_RATE))
    # (D) spectral top-2 + entropy            5
    ps, fhz = _norm_psd(mag)
    f1, p1, f2, p2 = _dominant_two(ps, fhz)
    feats.extend([f1, p1, f2, p2, _spec_entropy(ps)])
    # (E) peaks (count + median prominence)   2
    if HAVE_SCIPY:
        pks, _ = find_peaks(mag)
        feats.extend([float(len(pks)),
                      float(np.median(peak_prominences(mag, pks)[0]))
                      if pks.size else 0.0])
    else:
        pk = ((mag[1:-1] > mag[:-2]) & (mag[1:-1] > mag[2:])).sum() \
             if mag.size >= 3 else 0
        feats.extend([float(pk), 0.0])
    # (F) gravity + dynamic angles            9
    g = _lowpass_gravity(xyz, fc=0.5)
    d = xyz - g
    gr, gp, gy = _angles(g[:, 0], g[:, 1], g[:, 2])
    feats.extend([float(np.mean(gr)), float(np.mean(gp)), float(np.mean(gy))])
    for arr in _angles(d[:, 0], d[:, 1], d[:, 2]):
        feats.extend([float(np.mean(arr)), float(np.std(arr))])

    out = np.asarray(feats, dtype=np.float32)
    assert out.shape[0] == cfg.N_STAT_PER_LOC, \
        f"stat dim mismatch {out.shape[0]} != {cfg.N_STAT_PER_LOC}"
    return out


def topo_from_mag(mag: np.ndarray, m: int = 3, tau: int = 5) -> np.ndarray:
    """24-dim persistence-homology feature vector for one location's magnitude."""
    mag = mag.astype(np.float32)
    T   = mag.shape[0]
    L   = T - (m - 1) * tau
    X   = (np.stack([mag[i:i + L] for i in range(0, m * tau, tau)], axis=1)
           if L > 5 else np.zeros((5, m), dtype=np.float32))

    def _ent(D):
        if D.size == 0: return 0.0
        b, d = D[:, 0], D[:, 1]; fin = np.isfinite(d)
        p = np.maximum(d[fin] - b[fin], 0.); s = p.sum()
        if s <= 0: return 0.0
        p /= s
        return float(-(p * np.log(p + 1e-12)).sum() / (np.log(len(p)) + 1e-12))

    def _topk(D, k=3):
        if D.size == 0: return [0.0] * k
        b, d = D[:, 0], D[:, 1]; fin = np.isfinite(d)
        p = np.sort(np.maximum(d[fin] - b[fin], 0.))[::-1]
        o = np.zeros(k, dtype=np.float32); o[:min(k, p.size)] = p[:k]
        return o.tolist()

    if HAVE_RIPSER and X.shape[0] >= 8:
        res = ripser(X, maxdim=1)
        D0, D1 = res["dgms"][0], res["dgms"][1]
    else:
        # cheap fallback: threshold-based fake persistence
        dd  = np.abs(np.subtract.outer(mag, mag))
        thr = (np.quantile(dd[np.isfinite(dd)], [0.2, 0.5])
               if np.isfinite(dd).any() else [0.0, 0.0])
        def fake(dist, t):
            M = (dist < t).astype(np.float32); lts = []
            for k in range(-10, 11):
                diag = np.diag(M, k=k); run = 0
                for v in diag:
                    if v > 0.5: run += 1
                    elif run:   lts.append(run); run = 0
                if run: lts.append(run)
            if not lts: return np.empty((0, 2), dtype=np.float32)
            lts = np.asarray(lts, dtype=np.float32)
            return np.stack([np.zeros_like(lts), lts], axis=1)
        D0, D1 = fake(dd, thr[0]), fake(dd, thr[1])

    feats = []
    for D in (D0, D1):
        if D.size == 0:
            feats.extend([0.0] * 3 + [0.0] + [0.0] * 3 + [0.0, 0.0] + [0.0] * 3)
            continue
        b, d = D[:, 0], D[:, 1]; fin = np.isfinite(d)
        b, d = b[fin], d[fin]; p = np.maximum(d - b, 0.)
        feats.append(float(p.max()  if p.size else 0.))
        feats.append(float(p.mean() if p.size else 0.))
        feats.append(float(p.sum()  if p.size else 0.))
        feats.append(_ent(D))
        feats.extend(_topk(D, 3))
        feats.append(float(b.max() if b.size else 0.))
        feats.append(float(d.max() if d.size else 0.))
        if p.size >= 5:
            qs = np.quantile(p, [0.5, 0.75, 0.9])
            feats.extend([float((p > q).sum()) for q in qs])
        else:
            feats.extend([0.0, 0.0, 0.0])

    out = np.asarray(feats, dtype=np.float32)
    assert out.shape[0] == cfg.N_TOPO_PER_LOC, \
        f"topo dim mismatch {out.shape[0]} != {cfg.N_TOPO_PER_LOC}"
    return out


# =============================================================================
# 6.  Data loading & windowing
# =============================================================================
def build_seq_matrix(df_s: pd.DataFrame):
    """Extract (T,30) signal matrix and label array for one subject."""
    data = {}
    for c in SEQ_CHANNELS:
        data[c] = (df_s[c].to_numpy(dtype=np.float32)
                   if c in df_s.columns
                   else np.full(len(df_s), np.nan, dtype=np.float32))
    X = np.column_stack([data[c] for c in SEQ_CHANNELS])
    X = (pd.DataFrame(X)
         .replace([np.inf, -np.inf], np.nan)
         .ffill().bfill())
    X = X.fillna(X.median()).to_numpy(dtype=np.float32)
    y = df_s[cfg.LABEL_COL].astype(str).to_numpy()
    return X, y


def window_entries_for_subject(sid: int, X: np.ndarray, y_str: np.ndarray,
                               class_to_idx: dict) -> list:
    """
    Slice non-overlapping windows; instance-normalise; build acc16 triads.
    Entry format: (sid, window:(T,C), label_idx, seq_start, triads:tuple)
    """
    T, C = X.shape
    entries = []
    for start in range(0, T - cfg.WINDOW_SIZE + 1, cfg.STRIDE):
        stop  = start + cfg.WINDOW_SIZE
        yw    = y_str[start:stop]
        lab, counts = np.unique(yw, return_counts=True)
        top   = int(np.argmax(counts))
        maj   = lab[top]; frac = counts[top] / cfg.WINDOW_SIZE
        if maj not in class_to_idx or frac < cfg.MAJ_THR:
            continue

        win = X[start:stop].copy()
        mu  = np.nanmean(win, axis=0, keepdims=True)
        sd  = np.nanstd(win,  axis=0, keepdims=True)
        win = np.clip((win - mu) / (sd + 1e-8), -10, 10).astype(np.float32)

        # acc16 triads (normalised) for stats / topo computation in Dataset
        triads = tuple(
            win[:, [SEQ_CHANNELS.index(t) for t in trip]]
            for trip in ACC16_TRIPLETS
        )
        entries.append((int(sid), win, int(class_to_idx[maj]), int(start), triads))
    return entries


def build_all_entries(df: pd.DataFrame):
    labels       = sorted(df[cfg.LABEL_COL].astype(str).unique())
    class_to_idx = {c: i for i, c in enumerate(labels)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    entries = []
    for sid, df_s in df.groupby(cfg.SUBJECT_COL, sort=True):
        X, y = build_seq_matrix(df_s)
        entries.extend(window_entries_for_subject(int(sid), X, y, class_to_idx))
    return entries, labels, class_to_idx, idx_to_class


# =============================================================================
# 7.  [C5] Frequency-aware data augmentation
# =============================================================================
def _bandpass_jitter(sig: np.ndarray) -> np.ndarray:
    T, C = sig.shape; out = sig.copy()
    band = random.choice(["low", "mid", "high"])
    for c in range(C):
        f = np.fft.rfft(out[:, c]); n = len(f)
        if   band == "low":  f[n // 3:] = 0
        elif band == "mid":  f[:n // 4] = 0; f[n // 2:] = 0
        else:                f[:2 * n // 3] = 0
        out[:, c] = np.fft.irfft(f, n=T)
    return out

def _axis_permute(sig: np.ndarray) -> np.ndarray:
    idx = list(range(sig.shape[1])); random.shuffle(idx); return sig[:, idx]

def _magnitude_scale(sig: np.ndarray) -> np.ndarray:
    return sig * np.random.uniform(0.8, 1.2, (1, sig.shape[1]))

def _time_warp(sig: np.ndarray) -> np.ndarray:
    T, C = sig.shape; f = random.choice([0.9, 0.95, 1.05, 1.1])
    nT   = max(T, int(round(T * f)))
    warp = np.zeros((nT, C), dtype=sig.dtype)
    for c in range(C):
        warp[:, c] = np.interp(np.linspace(0, T - 1, nT), np.arange(T), sig[:, c])
    return warp[:T] if nT >= T else np.pad(warp, ((0, T - nT), (0, 0)))

def freq_augment(sig: np.ndarray) -> np.ndarray:
    """Apply one randomly chosen frequency-aware augmentation."""
    return random.choice([_bandpass_jitter, _axis_permute,
                          _magnitude_scale, _time_warp])(sig)


# =============================================================================
# 8.  Dataset & DataLoader
# =============================================================================
class PAMAP2Dataset(Dataset):
    """
    Per-window output:
      patches_list  list of (CHANNELS, N_Pi, PL_i)   one per patch scale
      sfeat         (N_STAT_FEATURES,)   = 3 × 56
      tafeat        (N_TOPO_FEATURES,)   = 3 × 24
      label         long scalar
      sid           long scalar
      seq           long scalar (window start index)
      raw_seg       (WINDOW_SIZE, CHANNELS)   used by C10 reconstruction loss

    Note: stats/topo are computed on the pre-augmentation acc16 triads stored
    in the entry.  Patches (and raw_seg) use the augmented signal when C5 is
    active.  This deliberate asymmetry keeps auxiliary features stable while
    the patch stream is augmented.
    """
    def __init__(self, entries: list, is_train: bool = False):
        self.entries    = entries
        self.is_train   = is_train
        self.patch_lens = (cfg.PATCH_LENS_MULTI if CC.C4_MULTISCALE_PATCHING
                           else [cfg.PATCH_LEN])

    def __len__(self): return len(self.entries)

    @staticmethod
    def _make_patches(seg: np.ndarray, pl: int) -> np.ndarray:
        """(T, C) → (C, N_P, PL)"""
        T, C = seg.shape; n = T // pl
        return seg[:n * pl].reshape(n, pl, C).transpose(2, 0, 1).astype(np.float32)

    def __getitem__(self, idx):
        sid, win, lab, seq, triads = self.entries[idx]

        # [C5] frequency-aware augmentation on training windows only
        seg = win.copy()
        if self.is_train and CC.C5_FREQ_AUGMENTATION:
            seg = freq_augment(seg)

        patches_list = [
            torch.from_numpy(self._make_patches(seg, pl))
            for pl in self.patch_lens
        ]

        # Stats & Topo computed on pre-augmentation triads (stable features)
        sfeats, tfeats = [], []
        for xyz in triads:
            sfeats.append(stats_from_xyz(xyz))
            tfeats.append(topo_from_mag(np.linalg.norm(xyz, axis=1)))

        sfeat  = torch.from_numpy(np.concatenate(sfeats).astype(np.float32))
        tafeat = torch.from_numpy(np.concatenate(tfeats).astype(np.float32))

        return (
            patches_list,
            sfeat,
            tafeat,
            torch.tensor(lab,  dtype=torch.long),
            torch.tensor(sid,  dtype=torch.long),
            torch.tensor(seq,  dtype=torch.long),
            torch.from_numpy(seg.astype(np.float32)),   # raw_seg for C10
        )


def _collate(batch):
    patches_lists, sfeats, tafeats, labels, sids, seqs, raws = zip(*batch)
    n_scales = len(patches_lists[0])
    return (
        [torch.stack([b[s] for b in patches_lists]) for s in range(n_scales)],
        torch.stack(sfeats),
        torch.stack(tafeats),
        torch.stack(labels),
        torch.stack(sids),
        torch.stack(seqs),
        torch.stack(raws),
    )


def make_loader(entries, batch_size=32, shuffle=False,
                sampler_weights=None, is_train=False):
    ds = PAMAP2Dataset(entries, is_train=is_train)
    if sampler_weights is not None:
        sampler = WeightedRandomSampler(sampler_weights,
                                        num_samples=len(sampler_weights),
                                        replacement=True)
        dl = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                        num_workers=2, pin_memory=GPU, collate_fn=_collate)
    else:
        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=2, pin_memory=GPU, collate_fn=_collate)
    return ds, dl


# =============================================================================
# 9.  Training utilities: WarmupCosine scheduler, EMA, class weights
# =============================================================================
class WarmupCosine:
    """Linear warmup → cosine annealing LR scheduler."""
    def __init__(self, opt, warmup_steps, total_steps, min_lr=1e-5):
        self.opt  = opt
        self.ws   = max(1, warmup_steps)
        self.ts   = max(self.ws + 1, total_steps)
        self.min  = min_lr
        self.base = cfg.LR
        self.i    = 0

    def step(self):
        self.i += 1
        if self.i < self.ws:
            lr = self.base * self.i / self.ws
        else:
            t  = (self.i - self.ws) / max(1, self.ts - self.ws)
            lr = self.min + 0.5 * (self.base - self.min) * (1 + math.cos(math.pi * t))
        for g in self.opt.param_groups:
            g["lr"] = lr


class EMA:
    """Exponential moving average of model parameters."""
    def __init__(self, model, decay=0.999):
        self.decay  = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = self.decay * self.shadow[n] + (1. - self.decay) * p.detach()

    def copy_to(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])


def class_weights_from_entries(entries, K):
    counts = np.zeros(K, dtype=np.int64)
    for _, _, lab, _, _ in entries: counts[lab] += 1
    w = counts.max() / np.clip(counts, 1, None)
    w = torch.tensor(w, dtype=torch.float32)
    return w / w.sum() * K


def sample_weights_from_entries(entries, K):
    counts = np.zeros(K, dtype=np.int64)
    for _, _, lab, _, _ in entries: counts[lab] += 1
    inv = counts.max() / np.clip(counts, 1, None)
    w   = np.array([inv[lab] for (_, _, lab, _, _) in entries], dtype=np.float32)
    return torch.tensor(w, dtype=torch.float32)


# =============================================================================
# 10.  Model building blocks
# =============================================================================

# ── Normalisation ─────────────────────────────────────────────────────────────
class ZCRMSNorm(nn.Module):
    """Zero-centred RMS normalisation."""
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__(); self.g = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        x0 = x - x.mean(-1, keepdim=True)
        return x0 / x0.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt() * self.g


# ── [C8] Stochastic depth ─────────────────────────────────────────────────────
class StochasticDepth(nn.Module):
    def __init__(self, layer: nn.Module, survival_prob: float):
        super().__init__(); self.layer = layer; self.p = survival_prob
    def forward(self, x, *args, **kwargs):
        if not self.training or self.p >= 1.0:
            return self.layer(x, *args, **kwargs)
        if random.random() > self.p: return x          # drop layer
        return self.layer(x, *args, **kwargs)


# ── Local temporal modelling ──────────────────────────────────────────────────
class GatedDeltaNet(nn.Module):
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.norm      = ZCRMSNorm(d)
        self.q_lin     = nn.Linear(d, d); self.k_lin = nn.Linear(d, d); self.v_lin = nn.Linear(d, d)
        self.q_conv    = nn.Conv1d(d, d, 3, padding=1, groups=d)
        self.k_conv    = nn.Conv1d(d, d, 3, padding=1, groups=d)
        self.v_conv    = nn.Conv1d(d, d, 3, padding=1, groups=d)
        self.act       = nn.Sigmoid()
        self.alpha     = nn.Linear(d, d); self.beta = nn.Linear(d, d)
        self.post_norm = ZCRMSNorm(d); self.post = nn.Linear(d, d)
        self.silu      = nn.SiLU(); self.gate = nn.Sigmoid()
        self.drop      = nn.Dropout(dropout)

    @staticmethod
    def _l2(x, eps=1e-8): return x / (x.pow(2).sum(-1, keepdim=True).add(eps).sqrt())

    def forward(self, x):
        h = self.norm(x)
        q = self.act(self.q_conv(self.q_lin(h).transpose(1, 2)).transpose(1, 2))
        k = self.act(self.k_conv(self.k_lin(h).transpose(1, 2)).transpose(1, 2))
        v = self.act(self.v_conv(self.v_lin(h).transpose(1, 2)).transpose(1, 2))
        q, k  = self._l2(q), self._l2(k)
        delta = q * (k * v)
        delta = torch.tanh(self.alpha(x)) * delta + self.beta(x)
        dhat  = self.post(self.post_norm(delta))
        return x + self.drop(self.gate(self.silu(dhat)) * dhat)


# ── Soft Mixture-of-Experts ───────────────────────────────────────────────────
class SoftMoE(nn.Module):
    def __init__(self, d: int, hidden: int, n_experts: int = 4, dropout: float = 0.1):
        super().__init__()
        self.router  = nn.Linear(d, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden), nn.SiLU(),
                          nn.Dropout(dropout), nn.Linear(hidden, d))
            for _ in range(n_experts)
        ])
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        w = torch.softmax(self.router(x), dim=-1)
        s = torch.stack([e(x) for e in self.experts], dim=-2)
        return self.drop((w.unsqueeze(-1) * s).sum(-2))


# ── Rotary positional embeddings (RoPE) ───────────────────────────────────────
def precompute_freqs(dim: int, n_tok: int, theta: float = 10000.0) -> torch.Tensor:
    assert dim % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    return torch.polar(torch.ones(n_tok, dim // 2),
                       torch.outer(torch.arange(n_tok, dtype=torch.float32), freqs))

def apply_rope(q, k, freqs):
    B, H, N, D = q.shape; d2 = D // 2
    f  = freqs[:N].to(q.device).view(1, 1, N, d2)
    qc = torch.view_as_complex(q.float().contiguous().view(B, H, N, d2, 2))
    kc = torch.view_as_complex(k.float().contiguous().view(B, H, N, d2, 2))
    return (torch.view_as_real(qc * f).view(B, H, N, D).type_as(q),
            torch.view_as_real(kc * f).view(B, H, N, D).type_as(k))


# ── Gated multi-head attention with RoPE ──────────────────────────────────────
class GatedAttention(nn.Module):
    def __init__(self, d: int, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        assert d % n_heads == 0 and (d // n_heads) % 2 == 0, \
            "Per-head dim must be divisible by 2 for RoPE"
        self.h    = n_heads; self.dh = d // n_heads
        self.norm = ZCRMSNorm(d)
        self.qkv  = nn.Linear(d, 3 * d)
        self.out  = nn.Linear(d, d); self.gate = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, freqs):
        h = self.norm(x); B, N, D = h.shape
        qkv = self.qkv(h).reshape(B, N, 3, self.h, self.dh).permute(0, 2, 1, 3, 4)
        q = qkv[:, 0].transpose(1, 2)
        k = qkv[:, 1].transpose(1, 2)
        v = qkv[:, 2].transpose(1, 2)
        q, k  = apply_rope(q, k, freqs)
        attn  = self.drop(torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.dh), dim=-1))
        y     = self.out((attn @ v).transpose(1, 2).contiguous().reshape(B, N, D))
        return x + torch.sigmoid(self.gate(h)) * y


# ── [C1] Dual-domain patch embedding ─────────────────────────────────────────
class DualDomainPatchEmbed(nn.Module):
    """
    Projects each patch in both time domain and frequency domain, fuses with
    a learnable per-output-dim gate:  out = σ(g)·time + (1−σ(g))·freq
    """
    def __init__(self, patch_len: int, channels: int, d: int):
        super().__init__()
        in_dim   = patch_len * channels
        freq_dim = (patch_len // 2 + 1) * channels
        self.time_proj = nn.Linear(in_dim, d)
        self.freq_proj = nn.Linear(freq_dim, d)
        self.gate_w    = nn.Parameter(torch.zeros(d))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        B, C, NP, PL = patches.shape
        t_emb = self.time_proj(patches.permute(0, 2, 1, 3).reshape(B, NP, C * PL))
        mag   = torch.fft.rfft(patches.permute(0, 2, 3, 1), dim=2).abs().reshape(B, NP, -1)
        f_emb = self.freq_proj(mag)
        g     = torch.sigmoid(self.gate_w)
        return g * t_emb + (1 - g) * f_emb


class SimplePatchEmbed(nn.Module):
    """Fallback time-domain linear projection (C1 OFF)."""
    def __init__(self, patch_len: int, channels: int, d: int):
        super().__init__()
        self.proj = nn.Linear(patch_len * channels, d)
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        B, C, NP, PL = patches.shape
        return self.proj(patches.permute(0, 2, 1, 3).reshape(B, NP, C * PL))


# ── [C2] CALANet-style skip aggregation ──────────────────────────────────────
class SkipAggregation(nn.Module):
    def __init__(self, d: int, n_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_layers) / n_layers)
        self.proj    = nn.Sequential(nn.Linear(d, d), nn.SiLU())

    def forward(self, hiddens: list) -> torch.Tensor:
        w   = torch.softmax(self.weights, dim=0)
        agg = sum(w[i] * hiddens[i] for i in range(len(hiddens)))
        return self.proj(agg)


# ── [C10] Lightweight reconstruction head ─────────────────────────────────────
class ReconHead(nn.Module):
    """
    Reconstructs per-patch channel means from the pooled representation z.
    Output dim = N_PATCHES × CHANNELS = 40 × 30 = 1 200  (cheap pretext task).
    Avoids the prohibitively large output layer that full-sample reconstruction
    would require with 30 channels.
    """
    def __init__(self, d: int, n_patches: int, channels: int):
        super().__init__()
        out_dim   = n_patches * channels
        self.mlp  = nn.Sequential(
            nn.Linear(d, 2 * d), nn.ReLU(),
            nn.Linear(2 * d, out_dim),
        )
        self.NP = n_patches; self.C = channels

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)   # (B, NP × C)


# =============================================================================
# 11.  PatchHAR v2  –  PAMAP2 edition
# =============================================================================
class PatchHARv2(nn.Module):
    """
    Full PatchHAR v2 with all 10 contributions, adapted for PAMAP2:
      • CHANNELS = 30
      • C3 circadian bias DISABLED (no timestamps used as features)
      • STAT_TOPO_TOKENS: statistics (168-d) + topology (72-d) tokens
        projected to D_MODEL and appended to the patch sequence
      • Token count: N_PATCHES + 2 (when STAT_TOPO_TOKENS ON)
    """
    def __init__(self, num_classes: int):
        super().__init__()
        d     = cfg.D_MODEL
        NP    = cfg.N_PATCHES
        n_tok = NP + (2 if CC.STAT_TOPO_TOKENS else 0)

        # ── Patch embeddings (one per granularity for C4) ─────────────────
        self.patch_lens = (cfg.PATCH_LENS_MULTI if CC.C4_MULTISCALE_PATCHING
                           else [cfg.PATCH_LEN])
        EmbedCls = DualDomainPatchEmbed if CC.C1_DUAL_DOMAIN_EMBEDDING else SimplePatchEmbed
        self.patch_embeds = nn.ModuleList([
            EmbedCls(pl, cfg.CHANNELS, d) for pl in self.patch_lens
        ])

        # [C4] fuse multi-scale tokens → (B, NP, D)
        if CC.C4_MULTISCALE_PATCHING and len(self.patch_lens) > 1:
            self.scale_fusion = nn.Linear(d * len(self.patch_lens), d)
        else:
            self.scale_fusion = None

        # [STAT_TOPO_TOKENS] project stat/topo feature vectors to D_MODEL
        if CC.STAT_TOPO_TOKENS:
            self.stat_proj = nn.Linear(cfg.N_STAT_FEATURES, d)
            self.topo_proj = nn.Linear(cfg.N_TOPO_FEATURES, d)

        # [C3] DISABLED for PAMAP2 – no time embedding needed
        # (circadian structure is absent in a lab protocol dataset)

        self.input_norm = nn.LayerNorm(d)

        # ── [C8] Stochastic-depth GatedDeltaNet layers ────────────────────
        raw_layers = [GatedDeltaNet(d, dropout=cfg.DROPOUT)
                      for _ in range(cfg.N_LAYERS)]
        if CC.C8_STOCHASTIC_DEPTH:
            survs = [1.0 - (i / cfg.N_LAYERS) * cfg.SD_DROP_MAX
                     for i in range(cfg.N_LAYERS)]
            self.delta_layers = nn.ModuleList([
                StochasticDepth(l, p) for l, p in zip(raw_layers, survs)
            ])
        else:
            self.delta_layers = nn.ModuleList(raw_layers)

        # ── [C2] Skip aggregation ──────────────────────────────────────────
        if CC.C2_CALANET_SKIP_AGG:
            self.skip_agg = SkipAggregation(d, cfg.N_LAYERS)

        # ── Transformer core ──────────────────────────────────────────────
        self.moe1 = SoftMoE(d, 2 * d, cfg.N_EXPERTS, cfg.DROPOUT)
        self.attn = GatedAttention(d, cfg.N_HEADS, cfg.DROPOUT)
        self.moe2 = SoftMoE(d, 2 * d, cfg.N_EXPERTS, cfg.DROPOUT)

        # RoPE frequency cache sized to accommodate extra stat/topo tokens
        freqs = precompute_freqs(d // cfg.N_HEADS, n_tok)
        self.register_buffer("freqs", freqs)

        # ── [C6] Learnable temperature τ ─────────────────────────────────
        if CC.C6_LABEL_SMOOTH_TEMP:
            self.log_tau = nn.Parameter(torch.zeros(1))

        # ── [C10] Reconstruction head ─────────────────────────────────────
        if CC.C10_RECON_AUX_GRAD_SURGERY:
            self.recon_head = ReconHead(d, NP, cfg.CHANNELS)

        # ── [C7] Prototype memory bank ────────────────────────────────────
        if CC.C7_PROTOTYPE_MEMORY:
            self.register_buffer("prototypes", torch.zeros(num_classes, d))
            self.proto_filled = False

        self.num_classes = num_classes

        # ── Classification head ───────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(d, d // 2), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d // 2, num_classes),
        )

    # ── Embed patches + optional stat/topo tokens ─────────────────────────
    def _embed(self, patches_list, sfeat, tafeat) -> torch.Tensor:
        """
        patches_list : list of (B, C, N_Pi, PL_i)
        sfeat        : (B, N_STAT_FEATURES)
        tafeat       : (B, N_TOPO_FEATURES)
        Returns      : (B, N_tok, D)
        """
        NP = cfg.N_PATCHES
        if CC.C4_MULTISCALE_PATCHING and len(patches_list) > 1:
            embs = []
            for embed, p in zip(self.patch_embeds, patches_list):
                e = embed(p)                                    # (B, N_Pi, D)
                e = F.interpolate(e.permute(0, 2, 1),
                                  size=NP, mode="linear",
                                  align_corners=False).permute(0, 2, 1)
                embs.append(e)
            x = self.scale_fusion(torch.cat(embs, dim=-1))     # (B, NP, D)
        else:
            x = self.patch_embeds[0](patches_list[0])           # (B, NP, D)

        if CC.STAT_TOPO_TOKENS:
            s = self.stat_proj(sfeat).unsqueeze(1)              # (B, 1, D)
            t = self.topo_proj(tafeat).unsqueeze(1)             # (B, 1, D)
            x = torch.cat([x, s, t], dim=1)                     # (B, NP+2, D)

        return x

    # ── Run backbone; return (pooled z, layer hiddens) ────────────────────
    def _backbone(self, x: torch.Tensor):
        """x: (B, N_tok, D)  →  z: (B, D), hiddens: list"""
        x = self.input_norm(x)
        hiddens = []
        for layer in self.delta_layers:
            x = layer(x); hiddens.append(x)
        if CC.C2_CALANET_SKIP_AGG:
            x = x + self.skip_agg(hiddens)
        x = x + self.moe1(x)
        x = self.attn(x, self.freqs)
        x = x + self.moe2(x)
        return x.mean(dim=1), hiddens                           # (B, D)

    # ── Full forward pass ─────────────────────────────────────────────────
    def forward(self, patches_list, sfeat, tafeat,
                return_embedding: bool = False):
        x = self._embed(patches_list, sfeat, tafeat)
        z, _ = self._backbone(x)

        if return_embedding:
            return z

        recon = None
        if CC.C10_RECON_AUX_GRAD_SURGERY and self.training:
            recon = self.recon_head(z)

        logits = self.head(z)

        if CC.C6_LABEL_SMOOTH_TEMP:
            tau    = torch.exp(self.log_tau).clamp(0.5, 2.0)
            logits = logits / tau

        # [C7] blend with prototype cosine score at inference
        if CC.C7_PROTOTYPE_MEMORY and not self.training and self.proto_filled:
            z_n    = F.normalize(z, dim=-1)
            pr_n   = F.normalize(self.prototypes, dim=-1)
            cosine = z_n @ pr_n.T
            logits = ((1 - cfg.PROTO_ALPHA) * logits
                      + cfg.PROTO_ALPHA * cosine)

        return logits, recon

    # ── [C7] EMA prototype update ─────────────────────────────────────────
    @torch.no_grad()
    def update_prototypes(self, embs: torch.Tensor, labels: torch.Tensor):
        m = cfg.PROTO_MOMENTUM
        for k in range(self.num_classes):
            mask = (labels == k)
            if mask.sum() == 0: continue
            mean = embs[mask].mean(0)
            self.prototypes[k] = (m * self.prototypes[k] + (1 - m) * mean
                                  if self.proto_filled else mean)
        self.proto_filled = True


# =============================================================================
# 12.  Loss & training helpers
# =============================================================================
class SmoothCE(nn.Module):
    """Label-smoothed cross-entropy with optional class weights ([C6])."""
    def __init__(self, weight=None, smoothing: float = 0.0):
        super().__init__(); self.eps = smoothing; self.w = weight

    def forward(self, logits, labels):
        K = logits.size(-1)
        with torch.no_grad():
            soft = torch.full_like(logits, self.eps / max(K - 1, 1))
            soft.scatter_(-1, labels.unsqueeze(-1), 1.0 - self.eps)
        loss = -(soft * F.log_softmax(logits, dim=-1)).sum(-1)
        if self.w is not None:
            loss = loss * self.w.to(logits.device)[labels]
        return loss.mean()


def recon_loss_fn(recon: torch.Tensor, raw_segs: torch.Tensor) -> torch.Tensor:
    """
    [C10] Per-patch channel-mean reconstruction loss.
    Target shape: (B, N_PATCHES × CHANNELS) = (B, 1200)  — very lightweight.
    """
    B, T, C = raw_segs.shape
    NP, PL  = cfg.N_PATCHES, cfg.PATCH_LEN
    target  = (raw_segs[:, :NP * PL, :]
               .reshape(B, NP, PL, C)
               .mean(dim=2)                # (B, NP, C)
               .reshape(B, NP * C))
    return F.mse_loss(recon, target.detach())


def tc_loss(logits: torch.Tensor) -> torch.Tensor:
    """Symmetrised KL between adjacent-window predictions."""
    if logits.size(0) < 2:
        return logits.new_zeros(1).squeeze()
    p = F.softmax(logits[:-1], dim=-1)
    q = F.softmax(logits[1:],  dim=-1)
    return 0.5 * (F.kl_div(q.log(), p, reduction="batchmean") +
                  F.kl_div(p.log(), q, reduction="batchmean"))


def manifold_mixup(z: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2):
    """[C9] Interpolate in embedding space; returns (z_mixed, la, lb, λ)."""
    if alpha <= 0: return z, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(z.size(0), device=z.device)
    return lam * z + (1 - lam) * z[idx], labels, labels[idx], lam


# =============================================================================
# 13.  Temperature scaling & HSMM-lite
# =============================================================================
class TemperatureScaling(nn.Module):
    def __init__(self, T_init: float = 1.0):
        super().__init__()
        self.logT = nn.Parameter(torch.tensor(math.log(float(T_init))))

    def forward(self, logits):
        T = torch.exp(self.logT).clamp(*cfg.TEMP_CLAMP)
        return logits / T

    def temperature(self):
        with torch.no_grad():
            return float(torch.exp(self.logT).clamp(*cfg.TEMP_CLAMP).item())


def fit_temperature(logits: torch.Tensor, labels: np.ndarray) -> TemperatureScaling:
    """Fit a single temperature scalar on validation logits via LBFGS."""
    ts  = TemperatureScaling().to(device)
    y   = torch.from_numpy(labels).to(device)
    x   = logits.to(device).detach()
    ce  = nn.CrossEntropyLoss()
    opt = optim.LBFGS(ts.parameters(), lr=cfg.TEMP_LR,
                      max_iter=cfg.TEMP_MAX_ITERS,
                      line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = ce(ts(x), y); loss.backward(); return loss

    opt.step(closure)
    with torch.no_grad():
        nll = ce(ts(x), y).item()
    print(f"    🌡️  T={ts.temperature():.3f} | val NLL={nll:.4f}")
    return ts


def collect_logits(model: PatchHARv2, loader: DataLoader):
    """Collect raw logits, true labels, subject IDs and seq starts."""
    model.eval()
    outs, labs, sids_all, seqs_all = [], [], [], []
    with torch.no_grad():
        for pl, sf, taf, lbl, sid, seq, _ in loader:
            pl  = [p.to(device).float() for p in pl]
            sf  = sf.to(device).float(); taf = taf.to(device).float()
            logits, _ = model(pl, sf, taf)
            outs.append(logits.cpu())
            labs.extend(lbl.numpy().tolist())
            sids_all.extend(sid.numpy().tolist())
            seqs_all.extend(seq.numpy().tolist())
    return (torch.cat(outs),
            np.array(labs),
            np.array(sids_all),
            np.array(seqs_all))


def estimate_hmm(entries: list, K: int):
    """Estimate HMM initial distribution π and transition matrix A from entries."""
    by_sid = defaultdict(list)
    for sid, _, lab, seq, _ in entries:
        by_sid[int(sid)].append((int(seq), int(lab)))

    A  = np.full((K, K), cfg.HMM_SMOOTH, dtype=np.float64)
    pi = np.full(K, cfg.HMM_SMOOTH, dtype=np.float64)

    for seqs in by_sid.values():
        seqs.sort(key=lambda x: x[0])
        if seqs: pi[seqs[0][1]] += 1
        for (_, a), (_, b) in zip(seqs[:-1], seqs[1:]):
            A[a, b] += 1

    A  = np.clip(A / A.sum(1, keepdims=True), cfg.HMM_MIN_PROB, 1.0)
    pi = np.clip(pi / pi.sum(),               cfg.HMM_MIN_PROB, 1.0)
    return pi, A


def viterbi(E_log: np.ndarray, log_pi: np.ndarray,
            log_A: np.ndarray, lam: float = 0.75) -> np.ndarray:
    """
    Viterbi decoding with a change penalty λ subtracted when state changes.
    E_log: (T, K) log-emission probabilities (log-softmax of calibrated logits).
    """
    T, K  = E_log.shape
    dp    = np.full((T, K), -np.inf); bp = np.full((T, K), -1, dtype=np.int32)
    dp[0] = log_pi + E_log[0]
    penalty = lam * (1 - np.eye(K))
    for t in range(1, T):
        prev  = dp[t - 1, :, None] + log_A - penalty
        bp[t] = np.argmax(prev, axis=0)
        dp[t] = prev[bp[t], np.arange(K)] + E_log[t]
    path        = np.zeros(T, dtype=np.int32)
    path[-1]    = int(np.argmax(dp[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]
    return path


def tune_lambda(ts_model, val_logits, val_true, val_seqs,
                train_entries, K):
    """Grid-search λ on the validation fold using the HMM from train."""
    pi, A  = estimate_hmm(train_entries, K)
    log_pi = np.log(np.clip(pi, cfg.HMM_MIN_PROB, 1.0))
    log_A  = np.log(np.clip(A,  cfg.HMM_MIN_PROB, 1.0))

    with torch.no_grad():
        probs = ts_model(val_logits.to(device)).softmax(1).cpu().numpy()

    order = np.argsort(val_seqs)
    E_log = np.log(np.clip(probs[order], cfg.HMM_MIN_PROB, 1.0))

    best_lam, best_f1 = 0.75, -1.0
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        path = viterbi(E_log, log_pi, log_A, lam)
        pred = np.empty_like(path); pred[order] = path
        f1   = f1_score(val_true, pred, average="macro", zero_division=0)
        if f1 > best_f1: best_f1, best_lam = f1, lam

    return best_lam, pi, A


# =============================================================================
# 14.  Metrics
# =============================================================================
def cohen_kappa(yt, yp):
    cm = confusion_matrix(yt, yp); n = cm.sum()
    if n == 0: return 0.0
    po = np.trace(cm) / n
    pe = np.dot(cm.sum(1), cm.sum(0)) / (n * n)
    return float((po - pe) / (1 - pe)) if abs(1 - pe) > 1e-12 else 0.0

def multiclass_mcc(yt, yp):
    cm  = confusion_matrix(yt, yp).astype(float); n = cm.sum()
    if n == 0: return 0.0
    s, t, p = np.trace(cm), cm.sum(1), cm.sum(0)
    num = s * n - np.sum(t * p)
    den = math.sqrt(max(n**2 - np.sum(t**2), 0.) * max(n**2 - np.sum(p**2), 0.))
    return float(num / den) if den > 0 else 0.0


# =============================================================================
# 15.  Single-fold training
# =============================================================================
def train_one_fold(model: PatchHARv2,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   class_w: torch.Tensor,
                   epochs: int,
                   patience: int) -> PatchHARv2:
    total  = epochs * max(1, len(train_loader))
    warmup = int(cfg.WARMUP_FRAC * total)
    opt    = optim.AdamW(model.parameters(),
                         lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    sched  = WarmupCosine(opt, warmup, total, min_lr=cfg.LR * 0.05)
    crit   = SmoothCE(weight=class_w.to(device),
                      smoothing=(cfg.LABEL_SMOOTH_EPS
                                 if CC.C6_LABEL_SMOOTH_TEMP else 0.0))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=GPU)
    except TypeError:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler(enabled=GPU)

    ema        = EMA(model, decay=cfg.EMA_DECAY)
    best_score = -1e9; best_state = None; pat_ctr = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            pl, sf, taf, labels, sid, seq, raw_segs = batch
            pl       = [p.to(device).float() for p in pl]
            sf       = sf.to(device).float()
            taf      = taf.to(device).float()
            labels   = labels.to(device).view(-1)
            raw_segs = raw_segs.to(device).float()

            opt.zero_grad(set_to_none=True)

            # ── [C9] Manifold Mixup: mix in embedding space ──────────────
            if CC.C9_MANIFOLD_MIXUP:
                with amp_ctx():
                    x = model._embed(pl, sf, taf)
                    z, _ = model._backbone(x)                  # (B, D)

                z_mix, la, lb, lam = manifold_mixup(z, labels, cfg.MIXUP_ALPHA)

                with amp_ctx():
                    logits = model.head(z_mix)
                    if CC.C6_LABEL_SMOOTH_TEMP:
                        tau    = torch.exp(model.log_tau).clamp(0.5, 2.0)
                        logits = logits / tau
                    loss = lam * crit(logits, la) + (1 - lam) * crit(logits, lb)
                    if cfg.TC_LAMBDA > 0:
                        loss = loss + cfg.TC_LAMBDA * tc_loss(logits)
                    # C10: reconstruct from un-mixed z (detached)
                    if CC.C10_RECON_AUX_GRAD_SURGERY:
                        recon = model.recon_head(z.detach())
                        loss  = loss + cfg.RECON_LAMBDA * recon_loss_fn(recon, raw_segs)
            else:
                # Standard input-space forward
                with amp_ctx():
                    logits, recon = model(pl, sf, taf)
                    loss = crit(logits, labels)
                    if cfg.TC_LAMBDA > 0:
                        loss = loss + cfg.TC_LAMBDA * tc_loss(logits)
                    if CC.C10_RECON_AUX_GRAD_SURGERY and recon is not None:
                        loss = loss + cfg.RECON_LAMBDA * recon_loss_fn(recon, raw_segs)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if torch.isfinite(loss):
                nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
                scaler.step(opt)
            else:
                opt.zero_grad(set_to_none=True)
            scaler.update(); ema.update(model); sched.step()

        # ── Validation with EMA parameters ───────────────────────────────
        bak = {n: p.detach().clone()
               for n, p in model.named_parameters() if p.requires_grad}
        ema.copy_to(model); model.eval()

        vp, vt, embs_v, labs_v = [], [], [], []
        with torch.no_grad():
            for pl, sf, taf, lbl, sid, seq, _ in val_loader:
                pl  = [p.to(device).float() for p in pl]
                sf  = sf.to(device).float(); taf = taf.to(device).float()
                if CC.C7_PROTOTYPE_MEMORY:
                    z      = model.forward(pl, sf, taf, return_embedding=True)
                    lg, _  = model(pl, sf, taf)
                    embs_v.append(z.cpu()); labs_v.append(lbl)
                else:
                    lg, _ = model(pl, sf, taf)
                vp.extend(lg.argmax(1).cpu().numpy())
                vt.extend(lbl.numpy())

        # [C7] update prototype bank
        if CC.C7_PROTOTYPE_MEMORY and embs_v:
            model.update_prototypes(torch.cat(embs_v).to(device),
                                    torch.cat(labs_v).to(device))

        # restore non-EMA parameters
        for n, p in model.named_parameters():
            if p.requires_grad: p.data.copy_(bak[n])

        vp  = np.array(vp); vt = np.array(vt)
        f1  = f1_score(vt, vp, average="macro", zero_division=0)
        kap = cohen_kappa(vt, vp)
        print(f"    Ep {epoch+1:02d}/{epochs} | F1={f1:.4f} | κ={kap:.4f}")

        score = f1 + kap
        if score > best_score + 1e-6:
            best_score = score; pat_ctr = 0
            ema.copy_to(model)
            best_state = copy.deepcopy(model.state_dict())
        else:
            pat_ctr += 1
            if pat_ctr >= patience:
                print("    ⏹️  Early stop"); break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# =============================================================================
# 16.  LOGO main loop
# =============================================================================
def run_logo():
    # ── Load data ─────────────────────────────────────────────────────────
    print("📂 Loading PAMAP2 dataset …")
    df = pd.read_parquet(cfg.DATA_PATH)
    df = df[[c for c in ALL_COLUMNS if c in df.columns]].copy()
    df[cfg.LABEL_COL] = df[cfg.LABEL_COL].astype(str)

    print("\n── Building windowed entries …")
    entries_all, labels, class_to_idx, idx_to_class = build_all_entries(df)
    num_classes = len(labels)
    subjects    = sorted(df[cfg.SUBJECT_COL].unique().tolist())

    print(f"  Classes ({num_classes}): {labels}")
    print(f"  Subjects: {subjects}")
    print(f"  Total windows: {len(entries_all)}")

    # ── Print active contributions ─────────────────────────────────────────
    print("\n── Active contributions:")
    for k, v in CC.__dict__.items():
        if k.startswith(("C", "S")) and not k.startswith("__"):
            print(f"    {'✓' if v else '✗'} {k}")

    # group entries by subject for fast LOGO indexing
    by_sid = defaultdict(list)
    for e in entries_all: by_sid[int(e[0])].append(e)

    raw_scores, hmm_scores = [], []

    for fold_i, test_sid in enumerate(subjects):
        val_sid    = subjects[(fold_i + 1) % len(subjects)]   # wrap-around
        train_sids = [s for s in subjects if s not in (test_sid, val_sid)]

        train_ent = sum([by_sid[s] for s in train_sids], [])
        val_ent   = by_sid[val_sid]
        test_ent  = by_sid[test_sid]

        print(f"\n{'═'*65}")
        print(f"  FOLD {fold_i+1}/{len(subjects)} | "
              f"Test: {test_sid}  Val: {val_sid}  Train: {train_sids}")
        print(f"  Windows → train:{len(train_ent)}  "
              f"val:{len(val_ent)}  test:{len(test_ent)}")

        if not val_ent or not test_ent:
            print("  ⚠️  Empty val or test set, skipping fold."); continue

        # ── Data loaders ──────────────────────────────────────────────────
        train_w      = sample_weights_from_entries(train_ent, num_classes)
        _, train_dl  = make_loader(train_ent, cfg.BATCH_SIZE,
                                   sampler_weights=train_w, is_train=True)
        _, val_dl    = make_loader(val_ent,  cfg.BATCH_SIZE, shuffle=False)
        _, test_dl   = make_loader(test_ent, cfg.BATCH_SIZE, shuffle=False)

        # ── Model ─────────────────────────────────────────────────────────
        model   = PatchHARv2(num_classes).to(device)
        class_w = class_weights_from_entries(train_ent, num_classes)
        n_param = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {n_param:,}")

        # ── Train ─────────────────────────────────────────────────────────
        model = train_one_fold(model, train_dl, val_dl, class_w,
                               cfg.EPOCHS, cfg.EARLY_STOP_PATIENCE)

        # ── Temperature scaling on val ────────────────────────────────────
        val_logits, val_true, _, val_seqs = collect_logits(model, val_dl)
        ts = fit_temperature(val_logits, val_true)

        # ── Tune HSMM λ on val; estimate π, A from train ─────────────────
        best_lam, pi, A = tune_lambda(ts, val_logits, val_true,
                                      val_seqs, train_ent, num_classes)
        log_pi = np.log(np.clip(pi, cfg.HMM_MIN_PROB, 1.0))
        log_A  = np.log(np.clip(A,  cfg.HMM_MIN_PROB, 1.0))

        # ── Evaluate on test: raw ─────────────────────────────────────────
        test_logits, test_true, _, test_seqs = collect_logits(model, test_dl)
        pred_raw = test_logits.argmax(1).numpy()
        f1r  = f1_score(test_true, pred_raw, average="macro", zero_division=0)
        kr   = cohen_kappa(test_true, pred_raw)
        mr   = multiclass_mcc(test_true, pred_raw)
        print(f"\n  ▶ Raw          | F1={f1r:.4f} | κ={kr:.4f} | MCC={mr:.4f}")

        # ── Evaluate on test: HSMM-lite ───────────────────────────────────
        with torch.no_grad():
            probs = ts(test_logits.to(device)).softmax(1).cpu().numpy()
        order    = np.argsort(test_seqs)
        E_log    = np.log(np.clip(probs[order], cfg.HMM_MIN_PROB, 1.0))
        path     = viterbi(E_log, log_pi, log_A, lam=best_lam)
        pred_hmm = np.empty_like(path); pred_hmm[order] = path
        f1h  = f1_score(test_true, pred_hmm, average="macro", zero_division=0)
        kh   = cohen_kappa(test_true, pred_hmm)
        mh   = multiclass_mcc(test_true, pred_hmm)
        print(f"  ▶ HSMM (λ={best_lam:.2f}) | F1={f1h:.4f} | κ={kh:.4f} | MCC={mh:.4f}")

        if len(np.unique(test_true)) == 1:
            print("  ℹ️  Single-class test fold: κ and MCC = 0 by definition.")

        # per-fold classification report (raw)
        print("\n" + classification_report(
            test_true, pred_raw,
            target_names=[idx_to_class[i] for i in range(num_classes)],
            zero_division=0))

        raw_scores.append((f1r, kr, mr))
        hmm_scores.append((f1h, kh, mh))

    # ── LOGO summary ──────────────────────────────────────────────────────
    def _summarise(scores):
        arr = np.asarray(scores, dtype=np.float64)
        m, s = arr.mean(0), arr.std(0)
        return dict(f1_mean=round(m[0], 4),    f1_std=round(s[0], 4),
                    kappa_mean=round(m[1], 4),  kappa_std=round(s[1], 4),
                    mcc_mean=round(m[2], 4),    mcc_std=round(s[2], 4))

    summary = {
        "raw":  _summarise(raw_scores),
        "hsmm": _summarise(hmm_scores),
        "config": {
            "window_size": cfg.WINDOW_SIZE,
            "patch_len":   cfg.PATCH_LEN,
            "n_patches":   cfg.N_PATCHES,
            "channels":    cfg.CHANNELS,
            "d_model":     cfg.D_MODEL,
            "n_heads":     cfg.N_HEADS,
            "n_layers":    cfg.N_LAYERS,
            "n_experts":   cfg.N_EXPERTS,
            "stat_features": cfg.N_STAT_FEATURES,
            "topo_features": cfg.N_TOPO_FEATURES,
        },
        "contributions": {k: v for k, v in CC.__dict__.items()
                          if not k.startswith("__")},
    }

    print(f"\n{'═'*65}")
    print("  LOGO SUMMARY")
    print(json.dumps(summary, indent=2))
    print("═" * 65)
    return summary


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    run_logo()
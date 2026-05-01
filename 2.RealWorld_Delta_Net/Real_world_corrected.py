"""
realworld2016_patchhar_v2_logo.py
==================================
PatchHAR v2 (all 10 contributions) adapted for the RealWorld2016 dataset
with Leave-One-Group-Out (subject-level) cross-validation.

Dataset
-------
  Source   : RealWorld2016  (proband1 … proband15, ZIP files per activity)
  Sensors  : acc + gyr + mag  @  50 Hz  →  9 channels  (3 per sensor)
  Body part: shin  (configurable)
  Windows  : 500 samples (10 s @ 50 Hz), 50% overlap
  Activities: climbingdown, climbingup, jumping, lying,
              running, sitting, standing, walking  (8 classes)

Model
-----
  PatchHAR v2  D_MODEL=64 | N_HEADS=2 | N_LAYERS=3 | N_EXPERTS=4
  Patch tokens only — no statistical or topological side-features.
  C3 (circadian bias) DISABLED – no timestamp features available.

Post-processing per LOGO fold
------------------------------
  1. Temperature scaling  (LBFGS on val logits)
  2. HSMM-lite Viterbi with lambda tuned on val set

Ablation
--------
  Set any ContribConfig flag to False to disable that contribution, e.g.:
      CC.C1_DUAL_DOMAIN_EMBEDDING = False
"""

from __future__ import annotations
import copy, math, os, random, json, warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Dict, List, Optional
from zipfile import ZipFile

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
    C3_CIRCADIAN_BIAS          = False  # DISABLED – no timestamp features in RealWorld2016
    C4_MULTISCALE_PATCHING     = True   # 3 patch granularities (25, 50, 100 samples)
    C5_FREQ_AUGMENTATION       = True   # band-pass jitter / axis-perm / scale / time-warp
    C6_LABEL_SMOOTH_TEMP       = True   # label-smoothed CE + learnable temperature tau
    C7_PROTOTYPE_MEMORY        = True   # EMA class prototypes blended at inference
    C8_STOCHASTIC_DEPTH        = True   # layer-drop regularisation (linear schedule)
    C9_MANIFOLD_MIXUP          = True   # mixup on pooled embeddings, not raw input
    C10_RECON_AUX_GRAD_SURGERY = True   # lightweight patch-mean reconstruction pretext

CC = ContribConfig()


# =============================================================================
# 2.  Configuration
# =============================================================================
class Config:
    # Data
    DATA_PATH   = "/mnt/share/ali/realworld2016_dataset/"
    SUBJECTS    = tuple(range(1, 16))       # proband1 … proband15
    SENSORS     = ("acc", "gyr", "mag")
    BODY_PART   = "shin"
    OVERLAP     = 0.5                       # sliding-window overlap fraction
    SIGNAL_RATE = 50                        # Hz
    WINDOW_SEC  = 10
    CHANNELS    = 9                         # 3 sensors × 3 axes

    # Derived window / patch sizes
    WINDOW_SIZE      = SIGNAL_RATE * WINDOW_SEC   # 500 samples
    PATCH_LEN        = 50                          # primary granularity → 10 tokens
    PATCH_LENS_MULTI = [25, 50, 100]               # C4 multi-scale
    N_PATCHES        = WINDOW_SIZE // PATCH_LEN    # 10

    # Model
    D_MODEL     = 64
    N_HEADS     = 2       # per-head dim = 32 (even -> RoPE OK)
    N_LAYERS    = 3
    N_EXPERTS   = 4
    DROPOUT     = 0.25
    SD_DROP_MAX = 0.10    # C8: max stochastic-depth drop probability

    # C6 / C7
    LABEL_SMOOTH_EPS = 0.05
    PROTO_MOMENTUM   = 0.95
    PROTO_ALPHA      = 0.30

    # C10
    RECON_LAMBDA = 0.10

    # Training
    BATCH_SIZE          = 64
    EPOCHS              = 120
    LR                  = 1e-3
    WEIGHT_DECAY        = 1e-4
    MAX_GRAD_NORM       = 1.0
    WARMUP_FRAC         = 0.07
    EARLY_STOP_PATIENCE = 30
    EMA_DECAY           = 0.995
    MIXUP_ALPHA         = 0.20
    TC_LAMBDA           = 0.05

    # Temperature scaling
    TEMP_MAX_ITERS = 500
    TEMP_LR        = 5e-2
    TEMP_CLAMP     = (0.05, 10.0)

    # HSMM-lite
    HMM_SMOOTH   = 1.0
    HMM_MIN_PROB = 1e-6

    SEED = 42


cfg = Config()

# Activity labels
ACTIVITY_LABELS: Dict[str, int] = {
    "climbingdown": 0, "climbingup": 1, "jumping": 2, "lying":    3,
    "running":      4, "sitting":   5,  "standing": 6, "walking": 7,
}
IDX_TO_CLASS = {v: k for k, v in ACTIVITY_LABELS.items()}
NUM_CLASSES  = len(ACTIVITY_LABELS)

SENSOR_PREFIX = {"acc": "acc", "gyr": "Gyroscope", "mag": "MagneticField"}


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
print(f"Device: {device}")
if GPU:
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


def amp_ctx():
    if GPU:
        try:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        except Exception:
            from torch.cuda.amp import autocast
            return autocast()
    return nullcontext()


# =============================================================================
# 4.  RealWorld2016 data loading  (faithful port of reference implementation)
# =============================================================================

# Sensor file-name prefixes (matches the ZIP internals)
SENSOR_CFG = {
    "acc": "acc",
    "gyr": "Gyroscope",
    "mag": "MagneticField",
}


def _cal_min_time_len(readme_file) -> int:
    """Parse readMe bytes and return the minimum entry count reported."""
    lengths = []
    for line in readme_file.readlines():
        if line.startswith(b"> entries"):
            try:
                lengths.append(int(line.split()[2]))
            except Exception:
                pass
    return min(lengths) if lengths else 1000


def _cal_sensor_data_from_zip(zip_file: Optional[ZipFile],
                               sensor_type: str,
                               ziplabel: str,
                               body_part: str,
                               min_time_len: int) -> np.ndarray:
    """
    Read one (sensor, body_part) CSV from an open ZipFile.
    Returns (min_time_len, 3) float32 array; NaN-filled on any failure.
    """
    if zip_file is None:
        return np.full((min_time_len, 3), np.nan, dtype=np.float32)
    prefix   = SENSOR_CFG[sensor_type]
    csv_name = f"{prefix}_{ziplabel}_{body_part}.csv"
    if csv_name not in zip_file.namelist():
        return np.full((min_time_len, 3), np.nan, dtype=np.float32)
    try:
        df  = pd.read_csv(zip_file.open(csv_name))
        x   = np.array(df["attr_x"]).reshape(-1, 1)
        y   = np.array(df["attr_y"]).reshape(-1, 1)
        z   = np.array(df["attr_z"]).reshape(-1, 1)
        xyz = np.concatenate((x, y, z), axis=1).astype(np.float32)
        return xyz[:min_time_len]
    except Exception:
        return np.full((min_time_len, 3), np.nan, dtype=np.float32)


def _cal_all_sensor_data(sensor_zip_files: Dict[str, Optional[ZipFile]],
                          ziplabel: str,
                          use_sensors: tuple,
                          body_parts: List[str]) -> Optional[np.ndarray]:
    """
    Build (T, C) array by concatenating sensors then body-parts.
    Mirrors the reference cal_all_sensor_data exactly:
      - inner loop: body_parts concatenated column-wise per sensor
      - outer loop: sensors concatenated column-wise
      - lengths aligned with min() at each concatenation step
    """
    # Determine min_time_len from the first readable readMe
    min_time_len = 1000
    for st in use_sensors:
        zf = sensor_zip_files.get(st)
        if zf is None:
            continue
        try:
            readme = zf.open("readMe")
            tl = _cal_min_time_len(readme)
            readme.close()
            if tl > 0:
                min_time_len = tl
                break
        except Exception:
            continue

    all_sensor_data: Optional[np.ndarray] = None

    for sensor_type in use_sensors:
        # --- concatenate body-parts horizontally for this sensor ---
        sensor_data: Optional[np.ndarray] = None
        for b_part in body_parts:
            xyz = _cal_sensor_data_from_zip(
                sensor_zip_files.get(sensor_type),
                sensor_type, ziplabel, b_part, min_time_len)
            if sensor_data is None:
                sensor_data = xyz
            else:
                m = min(xyz.shape[0], sensor_data.shape[0])
                sensor_data = np.concatenate(
                    (sensor_data[:m], xyz[:m]), axis=1)

        # --- concatenate sensors horizontally ---
        if sensor_data is None:
            continue
        if all_sensor_data is None:
            all_sensor_data = sensor_data
        else:
            m = min(sensor_data.shape[0], all_sensor_data.shape[0])
            all_sensor_data = np.concatenate(
                (all_sensor_data[:m], sensor_data[:m]), axis=1)

    return all_sensor_data


def _check_window_quality(window: np.ndarray,
                           max_nan_ratio:  float = 0.10,
                           max_zero_ratio: float = 0.90) -> bool:
    """
    Exact port of reference check_window_quality.
    Note: constant_channels > 10 threshold matches the original code.
    """
    total = window.size
    if np.isnan(window).sum() / max(total, 1) > max_nan_ratio:  return False
    if (window == 0).sum()    / max(total, 1) > max_zero_ratio: return False
    unique_per_ch = [
        len(np.unique(window[:, c][~np.isnan(window[:, c])]))
        for c in range(window.shape[1])
    ]
    if sum(u <= 1 for u in unique_per_ch) > 10: return False   # matches reference
    finite = window[np.isfinite(window)]
    if finite.size and (finite.max() > 1e4 or finite.min() < -1e4): return False
    return True


def _create_windows(data: np.ndarray, window_size: int,
                    overlap: float = 0.5) -> List[np.ndarray]:
    """Sliding-window extraction with quality filtering (raw, un-normalised)."""
    if data is None or len(data) < window_size:
        return []
    stride  = max(1, int(window_size * (1 - overlap)))
    windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        w = data[i : i + window_size]
        if _check_window_quality(w):
            windows.append(w.astype(np.float32))
    return windows


def load_subject_activity(sub_id: int, label: str) -> List[np.ndarray]:
    """
    Load and window one (subject, activity) pair.
    Returns a list of raw (WINDOW_SIZE, CHANNELS) float32 arrays.
    Normalisation is deferred to the Dataset __getitem__.
    """
    ziplabel = label.replace("_", "")

    # Open one ZipFile per sensor type
    sensor_zip_files: Dict[str, Optional[ZipFile]] = {}
    for s in cfg.SENSORS:
        path = os.path.join(
            cfg.DATA_PATH, f"proband{sub_id}", "data",
            f"{s}_{ziplabel}_csv.zip")
        try:
            sensor_zip_files[s] = ZipFile(path) if os.path.exists(path) else None
        except Exception:
            sensor_zip_files[s] = None

    data = _cal_all_sensor_data(
        sensor_zip_files, ziplabel,
        use_sensors=cfg.SENSORS,
        body_parts=[cfg.BODY_PART])

    for zf in sensor_zip_files.values():
        if zf:
            zf.close()

    return _create_windows(data, cfg.WINDOW_SIZE, cfg.OVERLAP)


def build_all_entries(subjects: tuple) -> list:
    """
    Returns a flat list of entries:
        (sid, window:(WINDOW_SIZE, CHANNELS), label_idx, seq_idx)

    seq_idx is a monotonically increasing counter within each subject,
    preserving temporal order for Viterbi decoding.
    Windows from different activities for the same subject share the
    same counter space so that the sequence index reflects the loading
    order (consistent with the reference generate_all_windows).
    """
    activities = list(ACTIVITY_LABELS.keys())
    total = len(subjects) * len(activities)
    done  = 0
    entries = []

    for sub_id in subjects:
        seq_ctr = 0
        for act in activities:
            windows = load_subject_activity(sub_id, act)
            for w in windows:
                entries.append(
                    (int(sub_id), w, int(ACTIVITY_LABELS[act]), seq_ctr))
                seq_ctr += 1
            done += 1
            print(f"\r  Loading {done}/{total}  windows so far: {len(entries)}",
                  end="", flush=True)
    print()
    return entries


# =============================================================================
# 5.  [C5] Frequency-aware data augmentation
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
    T, C = sig.shape; factor = random.choice([0.9, 0.95, 1.05, 1.1])
    nT   = max(T, int(round(T * factor)))
    warp = np.zeros((nT, C), dtype=sig.dtype)
    for c in range(C):
        warp[:, c] = np.interp(
            np.linspace(0, T - 1, nT), np.arange(T), sig[:, c])
    return warp[:T] if nT >= T else np.pad(warp, ((0, T - nT), (0, 0)))

def freq_augment(sig: np.ndarray) -> np.ndarray:
    return random.choice([_bandpass_jitter, _axis_permute,
                          _magnitude_scale, _time_warp])(sig)


# =============================================================================
# 6.  Dataset & DataLoader
# =============================================================================
class RealWorldDataset(Dataset):
    """
    Per-window output:
      patches_list  list of (CHANNELS, N_Pi, PL_i)   one per patch scale
      label         long scalar
      sid           long scalar
      seq_idx       long scalar  (window index for Viterbi ordering)
      raw_seg       (WINDOW_SIZE, CHANNELS)            used by C10 recon loss
    """
    def __init__(self, entries: list, is_train: bool = False):
        self.entries    = entries
        self.is_train   = is_train
        self.patch_lens = (cfg.PATCH_LENS_MULTI if CC.C4_MULTISCALE_PATCHING
                           else [cfg.PATCH_LEN])

    def __len__(self): return len(self.entries)

    @staticmethod
    def _make_patches(seg: np.ndarray, pl: int) -> np.ndarray:
        """(T, C) -> (C, N_P, PL)"""
        T, C = seg.shape; n = T // pl
        return seg[:n * pl].reshape(n, pl, C).transpose(2, 0, 1).astype(np.float32)

    @staticmethod
    def _normalise(win: np.ndarray) -> np.ndarray:
        """Per-channel instance normalisation, clipped to +-10."""
        mu = np.nanmean(win, axis=0, keepdims=True)
        sd = np.nanstd(win,  axis=0, keepdims=True)
        return np.clip((win - mu) / (sd + 1e-8), -10, 10).astype(np.float32)

    def __getitem__(self, idx):
        sid, win, lab, seq = self.entries[idx]

        # Normalise on the raw window first, then optionally augment
        seg = self._normalise(win)
        if self.is_train and CC.C5_FREQ_AUGMENTATION:
            seg = freq_augment(seg)

        patches_list = [
            torch.from_numpy(self._make_patches(seg, pl))
            for pl in self.patch_lens
        ]

        return (
            patches_list,
            torch.tensor(lab, dtype=torch.long),
            torch.tensor(sid, dtype=torch.long),
            torch.tensor(seq, dtype=torch.long),
            torch.from_numpy(seg.astype(np.float32)),
        )


def _collate(batch):
    patches_lists, labels, sids, seqs, raws = zip(*batch)
    n_scales = len(patches_lists[0])
    return (
        [torch.stack([b[s] for b in patches_lists]) for s in range(n_scales)],
        torch.stack(labels),
        torch.stack(sids),
        torch.stack(seqs),
        torch.stack(raws),
    )


def make_loader(entries, batch_size=64, shuffle=False,
                sampler_weights=None, is_train=False):
    ds = RealWorldDataset(entries, is_train=is_train)
    if sampler_weights is not None:
        sampler = WeightedRandomSampler(sampler_weights,
                                        num_samples=len(sampler_weights),
                                        replacement=True)
        dl = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                        num_workers=4, pin_memory=GPU, collate_fn=_collate)
    else:
        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=4, pin_memory=GPU, collate_fn=_collate)
    return ds, dl


# =============================================================================
# 7.  Training utilities: WarmupCosine, EMA, class/sample weights
# =============================================================================
class WarmupCosine:
    """Linear warmup -> cosine annealing LR scheduler."""
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
                self.shadow[n] = (self.decay * self.shadow[n]
                                  + (1. - self.decay) * p.detach())

    def copy_to(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])


def class_weights_from_entries(entries, K):
    counts = np.zeros(K, dtype=np.int64)
    for _, _, lab, _ in entries: counts[lab] += 1
    w = counts.max() / np.clip(counts, 1, None)
    w = torch.tensor(w, dtype=torch.float32)
    return w / w.sum() * K


def sample_weights_from_entries(entries, K):
    counts = np.zeros(K, dtype=np.int64)
    for _, _, lab, _ in entries: counts[lab] += 1
    inv = counts.max() / np.clip(counts, 1, None)
    w   = np.array([inv[lab] for (_, _, lab, _) in entries], dtype=np.float32)
    return torch.tensor(w, dtype=torch.float32)


# =============================================================================
# 8.  Model building blocks
# =============================================================================

class ZCRMSNorm(nn.Module):
    """Zero-centred RMS normalisation."""
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__(); self.g = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        x0 = x - x.mean(-1, keepdim=True)
        return x0 / x0.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt() * self.g


class StochasticDepth(nn.Module):
    """[C8] Wraps any layer with probabilistic layer-drop."""
    def __init__(self, layer: nn.Module, survival_prob: float):
        super().__init__(); self.layer = layer; self.p = survival_prob

    def forward(self, x, *args, **kwargs):
        if not self.training or self.p >= 1.0:
            return self.layer(x, *args, **kwargs)
        if random.random() > self.p: return x
        return self.layer(x, *args, **kwargs)


class GatedDeltaNet(nn.Module):
    """Local temporal modelling block."""
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
    def _l2(x, eps=1e-8):
        return x / (x.pow(2).sum(-1, keepdim=True).add(eps).sqrt())

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


class SoftMoE(nn.Module):
    """Soft Mixture-of-Experts."""
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


def precompute_freqs(dim: int, n_tok: int, theta: float = 10000.0) -> torch.Tensor:
    assert dim % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    return torch.polar(
        torch.ones(n_tok, dim // 2),
        torch.outer(torch.arange(n_tok, dtype=torch.float32), freqs))

def apply_rope(q, k, freqs):
    B, H, N, D = q.shape; d2 = D // 2
    f  = freqs[:N].to(q.device).view(1, 1, N, d2)
    qc = torch.view_as_complex(q.float().contiguous().view(B, H, N, d2, 2))
    kc = torch.view_as_complex(k.float().contiguous().view(B, H, N, d2, 2))
    return (torch.view_as_real(qc * f).view(B, H, N, D).type_as(q),
            torch.view_as_real(kc * f).view(B, H, N, D).type_as(k))


class GatedAttention(nn.Module):
    """Multi-head self-attention with RoPE and output gate."""
    def __init__(self, d: int, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        assert d % n_heads == 0 and (d // n_heads) % 2 == 0, \
            "Per-head dim must be even for RoPE"
        self.h    = n_heads; self.dh = d // n_heads
        self.norm = ZCRMSNorm(d)
        self.qkv  = nn.Linear(d, 3 * d)
        self.out  = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, freqs):
        h = self.norm(x); B, N, D = h.shape
        qkv = (self.qkv(h).reshape(B, N, 3, self.h, self.dh)
               .permute(0, 2, 1, 3, 4))
        q = qkv[:, 0].transpose(1, 2)
        k = qkv[:, 1].transpose(1, 2)
        v = qkv[:, 2].transpose(1, 2)
        q, k = apply_rope(q, k, freqs)
        attn = self.drop(torch.softmax(
            (q @ k.transpose(-2, -1)) / math.sqrt(self.dh), dim=-1))
        y = self.out(
            (attn @ v).transpose(1, 2).contiguous().reshape(B, N, D))
        return x + torch.sigmoid(self.gate(h)) * y


class DualDomainPatchEmbed(nn.Module):
    """
    [C1] Encodes each patch in time and frequency domains, fused with a
    learnable gate:  out = sigmoid(g) * time_proj + (1 - sigmoid(g)) * freq_proj
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
        t_emb = self.time_proj(
            patches.permute(0, 2, 1, 3).reshape(B, NP, C * PL))
        mag   = torch.fft.rfft(patches.permute(0, 2, 3, 1), dim=2).abs()
        f_emb = self.freq_proj(mag.reshape(B, NP, -1))
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


class SkipAggregation(nn.Module):
    """[C2] Learnable weighted sum of GatedDeltaNet layer outputs."""
    def __init__(self, d: int, n_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_layers) / n_layers)
        self.proj    = nn.Sequential(nn.Linear(d, d), nn.SiLU())

    def forward(self, hiddens: list) -> torch.Tensor:
        w   = torch.softmax(self.weights, dim=0)
        agg = sum(w[i] * hiddens[i] for i in range(len(hiddens)))
        return self.proj(agg)


class ReconHead(nn.Module):
    """
    [C10] Reconstructs per-patch channel means from pooled representation z.
    Output dim = N_PATCHES x CHANNELS = 10 x 9 = 90.
    """
    def __init__(self, d: int, n_patches: int, channels: int):
        super().__init__()
        out_dim  = n_patches * channels
        self.mlp = nn.Sequential(
            nn.Linear(d, 2 * d), nn.ReLU(),
            nn.Linear(2 * d, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)


# =============================================================================
# 9.  PatchHAR v2  -  RealWorld2016 edition
# =============================================================================
class PatchHARv2(nn.Module):
    """
    PatchHAR v2 with all 10 contributions, adapted for RealWorld2016:
      - CHANNELS = 9  (acc + gyr + mag, 3 axes each)
      - WINDOW_SIZE = 500  (10 s @ 50 Hz)
      - N_PATCHES = 10  (primary patch length = 50 samples)
      - C3 circadian bias DISABLED
    """
    def __init__(self, num_classes: int):
        super().__init__()
        d  = cfg.D_MODEL
        NP = cfg.N_PATCHES

        # Patch embeddings (one per granularity for C4)
        self.patch_lens = (cfg.PATCH_LENS_MULTI if CC.C4_MULTISCALE_PATCHING
                           else [cfg.PATCH_LEN])
        EmbedCls = (DualDomainPatchEmbed if CC.C1_DUAL_DOMAIN_EMBEDDING
                    else SimplePatchEmbed)
        self.patch_embeds = nn.ModuleList([
            EmbedCls(pl, cfg.CHANNELS, d) for pl in self.patch_lens
        ])

        # [C4] fuse multi-scale interpolated tokens -> (B, NP, D)
        if CC.C4_MULTISCALE_PATCHING and len(self.patch_lens) > 1:
            self.scale_fusion = nn.Linear(d * len(self.patch_lens), d)
        else:
            self.scale_fusion = None

        self.input_norm = nn.LayerNorm(d)

        # [C8] Stochastic-depth GatedDeltaNet layers
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

        # [C2] Skip aggregation
        if CC.C2_CALANET_SKIP_AGG:
            self.skip_agg = SkipAggregation(d, cfg.N_LAYERS)

        # Transformer core
        self.moe1 = SoftMoE(d, 2 * d, cfg.N_EXPERTS, cfg.DROPOUT)
        self.attn = GatedAttention(d, cfg.N_HEADS, cfg.DROPOUT)
        self.moe2 = SoftMoE(d, 2 * d, cfg.N_EXPERTS, cfg.DROPOUT)

        # RoPE frequency cache sized to N_PATCHES
        freqs = precompute_freqs(d // cfg.N_HEADS, NP)
        self.register_buffer("freqs", freqs)

        # [C6] Learnable temperature tau
        if CC.C6_LABEL_SMOOTH_TEMP:
            self.log_tau = nn.Parameter(torch.zeros(1))

        # [C10] Reconstruction head
        if CC.C10_RECON_AUX_GRAD_SURGERY:
            self.recon_head = ReconHead(d, NP, cfg.CHANNELS)

        # [C7] Prototype memory bank
        if CC.C7_PROTOTYPE_MEMORY:
            self.register_buffer("prototypes", torch.zeros(num_classes, d))
            self.proto_filled = False

        self.num_classes = num_classes

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(d, d // 2), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d // 2, num_classes),
        )

    def _embed(self, patches_list) -> torch.Tensor:
        """patches_list : list of (B, C, N_Pi, PL_i)  ->  (B, N_PATCHES, D)"""
        NP = cfg.N_PATCHES
        if CC.C4_MULTISCALE_PATCHING and len(patches_list) > 1:
            embs = []
            for embed, p in zip(self.patch_embeds, patches_list):
                e = embed(p)
                e = F.interpolate(e.permute(0, 2, 1),
                                  size=NP, mode="linear",
                                  align_corners=False).permute(0, 2, 1)
                embs.append(e)
            return self.scale_fusion(torch.cat(embs, dim=-1))
        else:
            return self.patch_embeds[0](patches_list[0])

    def _backbone(self, x: torch.Tensor):
        """x : (B, NP, D)  ->  z : (B, D)"""
        x = self.input_norm(x)
        hiddens = []
        for layer in self.delta_layers:
            x = layer(x); hiddens.append(x)
        if CC.C2_CALANET_SKIP_AGG:
            x = x + self.skip_agg(hiddens)
        x = x + self.moe1(x)
        x = self.attn(x, self.freqs)
        x = x + self.moe2(x)
        return x.mean(dim=1), hiddens

    def forward(self, patches_list, return_embedding: bool = False):
        x = self._embed(patches_list)
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

        if CC.C7_PROTOTYPE_MEMORY and not self.training and self.proto_filled:
            z_n    = F.normalize(z, dim=-1)
            pr_n   = F.normalize(self.prototypes, dim=-1)
            cosine = z_n @ pr_n.T
            logits = ((1 - cfg.PROTO_ALPHA) * logits
                      + cfg.PROTO_ALPHA * cosine)

        return logits, recon

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
# 10.  Loss & training helpers
# =============================================================================
class SmoothCE(nn.Module):
    """[C6] Label-smoothed cross-entropy with optional class weights."""
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
    """[C10] Per-patch channel-mean reconstruction loss. Target: (B, NP*C)."""
    B, T, C = raw_segs.shape
    NP, PL  = cfg.N_PATCHES, cfg.PATCH_LEN
    target  = (raw_segs[:, :NP * PL, :]
               .reshape(B, NP, PL, C)
               .mean(dim=2)
               .reshape(B, NP * C))
    return F.mse_loss(recon, target.detach())


def tc_loss(logits: torch.Tensor) -> torch.Tensor:
    """Symmetrised KL divergence between adjacent-window predictions."""
    if logits.size(0) < 2:
        return logits.new_zeros(1).squeeze()
    p = F.softmax(logits[:-1], dim=-1)
    q = F.softmax(logits[1:],  dim=-1)
    return 0.5 * (F.kl_div(q.log(), p, reduction="batchmean") +
                  F.kl_div(p.log(), q, reduction="batchmean"))


def manifold_mixup(z: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2):
    """[C9] Mix in embedding space; returns (z_mixed, la, lb, lambda)."""
    if alpha <= 0: return z, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(z.size(0), device=z.device)
    return lam * z + (1 - lam) * z[idx], labels, labels[idx], lam


# =============================================================================
# 11.  Temperature scaling & HSMM-lite
# =============================================================================
class TemperatureScaling(nn.Module):
    def __init__(self, T_init: float = 1.0):
        super().__init__()
        self.logT = nn.Parameter(torch.tensor(math.log(float(T_init))))

    def forward(self, logits):
        return logits / torch.exp(self.logT).clamp(*cfg.TEMP_CLAMP)

    def temperature(self):
        with torch.no_grad():
            return float(torch.exp(self.logT).clamp(*cfg.TEMP_CLAMP).item())


def fit_temperature(logits: torch.Tensor,
                    labels: np.ndarray) -> TemperatureScaling:
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
    print(f"    T={ts.temperature():.3f} | val NLL={nll:.4f}")
    return ts


def collect_logits(model: PatchHARv2, loader: DataLoader):
    model.eval()
    outs, labs, sids_out, seqs_out = [], [], [], []
    with torch.no_grad():
        for pl, lbl, sid, seq, _ in loader:
            pl        = [p.to(device).float() for p in pl]
            logits, _ = model(pl)
            outs.append(logits.cpu())
            labs.extend(lbl.numpy().tolist())
            sids_out.extend(sid.numpy().tolist())
            seqs_out.extend(seq.numpy().tolist())
    return (torch.cat(outs),
            np.array(labs),
            np.array(sids_out),
            np.array(seqs_out))


def estimate_hmm(entries: list, K: int):
    by_sid = defaultdict(list)
    for sid, _, lab, seq in entries:
        by_sid[int(sid)].append((int(seq), int(lab)))

    A  = np.full((K, K), cfg.HMM_SMOOTH, dtype=np.float64)
    pi = np.full(K,      cfg.HMM_SMOOTH, dtype=np.float64)

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
    T, K  = E_log.shape
    dp    = np.full((T, K), -np.inf)
    bp    = np.full((T, K), -1, dtype=np.int32)
    dp[0] = log_pi + E_log[0]
    penalty = lam * (1 - np.eye(K))
    for t in range(1, T):
        prev  = dp[t - 1, :, None] + log_A - penalty
        bp[t] = np.argmax(prev, axis=0)
        dp[t] = prev[bp[t], np.arange(K)] + E_log[t]
    path     = np.zeros(T, dtype=np.int32)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]
    return path


def tune_lambda(ts_model, val_logits, val_true, val_seqs, train_entries, K):
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
# 12.  Metrics
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
    den = math.sqrt(max(n**2 - np.sum(t**2), 0.) *
                    max(n**2 - np.sum(p**2), 0.))
    return float(num / den) if den > 0 else 0.0


# =============================================================================
# 13.  Single-fold training
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
            pl, labels, sid, seq, raw_segs = batch
            pl       = [p.to(device).float() for p in pl]
            labels   = labels.to(device).view(-1)
            raw_segs = raw_segs.to(device).float()

            opt.zero_grad(set_to_none=True)

            if CC.C9_MANIFOLD_MIXUP:
                with amp_ctx():
                    x = model._embed(pl)
                    z, _ = model._backbone(x)

                z_mix, la, lb, lam = manifold_mixup(z, labels, cfg.MIXUP_ALPHA)

                with amp_ctx():
                    logits = model.head(z_mix)
                    if CC.C6_LABEL_SMOOTH_TEMP:
                        tau    = torch.exp(model.log_tau).clamp(0.5, 2.0)
                        logits = logits / tau
                    loss = (lam * crit(logits, la)
                            + (1 - lam) * crit(logits, lb))
                    if cfg.TC_LAMBDA > 0:
                        loss = loss + cfg.TC_LAMBDA * tc_loss(logits)
                    if CC.C10_RECON_AUX_GRAD_SURGERY:
                        recon = model.recon_head(z.detach())
                        loss  = loss + cfg.RECON_LAMBDA * recon_loss_fn(
                            recon, raw_segs)
            else:
                with amp_ctx():
                    logits, recon = model(pl)
                    loss = crit(logits, labels)
                    if cfg.TC_LAMBDA > 0:
                        loss = loss + cfg.TC_LAMBDA * tc_loss(logits)
                    if CC.C10_RECON_AUX_GRAD_SURGERY and recon is not None:
                        loss = loss + cfg.RECON_LAMBDA * recon_loss_fn(
                            recon, raw_segs)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if torch.isfinite(loss):
                nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
                scaler.step(opt)
            else:
                opt.zero_grad(set_to_none=True)
            scaler.update(); ema.update(model); sched.step()

        # Validation with EMA weights
        bak = {n: p.detach().clone()
               for n, p in model.named_parameters() if p.requires_grad}
        ema.copy_to(model); model.eval()

        vp, vt, embs_v, labs_v = [], [], [], []
        with torch.no_grad():
            for pl, lbl, sid, seq, _ in val_loader:
                pl = [p.to(device).float() for p in pl]
                if CC.C7_PROTOTYPE_MEMORY:
                    z     = model.forward(pl, return_embedding=True)
                    lg, _ = model(pl)
                    embs_v.append(z.cpu()); labs_v.append(lbl)
                else:
                    lg, _ = model(pl)
                vp.extend(lg.argmax(1).cpu().numpy())
                vt.extend(lbl.numpy())

        if CC.C7_PROTOTYPE_MEMORY and embs_v:
            model.update_prototypes(torch.cat(embs_v).to(device),
                                    torch.cat(labs_v).to(device))

        for n, p in model.named_parameters():
            if p.requires_grad: p.data.copy_(bak[n])

        vp  = np.array(vp); vt = np.array(vt)
        f1  = f1_score(vt, vp, average="macro", zero_division=0)
        kap = cohen_kappa(vt, vp)
        print(f"    Ep {epoch+1:03d}/{epochs} | F1={f1:.4f} | kappa={kap:.4f}")

        score = f1 + kap
        if score > best_score + 1e-6:
            best_score = score; pat_ctr = 0
            ema.copy_to(model)
            best_state = copy.deepcopy(model.state_dict())
        else:
            pat_ctr += 1
            if pat_ctr >= patience:
                print("    Early stop"); break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# =============================================================================
# 14.  LOGO main loop
# =============================================================================
def run_logo():
    print("=" * 65)
    print("  PatchHAR v2  —  RealWorld2016  —  LOGO CV")
    print(f"  D_MODEL={cfg.D_MODEL} | N_HEADS={cfg.N_HEADS} | "
          f"N_LAYERS={cfg.N_LAYERS} | N_EXPERTS={cfg.N_EXPERTS}")
    print(f"  Window={cfg.WINDOW_SIZE} samples | "
          f"Patch lens={cfg.PATCH_LENS_MULTI} | Channels={cfg.CHANNELS}")
    print("  Active contributions:")
    for k, v in CC.__dict__.items():
        if not k.startswith("__"):
            print(f"    {'ON ' if v else 'OFF'} {k}")
    print("=" * 65)

    # ── Load all data once ────────────────────────────────────────────────
    print("\nLoading RealWorld2016 dataset ...")
    entries_all = build_all_entries(cfg.SUBJECTS)
    print(f"  Total windows: {len(entries_all)}")

    subjects = sorted(set(e[0] for e in entries_all))
    by_sid   = defaultdict(list)
    for e in entries_all: by_sid[e[0]].append(e)

    raw_scores, hmm_scores = [], []

    for fold_i, test_sid in enumerate(subjects):
        val_sid    = subjects[(fold_i + 1) % len(subjects)]
        train_sids = [s for s in subjects if s not in (test_sid, val_sid)]

        train_ent = sum([by_sid[s] for s in train_sids], [])
        val_ent   = by_sid[val_sid]
        test_ent  = by_sid[test_sid]

        print(f"\n{'='*65}")
        print(f"  FOLD {fold_i+1}/{len(subjects)}  "
              f"test=proband{test_sid}  val=proband{val_sid}  "
              f"train={[f'p{s}' for s in train_sids]}")
        print(f"  Windows  train:{len(train_ent)}  "
              f"val:{len(val_ent)}  test:{len(test_ent)}")

        if not val_ent or not test_ent:
            print("  Empty val or test set -- skipping."); continue

        train_w     = sample_weights_from_entries(train_ent, NUM_CLASSES)
        _, train_dl = make_loader(train_ent, cfg.BATCH_SIZE,
                                  sampler_weights=train_w, is_train=True)
        _, val_dl   = make_loader(val_ent,  cfg.BATCH_SIZE, shuffle=False)
        _, test_dl  = make_loader(test_ent, cfg.BATCH_SIZE, shuffle=False)

        model   = PatchHARv2(NUM_CLASSES).to(device)
        class_w = class_weights_from_entries(train_ent, NUM_CLASSES)
        n_param = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_param:,}")

        model = train_one_fold(model, train_dl, val_dl, class_w,
                               cfg.EPOCHS, cfg.EARLY_STOP_PATIENCE)

        # Temperature scaling on val
        val_logits, val_true, _, val_seqs = collect_logits(model, val_dl)
        ts = fit_temperature(val_logits, val_true)

        # Tune HSMM lambda on val; estimate pi, A from train
        best_lam, pi, A = tune_lambda(ts, val_logits, val_true,
                                      val_seqs, train_ent, NUM_CLASSES)
        log_pi = np.log(np.clip(pi, cfg.HMM_MIN_PROB, 1.0))
        log_A  = np.log(np.clip(A,  cfg.HMM_MIN_PROB, 1.0))

        # Evaluate: raw
        test_logits, test_true, _, test_seqs = collect_logits(model, test_dl)
        pred_raw = test_logits.argmax(1).numpy()
        f1r = f1_score(test_true, pred_raw, average="macro", zero_division=0)
        kr  = cohen_kappa(test_true, pred_raw)
        mr  = multiclass_mcc(test_true, pred_raw)
        print(f"\n  Raw          F1={f1r:.4f} | kappa={kr:.4f} | MCC={mr:.4f}")

        # Evaluate: HSMM-lite
        with torch.no_grad():
            probs = ts(test_logits.to(device)).softmax(1).cpu().numpy()
        order    = np.argsort(test_seqs)
        E_log    = np.log(np.clip(probs[order], cfg.HMM_MIN_PROB, 1.0))
        path     = viterbi(E_log, log_pi, log_A, lam=best_lam)
        pred_hmm = np.empty_like(path); pred_hmm[order] = path
        f1h = f1_score(test_true, pred_hmm, average="macro", zero_division=0)
        kh  = cohen_kappa(test_true, pred_hmm)
        mh  = multiclass_mcc(test_true, pred_hmm)
        print(f"  HSMM lam={best_lam:.2f}  F1={f1h:.4f} | kappa={kh:.4f} | MCC={mh:.4f}")

        print("\n" + classification_report(
            test_true, pred_raw,
            target_names=[IDX_TO_CLASS[i] for i in range(NUM_CLASSES)],
            zero_division=0))

        raw_scores.append((f1r, kr, mr))
        hmm_scores.append((f1h, kh, mh))

    # ── LOGO summary ──────────────────────────────────────────────────────
    def _summarise(scores):
        arr = np.asarray(scores, dtype=np.float64)
        m, s = arr.mean(0), arr.std(0)
        return dict(f1_mean=round(m[0], 4),   f1_std=round(s[0], 4),
                    kappa_mean=round(m[1], 4), kappa_std=round(s[1], 4),
                    mcc_mean=round(m[2], 4),   mcc_std=round(s[2], 4))

    summary = {
        "raw":  _summarise(raw_scores),
        "hsmm": _summarise(hmm_scores),
        "config": {
            "dataset":     "RealWorld2016",
            "body_part":   cfg.BODY_PART,
            "window_size": cfg.WINDOW_SIZE,
            "overlap":     cfg.OVERLAP,
            "patch_lens":  cfg.PATCH_LENS_MULTI,
            "n_patches":   cfg.N_PATCHES,
            "channels":    cfg.CHANNELS,
            "d_model":     cfg.D_MODEL,
            "n_heads":     cfg.N_HEADS,
            "n_layers":    cfg.N_LAYERS,
            "n_experts":   cfg.N_EXPERTS,
            "n_subjects":  len(subjects),
            "n_classes":   NUM_CLASSES,
        },
        "contributions": {k: v for k, v in CC.__dict__.items()
                          if not k.startswith("__")},
    }

    print(f"\n{'='*65}")
    print("  LOGO SUMMARY")
    print(json.dumps(summary, indent=2))
    print("=" * 65)
    return summary


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    run_logo()
"""
PatchHAR v3 — Wrist-Accelerometer HAR for CAPTURE-24
=====================================================
Contributions
  C1  DualDomainEmbedding     (off by default)
  C3  CircadianBias
  C4  MultiscalePatching       {25, 50, 100}
  C5  FreqAugmentation
  C6  LabelSmoothTemp
  C7  PrototypeMemory
  C8  StochasticDepth
  C9  ManifoldMixup
  C10 ReconAuxLoss

New in v3
  C11 AxisInteractionBlock     — cross-axis 3×3 attention per patch
  C12 AdversarialDisentangle   — GRL subject classifier
  C13 BoundaryDetection        — focal-BCE boundary CNN + attention gate
  C14 KinematicMemory          — correct recurrent delta-rule + freq-conditioned γ

v2 bugs fixed
  • TC loss removed              (shuffle=True made it measure noise)
  • GDN renamed/replaced         (was not a true delta rule)
  • repeat_interleave → interp   (coarse embedding upsampling, was lossy)
  • _time_warp stretch-only      (factor<1 produced zero-padded discontinuities)
  • SoftMoE → SoftRoutedMoE     (not the slot-based Puigcerver version)
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
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ══════════════════════════════════════════════════════════════════════════════
# 0.  CONTRIBUTION FLAGS
# ══════════════════════════════════════════════════════════════════════════════
class ContribConfig:
    # ── inherited ──────────────────────────────────────────────────────────
    C1_DUAL_DOMAIN_EMBEDDING    = False
    C3_CIRCADIAN_BIAS           = True
    C4_MULTISCALE_PATCHING      = True
    C5_FREQ_AUGMENTATION        = True
    C6_LABEL_SMOOTH_TEMP        = True
    C7_PROTOTYPE_MEMORY         = True
    C8_STOCHASTIC_DEPTH         = True
    C9_MANIFOLD_MIXUP           = True
    C10_RECON_AUX_GRAD_SURGERY  = True
    # ── new ────────────────────────────────────────────────────────────────
    C11_AXIS_INTERACTION        = True
    C12_ADVERSARIAL_DISENTANGLE = True
    C13_BOUNDARY_DETECTION      = True
    C14_KINEMATIC_MEMORY        = True

CC = ContribConfig()


def _parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--disable", nargs="*", default=[], metavar="Cn")
    args, _ = p.parse_known_args()
    for flag in (args.disable or []):
        attr = flag.upper()
        if hasattr(CC, attr):
            setattr(CC, attr, False)
            print(f"  [Ablation] {attr} DISABLED")
        else:
            print(f"  [Warn] Unknown flag: {flag}")

_parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 1.  GLOBAL CONFIG
# ══════════════════════════════════════════════════════════════════════════════
class Config:
    PROC_DIR   = Path("/mnt/share/ali/processed/")
    OUTPUT_DIR = Path("/mnt/share/ali/processed/patchhar_results/")

    TRAIN_N = 80
    VAL_N   = 20

    SIGNAL_RATE      = 100
    WINDOW_SIZE      = 3000
    PATCH_LEN        = 25
    CHANNELS         = 3
    N_PATCHES        = WINDOW_SIZE // PATCH_LEN   # 120

    PATCH_LENS_MULTI = [25, 50, 100]

    D_MODEL   = 64
    N_HEADS   = 2
    N_LAYERS  = 3
    N_EXPERTS = 4
    DROPOUT   = 0.25

    SD_DROP_MAX      = 0.10
    LABEL_SMOOTH_EPS = 0.10
    PROTO_MOMENTUM   = 0.95
    PROTO_ALPHA      = 0.30
    RECON_LAMBDA     = 0.10

    # C12 — adversarial disentanglement
    ADV_LAMBDA_MAX     = 0.5     # ceiling of GRL reversal scale
    ADV_SUBJECT_WEIGHT = 0.10    # subject-CE loss weight

    # C13 — boundary detection
    BOUNDARY_LAMBDA = 0.10
    FOCAL_ALPHA     = 0.75       # focal α for rare boundary=1 class
    FOCAL_GAMMA     = 2.0

    # C14 — kinematic memory
    N_FREQ_BANDS = 8

    BATCH_SIZE          = 32
    EPOCHS              = 30
    LR                  = 1e-3
    WEIGHT_DECAY        = 1e-4
    MAX_GRAD_NORM       = 1.0
    EARLY_STOP_PATIENCE = 8
    SEED                = 42
    MIXUP_ALPHA         = 0.2

    def _update(self, actual_window: int):
        self.WINDOW_SIZE = actual_window
        self.N_PATCHES   = actual_window // self.PATCH_LEN


cfg = Config()
cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATA DISCOVERY + SUBJECT MAPPING
# ══════════════════════════════════════════════════════════════════════════════
def discover(proc_dir: Path):
    files = sorted(proc_dir.glob("P*.npz"))
    if not files:
        raise FileNotFoundError(f"No P*.npz in {proc_dir}")
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
    print(f"Discovered {len(pids)} participants | {len(classes)} classes: {classes}")
    return pids, classes, class_to_idx, idx_to_class


pids_all, CLASSES, class_to_idx, idx_to_class = discover(cfg.PROC_DIR)
NUM_CLASSES = len(CLASSES)

n_train    = min(cfg.TRAIN_N, len(pids_all))
n_val      = min(cfg.VAL_N,   max(0, len(pids_all) - n_train))
train_pids = pids_all[:n_train]
val_pids   = pids_all[n_train : n_train + n_val]
test_pids  = pids_all[n_train + n_val :]
print(f"Split  : train={len(train_pids)} | val={len(val_pids)} | test={len(test_pids)}")

# Training-subject index mapping — val/test subjects receive -1
train_subject_to_idx = {pid: i for i, pid in enumerate(train_pids)}
N_TRAIN_SUBJECTS     = len(train_pids)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DATA UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def time_features(ns: int) -> np.ndarray:
    ts  = pd.to_datetime(int(ns), unit="ns", utc=True).tz_convert(None)
    out = np.zeros(5, dtype=np.float32)
    out[0] = ts.hour      / 24.0
    out[1] = ts.minute    / 60.0
    out[2] = ts.weekday() / 7.0
    out[3] = float(ts.weekday() >= 5)
    out[4] = float(ts.hour // 6) / 3.0
    return out


# ── Augmentation functions ────────────────────────────────────────────────────
def _bandpass_jitter(sig: np.ndarray) -> np.ndarray:
    T, C = sig.shape
    out  = sig.copy()
    band = random.choice(["low", "mid", "high"])
    for c in range(C):
        f = np.fft.rfft(out[:, c])
        n = len(f)
        if band == "low":
            f[n // 3 :]    = 0
        elif band == "mid":
            f[:n // 4]     = 0
            f[n // 2 :]    = 0
        else:
            f[:2*n // 3]   = 0
        out[:, c] = np.fft.irfft(f, n=T)
    return out


def _axis_permute(sig: np.ndarray) -> np.ndarray:
    idx = list(range(sig.shape[1]))
    random.shuffle(idx)
    return sig[:, idx]


def _magnitude_scale(sig: np.ndarray) -> np.ndarray:
    return sig * np.random.uniform(0.8, 1.2, size=(1, sig.shape[1]))


def _time_warp(sig: np.ndarray) -> np.ndarray:
    """Stretch-only warp (factor ≥ 1.0) — no zero-padding discontinuities."""
    T, C   = sig.shape
    factor = random.choice([1.0, 1.05, 1.10, 1.15, 1.20])   # stretch only
    new_T  = int(round(T * factor))   # new_T >= T always
    warped = np.zeros((new_T, C), dtype=sig.dtype)
    for c in range(C):
        warped[:, c] = np.interp(
            np.linspace(0, T - 1, new_T), np.arange(T), sig[:, c])
    return warped[:T]   # crop — always valid


def freq_augment(sig: np.ndarray) -> np.ndarray:
    fn = random.choice([_bandpass_jitter, _axis_permute,
                        _magnitude_scale, _time_warp])
    return fn(sig)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  DATASET
# ══════════════════════════════════════════════════════════════════════════════
class WindowDataset(Dataset):
    """
    Entry tuple stored per window:
      (pid, seg, tfeat, label_idx, first_ns, boundary_label, subject_idx)

    boundary_label: float 0/1 — this window immediately follows a label
                    change within the same subject (precomputed at load time).
    subject_idx:    int — index into the training-subject list, or -1 for
                    val/test subjects (where adversarial loss is skipped).
    """
    def __init__(self, pid_list, proc_dir, class_to_idx,
                 subject_to_idx, is_train=False):
        self.entries    = []
        self.is_train   = is_train
        proc_dir        = Path(proc_dir)
        self.patch_lens = (cfg.PATCH_LENS_MULTI if CC.C4_MULTISCALE_PATCHING
                           else [cfg.PATCH_LEN])
        _window_set     = False

        for pi, pid in enumerate(pid_list):
            path = proc_dir / f"{pid}.npz"
            if not path.exists():
                print(f"  [SKIP] {pid}.npz not found")
                continue

            npz   = np.load(path, allow_pickle=True)
            W     = npz["X"].astype(np.float32)
            L     = npz["y"].astype(str)
            F_    = npz["t"].astype("datetime64[ns]").astype(np.int64)

            order    = np.argsort(F_)
            W, L, F_ = W[order], L[order], F_[order]

            if not _window_set:
                if W.shape[1] != cfg.WINDOW_SIZE:
                    print(f"  [INFO] Actual window {W.shape[1]} → updating cfg")
                    cfg._update(W.shape[1])
                _window_set = True

            # Precompute boundary labels within this subject.
            # boundary[i] = 1.0 iff label[i] != label[i-1] (same subject).
            # No cross-subject boundary: boundary[0] always 0.
            boundaries = np.zeros(len(W), dtype=np.float32)
            for i in range(1, len(W)):
                if L[i] != L[i - 1]:
                    boundaries[i] = 1.0

            subj_idx = subject_to_idx.get(pid, -1)

            for w, lab, f, bnd in zip(W, L, F_, boundaries):
                if lab not in class_to_idx:
                    continue
                # Per-channel z-score normalisation
                normed = np.zeros_like(w, dtype=np.float32)
                for c in range(cfg.CHANNELS):
                    ch = w[:, c]
                    normed[:, c] = (ch - ch.mean()) / (ch.std() + 1e-8)
                normed = np.clip(normed, -10, 10)
                T   = cfg.WINDOW_SIZE
                seg = (normed[:T] if normed.shape[0] >= T
                       else np.pad(normed, ((0, T - normed.shape[0]), (0, 0))))
                self.entries.append((
                    pid,
                    seg,
                    time_features(int(f)),
                    int(class_to_idx[lab]),
                    int(f),
                    float(bnd),
                    int(subj_idx),
                ))

            if (pi + 1) % 10 == 0 or (pi + 1) == len(pid_list):
                print(f"  Loaded {pi+1}/{len(pid_list)} subjects — "
                      f"{len(self.entries):,} windows")

    @staticmethod
    def _make_patches(seg: np.ndarray, patch_len: int) -> np.ndarray:
        T, C = seg.shape
        n_p  = T // patch_len
        seg  = seg[:n_p * patch_len]
        # (NP, PL, C) → (C, NP, PL)
        return (seg.reshape(n_p, patch_len, C)
                   .transpose(2, 0, 1)
                   .astype(np.float32))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        pid, seg, tfeat, label, first_ns, bnd, subj_idx = self.entries[idx]
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
            torch.from_numpy(seg.astype(np.float32)),   # raw segment for recon
            torch.tensor(bnd,      dtype=torch.float32),
            torch.tensor(subj_idx, dtype=torch.long),
        )


def _collate(batch):
    (patches_lists, times, labels, pids,
     first_nss, segs, bnds, subj_idxs) = zip(*batch)
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
        torch.stack(bnds),
        torch.stack(subj_idxs),
    )


def make_loader(pid_list, shuffle=False, is_train=False):
    ds = WindowDataset(pid_list, cfg.PROC_DIR, class_to_idx,
                       train_subject_to_idx, is_train=is_train)
    dl = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=shuffle,
                    num_workers=0, pin_memory=GPU, collate_fn=_collate)
    return ds, dl


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MODEL BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

# ── Normalisation ─────────────────────────────────────────────────────────────
class ZCRMSNorm(nn.Module):
    """Zero-centred RMS normalisation (no running stats, no learnable bias)."""
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.g   = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        x0 = x - x.mean(-1, keepdim=True)
        return x0 / x0.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt() * self.g


# ── Stochastic depth ──────────────────────────────────────────────────────────
class StochasticDepth(nn.Module):
    def __init__(self, layer: nn.Module, survival_prob: float):
        super().__init__()
        self.layer = layer
        self.p     = survival_prob

    def forward(self, x):
        if not self.training or self.p >= 1.0:
            return self.layer(x)
        if random.random() > self.p:
            return x
        return self.layer(x)


# ══════════════════════════════════════════════════════════════════════════════
# C11 — Axis Interaction Block
# ══════════════════════════════════════════════════════════════════════════════
class AxisInteractionBlock(nn.Module):
    """
    Cross-axis self-attention at each patch position.

    Input:  (B, NP, C, D)  — C axes kept separate after per-axis embedding.
    Output: (B, NP, D)     — axes fused by learned linear projection.

    Runs 3×3 attention over the C=3 axis dimension independently for each of
    the NP patch positions (achieved by folding NP into the batch dimension).
    Cost: O(C² · NP · D) = O(9 · NP · D) — negligible vs O(NP²) for the
    full sequence.

    Physical motivation: walking creates phase-shifted correlations across
    axes (vertical leads, AP follows, ML damps).  Collapsing axes before this
    block would destroy the structure we want to model.
    """
    def __init__(self, d: int, n_axes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.qkv  = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        # Fuse C axes → D after interaction
        self.fuse = nn.Linear(n_axes * d, d)

    def forward(self, x):
        # x: (B, NP, C, D)
        B, NP, C, D = x.shape
        h   = self.norm(x.reshape(B * NP, C, D))          # (B·NP, C, D)
        qkv = (self.qkv(h)
               .reshape(B * NP, C, 3, D)
               .permute(2, 0, 1, 3))                       # (3, B·NP, C, D)
        q, k, v = qkv[0], qkv[1], qkv[2]                  # (B·NP, C, D)

        score = (q @ k.transpose(-2, -1)) / math.sqrt(D)  # (B·NP, C, C)
        attn  = self.drop(torch.softmax(score, dim=-1))
        out   = self.proj(attn @ v)                        # (B·NP, C, D)

        # Fuse axes
        out = out.reshape(B, NP, C * D)
        return self.fuse(out)                              # (B, NP, D)


# ══════════════════════════════════════════════════════════════════════════════
# Patch embedders
# ══════════════════════════════════════════════════════════════════════════════
class PerAxisPatchEmbed(nn.Module):
    """Independent per-axis linear projection — preserves axis identity."""
    def __init__(self, patch_len: int, d: int):
        super().__init__()
        self.proj = nn.Linear(patch_len, d)

    def forward(self, patches):
        # patches: (B, C, NP, PL)
        x = patches.permute(0, 2, 1, 3)   # (B, NP, C, PL)
        return self.proj(x)               # (B, NP, C, D)


class DualDomainPerAxisEmbed(nn.Module):
    """Per-axis time+FFT dual-domain embedding (C1 variant of C11 path)."""
    def __init__(self, patch_len: int, d: int):
        super().__init__()
        self.time_proj = nn.Linear(patch_len, d)
        self.freq_proj = nn.Linear(patch_len // 2 + 1, d)
        self.gate_w    = nn.Parameter(torch.zeros(d))

    def forward(self, patches):
        x     = patches.permute(0, 2, 1, 3)               # (B, NP, C, PL)
        t_emb = self.time_proj(x)
        f_emb = self.freq_proj(torch.fft.rfft(x, dim=-1).abs())
        g     = torch.sigmoid(self.gate_w)
        return g * t_emb + (1 - g) * f_emb               # (B, NP, C, D)


class SimplePatchEmbed(nn.Module):
    """Channel-concatenated patch embedding (fallback: C11 off, C1 off)."""
    def __init__(self, patch_len: int, channels: int, d: int):
        super().__init__()
        self.proj = nn.Linear(patch_len * channels, d)

    def forward(self, patches):
        B, C, NP, PL = patches.shape
        x = patches.permute(0, 2, 1, 3).reshape(B, NP, C * PL)
        return self.proj(x)


class DualDomainPatchEmbed(nn.Module):
    """Channel-concatenated dual-domain (C1 on, C11 off)."""
    def __init__(self, patch_len: int, channels: int, d: int):
        super().__init__()
        self.time_proj = nn.Linear(patch_len * channels, d)
        self.freq_proj = nn.Linear((patch_len // 2 + 1) * channels, d)
        self.gate_w    = nn.Parameter(torch.zeros(d))

    def forward(self, patches):
        B, C, NP, PL = patches.shape
        x_t   = patches.permute(0, 2, 1, 3).reshape(B, NP, C * PL)
        t_emb = self.time_proj(x_t)
        # FFT per time axis then concat
        x_f   = patches.permute(0, 2, 3, 1)               # (B, NP, PL, C)
        mag   = torch.fft.rfft(x_f, dim=2).abs()          # (B, NP, PL//2+1, C)
        f_emb = self.freq_proj(mag.reshape(B, NP, -1))
        g     = torch.sigmoid(self.gate_w)
        return g * t_emb + (1 - g) * f_emb


class AxisAwarePatchEmbed(nn.Module):
    """
    Single-scale embedding that keeps axes separate through projection then
    applies cross-axis interaction (C11 path).

    Per-axis projection → (B, NP, C, D)
    AxisInteractionBlock → (B, NP, D)
    """
    def __init__(self, patch_len: int, channels: int, d: int):
        super().__init__()
        if CC.C1_DUAL_DOMAIN_EMBEDDING:
            self.per_axis = DualDomainPerAxisEmbed(patch_len, d)
        else:
            self.per_axis = PerAxisPatchEmbed(patch_len, d)
        self.interact = AxisInteractionBlock(d, n_axes=channels,
                                             dropout=cfg.DROPOUT)

    def forward(self, patches):
        x = self.per_axis(patches)   # (B, NP, C, D)
        return self.interact(x)      # (B, NP, D)


def _make_scale_embedder(patch_len: int, channels: int, d: int) -> nn.Module:
    """Route to the appropriate embedder based on active contributions."""
    if CC.C11_AXIS_INTERACTION:
        return AxisAwarePatchEmbed(patch_len, channels, d)
    if CC.C1_DUAL_DOMAIN_EMBEDDING:
        return DualDomainPatchEmbed(patch_len, channels, d)
    return SimplePatchEmbed(patch_len, channels, d)


class HierarchicalPatchEmbed(nn.Module):
    """
    Multi-scale patch embedding with linear-interpolation scale fusion.

    Replaces repeat_interleave (was lossy — all fine patches in a coarse
    window got the identical coarse embedding) with F.interpolate
    (continuous bilinear upsample preserving within-coarse-window gradation).
    """
    def __init__(self, patch_lens, channels: int, d: int):
        super().__init__()
        assert len(patch_lens) == 3
        self.patch_lens = sorted(patch_lens)
        pl0, pl1, pl2   = self.patch_lens
        self.embed_fine   = _make_scale_embedder(pl0, channels, d)
        self.embed_mid    = _make_scale_embedder(pl1, channels, d)
        self.embed_coarse = _make_scale_embedder(pl2, channels, d)

    def forward(self, patches_list):
        p0, p1, p2 = patches_list
        e0 = self.embed_fine(p0)     # (B, NP_fine, D)
        e1 = self.embed_mid(p1)      # (B, NP_mid, D)
        e2 = self.embed_coarse(p2)   # (B, NP_coarse, D)

        NP = e0.shape[1]
        e1_up = F.interpolate(e1.transpose(1, 2), size=NP,
                              mode='linear', align_corners=False).transpose(1, 2)
        e2_up = F.interpolate(e2.transpose(1, 2), size=NP,
                              mode='linear', align_corners=False).transpose(1, 2)
        return e0 + e1_up + e2_up    # (B, NP_fine, D)


# ── Circadian bias ─────────────────────────────────────────────────────────────
class CircadianBias(nn.Module):
    def __init__(self, n_patches: int, d: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, d), nn.SiLU(), nn.Linear(d, n_patches * d))
        self.np = n_patches
        self.d  = d

    def forward(self, times):
        return self.mlp(times).view(times.shape[0], self.np, self.d)


# ══════════════════════════════════════════════════════════════════════════════
# C14 — Kinematic Delta-Net (correct recurrent delta rule)
# ══════════════════════════════════════════════════════════════════════════════
class KinematicDeltaNet(nn.Module):
    """
    Recurrent associative memory with the correct delta-rule update and a
    frequency-conditioned per-token forget gate.

    State update (sequential over NP=120 patches — cost is negligible):

        h_t    = norm(x_t)
        q_t    = L2(W_q h_t),   k_t = L2(W_k h_t),   v_t = W_v h_t
        γ_t    = freq_gate(FFT_bands(h_t))   ∈ [0.50, 0.99]

        pred_t = S_{t-1} k_t          # memory read
        δ_t    = v_t − pred_t         # error signal (precision term)
        S_t    = γ_t · S_{t-1} + δ_t ⊗ k_t    # forget + write
        o_t    = S_t q_t              # output

    γ_t from embedding FFT energy (patch-index domain):
        slowly varying embedding (sleep)  → low-freq dominant → high γ
        rapidly varying embedding (MVPA)  → high-freq dominant → low γ

    Physical motivation: sleep operates at ~0.01 Hz, walking at ~1 Hz,
    MVPA at 2–5 Hz.  A fixed scalar γ cannot be optimal across all classes.
    """
    def __init__(self, d: int, n_freq_bands: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm      = ZCRMSNorm(d)
        self.q_lin     = nn.Linear(d, d, bias=False)
        self.k_lin     = nn.Linear(d, d, bias=False)
        self.v_lin     = nn.Linear(d, d)
        self.out_proj  = nn.Linear(d, d)
        self.post_norm = ZCRMSNorm(d)
        self.drop      = nn.Dropout(dropout)
        self.n_bands   = n_freq_bands

        self.freq_gate = nn.Sequential(
            nn.Linear(n_freq_bands, 16), nn.SiLU(),
            nn.Linear(16, 1), nn.Sigmoid(),
        )
        # bias → sigmoid ≈ 0.62 → γ ≈ 0.80 at init (conservative; avoids early
        # state explosion — gamma rises naturally as training stabilises)
        nn.init.constant_(self.freq_gate[2].bias, 0.5)

    # Maximum element-wise magnitude for the recurrent state.
    # Bounds ‖S‖_F ≤ S_CLIP * D regardless of γ or sequence length.
    S_CLIP   = 10.0
    # Clamp on the error signal δ before writing the outer product into S.
    # Prevents a single outlier token from spiking the entire memory.
    DELTA_CLIP = 5.0

    @staticmethod
    def _l2(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return x / x.pow(2).sum(-1, keepdim=True).add(eps).sqrt()

    def _gamma(self, h: torch.Tensor) -> torch.Tensor:
        """
        Frequency-conditioned forget rate from the embedding sequence FFT.
        h: (B, N, D) float32  →  gamma: (B, N, 1) ∈ [0.50, 0.99]

        Note: h is already float32 (caller casts before calling _gamma).
        rfft on float16 is undefined on some CUDA kernels; always float32 here.
        """
        fft   = torch.fft.rfft(h.detach().float(), dim=-1).abs()  # (B, N, D//2+1)
        n_f   = fft.shape[-1]
        bs    = max(1, n_f // self.n_bands)
        bands = []
        for i in range(self.n_bands):
            s = i * bs
            e = min(s + bs, n_f)
            bands.append(fft[:, :, s:e].mean(-1))
        energy = torch.stack(bands, dim=-1)                # (B, N, n_bands)
        energy = energy / energy.sum(-1, keepdim=True).clamp(min=1e-8)
        raw    = self.freq_gate(energy)                    # (B, N, 1)
        return 0.50 + 0.49 * raw                          # map to [0.50, 0.99]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Force the entire recurrent loop to float32 regardless of AMP context.

        Why: The state S ∈ ℝ^{B×D×D} accumulates 120 outer products.
        With γ ≈ 0.99, steady-state ‖S‖ ≈ ‖δ‖/(1−γ) ≈ 100·‖v‖.
        During early training S is far from equilibrium; float16 overflows
        (max ≈ 65504) and produces NaN in the first few batches.
        Running in float32 adds negligible overhead for D=64, N=120.
        """
        orig_dtype = x.dtype
        x = x.float()           # ← promote once; return to orig_dtype at end

        B, N, D = x.shape
        h = self.norm(x)
        q = self._l2(self.q_lin(h))    # (B, N, D) float32, unit-norm keys/queries
        k = self._l2(self.k_lin(h))
        v = self.v_lin(h)              # (B, N, D) float32, unconstrained values
        gamma = self._gamma(h)         # (B, N, 1) float32

        # State initialised to zero each forward pass (non-persistent across
        # windows — each 30-second segment is independent).
        S       = x.new_zeros(B, D, D)   # float32
        outputs = []
        for t in range(N):
            q_t = q[:, t]       # (B, D)
            k_t = k[:, t]
            v_t = v[:, t]
            g_t = gamma[:, t]   # (B, 1)

            pred  = torch.einsum('bde,be->bd', S, k_t)         # memory read
            delta = (v_t - pred).clamp(-self.DELTA_CLIP,        # error signal
                                        self.DELTA_CLIP)        # clamp before write
            S = (g_t.unsqueeze(-1) * S                          # forget
                 + torch.einsum('bd,be->bde', delta, k_t))      # write
            S = S.clamp(-self.S_CLIP, self.S_CLIP)              # prevent explosion
            outputs.append(torch.einsum('bde,be->bd', S, q_t))  # read output

        out = torch.stack(outputs, dim=1)       # (B, N, D)
        out = self.out_proj(self.post_norm(out))
        # Return in the original dtype so the rest of the graph is unaffected
        return (x + self.drop(out)).to(orig_dtype)


# ── Standard GatedDeltaNet (fallback when C14 disabled) ───────────────────────
class GatedDeltaNet(nn.Module):
    """
    Gated local-attention block with depthwise-conv projections.
    Kept for ablation (C14=False).  Note: NOT a true recurrent delta rule
    (no persistent memory state S) — renamed for accuracy vs v2.
    """
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.norm   = ZCRMSNorm(d)
        self.q_lin  = nn.Linear(d, d)
        self.k_lin  = nn.Linear(d, d)
        self.v_lin  = nn.Linear(d, d)
        self.q_conv = nn.Conv1d(d, d, 3, padding=1, groups=d)
        self.k_conv = nn.Conv1d(d, d, 3, padding=1, groups=d)
        self.v_conv = nn.Conv1d(d, d, 3, padding=1, groups=d)
        self.act    = nn.Sigmoid()
        self.alpha  = nn.Linear(d, d)
        self.beta   = nn.Linear(d, d)
        self.post_n = ZCRMSNorm(d)
        self.post   = nn.Linear(d, d)
        self.gate   = nn.Sigmoid()
        self.silu   = nn.SiLU()
        self.drop   = nn.Dropout(dropout)

    @staticmethod
    def _l2(x, eps=1e-8):
        return x / x.pow(2).sum(-1, keepdim=True).add(eps).sqrt()

    def forward(self, x):
        h = self.norm(x)
        q = self.act(self.q_conv(self.q_lin(h).transpose(1, 2)).transpose(1, 2))
        k = self.act(self.k_conv(self.k_lin(h).transpose(1, 2)).transpose(1, 2))
        v = self.act(self.v_conv(self.v_lin(h).transpose(1, 2)).transpose(1, 2))
        q, k  = self._l2(q), self._l2(k)
        delta = q * (k * v)
        delta = torch.tanh(self.alpha(x)) * delta + self.beta(x)
        dhat  = self.post(self.post_n(delta))
        return x + self.drop(self.gate(self.silu(dhat)) * dhat)


def _make_delta_layer(d: int, dropout: float) -> nn.Module:
    if CC.C14_KINEMATIC_MEMORY:
        return KinematicDeltaNet(d, n_freq_bands=cfg.N_FREQ_BANDS,
                                 dropout=dropout)
    return GatedDeltaNet(d, dropout=dropout)


# ── Soft-Routed MoE (renamed from SoftMoE — not the Puigcerver slot version) ─
class SoftRoutedMoE(nn.Module):
    def __init__(self, d: int, hidden: int, n_experts: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.router  = nn.Linear(d, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden), nn.SiLU(),
                          nn.Dropout(dropout),  nn.Linear(hidden, d))
            for _ in range(n_experts)
        ])
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        w = torch.softmax(self.router(x), dim=-1)          # (B, N, E)
        s = torch.stack([e(x) for e in self.experts], -2)  # (B, N, E, D)
        return self.drop((w.unsqueeze(-1) * s).sum(-2))


# ── RoPE ──────────────────────────────────────────────────────────────────────
def precompute_freqs(dim: int, n_tok: int, theta: float = 10000.0):
    assert dim % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t     = torch.arange(n_tok)
    return torch.polar(torch.ones(n_tok, dim // 2), torch.outer(t, freqs))


def apply_rope(q, k, freqs):
    B, H, N, D = q.shape
    d2 = D // 2
    f  = freqs[:N].to(q.device).view(1, 1, N, d2)
    q_ = torch.view_as_complex(q.float().contiguous().view(B, H, N, d2, 2))
    k_ = torch.view_as_complex(k.float().contiguous().view(B, H, N, d2, 2))
    qo = torch.view_as_real(q_ * f).view(B, H, N, D)
    ko = torch.view_as_real(k_ * f).view(B, H, N, D)
    return qo.type_as(q), ko.type_as(k)


# ── Gated Attention (with optional boundary bias for C13) ─────────────────────
class GatedAttention(nn.Module):
    """
    Gated multi-head self-attention with RoPE.

    When C13 is active, boundary_bias (B, NP) is added as a *column* bias to
    the attention score matrix.  A column bias amplifies attention TO
    high-boundary patches from all positions — the model attends more to
    transition patches regardless of where the query comes from.
    """
    def __init__(self, d: int, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        assert d % n_heads == 0
        assert (d // n_heads) % 2 == 0
        self.h    = n_heads
        self.dh   = d // n_heads
        self.norm = ZCRMSNorm(d)
        self.qkv  = nn.Linear(d, 3 * d)
        self.out  = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, freqs, boundary_bias=None):
        h = self.norm(x)
        B, N, D = h.shape
        # qkv: (B, N, 3*D) → (B, N, 3, h, dh) → (B, 3, N, h, dh)
        qkv = (self.qkv(h)
               .reshape(B, N, 3, self.h, self.dh)
               .permute(0, 2, 1, 3, 4))
        q = qkv[:, 0].transpose(1, 2)   # (B, h, N, dh)
        k = qkv[:, 1].transpose(1, 2)
        v = qkv[:, 2].transpose(1, 2)
        q, k  = apply_rope(q, k, freqs)
        score = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)  # (B, h, N, N)

        if boundary_bias is not None:
            # (B, N) → (B, 1, 1, N): all positions attend more to boundary patches
            score = score + boundary_bias.unsqueeze(1).unsqueeze(2)

        attn = self.drop(torch.softmax(score, dim=-1))
        y    = self.out(
            (attn @ v).transpose(1, 2).contiguous().reshape(B, N, D))
        return x + torch.sigmoid(self.gate(h)) * y


# ══════════════════════════════════════════════════════════════════════════════
# C12 — Gradient Reversal + Subject Classifier
# ══════════════════════════════════════════════════════════════════════════════
class _GRLFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambda_ * grad, None


class GradientReversal(nn.Module):
    """
    Gradient Reversal Layer (Ganin & Lempitsky 2015).
    Forward: identity.  Backward: negate and scale gradient by lambda_.
    """
    def forward(self, x, lambda_=1.0):
        return _GRLFn.apply(x, lambda_)


class SubjectClassifier(nn.Module):
    """Adversarial subject-identity classifier placed on GRL(z)."""
    def __init__(self, d: int, n_subjects: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d, d // 2), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d // 2, n_subjects),
        )

    def forward(self, z):
        return self.head(z)


# ══════════════════════════════════════════════════════════════════════════════
# C13 — Boundary Detector + Focal BCE
# ══════════════════════════════════════════════════════════════════════════════
class BoundaryDetector(nn.Module):
    """
    Lightweight 1D CNN on the token sequence.

    Returns:
      win_logit     (B,)   — window-level boundary RAW LOGIT (for focal loss)
      patch_logits  (B, NP) — per-patch logits (used as attention column bias)

    IMPORTANT: returns the raw logit, NOT a sigmoid probability.
    FocalBCELoss uses binary_cross_entropy_with_logits which is AMP-safe.
    Using F.binary_cross_entropy on a float16 sigmoid output raises a
    RuntimeError under torch.autocast / AMP on CUDA.

    Window-level logit via max-pool of patch logits (MIL-style): the window
    is a boundary window if ANY patch has a high boundary score.
    """
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(d,      d // 2, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(d // 2, d // 4, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(d // 4, 1,      kernel_size=1),
        )

    def forward(self, x):
        # x: (B, NP, D)
        h            = x.transpose(1, 2)               # (B, D, NP)
        patch_logits = self.cnn(h).squeeze(1)          # (B, NP) — raw logits
        win_logit    = patch_logits.max(dim=-1).values # (B,)    — raw logit
        return win_logit, patch_logits


class FocalBCELoss(nn.Module):
    """
    Focal binary cross-entropy for rare-class boundary detection.

    Accepts RAW LOGITS (not sigmoid probabilities).
    Uses F.binary_cross_entropy_with_logits which is:
      (a) numerically stable (log-sum-exp trick in CUDA kernel)
      (b) AMP-safe — F.binary_cross_entropy on float16 sigmoid outputs
          raises RuntimeError under torch.autocast on CUDA.
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logit, target):
        # logit:  (B,) raw logit — NOT sigmoid-activated
        # target: (B,) ∈ {0.0, 1.0}
        bce = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
        # sigmoid in float32 for stable focal weight computation
        p       = torch.sigmoid(logit.float())
        pt      = torch.where(target == 1, p, 1 - p)
        alpha_t = torch.where(target == 1,
                              logit.new_full(logit.shape, self.alpha),
                              logit.new_full(logit.shape, 1 - self.alpha))
        return (alpha_t * (1 - pt).pow(self.gamma) * bce).mean()


# ── Reconstruction head ────────────────────────────────────────────────────────
class ReconHead(nn.Module):
    def __init__(self, d: int, n_patches: int, patch_len: int, channels: int):
        super().__init__()
        out = n_patches * patch_len * channels
        self.mlp = nn.Sequential(
            nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, out))

    def forward(self, z):
        return self.mlp(z)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN MODEL: PatchHARv3
# ══════════════════════════════════════════════════════════════════════════════
class PatchHARv3(nn.Module):
    def __init__(self):
        super().__init__()
        d  = cfg.D_MODEL
        NP = cfg.N_PATCHES

        # ── patch embedding ──────────────────────────────────────────────────
        if CC.C4_MULTISCALE_PATCHING:
            self.hier_embed = HierarchicalPatchEmbed(
                cfg.PATCH_LENS_MULTI, cfg.CHANNELS, d)
            self.patch_lens = cfg.PATCH_LENS_MULTI
        else:
            self.hier_embed   = None
            self.patch_lens   = [cfg.PATCH_LEN]
            self.single_embed = _make_scale_embedder(cfg.PATCH_LEN,
                                                     cfg.CHANNELS, d)

        # ── temporal / positional bias ────────────────────────────────────────
        if CC.C3_CIRCADIAN_BIAS:
            self.circ_bias = CircadianBias(NP, d)
        else:
            self.time_emb = nn.Sequential(
                nn.Linear(5, d), nn.ReLU(), nn.Dropout(0.1))

        self.input_norm = nn.LayerNorm(d)

        # ── backbone: kinematic (C14) or gated delta-net stack ────────────────
        raw_layers = [_make_delta_layer(d, cfg.DROPOUT) for _ in range(cfg.N_LAYERS)]
        if CC.C8_STOCHASTIC_DEPTH:
            survival = [1.0 - (i / cfg.N_LAYERS) * cfg.SD_DROP_MAX
                        for i in range(cfg.N_LAYERS)]
            self.delta_layers = nn.ModuleList([
                StochasticDepth(l, p) for l, p in zip(raw_layers, survival)
            ])
        else:
            self.delta_layers = nn.ModuleList(raw_layers)

        # ── MoE → (boundary-gated) attention → MoE ───────────────────────────
        self.moe1 = SoftRoutedMoE(d, 2*d, n_experts=cfg.N_EXPERTS,
                                   dropout=cfg.DROPOUT)
        self.attn = GatedAttention(d, n_heads=cfg.N_HEADS,
                                    dropout=cfg.DROPOUT)
        self.moe2 = SoftRoutedMoE(d, 2*d, n_experts=cfg.N_EXPERTS,
                                   dropout=cfg.DROPOUT)

        freqs = precompute_freqs(d // cfg.N_HEADS, NP)
        self.register_buffer("freqs", freqs)

        # ── C6: learned temperature ───────────────────────────────────────────
        if CC.C6_LABEL_SMOOTH_TEMP:
            self.log_tau = nn.Parameter(torch.zeros(1))

        # ── C7: EMA prototype memory ──────────────────────────────────────────
        if CC.C7_PROTOTYPE_MEMORY:
            self.register_buffer("prototypes", torch.zeros(NUM_CLASSES, d))
            self.proto_filled = False

        # ── C10: reconstruction head ──────────────────────────────────────────
        if CC.C10_RECON_AUX_GRAD_SURGERY:
            self.recon_head = ReconHead(d, NP, cfg.PATCH_LEN, cfg.CHANNELS)

        # ── C12: adversarial subject disentanglement ──────────────────────────
        if CC.C12_ADVERSARIAL_DISENTANGLE:
            self.grl                = GradientReversal()
            self.subject_classifier = SubjectClassifier(d, N_TRAIN_SUBJECTS)

        # ── C13: boundary detector ────────────────────────────────────────────
        if CC.C13_BOUNDARY_DETECTION:
            self.boundary_detector = BoundaryDetector(d, dropout=cfg.DROPOUT)
            self.boundary_loss_fn  = FocalBCELoss(cfg.FOCAL_ALPHA,
                                                   cfg.FOCAL_GAMMA)

        # ── activity classification head ──────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(d, d // 2), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d // 2, NUM_CLASSES),
        )

    # ── embedding helper (manifold-mixup path uses this) ─────────────────────
    def _embed_patches(self, patches_list):
        if CC.C4_MULTISCALE_PATCHING:
            return self.hier_embed(patches_list)
        return self.single_embed(patches_list[0])

    # ── main forward ──────────────────────────────────────────────────────────
    def forward(self, patches_list, times, adv_lambda=0.0,
                return_embedding=False):
        """
        Returns (logits, recon, subj_logits, boundary_prob).
        If return_embedding=True, returns z (B, D) for probing / prototypes.
        """
        x = self._embed_patches(patches_list)          # (B, NP, D)

        if CC.C3_CIRCADIAN_BIAS:
            x = x + self.circ_bias(times)
        else:
            x = x + self.time_emb(times).unsqueeze(1)

        x = self.input_norm(x)

        for layer in self.delta_layers:
            x = layer(x)

        x = x + self.moe1(x)

        # C13: run boundary detector to get attention column bias
        boundary_prob, b_logits = None, None
        if CC.C13_BOUNDARY_DETECTION:
            boundary_prob, b_logits = self.boundary_detector(x)

        x = self.attn(x, self.freqs,
                      boundary_bias=b_logits if CC.C13_BOUNDARY_DETECTION
                      else None)

        x = x + self.moe2(x)

        z = x.mean(dim=1)   # (B, D)

        if return_embedding:
            return z

        # C10: reconstruction (detach z so recon loss does not back-prop through encoder)
        recon = None
        if CC.C10_RECON_AUX_GRAD_SURGERY and self.training:
            recon = self.recon_head(z)

        # C12: adversarial subject head (training only, λ>0 only)
        subj_logits = None
        if CC.C12_ADVERSARIAL_DISENTANGLE and self.training and adv_lambda > 0:
            z_rev       = self.grl(z, adv_lambda)
            subj_logits = self.subject_classifier(z_rev)

        logits = self.head(z)

        if CC.C6_LABEL_SMOOTH_TEMP:
            tau    = torch.exp(self.log_tau).clamp(0.5, 2.0)
            logits = logits / tau

        if CC.C7_PROTOTYPE_MEMORY and not self.training and self.proto_filled:
            z_n    = F.normalize(z, dim=-1)
            pr_n   = F.normalize(self.prototypes, dim=-1)
            cosine = z_n @ pr_n.T
            logits = (1 - cfg.PROTO_ALPHA) * logits + cfg.PROTO_ALPHA * cosine

        return logits, recon, subj_logits, boundary_prob

    @torch.no_grad()
    def update_prototypes(self, embeddings, labels):
        m = cfg.PROTO_MOMENTUM
        for k in range(NUM_CLASSES):
            mask = (labels == k)
            if mask.sum() == 0:
                continue
            mean = embeddings[mask].mean(0)
            if self.proto_filled:
                self.prototypes[k] = m * self.prototypes[k] + (1 - m) * mean
            else:
                self.prototypes[k] = mean
        self.proto_filled = True


# ══════════════════════════════════════════════════════════════════════════════
# 7.  LOSS UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def compute_class_weights(ds: WindowDataset) -> torch.Tensor:
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for e in ds.entries:
        counts[e[3]] += 1
    w = np.clip(counts.max() / np.clip(counts, 1, None), 1.0, 10.0)
    w = torch.tensor(w, dtype=torch.float32)
    return w / w.sum() * NUM_CLASSES


def manifold_mixup(x, labels, alpha=0.2):
    if alpha <= 0:
        return x, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], labels, labels[idx], lam


def raw_mixup(patches_list, times, labels, alpha=0.2):
    if alpha <= 0:
        return patches_list, times, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(labels.size(0), device=labels.device)
    mixed = [lam * p + (1 - lam) * p[idx] for p in patches_list]
    return mixed, lam * times + (1 - lam) * times[idx], labels, labels[idx], lam


def recon_loss(recon, raw_segs):
    B, T, C = raw_segs.shape
    NP, PL  = cfg.N_PATCHES, cfg.PATCH_LEN
    target  = (raw_segs[:, :NP * PL, :]
               .reshape(B, NP, PL, C)
               .permute(0, 1, 3, 2)
               .reshape(B, NP * C * PL))
    return F.mse_loss(recon, target.detach())


class SmoothCE(nn.Module):
    def __init__(self, weight=None, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing
        self.weight    = weight

    def forward(self, logits, labels):
        K   = logits.size(-1)
        eps = self.smoothing
        with torch.no_grad():
            soft = torch.full_like(logits, eps / (K - 1))
            soft.scatter_(-1, labels.unsqueeze(-1), 1.0 - eps)
        log_p = F.log_softmax(logits, dim=-1)
        loss  = -(soft * log_p).sum(-1)
        if self.weight is not None:
            w    = self.weight.to(logits.device)
            loss = loss * w[labels]
        return loss.mean()


# ── Metrics ───────────────────────────────────────────────────────────────────
def kappa(yt, yp):
    cm = confusion_matrix(yt, yp)
    n  = cm.sum()
    if n == 0:
        return 0.0
    po = np.trace(cm) / n
    pe = np.dot(cm.sum(1), cm.sum(0)) / (n * n)
    return float((po - pe) / (1 - pe)) if abs(1 - pe) > 1e-12 else 0.0


def mcc(yt, yp):
    cm  = confusion_matrix(yt, yp).astype(float)
    n   = cm.sum()
    if n == 0:
        return 0.0
    s   = np.trace(cm)
    t   = cm.sum(1)
    p   = cm.sum(0)
    num = s * n - np.sum(t * p)
    den = math.sqrt(
        max(n**2 - np.sum(t**2), 0.0) * max(n**2 - np.sum(p**2), 0.0))
    return float(num / den) if den > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 8.  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
_DROP_KEYS = {"scale_fusion.weight", "scale_fusion.bias"}
_REMAP     = {
    "patch_embeds.0.": "hier_embed.embed_fine.",
    "patch_embeds.1.": "hier_embed.embed_mid.",
    "patch_embeds.2.": "hier_embed.embed_coarse.",
}


def _remap_checkpoint(ckpt, model):
    out = {}
    for k, v in ckpt.items():
        if k in _DROP_KEYS:
            continue
        nk = k
        for old, new in _REMAP.items():
            if k.startswith(old):
                nk = new + k[len(old):]
                break
        out[nk] = v
    mk = set(model.state_dict().keys())
    ck = set(out.keys())
    if mk == ck:
        return out
    print(f"  [Remap] missing={len(mk-ck)} unexpected={len(ck-mk)}")
    return None


def _compat_load(save_path: Path, model: nn.Module) -> bool:
    if not save_path.exists():
        print(f"  [Warn] No checkpoint at {save_path}")
        return False
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    try:
        model.load_state_dict(ckpt, strict=True)
        print(f"  Checkpoint loaded from {save_path}")
        return True
    except RuntimeError:
        pass
    print("  [Compat] Direct load failed — attempting key remapping …")
    remapped = _remap_checkpoint(ckpt, model)
    if remapped is not None:
        try:
            model.load_state_dict(remapped, strict=True)
            torch.save(model.state_dict(), save_path)
            print("  [Compat] Remapped and re-saved.")
            return True
        except RuntimeError as e:
            print(f"  [Compat] Still failed: {e}")
    print("  [Warn] Could not load checkpoint.")
    return False


# ══════════════════════════════════════════════════════════════════════════════
# 9.  TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def train(model: PatchHARv3, train_dl: DataLoader, val_dl: DataLoader,
          class_w: torch.Tensor, save_path: Path):

    _compat_load(save_path, model)   # warm-start if checkpoint exists

    optimizer = optim.AdamW(model.parameters(),
                             lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.LR,
        steps_per_epoch=len(train_dl), epochs=cfg.EPOCHS,
        pct_start=0.1, anneal_strategy='cos')

    smooth_eps = cfg.LABEL_SMOOTH_EPS if CC.C6_LABEL_SMOOTH_TEMP else 0.0
    criterion  = SmoothCE(weight=class_w.to(device), smoothing=smooth_eps)

    # AMP scaler
    if GPU:
        try:
            scaler = torch.amp.GradScaler(device="cuda")
        except Exception:
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
    else:
        scaler = None

    def _scale_backward(loss):
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    def _step():
        if scaler is not None:
            scaler.unscale_(optimizer)
            ok = (all(p.grad is None or torch.isfinite(p.grad).all()
                      for p in model.parameters())
                  and torch.isfinite(loss))
            if ok:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
                scaler.step(optimizer)
            else:
                optimizer.zero_grad(set_to_none=True)
            scaler.update()
        else:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
            optimizer.step()

    history      = []
    best_score   = -1.0
    patience_ctr = 0

    for epoch in range(cfg.EPOCHS):
        model.train()
        total_loss = 0.0

        # C12 — DANN λ ramp: sigmoid schedule p ∈ [0,1]
        p          = epoch / max(cfg.EPOCHS - 1, 1)
        adv_lambda = (cfg.ADV_LAMBDA_MAX
                      * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0))

        for batch in train_dl:
            (patches_list, times, labels, _pids, _first_nss,
             raw_segs, boundary_labels, subject_idx) = batch

            patches_list    = [p_.to(device).float() for p_ in patches_list]
            times           = times.to(device).float()
            labels          = labels.to(device)
            raw_segs        = raw_segs.to(device).float()
            boundary_labels = boundary_labels.to(device).float()
            subject_idx     = subject_idx.to(device).long()

            # ── C9: Manifold Mixup path ──────────────────────────────────────
            if CC.C9_MANIFOLD_MIXUP:
                # Walk backbone manually to obtain z_pool before head
                with amp_ctx():
                    xb = model._embed_patches(patches_list)
                    if CC.C3_CIRCADIAN_BIAS:
                        xb = xb + model.circ_bias(times)
                    else:
                        xb = xb + model.time_emb(times).unsqueeze(1)
                    xb = model.input_norm(xb)

                    for layer in model.delta_layers:
                        xb = layer(xb)
                    xb = xb + model.moe1(xb)

                    b_prob, b_logits = None, None
                    if CC.C13_BOUNDARY_DETECTION:
                        b_prob, b_logits = model.boundary_detector(xb)

                    xb = model.attn(xb, model.freqs,
                                    boundary_bias=b_logits
                                    if CC.C13_BOUNDARY_DETECTION else None)
                    xb    = xb + model.moe2(xb)
                    z_pool = xb.mean(1)

                z_mix, la, lb, lam = manifold_mixup(z_pool, labels,
                                                     cfg.MIXUP_ALPHA)
                optimizer.zero_grad(set_to_none=True)
                with amp_ctx():
                    logits = model.head(z_mix)
                    if CC.C6_LABEL_SMOOTH_TEMP:
                        tau    = torch.exp(model.log_tau).clamp(0.5, 2.0)
                        logits = logits / tau

                    loss = (lam * criterion(logits, la)
                            + (1 - lam) * criterion(logits, lb))

                    if CC.C13_BOUNDARY_DETECTION and b_prob is not None:
                        bl   = model.boundary_loss_fn(b_prob, boundary_labels)
                        loss = loss + cfg.BOUNDARY_LAMBDA * bl

                    if CC.C12_ADVERSARIAL_DISENTANGLE and adv_lambda > 0:
                        z_rev = model.grl(z_pool, adv_lambda)
                        sl    = model.subject_classifier(z_rev)
                        valid = subject_idx >= 0
                        if valid.any():
                            al   = F.cross_entropy(sl[valid], subject_idx[valid])
                            loss = loss + cfg.ADV_SUBJECT_WEIGHT * al

                    if CC.C10_RECON_AUX_GRAD_SURGERY:
                        recon = model.recon_head(z_pool.detach())
                        loss  = loss + cfg.RECON_LAMBDA * recon_loss(
                            recon, raw_segs)

            # ── Standard (raw) mixup path ─────────────────────────────────────
            else:
                (patches_list, times,
                 la, lb, lam) = raw_mixup(patches_list, times, labels,
                                          cfg.MIXUP_ALPHA)
                optimizer.zero_grad(set_to_none=True)
                with amp_ctx():
                    logits, recon, subj_logits, b_prob = model(
                        patches_list, times, adv_lambda=adv_lambda)

                    loss = (lam * criterion(logits, la)
                            + (1 - lam) * criterion(logits, lb))

                    if CC.C13_BOUNDARY_DETECTION and b_prob is not None:
                        bl   = model.boundary_loss_fn(b_prob, boundary_labels)
                        loss = loss + cfg.BOUNDARY_LAMBDA * bl

                    if (CC.C12_ADVERSARIAL_DISENTANGLE
                            and subj_logits is not None):
                        valid = subject_idx >= 0
                        if valid.any():
                            al   = F.cross_entropy(subj_logits[valid],
                                                   subject_idx[valid])
                            loss = loss + cfg.ADV_SUBJECT_WEIGHT * al

                    if (CC.C10_RECON_AUX_GRAD_SURGERY and recon is not None):
                        loss = loss + cfg.RECON_LAMBDA * recon_loss(
                            recon, raw_segs)

            _scale_backward(loss)
            _step()
            scheduler.step()

            if torch.isfinite(loss):
                total_loss += float(loss.item())

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        vp, vt           = [], []
        embs_val, labs_val = [], []

        with torch.no_grad():
            for vbatch in val_dl:
                vpl, vtimes, vlabels = vbatch[0], vbatch[1], vbatch[2]
                vpl     = [p_.to(device).float() for p_ in vpl]
                vtimes  = vtimes.to(device).float()

                if CC.C7_PROTOTYPE_MEMORY:
                    z        = model.forward(vpl, vtimes, return_embedding=True)
                    vlogits, *_ = model(vpl, vtimes)
                    embs_val.append(z.cpu())
                    labs_val.append(vlabels)
                else:
                    vlogits, *_ = model(vpl, vtimes)

                vp.extend(vlogits.argmax(1).cpu().numpy())
                vt.extend(vlabels.numpy())

        if CC.C7_PROTOTYPE_MEMORY and embs_val:
            model.update_prototypes(
                torch.cat(embs_val).to(device),
                torch.cat(labs_val).to(device))

        vp  = np.array(vp)
        vt  = np.array(vt)
        f1  = float(f1_score(vt, vp, average="macro", zero_division=0))
        kap = kappa(vt, vp)
        avg = total_loss / max(1, len(train_dl))
        lr  = optimizer.param_groups[0]["lr"]

        print(f"  Epoch {epoch+1:02d}/{cfg.EPOCHS} | "
              f"lr={lr:.2e} | loss={avg:.4f} | "
              f"val_F1={f1:.4f} | κ={kap:.4f} | λ_adv={adv_lambda:.3f}")

        row = dict(epoch=epoch+1, loss=round(avg, 5),
                   val_f1=round(f1, 5), val_kappa=round(kap, 5),
                   adv_lambda=round(adv_lambda, 4), lr=round(lr, 8))
        history.append(row)

        score = f1 + kap
        if score > best_score + 1e-6:
            best_score   = score
            patience_ctr = 0
            torch.save(model.state_dict(), save_path)
            print(f"    → checkpoint saved (score={score:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.EARLY_STOP_PATIENCE:
                print(f"  Early stop at epoch {epoch+1}")
                break

    _compat_load(save_path, model)
    return history


# ══════════════════════════════════════════════════════════════════════════════
# 10.  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model: PatchHARv3, test_dl: DataLoader) -> dict:
    model.eval()
    all_pred, all_true = [], []
    for batch in test_dl:
        pl, times, labels = batch[0], batch[1], batch[2]
        pl    = [p_.to(device).float() for p_ in pl]
        logits, *_ = model(pl, times.to(device).float())
        all_pred.extend(logits.argmax(1).cpu().numpy())
        all_true.extend(labels.numpy())

    yt = np.array(all_true)
    yp = np.array(all_pred)
    metrics = dict(
        macro_f1=round(float(f1_score(yt, yp, average="macro",
                                      zero_division=0)), 4),
        kappa=round(kappa(yt, yp), 4),
        mcc=round(mcc(yt, yp), 4),
        accuracy=round(float((yt == yp).mean()), 4),
    )
    print(f"\n  Macro-F1={metrics['macro_f1']:.4f} | "
          f"κ={metrics['kappa']:.4f} | "
          f"MCC={metrics['mcc']:.4f} | "
          f"Acc={metrics['accuracy']:.4f}")
    print(classification_report(yt, yp, target_names=CLASSES, zero_division=0))
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 11.  ENTANGLEMENT PROBE (C12 diagnostic)
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_entanglement_probe(model: PatchHARv3,
                           train_dl: DataLoader) -> dict:
    """
    Probe whether the learned embedding z encodes subject identity.

    Procedure:
      1. Extract z from all training windows using the frozen model.
      2. 80/20 split within training subjects.
      3. Fit logistic regression to predict subject_idx.
      4. Report accuracy vs chance (1/N_TRAIN_SUBJECTS).

    Verdict: ENTANGLED   if probe_acc > 3 × chance
             DISENTANGLED otherwise

    This is a diagnostic — run it after training to verify C12 worked.
    High probe accuracy before C12, lower probe accuracy after C12, is the
    strongest evidence the GRL is suppressing subject identity in z.
    """
    model.eval()
    embs, subj_ids = [], []

    for batch in train_dl:
        pl, times, _, _, _, _, _, subject_idx = batch
        valid = subject_idx >= 0
        if not valid.any():
            continue
        pl_d = [p_[valid].to(device).float() for p_ in pl]
        td   = times[valid].to(device).float()
        z    = model.forward(pl_d, td, return_embedding=True)
        embs.append(z.cpu().numpy())
        subj_ids.append(subject_idx[valid].numpy())

    if not embs:
        print("  [Probe] No valid training embeddings found.")
        return {}

    X = np.concatenate(embs)
    y = np.concatenate(subj_ids)

    idx   = np.random.RandomState(0).permutation(len(y))
    split = int(0.8 * len(y))
    X_tr, X_te = X[idx[:split]], X[idx[split:]]
    y_tr, y_te = y[idx[:split]], y[idx[split:]]

    sc   = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    probe = LogisticRegression(max_iter=500, C=1.0, n_jobs=-1, random_state=0)
    probe.fit(X_tr, y_tr)
    acc    = float(probe.score(X_te, y_te))
    chance = 1.0 / N_TRAIN_SUBJECTS

    print(f"\n  ── Entanglement Probe ────────────────────────────")
    print(f"    Probe accuracy : {acc:.4f}")
    print(f"    Chance level   : {chance:.4f}  (1 / {N_TRAIN_SUBJECTS})")
    print(f"    Ratio          : {acc / max(chance, 1e-8):.1f}×")
    verdict = "ENTANGLED" if acc > 3 * chance else "DISENTANGLED"
    print(f"    Verdict        : {verdict}")
    print(f"  ──────────────────────────────────────────────────")

    return dict(probe_acc=acc, chance=chance,
                ratio=acc / max(chance, 1e-8), verdict=verdict)


# ══════════════════════════════════════════════════════════════════════════════
# 12.  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    banner = "═" * 70
    print(f"\n{banner}")
    print("  PatchHAR v3")
    active = [k for k, v in vars(CC).items()
              if k.startswith("C") and not k.startswith("__") and v]
    print(f"  Active contributions: {', '.join(active)}")
    print(banner)

    # ── data loaders ─────────────────────────────────────────────────────────
    print("\n── Loading data ──")
    train_ds, train_dl = make_loader(train_pids, shuffle=True,  is_train=True)
    val_ds,   val_dl   = make_loader(val_pids,   shuffle=False, is_train=False)
    test_ds,  test_dl  = make_loader(test_pids,  shuffle=False, is_train=False)

    if len(train_ds) == 0:
        raise RuntimeError("Training dataset is empty.")

    # ── model ─────────────────────────────────────────────────────────────────
    print("\n── Building model ──")
    model   = PatchHARv3().to(device)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_param:,}")

    save_path = cfg.OUTPUT_DIR / "weights_patchhar_v3.pth"
    class_w   = compute_class_weights(train_ds)
    print(f"  Class weights: {class_w.round(decimals=3).tolist()}")

    # ── training ──────────────────────────────────────────────────────────────
    print("\n── Training ──")
    history = train(model, train_dl, val_dl, class_w, save_path)

    # ── evaluation ────────────────────────────────────────────────────────────
    print("\n── Evaluation ──")
    metrics = evaluate(model, test_dl)

    # ── entanglement probe ────────────────────────────────────────────────────
    probe_results = {}
    if CC.C12_ADVERSARIAL_DISENTANGLE:
        print("\n── Entanglement Probe ──")
        probe_results = run_entanglement_probe(model, train_dl)

    # ── save results ──────────────────────────────────────────────────────────
    results = dict(
        metrics=metrics,
        training=history,
        contributions={k: v for k, v in vars(CC).items()
                       if k.startswith("C") and not k.startswith("__")},
        entanglement_probe=probe_results,
        config=dict(
            d_model=cfg.D_MODEL, n_layers=cfg.N_LAYERS,
            n_heads=cfg.N_HEADS, n_experts=cfg.N_EXPERTS,
            patch_lens=cfg.PATCH_LENS_MULTI,
            n_train_subjects=N_TRAIN_SUBJECTS,
            epochs=cfg.EPOCHS, lr=cfg.LR,
        ),
    )
    out_path = cfg.OUTPUT_DIR / "patchhar_v3_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved → {out_path}")
    print(f"  Weights  saved → {save_path}")
    print(banner + "\n")
    return results


if __name__ == "__main__":
    main()
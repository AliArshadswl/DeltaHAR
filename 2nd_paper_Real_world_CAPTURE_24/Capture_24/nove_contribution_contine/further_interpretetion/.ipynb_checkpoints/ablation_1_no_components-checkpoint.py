"""
Ablation 1: ALL custom components DISABLED — pure baseline.

All ContribConfig flags are forced to False before any model code runs.
Equivalent to running the original script with:
    --disable C1_DUAL_DOMAIN_EMBEDDING C3_CIRCADIAN_BIAS C4_MULTISCALE_PATCHING
              C5_FREQ_AUGMENTATION C6_LABEL_SMOOTH_TEMP C7_PROTOTYPE_MEMORY
              C8_STOCHASTIC_DEPTH C9_MANIFOLD_MIXUP C10_RECON_AUX_GRAD_SURGERY

What you get:
  - Single-scale patches (PATCH_LEN=25 only)
  - SimplePatchEmbed (time-domain projection only)
  - Static time embedding via a small MLP → broadcast over patches
  - Plain GatedDeltaNet layers (no stochastic depth)
  - Plain SoftMoE experts
  - No label smoothing, no temperature scaling
  - No prototype memory at inference
  - No frequency augmentation
  - Raw mixup (not manifold mixup)
  - No reconstruction auxiliary loss
"""
from __future__ import annotations
import math, random, json, warnings, sys
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


# ── Force all contributions OFF ──────────────────────────────────────────────
class ContribConfig:
    C1_DUAL_DOMAIN_EMBEDDING   = False
    C3_CIRCADIAN_BIAS          = False
    C4_MULTISCALE_PATCHING     = False
    C5_FREQ_AUGMENTATION       = False
    C6_LABEL_SMOOTH_TEMP       = False
    C7_PROTOTYPE_MEMORY        = False
    C8_STOCHASTIC_DEPTH        = False
    C9_MANIFOLD_MIXUP          = False
    C10_RECON_AUX_GRAD_SURGERY = False

CC = ContribConfig()
print("  [Ablation 1] ALL components DISABLED — pure baseline")


class Config:
    PROC_DIR   = Path("/mnt/share/ali/processed/")
    OUTPUT_DIR = Path("/mnt/share/ali/processed/patchhar_results/ablation1_no_components/")

    TRAIN_N = 80
    VAL_N   = 20

    SIGNAL_RATE = 100
    WINDOW_SIZE = 3000
    PATCH_LEN   = 25
    CHANNELS    = 3
    N_PATCHES   = WINDOW_SIZE // PATCH_LEN

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


def amp_ctx():
    if GPU:
        try:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        except Exception:
            from torch.cuda.amp import autocast
            return autocast()
    return nullcontext()


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
    print(f"Discovered {len(pids)} participants, {len(classes)} classes: {classes}")
    return pids, classes, class_to_idx, idx_to_class


pids_all, CLASSES, class_to_idx, idx_to_class = discover(cfg.PROC_DIR)
NUM_CLASSES = len(CLASSES)

n_train    = min(cfg.TRAIN_N, len(pids_all))
n_val      = min(cfg.VAL_N, max(0, len(pids_all) - n_train))
train_pids = pids_all[:n_train]
val_pids   = pids_all[n_train : n_train + n_val]
test_pids  = pids_all[n_train + n_val :]
print(f"Split  : train={len(train_pids)} | val={len(val_pids)} | test={len(test_pids)}")


def time_features(ns: int) -> np.ndarray:
    ts  = pd.to_datetime(int(ns), unit="ns", utc=True).tz_convert(None)
    out = np.zeros(5, dtype=np.float32)
    out[0] = ts.hour      / 24.0
    out[1] = ts.minute    / 60.0
    out[2] = ts.weekday() / 7.0
    out[3] = float(ts.weekday() >= 5)
    out[4] = float(ts.hour // 6) / 3.0
    return out


class WindowDataset(Dataset):
    """No frequency augmentation (C5 is off)."""
    def __init__(self, pid_list, proc_dir, class_to_idx, is_train=False):
        self.entries  = []
        self.is_train = is_train
        proc_dir      = Path(proc_dir)
        _set          = False
        # Only single patch length (C4 off)
        self.patch_lens = [cfg.PATCH_LEN]

        for pi, pid in enumerate(pid_list):
            path = proc_dir / f"{pid}.npz"
            if not path.exists():
                continue
            npz   = np.load(path, allow_pickle=True)
            W     = npz["X"].astype(np.float32)
            L     = npz["y"].astype(str)
            F     = npz["t"].astype("datetime64[ns]").astype(np.int64)
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
                       else np.pad(normed, ((0, T - normed.shape[0]), (0, 0))))
                self.entries.append((pid, seg, time_features(int(f)),
                                     int(class_to_idx[lab]), int(f)))
            if (pi + 1) % 10 == 0 or (pi + 1) == len(pid_list):
                print(f"  Loaded {pi+1}/{len(pid_list)} — {len(self.entries):,} windows")

    @staticmethod
    def _make_patches(seg, patch_len):
        T, C = seg.shape
        n_p  = T // patch_len
        seg  = seg[:n_p * patch_len]
        return seg.reshape(n_p, patch_len, C).transpose(2, 0, 1).astype(np.float32)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        pid, seg, tfeat, label, first_ns = self.entries[idx]
        # No freq augmentation
        patches_list = [torch.from_numpy(self._make_patches(seg, pl))
                        for pl in self.patch_lens]
        return (patches_list,
                torch.from_numpy(tfeat),
                torch.tensor(label, dtype=torch.long),
                pid,
                torch.tensor(first_ns, dtype=torch.long),
                torch.from_numpy(seg.astype(np.float32)))


def _collate(batch):
    patches_lists, times, labels, pids, first_nss, segs = zip(*batch)
    n_scales = len(patches_lists[0])
    patches_stacked = [torch.stack([b[s] for b in patches_lists]) for s in range(n_scales)]
    return (patches_stacked, torch.stack(times), torch.stack(labels),
            list(pids), torch.stack(first_nss), torch.stack(segs))


def make_loader(pid_list, shuffle=False, is_train=False):
    ds = WindowDataset(pid_list, cfg.PROC_DIR, class_to_idx, is_train=is_train)
    dl = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=shuffle,
                    num_workers=0, pin_memory=GPU, collate_fn=_collate)
    return ds, dl


# ── Model components ──────────────────────────────────────────────────────────

class ZCRMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.g   = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        x0 = x - x.mean(-1, keepdim=True)
        return (x0 / x0.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()) * self.g


class GatedDeltaNet(nn.Module):
    """Unchanged — only stochastic depth wrapping is removed (C8 off)."""
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
                          nn.Dropout(dropout), nn.Linear(hidden, d))
            for _ in range(n_experts)
        ])
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        w = torch.softmax(self.router(x), dim=-1)
        s = torch.stack([e(x) for e in self.experts], dim=-2)
        return self.drop((w.unsqueeze(-1) * s).sum(-2))


def precompute_freqs(dim, n_tok, theta=10000.0):
    assert dim % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t     = torch.arange(n_tok)
    return torch.polar(torch.ones(n_tok, dim // 2), torch.outer(t, freqs))


def apply_rope(q, k, freqs):
    B, H, N, D = q.shape
    d2  = D // 2
    f   = freqs[:N].to(q.device).view(1, 1, N, d2)
    q_  = torch.view_as_complex(q.float().contiguous().view(B, H, N, d2, 2))
    k_  = torch.view_as_complex(k.float().contiguous().view(B, H, N, d2, 2))
    return (torch.view_as_real(q_ * f).view(B, H, N, D).type_as(q),
            torch.view_as_real(k_ * f).view(B, H, N, D).type_as(k))


class GatedAttention(nn.Module):
    def __init__(self, d, n_heads=2, dropout=0.1):
        super().__init__()
        assert d % n_heads == 0 and (d // n_heads) % 2 == 0
        self.h    = n_heads
        self.dh   = d // n_heads
        self.norm = ZCRMSNorm(d)
        self.qkv  = nn.Linear(d, 3 * d)
        self.out  = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, freqs):
        h = self.norm(x)
        B, N, D = h.shape
        qkv = self.qkv(h).reshape(B, N, 3, self.h, self.dh).permute(0, 2, 1, 3, 4)
        q = qkv[:, 0].transpose(1, 2)
        k = qkv[:, 1].transpose(1, 2)
        v = qkv[:, 2].transpose(1, 2)
        q, k  = apply_rope(q, k, freqs)
        score = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        attn  = self.drop(torch.softmax(score, dim=-1))
        y     = self.out((attn @ v).transpose(1, 2).contiguous().reshape(B, N, D))
        return x + torch.sigmoid(self.gate(h)) * y


class SimplePatchEmbed(nn.Module):
    def __init__(self, patch_len, channels, d):
        super().__init__()
        self.proj = nn.Linear(patch_len * channels, d)

    def forward(self, patches):
        B, C, NP, PL = patches.shape
        x = patches.permute(0, 2, 1, 3).reshape(B, NP, C * PL)
        return self.proj(x)


class PatchHARv2(nn.Module):
    def __init__(self):
        super().__init__()
        d  = cfg.D_MODEL
        NP = cfg.N_PATCHES

        # C4 off → single-scale embed
        self.single_embed = SimplePatchEmbed(cfg.PATCH_LEN, cfg.CHANNELS, d)

        # C3 off → static time MLP
        self.time_emb = nn.Sequential(nn.Linear(5, d), nn.ReLU(), nn.Dropout(0.1))

        self.input_norm = nn.LayerNorm(d)

        # C8 off → plain delta layers, no stochastic depth
        self.delta_layers = nn.ModuleList(
            [GatedDeltaNet(d, dropout=cfg.DROPOUT) for _ in range(cfg.N_LAYERS)])

        self.moe1 = SoftMoE(d, 2*d, n_experts=cfg.N_EXPERTS, dropout=cfg.DROPOUT)
        self.attn = GatedAttention(d, n_heads=cfg.N_HEADS, dropout=cfg.DROPOUT)
        self.moe2 = SoftMoE(d, 2*d, n_experts=cfg.N_EXPERTS, dropout=cfg.DROPOUT)

        freqs = precompute_freqs(d // cfg.N_HEADS, NP)
        self.register_buffer("freqs", freqs)

        # C6 off → no temperature parameter
        # C7 off → no prototype memory
        # C10 off → no recon head

        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d // 2, NUM_CLASSES),
        )

    def _embed_patches(self, patches_list):
        return self.single_embed(patches_list[0])

    def forward(self, patches_list, times, return_embedding=False):
        x = self._embed_patches(patches_list)
        x = x + self.time_emb(times).unsqueeze(1)
        x = self.input_norm(x)
        for layer in self.delta_layers:
            x = layer(x)
        x = x + self.moe1(x)
        x = self.attn(x, self.freqs)
        x = x + self.moe2(x)
        z = x.mean(dim=1)
        if return_embedding:
            return z
        logits = self.head(z)
        return logits, None  # no recon


def compute_class_weights(ds):
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for e in ds.entries:
        counts[e[3]] += 1
    w = np.clip(counts.max() / np.clip(counts, 1, None), 1.0, 10.0)
    w = torch.tensor(w, dtype=torch.float32)
    return w / w.sum() * NUM_CLASSES


def raw_mixup(patches_list, times, labels, alpha=0.2):
    if alpha <= 0:
        return patches_list, times, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(labels.size(0), device=labels.device)
    mixed = [lam * p + (1-lam) * p[idx] for p in patches_list]
    return mixed, lam * times + (1-lam) * times[idx], labels, labels[idx], lam


def tc_loss(logits):
    if logits.size(0) < 2:
        return logits.new_zeros(1).squeeze()
    p = F.softmax(logits[:-1], dim=-1)
    q = F.softmax(logits[1:],  dim=-1)
    return 0.5 * (F.kl_div(q.log(), p, reduction="batchmean") +
                  F.kl_div(p.log(), q, reduction="batchmean"))


class SmoothCE(nn.Module):
    """No smoothing (eps=0 when C6 is off)."""
    def __init__(self, weight=None, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing
        self.weight    = weight

    def forward(self, logits, labels):
        K   = logits.size(-1)
        eps = self.smoothing
        with torch.no_grad():
            soft = torch.full_like(logits, eps / max(K - 1, 1))
            soft.scatter_(-1, labels.unsqueeze(-1), 1.0 - eps)
        log_prob = F.log_softmax(logits, dim=-1)
        loss = -(soft * log_prob).sum(-1)
        if self.weight is not None:
            w    = self.weight.to(logits.device)
            loss = loss * w[labels]
        return loss.mean()


def kappa(yt, yp):
    cm = confusion_matrix(yt, yp)
    n  = cm.sum()
    if n == 0: return 0.0
    po = np.trace(cm) / n
    pe = np.dot(cm.sum(1), cm.sum(0)) / (n * n)
    return float((po - pe) / (1 - pe)) if abs(1 - pe) > 1e-12 else 0.0


def mcc(yt, yp):
    cm  = confusion_matrix(yt, yp).astype(float)
    n   = cm.sum()
    if n == 0: return 0.0
    s   = np.trace(cm)
    t   = cm.sum(1); p = cm.sum(0)
    num = s * n - np.sum(t * p)
    den = math.sqrt(max(n**2 - np.sum(t**2), 0.0) * max(n**2 - np.sum(p**2), 0.0))
    return float(num / den) if den > 0 else 0.0


def _compat_load(save_path, model):
    if not save_path.exists():
        return False
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    try:
        model.load_state_dict(ckpt, strict=True)
        print(f"  Best checkpoint loaded.")
        return True
    except RuntimeError as e:
        print(f"  [Warn] Could not load checkpoint: {e}")
        return False


def train(model, train_dl, val_dl, class_w, save_path):
    if save_path.exists():
        try:
            ckpt       = torch.load(save_path, map_location="cpu", weights_only=False)
            if set(ckpt.keys()) != set(model.state_dict().keys()):
                save_path.unlink()
        except Exception:
            save_path.unlink()

    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.LR, steps_per_epoch=len(train_dl),
        epochs=cfg.EPOCHS, pct_start=0.10, anneal_strategy="cos")
    criterion = SmoothCE(weight=class_w.to(device), smoothing=0.0)  # no smoothing

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=GPU)
    except TypeError:
        from torch.cuda.amp import GradScaler as _GS
        scaler = _GS(enabled=GPU)

    best_score, patience_ctr, history = -1e9, 0, []

    for epoch in range(cfg.EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_dl:
            patches_list, times, labels, _, _, raw_segs = batch
            # Raw mixup only (C9 off)
            patches_list, times, la, lb, lam = raw_mixup(
                [p.to(device).float() for p in patches_list],
                times.to(device).float(),
                labels.to(device).view(-1),
                cfg.MIXUP_ALPHA)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx():
                logits, _ = model(patches_list, times)
                loss = lam * criterion(logits, la) + (1-lam) * criterion(logits, lb)
                if cfg.TC_LAMBDA > 0:
                    loss = loss + cfg.TC_LAMBDA * tc_loss(logits)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_ok = (all(p.grad is None or torch.isfinite(p.grad).all()
                           for p in model.parameters()) and torch.isfinite(loss))
            if grad_ok:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
                scaler.step(optimizer)
            else:
                optimizer.zero_grad(set_to_none=True)
            scaler.update()
            scheduler.step()
            if torch.isfinite(loss):
                total_loss += float(loss.item())

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for batch in val_dl:
                patches_list, times, labels, _, _, _ = batch
                patches_list = [p.to(device).float() for p in patches_list]
                times_d      = times.to(device).float()
                logits, _    = model(patches_list, times_d)
                vp.extend(logits.argmax(1).cpu().numpy())
                vt.extend(labels.numpy())

        vp  = np.array(vp); vt = np.array(vt)
        f1  = float(f1_score(vt, vp, average="macro", zero_division=0))
        kap = kappa(vt, vp)
        avg = total_loss / max(1, len(train_dl))
        lr  = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1:02d}/{cfg.EPOCHS} | lr={lr:.2e} | "
              f"loss={avg:.4f} | F1={f1:.4f} | κ={kap:.4f}")
        history.append({"epoch": epoch+1, "loss": round(avg,6),
                         "val_f1": round(f1,6), "val_kappa": round(kap,6)})

        score = f1 + kap
        if score > best_score + 1e-6:
            best_score, patience_ctr = score, 0
            torch.save(model.state_dict(), save_path)
            print(f"    ✓ Checkpoint saved  (F1={f1:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    _compat_load(save_path, model)
    return history


@torch.no_grad()
def evaluate(model, test_dl):
    model.eval()
    all_pred, all_true = [], []
    for batch in test_dl:
        patches_list, times, labels, _, _, _ = batch
        patches_list = [p.to(device).float() for p in patches_list]
        logits, _    = model(patches_list, times.to(device).float())
        all_pred.extend(logits.argmax(1).cpu().numpy())
        all_true.extend(labels.numpy())
    yt = np.array(all_true); yp = np.array(all_pred)
    metrics = {
        "macro_f1": round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
        "kappa":    round(kappa(yt, yp), 4),
        "mcc":      round(mcc(yt, yp), 4),
        "accuracy": round(float((yt == yp).mean()), 4),
    }
    print(f"\n  Macro-F1={metrics['macro_f1']:.4f} | κ={metrics['kappa']:.4f} | "
          f"MCC={metrics['mcc']:.4f} | Acc={metrics['accuracy']:.4f}\n")
    print(classification_report(yt, yp, target_names=CLASSES, zero_division=0))
    return metrics


def main():
    print("=" * 70)
    print("  PatchHAR v2 — Ablation 1: ALL COMPONENTS DISABLED")
    print("=" * 70)

    train_ds, train_dl = make_loader(train_pids, shuffle=True,  is_train=True)
    val_ds,   val_dl   = make_loader(val_pids,   shuffle=False, is_train=False)
    test_ds,  test_dl  = make_loader(test_pids,  shuffle=False, is_train=False)
    print(f"  Train {len(train_ds):,} | Val {len(val_ds):,} | Test {len(test_ds):,}")

    model    = PatchHARv2().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {n_params:,}")

    save_path = cfg.OUTPUT_DIR / "weights_ablation1.pth"
    class_w   = compute_class_weights(train_ds)
    history   = train(model, train_dl, val_dl, class_w, save_path)

    print("\n── Evaluation ──")
    metrics = evaluate(model, test_dl)

    results = {
        "ablation":      "1_no_components",
        "description":   "All custom components disabled — pure baseline",
        "metrics":       metrics,
        "training":      history,
        "contributions": {k: False for k in vars(ContribConfig)
                          if k.startswith("C") and not k.startswith("__")},
        "config": {
            "window_size": cfg.WINDOW_SIZE, "patch_len": cfg.PATCH_LEN,
            "d_model": cfg.D_MODEL, "n_heads": cfg.N_HEADS,
            "n_layers": cfg.N_LAYERS, "n_experts": cfg.N_EXPERTS,
            "n_params": n_params, "device": str(device),
        },
    }
    out = cfg.OUTPUT_DIR / "results_ablation1.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved: {out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
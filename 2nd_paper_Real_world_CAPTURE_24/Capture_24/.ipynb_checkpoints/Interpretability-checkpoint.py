"""
patchhar_interpretability.py
============================
Self-contained interpretability analysis for PatchHAR v2.

Eight methods, each producing one publication-ready figure or table:

  I1  Attention head maps          (GatedAttention layer)
  I2  Gradient x input saliency    (GDN layers, per patch per axis)
  I3  SoftMoE router heatmap       (expert specialisation over time)
  I4  Dual-domain gate values      (FFT vs time branch per class)
  I5  Circadian bias sweep         (bias norm over 24 h x patch position)
  I6  t-SNE / UMAP embedding plot  (with prototype centroids)
  I7  Occlusion sensitivity        (patch-level perturbation)
  I8  Skip aggregation weights     (GDN layer importance per class)

Usage
-----
  python patchhar_interpretability.py
  python patchhar_interpretability.py --methods I1 I3 I6
  python patchhar_interpretability.py --ckpt /path/to/weights.pth
  python patchhar_interpretability.py --out-dir /path/to/figs
"""

from __future__ import annotations
import sys
import os

# ─────────────────────────────────────────────────────────────────────────────
# Import from patchhar_v2.py
# Reset sys.argv BEFORE the import so _parse_args() in patchhar_v2 does not
# see this script's CLI flags and misinterpret them.
# ─────────────────────────────────────────────────────────────────────────────
_real_argv = sys.argv[:]
sys.argv   = ["patchhar_v2.py"]

sys.path.insert(0, "/mnt/share/ali/1.Real_world_CAPTURE_24/Capture_24/")
from patchhar_v2 import (
    CC, cfg, device,
    CLASSES, NUM_CLASSES, class_to_idx, idx_to_class,
    train_pids, val_pids, test_pids,
    make_loader, _collate,
    GatedDeltaNet, GatedAttention, SoftMoE, PatchHARv2,
    StochasticDepth, apply_rope,
    precompute_freqs, ZCRMSNorm,
)

sys.argv = _real_argv   # restore so argparse below works correctly

# ─────────────────────────────────────────────────────────────────────────────
# Standard imports (after the patchhar_v2 import)
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import math
import random
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
ALL_METHODS = ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8"]


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="*", default=ALL_METHODS,
                   help="Which interpretability methods to run")
    p.add_argument("--ckpt", default=None, type=str,
                   help="Path to checkpoint .pth  (default: latest in OUTPUT_DIR)")
    p.add_argument("--out-dir", default=None, type=str,
                   help="Directory for figures  (default: OUTPUT_DIR/interp)")
    p.add_argument("--n-samples", default=500, type=int,
                   help="Max windows to use for averaging (per method)")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--umap", action="store_true",
                   help="Use UMAP instead of t-SNE for I6 (requires umap-learn)")
    return p.parse_args()


args    = _parse()
METHODS = [m.upper() for m in args.methods]
SEED    = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Resolve paths
# ─────────────────────────────────────────────────────────────────────────────
if args.ckpt:
    CKPT_PATH = Path(args.ckpt)
else:
    candidates = sorted(cfg.OUTPUT_DIR.glob("weights_full*.pth"))
    if not candidates:
        candidates = sorted(cfg.OUTPUT_DIR.glob("weights*.pth"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found in {cfg.OUTPUT_DIR}. "
            "Pass --ckpt explicitly.")
    CKPT_PATH = candidates[-1]

OUT_DIR = Path(args.out_dir) if args.out_dir else cfg.OUTPUT_DIR / "interp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Checkpoint : {CKPT_PATH}")
print(f"Output dir : {OUT_DIR}")
print(f"Methods    : {METHODS}")
print(f"N samples  : {args.n_samples}")

CLASS_COLORS = plt.cm.tab20(np.linspace(0, 1, NUM_CLASSES))


# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path: Path) -> PatchHARv2:
    model = PatchHARv2().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, name: str):
    for ext in ("pdf", "png"):
        p = OUT_DIR / f"{name}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  Saved: {name}.pdf / .png")
    plt.close(fig)


def _make_loader(pid_list) -> DataLoader:
    _, dl = make_loader(pid_list, shuffle=True, is_train=False)
    return dl


def _iter_batches(dl, n_samples: int):
    """Yield batches until at least n_samples windows have been seen."""
    collected = 0
    for batch in dl:
        yield batch
        collected += batch[2].shape[0]
        if collected >= n_samples:
            break


# ─────────────────────────────────────────────────────────────────────────────
# I1 — Attention head maps
# ─────────────────────────────────────────────────────────────────────────────
def run_I1(model: PatchHARv2, dl: DataLoader):
    print("\n── I1: Attention head maps ──")
    NP = cfg.N_PATCHES
    NH = cfg.N_HEADS

    acc   = np.zeros((NUM_CLASSES, NH, NP, NP), dtype=np.float64)
    count = np.zeros(NUM_CLASSES, dtype=np.int64)

    attn_cache: list[torch.Tensor] = []

    # Monkey-patch GatedAttention.forward to expose attention weights
    original_forward = model.attn.__class__.forward

    def _patched_forward(self, x, freqs):
        h = self.norm(x)
        B, N, D = h.shape
        qkv = (self.qkv(h)
               .reshape(B, N, 3, self.h, self.dh)
               .permute(0, 2, 1, 3, 4))
        q = qkv[:, 0].transpose(1, 2)
        k = qkv[:, 1].transpose(1, 2)
        v = qkv[:, 2].transpose(1, 2)
        q, k   = apply_rope(q, k, freqs)
        score  = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        attn_w = torch.softmax(score, dim=-1)          # (B, NH, NP, NP)
        attn_cache.append(attn_w.detach().cpu())
        y = self.out(
            (attn_w @ v).transpose(1, 2).contiguous().reshape(B, N, D))
        return x + torch.sigmoid(self.gate(h)) * y

    model.attn.__class__.forward = _patched_forward

    with torch.no_grad():
        for batch in _iter_batches(dl, args.n_samples):
            patches_list, times, labels, *_ = batch
            patches_list = [p.to(device).float() for p in patches_list]
            times_d      = times.to(device).float()
            attn_cache.clear()
            _ = model(patches_list, times_d)
            if not attn_cache:
                continue
            aw = attn_cache[0].numpy()   # (B, NH, NP, NP)
            for b, lab in enumerate(labels.numpy()):
                acc[lab]   += aw[b]
                count[lab] += 1

    model.attn.__class__.forward = original_forward

    nrows, ncols = NUM_CLASSES, NH
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3 * ncols, 2.8 * nrows),
                             squeeze=False)
    for k in range(NUM_CLASSES):
        for h in range(NH):
            ax  = axes[k][h]
            mat = acc[k, h] / max(count[k], 1)
            sns.heatmap(mat, ax=ax, cmap="viridis",
                        xticklabels=False, yticklabels=False,
                        cbar=(h == NH - 1))
            if h == 0:
                ax.set_ylabel(CLASSES[k], fontsize=7, rotation=0,
                              labelpad=60, va="center")
            if k == 0:
                ax.set_title(f"Head {h+1}", fontsize=8)
            ax.set_xlabel("key patch", fontsize=6)

    fig.suptitle("I1 — Attention maps per class x head\n"
                 "(rows=query patch, cols=key patch)", fontsize=10)
    _save(fig, "I1_attention_maps")


# ─────────────────────────────────────────────────────────────────────────────
# I2 — Gradient x input saliency
# ─────────────────────────────────────────────────────────────────────────────
def run_I2(model: PatchHARv2, dl: DataLoader):
    print("\n── I2: Gradient x input saliency ──")
    NP = cfg.N_PATCHES
    C  = cfg.CHANNELS

    acc   = np.zeros((NUM_CLASSES, NP, C), dtype=np.float64)
    count = np.zeros(NUM_CLASSES, dtype=np.int64)

    model.train(False)
    for batch in _iter_batches(dl, args.n_samples):
        patches_list, times, labels, *_ = batch

        patches_list_d = [p.to(device).float().requires_grad_(True)
                          for p in patches_list]
        times_d = times.to(device).float()

        for p in patches_list_d:
            p.retain_grad()

        logits, _ = model(patches_list_d, times_d)
        model.zero_grad()

        for b_idx, lab in enumerate(labels.numpy()):
            logits[b_idx, lab].backward(retain_graph=True)

        p0 = patches_list_d[0]
        if p0.grad is None:
            continue

        # (B, C, N_P, PL) -> |grad * input|, sum over PL -> (B, N_P, C)
        sal = (p0.grad.detach().cpu().abs() * p0.detach().cpu().abs())
        sal = sal.sum(-1).permute(0, 2, 1).numpy()

        for b, lab in enumerate(labels.numpy()):
            acc[lab]   += sal[b]
            count[lab] += 1

        model.zero_grad()

    sal_1d = np.where(
        count[:, None] > 0,
        acc.sum(-1) / np.maximum(count[:, None], 1),
        0.0,
    )

    t_axis = np.arange(NP) * cfg.PATCH_LEN / cfg.SIGNAL_RATE

    # All-class saliency curves
    fig, axes = plt.subplots(NUM_CLASSES, 1,
                             figsize=(10, 2.5 * NUM_CLASSES),
                             squeeze=False)
    for k, ax in enumerate([a[0] for a in axes]):
        ax.fill_between(t_axis, sal_1d[k], alpha=0.7,
                        color=CLASS_COLORS[k])
        ax.set_title(CLASSES[k], fontsize=8, loc="left")
        ax.set_xlim(0, t_axis[-1])
        ax.set_yticks([])
        if k == NUM_CLASSES - 1:
            ax.set_xlabel("Time (s)")

    fig.suptitle("I2 — Gradient x input saliency per class\n"
                 "(summed over x/y/z axes)", fontsize=10)
    fig.tight_layout()
    _save(fig, "I2_gradient_saliency")

    # Per-axis version for the first 4 classes
    axis_labels = ["x", "y", "z"]
    n_show = min(NUM_CLASSES, 4)
    fig2, axes2 = plt.subplots(n_show, C,
                               figsize=(3 * C, 2.5 * n_show),
                               squeeze=False)
    for k in range(n_show):
        for c in range(C):
            ax   = axes2[k][c]
            vals = acc[k, :, c] / max(count[k], 1)
            ax.fill_between(t_axis, vals, alpha=0.75,
                            color=CLASS_COLORS[k])
            ax.set_title(f"{CLASSES[k]} — {axis_labels[c]}", fontsize=7)
            ax.set_xlim(0, t_axis[-1])
            ax.set_yticks([])

    fig2.suptitle("I2 — Per-axis saliency (first 4 classes)", fontsize=10)
    fig2.tight_layout()
    _save(fig2, "I2_gradient_saliency_per_axis")


# ─────────────────────────────────────────────────────────────────────────────
# I3 — SoftMoE router heatmap
# ─────────────────────────────────────────────────────────────────────────────
def run_I3(model: PatchHARv2, dl: DataLoader):
    print("\n── I3: SoftMoE router heatmap ──")
    NP = cfg.N_PATCHES
    NE = cfg.N_EXPERTS

    acc1  = np.zeros((NUM_CLASSES, NP, NE), dtype=np.float64)
    acc2  = np.zeros((NUM_CLASSES, NP, NE), dtype=np.float64)
    count = np.zeros(NUM_CLASSES, dtype=np.int64)

    cache: list[list] = [[], []]

    # Patch moe1 at class level
    orig_moe_fwd = model.moe1.__class__.forward

    def _patched_moe_fwd(self, x):
        w = torch.softmax(self.router(x), dim=-1)   # (B, NP, NE)
        cache[0].append(w.detach().cpu())
        s = torch.stack([e(x) for e in self.experts], dim=-2)
        return self.drop((w.unsqueeze(-1) * s).sum(-2))

    model.moe1.__class__.forward = _patched_moe_fwd

    # Patch moe2 at instance level to avoid class-level collision
    _orig_moe2_fwd = model.moe2.forward

    def _moe2_fwd(x):
        w = torch.softmax(model.moe2.router(x), dim=-1)
        cache[1].append(w.detach().cpu())
        s = torch.stack([e(x) for e in model.moe2.experts], dim=-2)
        return model.moe2.drop((w.unsqueeze(-1) * s).sum(-2))

    model.moe2.forward = _moe2_fwd

    with torch.no_grad():
        for batch in _iter_batches(dl, args.n_samples):
            patches_list, times, labels, *_ = batch
            patches_list = [p.to(device).float() for p in patches_list]
            times_d      = times.to(device).float()
            cache[0].clear()
            cache[1].clear()
            _ = model(patches_list, times_d)

            if not cache[0] or not cache[1]:
                continue

            w1 = cache[0][0].numpy()   # (B, NP, NE)
            w2 = cache[1][0].numpy()

            for b, lab in enumerate(labels.numpy()):
                acc1[lab]  += w1[b]
                acc2[lab]  += w2[b]
                count[lab] += 1

    model.moe1.__class__.forward = orig_moe_fwd
    model.moe2.forward           = _orig_moe2_fwd

    # Router heatmaps per MoE block
    for tag, acc in [("moe1", acc1), ("moe2", acc2)]:
        fig, axes = plt.subplots(1, NUM_CLASSES,
                                 figsize=(3.5 * NUM_CLASSES, 4),
                                 squeeze=False)
        for k, ax in enumerate(axes[0]):
            mat = (acc[k] / max(count[k], 1)).T   # (NE, NP)
            sns.heatmap(mat, ax=ax, cmap="YlOrRd",
                        xticklabels=False,
                        yticklabels=[f"E{i+1}" for i in range(NE)],
                        cbar=True)
            ax.set_title(CLASSES[k], fontsize=7)
            ax.set_xlabel("patch index", fontsize=7)
        fig.suptitle(
            f"I3 — SoftMoE ({tag}) router weights per class\n"
            "(rows=experts, cols=patch position; colour=mean routing weight)",
            fontsize=9)
        fig.tight_layout()
        _save(fig, f"I3_softmoe_{tag}_router")

    # Expert specialisation entropy
    ent_data = np.zeros((2, NUM_CLASSES), dtype=np.float64)
    for ti, (tag, acc) in enumerate([("moe1", acc1), ("moe2", acc2)]):
        for k in range(NUM_CLASSES):
            mean_w = acc[k] / max(count[k], 1)   # (NP, NE)
            p = mean_w.mean(0)
            p = p / (p.sum() + 1e-12)
            ent_data[ti, k] = -np.sum(p * np.log(p + 1e-12))

    fig3, ax3 = plt.subplots(figsize=(max(6, NUM_CLASSES * 0.8), 3))
    im = ax3.imshow(ent_data, aspect="auto", cmap="coolwarm_r")
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["MoE-1", "MoE-2"])
    ax3.set_xticks(range(NUM_CLASSES))
    ax3.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=7)
    plt.colorbar(im, ax=ax3,
                 label="Routing entropy (lower = more specialised)")
    ax3.set_title("I3 — Expert routing entropy per class / MoE block",
                  fontsize=9)
    fig3.tight_layout()
    _save(fig3, "I3_expert_entropy")


# ─────────────────────────────────────────────────────────────────────────────
# I4 — Dual-domain gate values (C1)
# ─────────────────────────────────────────────────────────────────────────────
def run_I4(model: PatchHARv2, dl: DataLoader):
    print("\n── I4: Dual-domain gate values ──")

    if not CC.C1_DUAL_DOMAIN_EMBEDDING:
        print("  C1 is OFF — skipping I4")
        return

    gate_vals = []
    for i, embed in enumerate(model.patch_embeds):
        if not hasattr(embed, "gate_w"):
            continue
        g = torch.sigmoid(embed.gate_w).detach().cpu().numpy()   # (D,)
        gate_vals.append((i, g))

    if not gate_vals:
        print("  No DualDomainPatchEmbed found — skipping I4")
        return

    # Figure 1: gate distribution per granularity
    fig, axes = plt.subplots(1, len(gate_vals),
                             figsize=(5 * len(gate_vals), 3.5),
                             squeeze=False)
    for col, (i, g) in enumerate(gate_vals):
        ax = axes[0][col]
        pl = (cfg.PATCH_LENS_MULTI[i]
              if CC.C4_MULTISCALE_PATCHING else cfg.PATCH_LEN)
        ax.hist(g, bins=20, color="steelblue", alpha=0.8,
                edgecolor="white")
        ax.axvline(g.mean(), color="crimson", lw=1.5,
                   label=f"mean={g.mean():.3f}")
        ax.set_xlim(0, 1)
        ax.set_xlabel(
            "sigma(gate_w)  [0=FFT dominates, 1=time dominates]")
        ax.set_title(f"Scale {i+1}  (PL={pl})", fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle("I4 — Dual-domain gate distribution across D dimensions",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, "I4_gate_distribution")

    # Figure 2: all dimensions sorted descending
    all_g    = np.concatenate([g for _, g in gate_vals])
    sorted_g = np.sort(all_g)[::-1]

    fig2, ax2 = plt.subplots(figsize=(10, 3))
    colors = np.where(sorted_g > 0.5, "steelblue", "darkorange")
    ax2.bar(range(len(sorted_g)), sorted_g,
            color=colors, width=1.0, edgecolor="none")
    ax2.axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.5)
    ax2.set_xlabel("D dimension (sorted by gate value)")
    ax2.set_ylabel("sigma(gate_w)")
    ax2.set_title("I4 — Sorted gate values: "
                  "blue > 0.5 = time branch, orange < 0.5 = FFT branch")
    fig2.tight_layout()
    _save(fig2, "I4_gate_sorted")


# ─────────────────────────────────────────────────────────────────────────────
# I5 — Circadian bias sweep (C3)
# ─────────────────────────────────────────────────────────────────────────────
def run_I5(model: PatchHARv2, dl: DataLoader):
    print("\n── I5: Circadian bias sweep ──")

    if not CC.C3_CIRCADIAN_BIAS:
        print("  C3 is OFF — skipping I5")
        return

    NP          = cfg.N_PATCHES
    bias_matrix = np.zeros((24, NP), dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for hour in range(24):
            tvec = torch.tensor([[
                hour / 24.0,
                0.0,
                2.0 / 7.0,
                0.0,
                float(hour // 6) / 3.0,
            ]], dtype=torch.float32, device=device)   # (1, 5)

            bias = model.circ_bias(tvec)              # (1, N_P, D)
            norm = bias.squeeze(0).norm(dim=-1).cpu().numpy()
            bias_matrix[hour] = norm

    t_axis = np.arange(NP) * cfg.PATCH_LEN / cfg.SIGNAL_RATE

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(bias_matrix, aspect="auto", origin="upper",
                   cmap="plasma",
                   extent=[0, t_axis[-1], 23.5, -0.5])
    ax.set_xlabel("Patch position (s within 30-s window)")
    ax.set_ylabel("Hour of day")
    ax.set_yticks(range(0, 24, 3))
    ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])
    plt.colorbar(im, ax=ax, label="||circadian bias||_2")
    ax.set_title("I5 — Circadian positional bias magnitude\n"
                 "(rows=hour, cols=patch position within window)")
    _save(fig, "I5_circadian_bias_sweep")

    # Mean bias norm per hour (collapsed over patch axis)
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.plot(range(24), bias_matrix.mean(axis=1), "o-",
             color="purple", lw=1.5, markersize=4)
    ax2.set_xlabel("Hour of day")
    ax2.set_ylabel("Mean ||bias||_2 across patches")
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_xticklabels(
        [f"{h:02d}:00" for h in range(0, 24, 2)],
        rotation=45, ha="right")
    ax2.set_title("I5 — Mean circadian bias norm over 24 hours")
    fig2.tight_layout()
    _save(fig2, "I5_circadian_bias_hourly")


# ─────────────────────────────────────────────────────────────────────────────
# I6 — t-SNE / UMAP embedding with prototype centroids
# ─────────────────────────────────────────────────────────────────────────────
def run_I6(model: PatchHARv2, dl: DataLoader):
    print("\n── I6: Embedding visualisation ──")

    all_z, all_y = [], []
    model.eval()
    with torch.no_grad():
        for batch in _iter_batches(dl, args.n_samples):
            patches_list, times, labels, *_ = batch
            patches_list = [p.to(device).float() for p in patches_list]
            times_d      = times.to(device).float()
            z = model(patches_list, times_d, return_embedding=True)
            all_z.append(z.cpu().numpy())
            all_y.append(labels.numpy())

    Z = np.concatenate(all_z, axis=0)
    Y = np.concatenate(all_y, axis=0)
    print(f"  Collected {Z.shape[0]} embeddings")

    Z_n = normalize(Z, norm="l2")

    use_umap = False
    if args.umap:
        try:
            import umap as umap_lib
            reducer     = umap_lib.UMAP(n_components=2,
                                        random_state=SEED,
                                        min_dist=0.1,
                                        n_neighbors=30)
            Z2d         = reducer.fit_transform(Z_n)
            method_name = "UMAP"
            use_umap    = True
        except ImportError:
            print("  umap-learn not installed; falling back to t-SNE")

    if not use_umap:
        ts  = TSNE(n_components=2, random_state=SEED,
                   perplexity=min(30, max(5, Z.shape[0] // 4)),
                   n_iter=1000, verbose=0)
        Z2d         = ts.fit_transform(Z_n)
        method_name = "t-SNE"

    fig, ax = plt.subplots(figsize=(9, 7))
    for k in range(NUM_CLASSES):
        mask = (Y == k)
        ax.scatter(Z2d[mask, 0], Z2d[mask, 1],
                   s=10, alpha=0.55,
                   color=CLASS_COLORS[k],
                   label=CLASSES[k], linewidths=0)

    # Overlay prototype centroids if available
    if CC.C7_PROTOTYPE_MEMORY and model.proto_filled:
        protos   = model.prototypes.detach().cpu().numpy()
        protos_n = normalize(protos, norm="l2")
        if use_umap:
            p2d = reducer.transform(protos_n)
        else:
            # t-SNE has no transform — use mean of each class cluster in 2-D
            p2d = np.array([
                Z2d[Y == k].mean(0) if (Y == k).sum() > 0
                else np.zeros(2)
                for k in range(NUM_CLASSES)
            ])
        for k in range(NUM_CLASSES):
            ax.scatter(p2d[k, 0], p2d[k, 1],
                       s=180, marker="*",
                       color=CLASS_COLORS[k],
                       edgecolors="black", linewidths=0.5, zorder=5)

    ax.legend(fontsize=7, markerscale=2,
              bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_xlabel(f"{method_name} dim 1")
    ax.set_ylabel(f"{method_name} dim 2")
    ax.set_title(f"I6 — {method_name} of pooled embeddings\n"
                 "(star markers = prototype centroids where available)")
    fig.tight_layout()
    _save(fig, f"I6_embedding_{method_name.lower()}")


# ─────────────────────────────────────────────────────────────────────────────
# I7 — Occlusion sensitivity
# ─────────────────────────────────────────────────────────────────────────────
def run_I7(model: PatchHARv2, dl: DataLoader):
    print("\n── I7: Occlusion sensitivity ──")

    NP = cfg.N_PATCHES

    acc   = np.zeros((NUM_CLASSES, NP), dtype=np.float64)
    count = np.zeros(NUM_CLASSES, dtype=np.int64)

    model.eval()
    with torch.no_grad():
        for batch in _iter_batches(dl, min(args.n_samples, 200)):
            patches_list, times, labels, *_ = batch
            B = labels.shape[0]

            patches_list_d = [p.to(device).float() for p in patches_list]
            times_d        = times.to(device).float()

            logits_base, _ = model(patches_list_d, times_d)
            conf_base      = F.softmax(logits_base, dim=-1)   # (B, K)

            for pos in range(NP):
                patched = [p.clone() for p in patches_list_d]
                patched[0][:, :, pos, :] = 0.0               # zero patch

                logits_occ, _ = model(patched, times_d)
                conf_occ      = F.softmax(logits_occ, dim=-1)

                for b in range(B):
                    lab           = int(labels[b])
                    drop          = float(conf_base[b, lab] - conf_occ[b, lab])
                    acc[lab, pos] += drop

            for b in range(B):
                count[int(labels[b])] += 1

    occ = np.where(count[:, None] > 0,
                   acc / np.maximum(count[:, None], 1), 0.0)

    t_axis = np.arange(NP) * cfg.PATCH_LEN / cfg.SIGNAL_RATE

    # Summary heatmap (all classes)
    fig, ax = plt.subplots(figsize=(12, max(3, NUM_CLASSES * 0.5)))
    im = ax.imshow(occ, aspect="auto", cmap="Reds",
                   extent=[0, t_axis[-1], NUM_CLASSES - 0.5, -0.5])
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(CLASSES, fontsize=7)
    ax.set_xlabel("Patch position (s within 30-s window)")
    plt.colorbar(im, ax=ax, label="Mean confidence drop when occluded")
    ax.set_title("I7 — Occlusion sensitivity\n"
                 "(brighter = more important patch position)")
    _save(fig, "I7_occlusion_sensitivity")

    # Per-class line plots
    fig2, axes2 = plt.subplots(NUM_CLASSES, 1,
                               figsize=(10, 2.2 * NUM_CLASSES),
                               squeeze=False, sharex=True)
    for k, ax in enumerate([a[0] for a in axes2]):
        ax.fill_between(t_axis, occ[k], alpha=0.65,
                        color=CLASS_COLORS[k])
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(CLASSES[k], fontsize=8, loc="left")
        ax.set_yticks([])

    axes2[-1][0].set_xlabel("Time within window (s)")
    fig2.suptitle("I7 — Occlusion sensitivity per class", fontsize=10)
    fig2.tight_layout()
    _save(fig2, "I7_occlusion_per_class")


# ─────────────────────────────────────────────────────────────────────────────
# I8 — Skip aggregation layer weights (C2)
# ─────────────────────────────────────────────────────────────────────────────
def run_I8(model: PatchHARv2, dl: DataLoader):
    print("\n── I8: Skip aggregation weights ──")

    if not CC.C2_CALANET_SKIP_AGG:
        print("  C2 is OFF — skipping I8")
        return

    w = torch.softmax(model.skip_agg.weights,
                      dim=0).detach().cpu().numpy()
    L = len(w)

    # Figure 1: global layer importance bar chart
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(range(1, L + 1), w,
                  color=plt.cm.Blues(np.linspace(0.4, 0.9, L)),
                  edgecolor="white")
    ax.set_xlabel("GatedDeltaNet layer (depth)")
    ax.set_ylabel("softmax weight")
    ax.set_xticks(range(1, L + 1))
    for bar, val in zip(bars, w):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=8)
    ax.set_title("I8 — Skip aggregation layer weights (global)")
    fig.tight_layout()
    _save(fig, "I8_skip_weights_global")

    # Figure 2: per-class layer importance via hidden-state norms
    acc_corr  = np.zeros((NUM_CLASSES, L), dtype=np.float64)
    count     = np.zeros(NUM_CLASSES, dtype=np.int64)
    local_cache: list = [None] * L

    def _make_hook(li):
        def _hook(module, inp, out):
            local_cache[li] = out.detach().cpu()
        return _hook

    hooks = []
    for li, layer in enumerate(model.delta_layers):
        inner = layer.layer if isinstance(layer, StochasticDepth) else layer
        hooks.append(inner.register_forward_hook(_make_hook(li)))

    model.eval()
    with torch.no_grad():
        for batch in _iter_batches(dl, args.n_samples):
            patches_list, times, labels, *_ = batch
            patches_list = [p.to(device).float() for p in patches_list]
            times_d      = times.to(device).float()

            logits, _ = model(patches_list, times_d)
            probs = F.softmax(logits, dim=-1).cpu().numpy()   # (B, K)

            if any(h is None for h in local_cache):
                continue

            for b, lab in enumerate(labels.numpy()):
                for li in range(L):
                    h_mean = local_cache[li][b].norm(dim=-1).mean().item()
                    acc_corr[lab, li] += h_mean * probs[b, lab]
                count[lab] += 1

    for hook in hooks:
        hook.remove()

    acc_norm = np.where(
        count[:, None] > 0,
        acc_corr / np.maximum(count[:, None], 1),
        0.0,
    )
    row_sums = acc_norm.sum(axis=1, keepdims=True)
    acc_norm = np.where(row_sums > 0, acc_norm / row_sums, 0.0)

    fig2, ax2 = plt.subplots(
        figsize=(max(5, L * 1.5), max(3, NUM_CLASSES * 0.5)))
    im = ax2.imshow(acc_norm, aspect="auto", cmap="YlGnBu")
    ax2.set_xticks(range(L))
    ax2.set_xticklabels([f"GDN-{i+1}" for i in range(L)], fontsize=8)
    ax2.set_yticks(range(NUM_CLASSES))
    ax2.set_yticklabels(CLASSES, fontsize=7)
    plt.colorbar(im, ax=ax2,
                 label="Relative layer contribution (normalised)")
    ax2.set_title("I8 — Per-class GDN layer importance")
    fig2.tight_layout()
    _save(fig2, "I8_skip_weights_per_class")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  PatchHAR v2 — Interpretability Analysis")
    print("=" * 70)

    model = load_model(CKPT_PATH)
    print(f"  Loaded: {CKPT_PATH.name}")
    print("  Active contributions:")
    for k, v in CC.__dict__.items():
        if k.startswith("C") and not k.startswith("__"):
            print(f"    {'ok' if v else '--'} {k}")

    pids = test_pids if test_pids else val_pids
    dl   = _make_loader(pids)

    dispatch = {
        "I1": run_I1,
        "I2": run_I2,
        "I3": run_I3,
        "I4": run_I4,
        "I5": run_I5,
        "I6": run_I6,
        "I7": run_I7,
        "I8": run_I8,
    }

    for method_id in METHODS:
        if method_id not in dispatch:
            print(f"  [WARN] Unknown method: {method_id}")
            continue
        try:
            dispatch[method_id](model, dl)
        except Exception as e:
            print(f"  [ERROR] {method_id} failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"  Done — figures saved to {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
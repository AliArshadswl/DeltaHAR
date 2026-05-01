"""
patchhar_interpretability.py
============================
Full interpretability suite for PatchHAR v2.
Run after training:

    python patchhar_interpretability.py

All figures are saved to cfg.OUTPUT_DIR / "interpretability/".

Sections
--------
  1.  Attention Rollout & Patch Saliency Heatmap
  2.  MoE Expert Specialisation Analysis
  3.  ECE / Reliability Diagrams (per model variant)
  4.  Dual-Domain Embedding Gate Analysis  (C1)
  5.  Prototype Memory UMAP + Confusion Analysis  (C7)
  6.  Per-Class Error Diagnosis
      6a. Normalised Confusion Matrix
      6b. Circadian Error Analysis  (C3)
      6c. Signal-Difficulty vs F1
      6d. Grad-CAM on Worst Mis-predictions
  7.  Reconstruction Quality Analysis  (C10)
  8.  PCGrad Gradient-Conflict Logging  (C10)
  9.  Multi-Scale Patch Ablation  (C4)
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import sys, json, math, random, warnings, copy
from pathlib import Path
from contextlib import nullcontext

warnings.filterwarnings("ignore")

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, confusion_matrix, classification_report,
)
from sklearn.calibration import calibration_curve

try:
    import umap
    HAS_UMAP = True
except ImportError:
    from sklearn.manifold import TSNE as _TSNE
    HAS_UMAP = False
    print("[INFO] umap-learn not found – using t-SNE for embeddings.")

# ── local: import everything from your training script ────────────────────────
# Assumes this file lives in the same directory as your training script.
# Adjust the import name if needed.
try:
    from patchhar_v2 import (          # ← rename to match your filename
        PatchHARv2, Config, ContribConfig, CC, cfg,
        make_loader, device, CLASSES, NUM_CLASSES,
        class_to_idx, idx_to_class,
        train_pids, val_pids, test_pids,
        amp_ctx,
    )
except ModuleNotFoundError:
    raise SystemExit(
        "Could not import from patchhar_v2. "
        "Rename the import above to match your training-script filename."
    )

# ── output folder ─────────────────────────────────────────────────────────────
INTERP_DIR = cfg.OUTPUT_DIR / "interpretability"
INTERP_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH = cfg.OUTPUT_DIR / "weights_patchhar_v2.pth"

# =============================================================================
# =============================================================================
# IEEE-compatible publication style
# =============================================================================
# Fonts available on this server:
#   "Liberation Serif"  — metric-compatible Times New Roman substitute
#   "TeX Gyre Termes"   — another Times New Roman clone (our second choice)
#   "Latin Modern Roman"— the LaTeX default (clean, academic)
# IEEE column widths: single = 3.5 in, double = 7.16 in
# =============================================================================

# Priority list: first available font wins
_FONT_PRIORITY = [
    "Liberation Serif",   # Times New Roman metric clone — best match
    "TeX Gyre Termes",    # Times New Roman clone
    "Latin Modern Roman", # LaTeX default serif
    "DejaVu Serif",       # always present, good fallback
]

def _pick_font(priority):
    """
    Return the first font name from `priority` that matplotlib
    can actually locate on this system.
    We match against the full font manager list (name attribute),
    which is what matplotlib uses when font.family is set to a
    specific name rather than a generic alias like 'serif'.
    """
    import matplotlib.font_manager as _fm
    import warnings as _w
    # Suppress the "findfont: Generic family" warnings permanently
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    _w.filterwarnings("ignore", message="findfont")

    available = {f.name for f in _fm.fontManager.ttflist}
    for name in priority:
        if name in available:
            print(f"[Style] Font selected: '{name}'")
            return name
    # Hard fallback: DejaVu Sans is always bundled with matplotlib
    print("[Style] No preferred font found — falling back to 'DejaVu Sans'.")
    return "DejaVu Sans"

_CHOSEN_FONT = _pick_font(_FONT_PRIORITY)

def _setup_style(font_name):
    """
    Apply IEEE publication-quality rcParams.
    Key fix: set font.family directly to the chosen font name
    (e.g. "Liberation Serif") rather than to the generic alias
    "serif", which matplotlib cannot resolve on this server.
    """
    # Use the chosen font name directly — do NOT use "serif" as family
    _f = font_name if font_name else "DejaVu Sans"

    plt.rcParams.update({
        # ── font — use name directly, not generic alias ────────────────────
        "font.family"               : _f,
        "mathtext.fontset"          : "stix",
        "text.usetex"               : False,

        # ── IEEE sizes (8–10 pt body → scale labels to ~10–12 pt) ─────────
        "font.size"                 : 11,
        "axes.titlesize"            : 12,
        "axes.labelsize"            : 11,
        "xtick.labelsize"           : 10,
        "ytick.labelsize"           : 10,
        "legend.fontsize"           : 10,
        "legend.title_fontsize"     : 11,
        "figure.titlesize"          : 13,

        # ── axes ──────────────────────────────────────────────────────────
        "axes.linewidth"            : 1.0,
        "axes.spines.top"           : False,
        "axes.spines.right"         : False,
        "axes.grid"                 : True,
        "axes.grid.which"           : "major",
        "grid.alpha"                : 0.30,
        "grid.linewidth"            : 0.6,
        "grid.color"                : "#d0d0d0",
        "axes.axisbelow"            : True,

        # ── ticks ─────────────────────────────────────────────────────────
        "xtick.major.width"         : 1.0,
        "ytick.major.width"         : 1.0,
        "xtick.major.size"          : 4.0,
        "ytick.major.size"          : 4.0,
        "xtick.direction"           : "out",
        "ytick.direction"           : "out",
        "xtick.major.pad"           : 4.0,
        "ytick.major.pad"           : 4.0,

        # ── lines / markers / patches ─────────────────────────────────────
        "lines.linewidth"           : 1.8,
        "lines.markersize"          : 6.0,
        "patch.linewidth"           : 0.7,

        # ── legend ────────────────────────────────────────────────────────
        "legend.frameon"            : True,
        "legend.framealpha"         : 0.92,
        "legend.edgecolor"          : "#bbbbbb",
        "legend.borderpad"          : 0.45,
        "legend.labelspacing"       : 0.35,
        "legend.handlelength"       : 1.6,
        "legend.handletextpad"      : 0.5,

        # ── figure / saving ───────────────────────────────────────────────
        "figure.dpi"                : 120,
        "savefig.dpi"               : 300,
        "savefig.bbox"              : "tight",
        "savefig.pad_inches"        : 0.08,
        # IMPORTANT: disable constrained_layout globally.
        # Seaborn heatmaps create their own internal layout engine for
        # colorbars.  If constrained_layout is on globally, tight_layout()
        # raises "Colorbar layout not compatible with old engine".
        "figure.constrained_layout.use": False,

        # ── image / PDF embedding (required by IEEE) ───────────────────────
        "image.interpolation"       : "nearest",
        "pdf.fonttype"              : 42,
        "ps.fonttype"               : 42,
    })

    # Seaborn — must come AFTER rcParams update
    # Pass font as 'font' kwarg (not via rc font.family which uses alias)
    sns.set_theme(
        style      = "ticks",
        context    = "paper",
        font       = _f,
        font_scale = 1.15,
        rc = {
            "font.family"        : _f,
            "axes.spines.top"    : False,
            "axes.spines.right"  : False,
            "axes.grid"          : True,
            "grid.alpha"         : 0.30,
            "figure.constrained_layout.use": False,
            "pdf.fonttype"       : 42,
            "ps.fonttype"        : 42,
        },
    )
_setup_style(_CHOSEN_FONT)


# ── IEEE column-width figure size constants (inches) ─────────────────────────
FIG_W_SINGLE = 3.50   # one column
FIG_W_DOUBLE = 7.16   # full text width (two columns)
FIG_W_15COL  = 5.40   # 1.5 columns


# ── Convenience: create a tight figure with sensible defaults ─────────────────
def _fig(*args, pad=0.4, h_pad=0.8, w_pad=0.6, **kwargs):
    """
    Drop-in replacement for _fig().
    Applies tight_layout automatically with IEEE-friendly padding.
    Returns (fig, axes) exactly like _fig().
    """
    fig, axes = plt.subplots(*args, **kwargs)
    fig.set_constrained_layout(False)
    return fig, axes


def _save(fig, path, pad=0.5, h_pad=1.0, w_pad=0.8):
    """
    Save figure with smart layout tightening.

    Handles the seaborn/colorbar conflict:
    When sns.heatmap() or plt.colorbar() is used, matplotlib sets an
    internal constrained-layout engine for the colorbar axis.
    Calling tight_layout() afterwards raises RuntimeError.
    We catch this and fall back to subplots_adjust with generous margins.
    """
    try:
        fig.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    except RuntimeError:
        # Colorbar-based layout engine conflict — use subplots_adjust instead
        try:
            fig.subplots_adjust(left=0.12, right=0.92,
                                bottom=0.15, top=0.92,
                                hspace=0.45, wspace=0.35)
        except Exception:
            pass
    except Exception:
        pass
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print(f"   Saved \u2192 {path}")

# ── reproducibility ───────────────────────────────────────────────────────────
random.seed(cfg.SEED); np.random.seed(cfg.SEED); torch.manual_seed(cfg.SEED)


# =============================================================================
# Helper: load model
# =============================================================================
def load_model(ckpt: Path = CKPT_PATH) -> PatchHARv2:
    model = PatchHARv2().to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    print(f"[✓] Loaded checkpoint: {ckpt}")
    return model


# =============================================================================
# Helper: collect predictions, embeddings, router weights, times
# =============================================================================
@torch.no_grad()
def collect_outputs(
    model: PatchHARv2,
    loader: DataLoader,
    max_batches: int | None = None,
) -> dict:
    """
    Returns a dict with numpy arrays:
        logits, probs, preds, labels, embeddings,
        moe1_weights, moe2_weights,
        times (raw 5-d), hours, raw_segs, recon

    Design note: we do exactly ONE forward pass per batch.
    A hook on the mean-pooling step captures the embedding z so we never
    call model.forward() twice (which would double-count MoE hook firings).
    """
    all_logits, all_labels = [], []
    all_times, all_raw_segs, all_recon = [], [], []

    # ── hook MoE router weights ───────────────────────────────────────────
    hooks    = []
    moe1_buf = []
    moe2_buf = []
    emb_buf  = []      # captures pooled z from the single forward pass

    def _moe1_hook(mod, inp, out):
        w = torch.softmax(mod.router(inp[0]), dim=-1)   # (B, N_tok, E)
        moe1_buf.append(w.detach().cpu())

    def _moe2_hook(mod, inp, out):
        w = torch.softmax(mod.router(inp[0]), dim=-1)
        moe2_buf.append(w.detach().cpu())

    # Hook the classification head's first dropout to intercept z = x.mean(1)
    # The head's first module is nn.Dropout, its input is z.
    def _head_hook(mod, inp, out):
        # inp[0] is z  (B, D)
        emb_buf.append(inp[0].detach().cpu())

    hooks.append(model.moe1.register_forward_hook(_moe1_hook))
    hooks.append(model.moe2.register_forward_hook(_moe2_hook))
    hooks.append(model.head[0].register_forward_hook(_head_hook))
    # model.head[0] is the first nn.Dropout — its input is z

    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        patches_list, times, labels, _, _, raw_segs = batch
        patches_list = [p.to(device).float() for p in patches_list]
        times_d      = times.to(device).float()

        # ── single forward pass ────────────────────────────────────────────
        logits, recon = model(patches_list, times_d)
        # embedding is captured by _head_hook into emb_buf

        all_logits.append(logits.cpu())
        all_labels.append(labels)
        all_times.append(times.cpu())
        all_raw_segs.append(raw_segs.cpu())
        if recon is not None:
            all_recon.append(recon.cpu())

    for h in hooks:
        h.remove()

    logits_t = torch.cat(all_logits)
    probs_t  = torch.softmax(logits_t, dim=-1)
    preds_t  = probs_t.argmax(-1)
    labels_t = torch.cat(all_labels)
    times_t  = torch.cat(all_times)
    hours    = (times_t[:, 0] * 24).numpy().astype(int)  # reverse-normalise

    moe1_w = torch.cat(moe1_buf).numpy() if moe1_buf else None   # (N, T, E)
    moe2_w = torch.cat(moe2_buf).numpy() if moe2_buf else None

    recon_t = torch.cat(all_recon) if all_recon else None

    return dict(
        logits     = logits_t.numpy(),
        probs      = probs_t.numpy(),
        preds      = preds_t.numpy(),
        labels     = labels_t.numpy(),
        embeddings = (torch.cat(emb_buf).numpy() if emb_buf
                      else np.zeros((len(logits_t), cfg.D_MODEL), dtype=np.float32)),
        moe1_w     = moe1_w,
        moe2_w     = moe2_w,
        times      = times_t.numpy(),
        hours      = hours,
        raw_segs   = torch.cat(all_raw_segs).numpy(),
        recon      = recon_t.numpy() if recon_t is not None else None,
    )


# =============================================================================
# 1. Attention Rollout & Patch Saliency
# =============================================================================
def compute_attention_rollout(
    model: PatchHARv2,
    patches_list: list[torch.Tensor],
    times: torch.Tensor,
) -> np.ndarray:
    """
    Returns per-patch importance scores of shape (B, N_P).
    Uses attention rollout: propagate A through residual connections.
    """
    attn_maps = []

    def _hook(mod, inp, out):
        # Recompute attention weights from the cached qkv
        h = mod.norm(inp[0])
        B, N, D = h.shape
        qkv = (mod.qkv(h)
               .reshape(B, N, 3, mod.h, mod.dh)
               .permute(0, 2, 1, 3, 4))
        q = qkv[:, 0].transpose(1, 2)
        k = qkv[:, 1].transpose(1, 2)
        from patchhar_v2 import apply_rope
        q, k  = apply_rope(q, k, mod.freqs if hasattr(mod, "freqs")
                           else model.freqs)
        score = (q @ k.transpose(-2, -1)) / math.sqrt(mod.dh)
        attn  = torch.softmax(score, dim=-1)                 # (B,H,N,N)
        attn_maps.append(attn.mean(1).detach().cpu())        # avg heads

    hook = model.attn.register_forward_hook(_hook)

    with torch.no_grad():
        model(patches_list, times)

    hook.remove()

    if not attn_maps:
        raise RuntimeError("Attention hook did not fire.")

    A = attn_maps[0].numpy()    # (B, N, N)
    # Rollout: add residual identity, renormalise rows
    I  = np.eye(A.shape[-1])[None]             # (1, N, N)
    Ar = 0.5 * A + 0.5 * I
    Ar = Ar / Ar.sum(-1, keepdims=True)
    # Importance = mean over destination tokens (→ source patch scores)
    return Ar.mean(1)                           # (B, N_P)


def plot_attention_rollout(
    model: PatchHARv2,
    loader: DataLoader,
    n_samples_per_class: int = 30,
    max_batches: int = 50,
):
    """
    Figure: (NUM_CLASSES × N_P) heatmap of mean patch attention per class.
    """
    print("[1] Attention Rollout …")
    class_scores = {k: [] for k in range(NUM_CLASSES)}

    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        patches_list, times, labels, _, _, _ = batch
        patches_list = [p.to(device).float() for p in patches_list]
        times_d      = times.to(device).float()

        scores = compute_attention_rollout(model, patches_list, times_d)
        for i, lab in enumerate(labels.numpy()):
            if len(class_scores[lab]) < n_samples_per_class:
                class_scores[lab].append(scores[i])

    # Build matrix  (K × N_P)
    K  = NUM_CLASSES
    NP = cfg.N_PATCHES
    mat = np.zeros((K, NP))
    for k in range(K):
        if class_scores[k]:
            mat[k] = np.stack(class_scores[k]).mean(0)

    # Normalise each row to [0,1] for visual clarity
    row_min = mat.min(1, keepdims=True)
    row_max = mat.max(1, keepdims=True)
    mat_n   = (mat - row_min) / (row_max - row_min + 1e-8)

    # x-axis: patch mid-point in seconds
    patch_sec = (np.arange(NP) * cfg.PATCH_LEN + cfg.PATCH_LEN / 2) / cfg.SIGNAL_RATE
    step = max(1, NP // 15)

    fig, ax = _fig(figsize=(FIG_W_DOUBLE, max(2.8, K * 0.85)))

    fig.set_constrained_layout(False)
    im = ax.imshow(mat_n, aspect="auto", cmap=CMAP_SEQ, vmin=0, vmax=1)
    ax.set_yticks(range(K))
    ax.set_yticklabels(CLASSES, fontsize=11)
    ax.set_xticks(range(0, NP, step))
    ax.set_xticklabels([f"{t:.1f}s" for t in patch_sec[::step]], fontsize=10, rotation=40)
    ax.set_xlabel("Patch position (seconds)", fontsize=10)
    ax.set_title("Attention Rollout: mean patch importance per class (row-normalised)", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    out = INTERP_DIR / "1_attention_rollout.pdf"
    _save(fig, out)


# =============================================================================
# 2. MoE Expert Specialisation
# =============================================================================
def plot_moe_analysis(outputs: dict):
    """
    Figures:
      2a. Per-class router weight heatmap (MoE-1 and MoE-2)
      2b. Router entropy per class
      2c. UMAP/t-SNE coloured by expert preference
    """
    print("[2] MoE Expert Analysis …")
    labels = outputs["labels"]
    K      = NUM_CLASSES
    E      = outputs["moe1_w"].shape[-1] if outputs["moe1_w"] is not None else 0

    if E == 0:
        print("   [SKIP] MoE weights not collected.")
        return

    for tag, moe_w in [("MoE1", outputs["moe1_w"]),
                        ("MoE2", outputs["moe2_w"])]:
        if moe_w is None:
            continue
        # moe_w : (N, N_tok, E) → mean over tokens → (N, E)
        mean_w = moe_w.mean(1)

        # ── 2a: per-class routing heatmap ─────────────────────────────
        class_mat = np.zeros((K, E))
        for k in range(K):
            mask = labels == k
            if mask.sum():
                class_mat[k] = mean_w[mask].mean(0)

        fig, ax = _fig(figsize=(FIG_W_SINGLE, max(3.5, K * 0.65)))

        fig.set_constrained_layout(False)
        sns.heatmap(class_mat, annot=True, fmt=".2f",
                    xticklabels=[f"E{i}" for i in range(E)],
                    yticklabels=CLASSES,
                    cmap="Blues", ax=ax, linewidths=0.4, linecolor="#eeeeee")
        ax.set_title(f"{tag} — Mean Routing Weight per Class", fontsize=11)
        ax.set_xlabel("Expert"); ax.set_ylabel("Class")
        out = INTERP_DIR / f"2a_{tag.lower()}_routing_heatmap.pdf"
        _save(fig, out)

        # ── 2b: per-class entropy ─────────────────────────────────────
        entropies = -np.sum(mean_w * np.log(mean_w + 1e-12), axis=-1)  # (N,)
        class_ent = [entropies[labels == k] for k in range(K)]

        fig, ax = _fig(figsize=(FIG_W_DOUBLE, 4.2))

        fig.set_constrained_layout(False)
        ax.boxplot(class_ent, labels=CLASSES, patch_artist=True,
                   boxprops=dict(facecolor="steelblue", alpha=0.6))
        ax.set_xlabel("Class"); ax.set_ylabel("Router Entropy (nats)")
        ax.set_title(f"{tag} — Router Entropy per Class\n"
                     "(low = hard routing / specialisation)", fontsize=11)
        plt.xticks(rotation=45, ha="right", fontsize=11)
        out = INTERP_DIR / f"2b_{tag.lower()}_entropy.pdf"
        _save(fig, out)

    # ── 2c: 2-D projection coloured by dominant expert (MoE1) ─────────────
    if outputs["moe1_w"] is not None:
        mean_w  = outputs["moe1_w"].mean(1)          # (N, E)
        dom_exp = mean_w.argmax(-1)                  # (N,)
        emb     = outputs["embeddings"]

        # Subsample for t-SNE speed
        MAX_VIS = 8000
        n_total = len(emb)
        if not HAS_UMAP and n_total > MAX_VIS:
            rng    = np.random.default_rng(cfg.SEED)
            idx_s  = rng.choice(n_total, MAX_VIS, replace=False)
            emb_s  = emb[idx_s]
            labs_s = labels[idx_s]
            exp_s  = dom_exp[idx_s]
        else:
            emb_s, labs_s, exp_s = emb, labels, dom_exp

        if HAS_UMAP:
            reducer = umap.UMAP(n_components=2, random_state=cfg.SEED)
            proj    = reducer.fit_transform(emb_s)
        else:
            proj = _TSNE(n_components=2, random_state=cfg.SEED,
                         n_jobs=-1).fit_transform(emb_s)

        fig, axes = _fig(1, 2, figsize=(FIG_W_DOUBLE, 5.0))

        fig.set_constrained_layout(False)
        sc0 = axes[0].scatter(proj[:, 0], proj[:, 1], c=labs_s,
                              cmap="tab20", s=4, alpha=0.6)
        axes[0].set_title("Embedding coloured by true class")
        plt.colorbar(sc0, ax=axes[0], ticks=range(K),
                     label="Class").ax.set_yticklabels(CLASSES, fontsize=10)

        cmap_exp = plt.colormaps.get_cmap("Set1").resampled(E)
        sc1 = axes[1].scatter(proj[:, 0], proj[:, 1], c=exp_s,
                              cmap=cmap_exp, s=4, alpha=0.6,
                              vmin=-0.5, vmax=E - 0.5)
        axes[1].set_title("Embedding coloured by dominant MoE-1 expert")
        plt.colorbar(sc1, ax=axes[1], ticks=range(E),
                     label="Expert").ax.set_yticklabels(
                         [f"E{i}" for i in range(E)], fontsize=11)

        plt.suptitle("MoE Expert Specialisation in Embedding Space", fontsize=12, y=1.01)
        out = INTERP_DIR / "2c_moe_embedding_projection.pdf"
        _save(fig, out)


# =============================================================================
# 3. Calibration (ECE) & Reliability Diagrams
# =============================================================================
def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(-1)
    predictions = probs.argmax(-1)
    correct     = (predictions == labels).astype(float)
    bin_edges   = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc  = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += mask.sum() / len(labels) * abs(acc - conf)
    return float(ece)


def _per_class_ece(probs: np.ndarray,
                   labels: np.ndarray,
                   n_bins: int = 15) -> dict:
    scores = {}
    for k in range(NUM_CLASSES):
        mask = labels == k
        if mask.sum() < 5:
            scores[CLASSES[k]] = float("nan")
            continue
        p_k = probs[mask][:, k]
        y_k = (labels[mask] == k).astype(float)
        # one-vs-rest calibration for class k
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            m2 = (p_k >= lo) & (p_k < hi)
            if m2.sum() == 0:
                continue
            acc  = y_k[m2].mean()
            conf = p_k[m2].mean()
            ece += m2.sum() / len(p_k) * abs(acc - conf)
        scores[CLASSES[k]] = float(ece)
    return scores


def plot_calibration(outputs: dict):
    """
    Figures:
      3a. Global reliability diagram + ECE annotation
      3b. Per-class ECE bar chart
    """
    print("[3] Calibration / ECE …")
    probs  = outputs["probs"]
    labels = outputs["labels"]

    global_ece = _ece(probs, labels)
    print(f"   Global ECE = {global_ece:.4f}")

    # ── 3a: reliability diagram ───────────────────────────────────────────
    confidences = probs.max(-1)
    frac_pos, mean_pred = calibration_curve(
        (probs.argmax(-1) == labels).astype(int),
        confidences, n_bins=15, strategy="uniform")

    fig, ax = _fig(figsize=(FIG_W_SINGLE + 0.5, 4.5))

    fig.set_constrained_layout(False)
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color="steelblue",
            lw=2, ms=6, label=f"Model  (ECE={global_ece:.3f})")
    ax.fill_between(mean_pred, mean_pred, frac_pos, alpha=0.15, color="red",
                    label="Gap (over/under-confidence)")
    ax.set_xlabel("Mean predicted confidence"); ax.set_ylabel("Fraction correct")
    ax.set_title("Reliability Diagram — Overall Calibration", fontsize=11)
    ax.legend(fontsize=9); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    out = INTERP_DIR / "3a_reliability_diagram.pdf"
    _save(fig, out)

    # ── 3b: per-class ECE ─────────────────────────────────────────────────
    pc_ece  = _per_class_ece(probs, labels)
    cls_names = list(pc_ece.keys())
    ece_vals  = [pc_ece[c] for c in cls_names]

    fig, ax = _fig(figsize=(FIG_W_DOUBLE, 4.2))

    fig.set_constrained_layout(False)
    colors = ["tomato" if v > 0.1 else "steelblue" for v in ece_vals]
    bars = ax.bar(cls_names, ece_vals, color=colors, edgecolor="k", lw=0.5)
    ax.axhline(global_ece, color="k", ls="--", lw=1.5,
               label=f"Global ECE = {global_ece:.3f}")
    ax.axhline(0.1, color="tomato", ls=":", lw=1.2, label="ECE = 0.10 threshold")
    for bar, val in zip(bars, ece_vals):
        if not math.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.002, f"{val:.3f}", ha="center",
                    va="bottom", fontsize=10)
    ax.set_ylabel("ECE"); ax.set_title("Per-Class Expected Calibration Error", fontsize=11)
    ax.legend(fontsize=9)
    plt.xticks(rotation=45, ha="right", fontsize=11)
    out = INTERP_DIR / "3b_per_class_ece.pdf"
    _save(fig, out)

    # Save ECE table
    ece_table = pd.DataFrame({
        "class": cls_names,
        "ece":   ece_vals,
    })
    ece_table.to_csv(INTERP_DIR / "3_ece_table.csv", index=False)


# =============================================================================
# 4. Dual-Domain Embedding Gate Analysis  (C1)
# =============================================================================
def plot_gate_analysis(model: PatchHARv2, outputs: dict):
    """
    Figures:
      4a. Gate weight histogram (pooled over fine/mid/coarse embedders)
      4b. Per-class box-plot of mean gate value (time-domain bias)
      4c. Per-class F1 drop when frequency branch is zeroed
    """
    print("[4] Dual-Domain Gate Analysis …")
    if not CC.C1_DUAL_DOMAIN_EMBEDDING:
        print("   [SKIP] C1 not active.")
        return

    # ── 4a: gate histogram ────────────────────────────────────────────────
    embedders = []
    if CC.C4_MULTISCALE_PATCHING:
        embedders = [model.hier_embed.embed_fine,
                     model.hier_embed.embed_mid,
                     model.hier_embed.embed_coarse]
    else:
        embedders = [model.single_embed]

    all_gates = []
    for emb in embedders:
        if hasattr(emb, "gate_w"):
            g = torch.sigmoid(emb.gate_w).detach().cpu().numpy()
            all_gates.append(g)

    if not all_gates:
        print("   [SKIP] No gate_w found.")
        return

    fig, ax = _fig(figsize=(FIG_W_15COL, 4.2))

    fig.set_constrained_layout(False)
    colors = ["steelblue", "darkorange", "forestgreen"]
    labels_emb = (["Fine (25)", "Mid (50)", "Coarse (100)"]
                  if len(all_gates) == 3 else [f"Scale {i}" for i in range(len(all_gates))])
    for g, col, lbl in zip(all_gates, colors, labels_emb):
        ax.hist(g, bins=40, alpha=0.55, color=col, label=lbl, density=True)
    ax.axvline(0.5, color="k", ls="--", lw=1.5, label="g=0.5 (equal weight)")
    ax.set_xlabel("Gate value σ(gate_w)  →  1 = time-domain, 0 = freq-domain")
    ax.set_ylabel("Density")
    ax.set_title("Dual-Domain Gate Distribution per Patch Granularity", fontsize=11)
    ax.legend(fontsize=9)
    out = INTERP_DIR / "4a_gate_histogram.pdf"
    _save(fig, out)

    # ── 4b: per-sample mean gate → per-class box-plot ─────────────────────
    # Compute per-sample embedding-gate mean using hooks
    gate_samples = []
    hook_handles  = []

    def _make_gate_hook(buf: list):
        def _h(mod, inp, out):
            g = torch.sigmoid(mod.gate_w).mean().item()
            B = inp[0].shape[0]
            buf.extend([g] * B)
        return _h

    for emb in embedders:
        if hasattr(emb, "gate_w"):
            hook_handles.append(
                emb.register_forward_hook(_make_gate_hook(gate_samples)))

    # We only need one forward pass per sample → iterate test loader once
    _, test_dl = make_loader(test_pids, shuffle=False, is_train=False)
    model.eval()
    tmp_labels = []
    with torch.no_grad():
        for batch in test_dl:
            patches_list, times, labels, _, _, _ = batch
            patches_list = [p.to(device).float() for p in patches_list]
            model(patches_list, times.to(device).float())
            tmp_labels.extend(labels.numpy().tolist())

    for h in hook_handles:
        h.remove()

    if gate_samples:
        # One gate value per sample per embedder → average over embedders
        n_emb = len(embedders)
        n_smp = len(tmp_labels)
        if len(gate_samples) == n_smp * n_emb:
            gate_arr = np.array(gate_samples).reshape(n_smp, n_emb).mean(1)
        else:
            gate_arr = np.array(gate_samples[:n_smp])

        lbl_arr = np.array(tmp_labels)
        class_gates = [gate_arr[lbl_arr == k] for k in range(NUM_CLASSES)]

        fig, ax = _fig(figsize=(FIG_W_DOUBLE, 4.2))

        fig.set_constrained_layout(False)
        bp = ax.boxplot(class_gates, labels=CLASSES, patch_artist=True,
                        boxprops=dict(alpha=0.6))
        colors_bp = plt.colormaps.get_cmap("tab20").resampled(NUM_CLASSES)(
                        np.linspace(0, 1, NUM_CLASSES))
        for patch, c in zip(bp["boxes"], colors_bp):
            patch.set_facecolor(c)
        ax.axhline(0.5, color="k", ls="--", lw=1.2)
        ax.set_ylabel("Mean gate value  (1=time, 0=freq)")
        ax.set_title("Per-Class Time/Frequency Domain Reliance (C1 gate)", fontsize=11)
        plt.xticks(rotation=45, ha="right", fontsize=11)
        out = INTERP_DIR / "4b_per_class_gate_boxplot.pdf"
        _save(fig, out)


# =============================================================================
# 5. Prototype Memory UMAP + Confusion by Prototype Proximity  (C7)
# =============================================================================
def plot_prototype_analysis(model: PatchHARv2, outputs: dict):
    """
    Figures:
      5a. UMAP of test embeddings + prototype centroids
      5b. Inter-prototype cosine similarity matrix
      5c. For mis-classified samples: sim(z, proto_pred) vs sim(z, proto_true)
    """
    print("[5] Prototype Memory Analysis …")
    if not CC.C7_PROTOTYPE_MEMORY or not model.proto_filled:
        print("   [SKIP] C7 not active or prototypes not yet filled.")
        return

    emb   = outputs["embeddings"]               # (N, D)
    labels = outputs["labels"]
    preds  = outputs["preds"]
    protos = model.prototypes.detach().cpu().numpy()   # (K, D)

    K = NUM_CLASSES

    # ── 5a: 2-D projection (subsample for speed with t-SNE) ───────────────
    MAX_VIS = 8000   # cap for t-SNE; UMAP handles larger sets fine
    n_total = len(emb)
    if not HAS_UMAP and n_total > MAX_VIS:
        rng   = np.random.default_rng(cfg.SEED)
        idx_s = rng.choice(n_total, MAX_VIS, replace=False)
        emb_s   = emb[idx_s]
        labels_s = labels[idx_s]
        print(f"   [INFO] Subsampled {MAX_VIS}/{n_total} points for t-SNE.")
    else:
        emb_s    = emb
        labels_s = labels

    all_pts = np.vstack([emb_s, protos])           # (N_sub+K, D)
    if HAS_UMAP:
        proj = umap.UMAP(n_components=2, random_state=cfg.SEED).fit_transform(all_pts)
    else:
        proj = _TSNE(n_components=2, random_state=cfg.SEED,
                     n_jobs=-1).fit_transform(all_pts)

    emb_2d   = proj[:len(emb_s)]
    proto_2d = proj[len(emb_s):]

    fig, ax = _fig(figsize=(FIG_W_15COL, 5.5))

    fig.set_constrained_layout(False)
    cmap20 = plt.colormaps.get_cmap("tab20").resampled(K)
    for k in range(K):
        mask = labels_s == k
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=[cmap20(k)], s=5, alpha=0.4, label=CLASSES[k])

    for k in range(K):
        ax.scatter(*proto_2d[k], marker="*", s=300, edgecolors="k",
                   c=[cmap20(k)], linewidths=0.8, zorder=10)
        ax.annotate(CLASSES[k], proto_2d[k],
                    fontsize=10, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")

    ax.set_title("Test Embeddings + Class Prototypes (★) in 2-D", fontsize=11)
    ax.axis("off")
    out = INTERP_DIR / "5a_prototype_umap.pdf"
    _save(fig, out)

    # ── 5b: inter-prototype cosine similarity ─────────────────────────────
    p_norm = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
    sim_mat = p_norm @ p_norm.T

    fig, ax = _fig(figsize=(FIG_W_SINGLE + 0.5, max(4.0, K * 0.70)))

    fig.set_constrained_layout(False)
    sns.heatmap(sim_mat, annot=True, fmt=".2f",
                xticklabels=CLASSES, yticklabels=CLASSES,
                cmap="RdYlGn_r", center=0, vmin=-1, vmax=1,
                ax=ax, linewidths=0.3, annot_kws={"size": 10})
    ax.set_title("Inter-Prototype Cosine Similarity\n"
                 "(high similarity → likely confused classes)", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0)
    out = INTERP_DIR / "5b_inter_prototype_cosine.pdf"
    _save(fig, out)

    # ── 5c: prototype pulling wrong predictions ────────────────────────────
    emb_t = torch.from_numpy(emb)
    pro_t = torch.from_numpy(protos)
    cos   = F.normalize(emb_t, dim=-1) @ F.normalize(pro_t, dim=-1).T
    cos_np = cos.numpy()   # (N, K)

    wrong = preds != labels
    if wrong.sum() == 0:
        print("   No mis-classifications on this split.")
        return

    sim_pred = cos_np[np.arange(len(labels)), preds][wrong]
    sim_true = cos_np[np.arange(len(labels)), labels][wrong]

    fig, ax = _fig(figsize=(FIG_W_SINGLE + 0.5, 4.5))

    fig.set_constrained_layout(False)
    ax.scatter(sim_true, sim_pred, s=8, alpha=0.35, color="tomato")
    lim = max(abs(sim_true).max(), abs(sim_pred).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.2, label="sim_pred = sim_true")
    ax.set_xlabel("cos(z, prototype_true_class)")
    ax.set_ylabel("cos(z, prototype_predicted_class)")
    ax.set_title("Prototype Confusion Analysis\n"
                 "(points above diagonal: prototype pulled model to wrong class)", fontsize=10)
    frac_above = (sim_pred > sim_true).mean()
    ax.text(0.05, 0.95, f"Proto-pulled errors: {frac_above:.1%}",
            transform=ax.transAxes, fontsize=10,
            va="top", bbox=dict(boxstyle="round", fc="w", alpha=0.7))
    ax.legend(fontsize=9)
    out = INTERP_DIR / "5c_prototype_confusion.pdf"
    _save(fig, out)


# =============================================================================
# 6. Per-Class Error Diagnosis
# =============================================================================
def plot_confusion_matrix(outputs: dict):
    """6a: Row-normalised confusion matrix."""
    print("[6a] Confusion Matrix …")
    cm = confusion_matrix(outputs["labels"], outputs["preds"])
    cm_norm = cm.astype(float) / cm.sum(1, keepdims=True).clip(1)

    fig, ax = _fig(figsize=(FIG_W_DOUBLE * 0.72, max(5.0, NUM_CLASSES * 0.80)))

    fig.set_constrained_layout(False)
    sns.heatmap(cm_norm, annot=True, fmt=".2f",
                xticklabels=CLASSES, yticklabels=CLASSES,
                cmap="Blues", linewidths=0.3, linecolor="#eeeeee",
                    ax=ax, annot_kws={"size": 10})
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Row-Normalised Confusion Matrix (recall on diagonal)", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.yticks(rotation=0)
    out = INTERP_DIR / "6a_confusion_matrix.pdf"
    _save(fig, out)


def plot_circadian_errors(outputs: dict):
    """6b: Error rate by hour-of-day, per class."""
    print("[6b] Circadian Error Analysis …")
    hours  = outputs["hours"]
    labels = outputs["labels"]
    preds  = outputs["preds"]

    # select top-4 worst macro-F1 classes
    f1s = f1_score(labels, preds, average=None, zero_division=0)
    worst_k = np.argsort(f1s)[:min(4, NUM_CLASSES)]

    fig, axes = _fig(1, len(worst_k),
                             figsize=(FIG_W_DOUBLE, 4.8), sharey=False)
    if len(worst_k) == 1:
        axes = [axes]

    for ax, k in zip(axes, worst_k):
        mask  = labels == k
        h_k   = hours[mask]
        err_k = (preds[mask] != labels[mask]).astype(float)

        # bin by hour
        hourly_err = np.full(24, np.nan)
        for h in range(24):
            m = h_k == h
            if m.sum() >= 3:
                hourly_err[h] = err_k[m].mean()

        ax.bar(range(24), np.nan_to_num(hourly_err, nan=0),
               color="steelblue", alpha=0.7, edgecolor="k", lw=0.4)
        ax.axhline(err_k.mean(), color="tomato", ls="--", lw=1.5,
                   label=f"Mean err = {err_k.mean():.2f}")
        ax.fill_betweenx([0, 1], 22, 24, alpha=0.08, color="navy", label="Night")
        ax.fill_betweenx([0, 1],  0,  6, alpha=0.08, color="navy")
        ax.set_xlim(0, 23); ax.set_ylim(0, 1)
        ax.set_xlabel("Hour of day"); ax.set_ylabel("Error rate")
        ax.set_title(f"Class: '{CLASSES[k]}'\n(F1={f1s[k]:.3f})", fontsize=10)
        ax.legend(fontsize=11)

    plt.suptitle("Circadian Error Analysis — Worst Performing Classes", fontsize=12, y=1.01)
    out = INTERP_DIR / "6b_circadian_errors.pdf"
    _save(fig, out)


def plot_signal_difficulty(outputs: dict):
    """6c: Per-sample signal entropy vs per-class F1."""
    print("[6c] Signal Difficulty Analysis …")
    raw_segs = outputs["raw_segs"]   # (N, T, C)
    labels   = outputs["labels"]
    preds    = outputs["preds"]

    # Variance-based difficulty proxy
    difficulty = raw_segs.var(axis=(1, 2))   # (N,)
    q33, q66   = np.percentile(difficulty, [33, 66])

    def _f1_in_bin(mask):
        return f1_score(labels[mask], preds[mask],
                        average="macro", zero_division=0)

    bins   = ["Low\n(quiescent)", "Mid\n(moderate)", "High\n(dynamic)"]
    masks  = [
        difficulty <= q33,
        (difficulty > q33) & (difficulty <= q66),
        difficulty > q66,
    ]
    f1_bins = [_f1_in_bin(m) if m.sum() > 0 else 0.0 for m in masks]

    fig, axes = _fig(1, 2, figsize=(FIG_W_DOUBLE, 4.5))

    fig.set_constrained_layout(False)

    # bar chart
    axes[0].bar(bins, f1_bins, color=["#4dac26", "#f1b6da", "#d01c8b"],
                edgecolor="k", lw=0.7)
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("Macro F1")
    axes[0].set_title("Macro F1 by Signal Variance Tercile", fontsize=11)
    for b, v in zip(range(3), f1_bins):
        axes[0].text(b, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

    # per-class scatter: mean difficulty vs recall
    cls_diff = [difficulty[labels == k].mean() if (labels == k).sum() else 0
                for k in range(NUM_CLASSES)]
    recall   = [confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))[k, k]
                / max((labels == k).sum(), 1)
                for k in range(NUM_CLASSES)]

    axes[1].scatter(cls_diff, recall, s=70, color="steelblue", zorder=5)
    for k, (x, y) in enumerate(zip(cls_diff, recall)):
        axes[1].annotate(CLASSES[k], (x, y), fontsize=10,
                         xytext=(4, 2), textcoords="offset points")
    axes[1].set_xlabel("Mean signal variance (difficulty proxy)")
    axes[1].set_ylabel("Per-class recall")
    axes[1].set_title("Per-Class Recall vs Signal Difficulty", fontsize=11)

    plt.suptitle("Signal Difficulty Analysis", fontsize=12, y=1.01)
    out = INTERP_DIR / "6c_signal_difficulty.pdf"
    _save(fig, out)


def plot_gradcam_worst(
    model: PatchHARv2,
    loader: DataLoader,
    n_pairs: int = 3,
    n_per_pair: int = 3,
):
    """
    6d: Grad-CAM on patch embeddings for most-confused class pairs.
    Saves a grid of (true class, predicted class, patch saliency) triplets.
    """
    print("[6d] Grad-CAM on worst mis-predictions …")

    # ── step 1: find most confused pairs from full test set ────────────────
    all_logits, all_labels, all_probs = [], [], []
    for batch in loader:
        patches_list, times, labels, _, _, _ = batch
        patches_list = [p.to(device).float() for p in patches_list]
        with torch.no_grad():
            logits, _ = model(patches_list, times.to(device).float())
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    logits_all = torch.cat(all_logits)
    labels_all = torch.cat(all_labels).numpy()
    preds_all  = logits_all.argmax(-1).numpy()
    cm_raw     = confusion_matrix(labels_all, preds_all,
                                  labels=list(range(NUM_CLASSES)))
    np.fill_diagonal(cm_raw, 0)
    flat_idx   = cm_raw.flatten().argsort()[::-1]
    top_pairs  = [(i // NUM_CLASSES, i % NUM_CLASSES)
                  for i in flat_idx[:n_pairs]]

    # ── step 2: Grad-CAM for each pair ────────────────────────────────────
    # Target: embedding tensor after patch projection (before transformer)
    # We register a hook on the patch embedder output

    fig, axes = _fig(n_pairs, n_per_pair,
                             figsize=(FIG_W_DOUBLE, 3.2 * n_pairs))
    if n_pairs == 1:
        axes = axes[None]
    if n_per_pair == 1:
        axes = axes[:, None]

    for pi, (true_k, pred_k) in enumerate(top_pairs):
        # Gather candidate batches for this pair
        samples = []
        for batch in loader:
            if len(samples) >= n_per_pair:
                break
            patches_list, times, labels, _, _, _ = batch
            patches_list_dev = [p.to(device).float() for p in patches_list]
            times_dev        = times.to(device).float()

            for bi in range(len(labels)):
                if labels[bi].item() != true_k:
                    continue
                # quick prediction check
                with torch.no_grad():
                    logits_b, _ = model(
                        [p[bi:bi+1] for p in patches_list_dev],
                        times_dev[bi:bi+1])
                if logits_b.argmax(-1).item() != pred_k:
                    continue
                samples.append(([p[bi:bi+1] for p in patches_list_dev],
                                 times_dev[bi:bi+1]))
                if len(samples) >= n_per_pair:
                    break

        for si, (pl, t) in enumerate(samples):
            if si >= n_per_pair:
                break
            ax = axes[pi][si]

            # Grad-CAM: gradient of predicted-class score w.r.t. embed_output
            emb_act = []
            def _fwd_hook(mod, inp, out):
                emb_act.clear()
                out.retain_grad()
                emb_act.append(out)

            handle = (model.hier_embed.embed_fine
                      if CC.C4_MULTISCALE_PATCHING
                      else model.single_embed).register_forward_hook(_fwd_hook)

            model.zero_grad()
            logits_b, _ = model(pl, t)
            score        = logits_b[0, pred_k]
            score.backward()

            handle.remove()

            if emb_act and emb_act[0].grad is not None:
                grad    = emb_act[0].grad[0].detach().cpu().numpy()  # (NP, D)
                act     = emb_act[0][0].detach().cpu().numpy()       # (NP, D)
                cam     = (grad * act).mean(-1)                      # (NP,)
                cam     = np.maximum(cam, 0)
                cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            else:
                cam = np.zeros(cfg.N_PATCHES)

            NP  = len(cam)
            sec = np.arange(NP) * cfg.PATCH_LEN / cfg.SIGNAL_RATE

            ax.bar(sec, cam, width=cfg.PATCH_LEN / cfg.SIGNAL_RATE * 0.9,
                   color=plt.cm.YlOrRd(cam), edgecolor="none", align="edge")
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Time (s)", fontsize=11)
            ax.set_ylabel("Saliency", fontsize=11)
            ax.set_title(
                f"True: {CLASSES[true_k]}\nPred: {CLASSES[pred_k]}",
                fontsize=11, color="tomato")
            ax.tick_params(labelsize=7)

        # blank unused subplots
        for si in range(len(samples), n_per_pair):
            axes[pi][si].axis("off")

    plt.suptitle("Grad-CAM on Worst Mis-classifications\n"
                 "(saliency = which patches drove the wrong prediction)",
                 fontsize=11)
    out = INTERP_DIR / "6d_gradcam_worst_errors.pdf"
    _save(fig, out)


# =============================================================================
# 7. Reconstruction Quality Analysis  (C10)
# =============================================================================
def plot_reconstruction_analysis(
    model: PatchHARv2,
    loader: DataLoader,
    outputs: dict,
):
    """
    Figures:
      7a. Per-class reconstruction MSE bar chart
      7b. Scatter: recon_mse vs classification confidence
      7c. Qualitative: raw vs reconstructed signal for 2 classes
    """
    print("[7] Reconstruction Analysis …")
    if not CC.C10_RECON_AUX_GRAD_SURGERY:
        print("   [SKIP] C10 not active.")
        return

    # ── collect recon in train mode (recon only fires during training) ────
    # We temporarily switch to train mode but disable grad updates
    model.train()
    all_recon, all_segs, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            patches_list, times, labels, _, _, raw_segs = batch
            patches_list = [p.to(device).float() for p in patches_list]
            times_d      = times.to(device).float()
            _, recon     = model(patches_list, times_d)
            if recon is not None:
                all_recon.append(recon.cpu())
                all_segs.append(raw_segs)
                all_labels.append(labels)

    model.eval()

    if not all_recon:
        print("   [SKIP] No reconstruction output collected.")
        return

    recon_t  = torch.cat(all_recon).numpy()
    segs_t   = torch.cat(all_segs).numpy()
    labels_t = torch.cat(all_labels).numpy()

    # Per-sample MSE
    NP, PL, C = cfg.N_PATCHES, cfg.PATCH_LEN, cfg.CHANNELS
    B = recon_t.shape[0]
    target = (segs_t[:, :NP*PL, :]
              .reshape(B, NP, PL, C)
              .transpose(0, 1, 3, 2)
              .reshape(B, NP * C * PL))
    mse_per_sample = ((recon_t - target) ** 2).mean(1)  # (N,)

    # ── 7a: per-class MSE ─────────────────────────────────────────────────
    class_mse = [mse_per_sample[labels_t == k].mean()
                 if (labels_t == k).sum() else 0.0
                 for k in range(NUM_CLASSES)]

    fig, ax = _fig(figsize=(FIG_W_DOUBLE, 4.2))

    fig.set_constrained_layout(False)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, NUM_CLASSES))
    ax.bar(CLASSES, class_mse, color=colors, edgecolor="k", lw=0.5)
    ax.set_ylabel("Mean Reconstruction MSE")
    ax.set_title("Per-Class Reconstruction MSE\n"
                 "(low = model learned signal structure of that class)", fontsize=11)
    ax.axhline(np.mean(class_mse), color="tomato", ls="--",
               lw=1.5, label=f"Global mean = {np.mean(class_mse):.4f}")
    ax.legend(fontsize=9)
    plt.xticks(rotation=45, ha="right", fontsize=11)
    out = INTERP_DIR / "7a_per_class_recon_mse.pdf"
    _save(fig, out)

    # ── 7b: recon MSE vs classification confidence ────────────────────────
    conf = outputs["probs"].max(-1)[:len(mse_per_sample)]

    fig, ax = _fig(figsize=(FIG_W_SINGLE + 0.5, 4.5))

    fig.set_constrained_layout(False)
    sc = ax.scatter(mse_per_sample, conf, c=labels_t[:len(mse_per_sample)],
                    cmap="tab20", s=6, alpha=0.5)
    from scipy.stats import pearsonr
    r, p = pearsonr(mse_per_sample, conf)
    ax.set_xlabel("Reconstruction MSE"); ax.set_ylabel("Prediction confidence")
    ax.set_title(f"Reconstruction Quality vs Prediction Confidence\n"
                 f"Pearson r = {r:.3f}  (p={p:.3e})", fontsize=11)
    plt.colorbar(sc, ax=ax, label="True class")
    out = INTERP_DIR / "7b_recon_vs_confidence.pdf"
    _save(fig, out)

    # ── 7c: qualitative raw vs reconstructed ──────────────────────────────
    show_classes = list(range(min(3, NUM_CLASSES)))
    fig, axes    = _fig(len(show_classes), cfg.CHANNELS,
                                figsize=(FIG_W_DOUBLE, 3.0 * len(show_classes)))
    if len(show_classes) == 1:
        axes = axes[None]
    if cfg.CHANNELS == 1:
        axes = axes[:, None]

    for ri, k in enumerate(show_classes):
        mask = labels_t == k
        if mask.sum() == 0:
            continue
        idx = np.where(mask)[0][0]
        raw_sig = segs_t[idx, :NP*PL, :]                           # (NP*PL, C)
        rec_arr = (recon_t[idx]
                   .reshape(NP, C, PL)
                   .transpose(0, 2, 1)
                   .reshape(NP * PL, C))

        t_axis = np.arange(NP * PL) / cfg.SIGNAL_RATE

        for ci in range(cfg.CHANNELS):
            ax = axes[ri][ci]
            ax.plot(t_axis, raw_sig[:, ci], color="steelblue",
                    lw=1.0, label="Raw")
            ax.plot(t_axis, rec_arr[:, ci], color="tomato",
                    lw=1.0, ls="--", label="Reconstructed", alpha=0.85)
            ax.set_title(f"{CLASSES[k]} | CH{ci}", fontsize=9)
            ax.set_xlabel("Time (s)", fontsize=11)
            if ci == 0:
                ax.legend(fontsize=10)
            ax.tick_params(labelsize=7)

    plt.suptitle("Raw vs Reconstructed Accelerometer Signal  (C10)", fontsize=12, y=1.01)
    out = INTERP_DIR / "7c_raw_vs_reconstructed.pdf"
    _save(fig, out)


# =============================================================================
# 8. PCGrad Gradient Conflict Logging  (C10)
# =============================================================================
def train_with_gradient_conflict_logging(
    model: PatchHARv2,
    loader: DataLoader,
    n_batches: int = 100,
) -> None:
    """
    8: For n_batches, compute cos(∇_cls, ∇_recon) with and without
    PCGrad projection, and plot the resulting histograms.
    """
    print("[8] PCGrad Gradient Conflict Analysis …")
    if not CC.C10_RECON_AUX_GRAD_SURGERY:
        print("   [SKIP] C10 not active.")
        return

    from patchhar_v2 import (SmoothCE, recon_loss, compute_class_weights,
                              _pcgrad_surgery)

    # Minimal setup
    model_tmp = copy.deepcopy(model).to(device).train()
    opt       = torch.optim.AdamW(model_tmp.parameters(), lr=1e-4)
    train_ds, _ = make_loader(train_pids, shuffle=True, is_train=True)
    cw          = compute_class_weights(train_ds).to(device)
    criterion   = SmoothCE(weight=cw, smoothing=0.0)

    cosines_raw  = []   # cosine BEFORE surgery
    cosines_proj = []   # cosine AFTER surgery
    frac_conflict_raw  = []
    frac_conflict_proj = []

    params = [p for p in model_tmp.parameters() if p.requires_grad]

    for bi, batch in enumerate(loader):
        if bi >= n_batches:
            break
        patches_list, times, labels, _, _, raw_segs = batch
        patches_list = [p.to(device).float() for p in patches_list]
        times_d   = times.to(device).float()
        labels_d  = labels.to(device).view(-1)
        raw_segs_d = raw_segs.to(device).float()

        # ── classification gradient ───────────────────────────────────────
        opt.zero_grad()
        logits, _ = model_tmp(patches_list, times_d)
        cls_loss   = criterion(logits, labels_d)
        cls_loss.backward(retain_graph=True)
        cls_grads  = torch.cat([
            p.grad.view(-1) if p.grad is not None else torch.zeros(p.numel(), device=device)
            for p in params
        ]).clone()

        # ── reconstruction gradient ───────────────────────────────────────
        opt.zero_grad()
        model_tmp.train()
        _, recon   = model_tmp(patches_list, times_d)
        if recon is None:
            continue
        rl         = recon_loss(recon, raw_segs_d)
        rl.backward()
        aux_grads  = torch.cat([
            p.grad.view(-1) if p.grad is not None else torch.zeros(p.numel(), device=device)
            for p in params
        ]).clone()

        # Cosine before surgery
        norm_cls  = cls_grads.norm()
        norm_aux  = aux_grads.norm()
        cos_raw   = (cls_grads * aux_grads).sum() / (norm_cls * norm_aux + 1e-12)
        cosines_raw.append(cos_raw.item())

        # Apply PCGrad surgery
        aux_proj  = _pcgrad_surgery(cls_grads, aux_grads)
        cos_proj  = (cls_grads * aux_proj).sum() / (norm_cls * (aux_proj.norm() + 1e-12) + 1e-12)
        cosines_proj.append(cos_proj.item())

        opt.zero_grad()

    del model_tmp

    cosines_raw  = np.array(cosines_raw)
    cosines_proj = np.array(cosines_proj)

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = _fig(1, 2, figsize=(FIG_W_DOUBLE, 4.5))
    fig.set_constrained_layout(False)

    axes[0].hist(cosines_raw,  bins=30, color="tomato",    alpha=0.7,
                 edgecolor="k", lw=0.4, label="Before PCGrad")
    axes[0].hist(cosines_proj, bins=30, color="steelblue", alpha=0.7,
                 edgecolor="k", lw=0.4, label="After PCGrad")
    axes[0].axvline(0, color="k", ls="--", lw=1.5)
    axes[0].set_xlabel("cos(∇_cls, ∇_recon)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Gradient Conflict Cosine Distribution", fontsize=11)
    axes[0].legend(fontsize=9)

    # Fraction of batches with conflict per epoch-window
    window = max(1, len(cosines_raw) // 10)
    epochs_raw  = [np.mean(cosines_raw[i:i+window]  < 0)
                   for i in range(0, len(cosines_raw),  window)]
    epochs_proj = [np.mean(cosines_proj[i:i+window] < 0)
                   for i in range(0, len(cosines_proj), window)]

    axes[1].plot(epochs_raw,  "o-", color="tomato",    label="Before PCGrad")
    axes[1].plot(epochs_proj, "s-", color="steelblue", label="After PCGrad")
    axes[1].set_xlabel(f"Batch window (every {window} batches)")
    axes[1].set_ylabel("Fraction of conflicting gradients")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("PCGrad: Fraction of Gradient Conflicts Over Training", fontsize=11)
    axes[1].legend(fontsize=9)

    plt.suptitle("PCGrad Gradient Surgery Analysis  (C10)", fontsize=12, y=1.01)
    out = INTERP_DIR / "8_pcgrad_conflict.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    stats = {
        "conflict_rate_before": float((cosines_raw  < 0).mean()),
        "conflict_rate_after":  float((cosines_proj < 0).mean()),
        "mean_cosine_before":   float(cosines_raw.mean()),
        "mean_cosine_after":    float(cosines_proj.mean()),
    }
    print(f"   Conflict rate  before={stats['conflict_rate_before']:.2%}  "
          f"after={stats['conflict_rate_after']:.2%}")
    (INTERP_DIR / "8_pcgrad_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"   Saved → {out}")


# =============================================================================
# 9. Multi-Scale Patch Ablation  (C4)
# =============================================================================
def plot_multiscale_ablation(model: PatchHARv2, loader: DataLoader):
    """
    9: Zero-out each granularity broadcast, measure per-class F1 drop.
    Figure: grouped bar chart — full model vs -mid vs -coarse vs -fine.
    """
    print("[9] Multi-Scale Patch Ablation …")
    if not CC.C4_MULTISCALE_PATCHING:
        print("   [SKIP] C4 not active.")
        return

    @torch.no_grad()
    def _eval_with_ablation(zero_scale: str | None) -> np.ndarray:
        """Returns per-class F1 array."""
        preds_, labels_ = [], []
        for batch in loader:
            patches_list, times, labels, _, _, _ = batch
            patches_list = [p.to(device).float() for p in patches_list]
            times_d      = times.to(device).float()

            # Embed normally, then zero out one scale's contribution
            p_fine, p_mid, p_coarse = patches_list

            e_fine   = model.hier_embed.embed_fine(p_fine)
            e_mid    = model.hier_embed.embed_mid(p_mid)
            e_coarse = model.hier_embed.embed_coarse(p_coarse)

            rf = model.hier_embed.repeat_mid
            rc = model.hier_embed.repeat_coarse
            e_mid_b   = e_mid.repeat_interleave(rf, dim=1)
            e_coarse_b = e_coarse.repeat_interleave(rc, dim=1)

            if zero_scale == "mid":
                e_mid_b    = torch.zeros_like(e_mid_b)
            elif zero_scale == "coarse":
                e_coarse_b = torch.zeros_like(e_coarse_b)
            elif zero_scale == "fine":
                e_fine     = torch.zeros_like(e_fine)

            x = e_fine + e_mid_b + e_coarse_b

            # Remaining forward: circ bias → transformer → head
            if CC.C3_CIRCADIAN_BIAS:
                x = x + model.circ_bias(times_d)
            else:
                x = x + model.time_emb(times_d).unsqueeze(1)
            x = model.input_norm(x)
            hiddens = []
            for layer in model.delta_layers:
                x = layer(x); hiddens.append(x)
            if CC.C2_CALANET_SKIP_AGG:
                x = x + model.skip_agg(hiddens)
            x = x + model.moe1(x)
            x = model.attn(x, model.freqs)
            x = x + model.moe2(x)
            z = x.mean(1)
            logits = model.head(z)
            preds_.extend(logits.argmax(1).cpu().numpy())
            labels_.extend(labels.numpy())

        return f1_score(np.array(labels_), np.array(preds_),
                        average=None, zero_division=0)

    f1_full   = _eval_with_ablation(None)
    f1_no_mid = _eval_with_ablation("mid")
    f1_no_crs = _eval_with_ablation("coarse")
    f1_no_fin = _eval_with_ablation("fine")

    drop_mid    = f1_full - f1_no_mid
    drop_coarse = f1_full - f1_no_crs
    drop_fine   = f1_full - f1_no_fin

    x    = np.arange(NUM_CLASSES)
    w    = 0.22
    fig, axes = _fig(2, 1, figsize=(FIG_W_DOUBLE, 8.5))
    fig.set_constrained_layout(False)

    # Absolute F1
    axes[0].bar(x - w,   f1_full,   w, label="Full model",     color="#4dac26", alpha=0.8)
    axes[0].bar(x,       f1_no_mid, w, label="−Mid scale",     color="#f1b6da", alpha=0.8)
    axes[0].bar(x + w,   f1_no_crs, w, label="−Coarse scale",  color="#d01c8b", alpha=0.8)
    axes[0].bar(x + 2*w, f1_no_fin, w, label="−Fine scale",    color="#b8e186", alpha=0.8)
    axes[0].set_xticks(x + w / 2)
    axes[0].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=11)
    axes[0].set_ylabel("Per-class F1")
    axes[0].set_title("C4 Multi-Scale Ablation: Per-Class F1 (absolute)", fontsize=11)
    axes[0].legend(fontsize=9); axes[0].set_ylim(0, 1.05)

    # F1 drop
    axes[1].bar(x - w/2,  drop_mid,    w, label="Drop (−Mid)",    color="#f1b6da", alpha=0.9, edgecolor="k", lw=0.4)
    axes[1].bar(x + w/2,  drop_coarse, w, label="Drop (−Coarse)", color="#d01c8b", alpha=0.9, edgecolor="k", lw=0.4)
    axes[1].bar(x + 3*w/2,drop_fine,   w, label="Drop (−Fine)",   color="#b8e186", alpha=0.9, edgecolor="k", lw=0.4)
    axes[1].axhline(0, color="k", lw=1.0)
    axes[1].set_xticks(x + w / 2)
    axes[1].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=11)
    axes[1].set_ylabel("F1 drop (full − ablated)")
    axes[1].set_title("F1 Drop per Scale Removal\n(positive = that scale helps this class)", fontsize=11)
    axes[1].legend(fontsize=9)
    out = INTERP_DIR / "9_multiscale_ablation.pdf"
    _save(fig, out)

    # Save numeric table
    df_abl = pd.DataFrame({
        "class":         CLASSES,
        "f1_full":       f1_full,
        "f1_no_mid":     f1_no_mid,
        "f1_no_coarse":  f1_no_crs,
        "f1_no_fine":    f1_no_fin,
        "drop_mid":      drop_mid,
        "drop_coarse":   drop_coarse,
        "drop_fine":     drop_fine,
    })
    df_abl.to_csv(INTERP_DIR / "9_multiscale_ablation.csv", index=False)
    print(f"   Numeric table saved.")


# =============================================================================
# Main  — run all sections
# =============================================================================
# CLI usage examples:
#   python patchhar_interpretability.py              # run everything
#   python patchhar_interpretability.py --from 2     # resume from section 2
#   python patchhar_interpretability.py --only 3 6   # run only sections 3 and 6
#   python patchhar_interpretability.py --skip 8 9   # skip sections 8 and 9
# =============================================================================

import argparse as _argparse
import time     as _time
import traceback as _tb


def _parse_interp_args():
    p = _argparse.ArgumentParser(add_help=False)
    p.add_argument("--from",   dest="from_sec", type=int, default=1,
                   metavar="N", help="Resume from section N (skip 1..N-1)")
    p.add_argument("--only",   dest="only",  nargs="+", type=int, default=None,
                   metavar="N", help="Run only these section numbers")
    p.add_argument("--skip",   dest="skip",  nargs="+", type=int, default=None,
                   metavar="N", help="Skip these section numbers")
    p.add_argument("--no-collect", dest="no_collect", action="store_true",
                   help="Load cached outputs.npz instead of re-collecting")
    args, _ = p.parse_known_args()
    return args


def _should_run(sec: int, args) -> bool:
    """Return True if this section number should execute."""
    if args.only is not None:
        return sec in args.only
    if args.skip is not None and sec in args.skip:
        return False
    return sec >= args.from_sec


def _run_section(sec: int, name: str, fn, *fn_args, **fn_kwargs):
    """Execute fn with pretty header, timing, and isolated error handling."""
    bar = "─" * 68
    print(f"\n{bar}")
    print(f"  §{sec}  {name}")
    print(bar)
    t0 = _time.time()
    try:
        fn(*fn_args, **fn_kwargs)
        elapsed = _time.time() - t0
        print(f"  ✓  §{sec} done in {elapsed:.1f}s")
        return True
    except Exception as exc:
        elapsed = _time.time() - t0
        print(f"\n  ✗  §{sec} FAILED after {elapsed:.1f}s — {exc}")
        print("     Full traceback:")
        _tb.print_exc()
        print(f"     Continuing with remaining sections …\n")
        return False


# ── Cached outputs helper ─────────────────────────────────────────────────────
_OUTPUTS_CACHE = INTERP_DIR / "_outputs_cache.npz"

def _save_outputs(outputs: dict):
    """Persist collected outputs to disk so --no-collect can reuse them."""
    arrays = {k: v for k, v in outputs.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(_OUTPUTS_CACHE, **arrays)
    # Save non-array entries (None flags) as a tiny json sidecar
    meta = {k: (None if v is None else "__array__")
            for k, v in outputs.items()}
    (_OUTPUTS_CACHE.parent / "_outputs_meta.json").write_text(
        json.dumps(meta, indent=2))
    print(f"   [cache] outputs saved → {_OUTPUTS_CACHE}")


def _load_outputs() -> dict:
    """Reload cached outputs from disk."""
    npz  = np.load(_OUTPUTS_CACHE, allow_pickle=False)
    meta = json.loads((_OUTPUTS_CACHE.parent / "_outputs_meta.json").read_text())
    out  = {}
    for k, flag in meta.items():
        out[k] = npz[k] if (flag == "__array__" and k in npz) else None
    print(f"   [cache] outputs loaded ← {_OUTPUTS_CACHE}")
    return out


def main():
    args = _parse_interp_args()

    print("=" * 70)
    print("  PatchHAR v2 — Interpretability Suite")
    print(f"  Output : {INTERP_DIR}")
    if args.from_sec > 1:
        print(f"  Resuming from section {args.from_sec}")
    if args.only:
        print(f"  Running only sections: {args.only}")
    if args.skip:
        print(f"  Skipping sections: {args.skip}")
    print("=" * 70)

    model = load_model()

    # ── loaders ───────────────────────────────────────────────────────────
    _, test_dl = make_loader(test_pids, shuffle=False, is_train=False)
    _, val_dl  = make_loader(val_pids,  shuffle=False, is_train=False)

    # ── collect / reload outputs ──────────────────────────────────────────
    if args.no_collect and _OUTPUTS_CACHE.exists():
        outputs = _load_outputs()
    else:
        print("\n[*] Collecting model outputs on test set …")
        t0 = _time.time()
        outputs = collect_outputs(model, test_dl)
        print(f"    N={len(outputs['labels']):,} samples  |  "
              f"Macro-F1={f1_score(outputs['labels'], outputs['preds'], average='macro', zero_division=0):.4f}  "
              f"({_time.time()-t0:.0f}s)")
        _save_outputs(outputs)

    # ── section registry ──────────────────────────────────────────────────
    #  (number, display-name, callable, positional-args-tuple)
    sections = [
        (1,  "Attention Rollout & Patch Saliency",
             plot_attention_rollout,       (model, test_dl)),
        (2,  "MoE Expert Specialisation",
             plot_moe_analysis,            (outputs,)),
        (3,  "ECE & Reliability Diagrams",
             plot_calibration,             (outputs,)),
        (4,  "Dual-Domain Gate Analysis  (C1)",
             plot_gate_analysis,           (model, outputs)),
        (5,  "Prototype Memory UMAP  (C7)",
             plot_prototype_analysis,      (model, outputs)),
        (6,  "Confusion Matrix",
             plot_confusion_matrix,        (outputs,)),
        (7,  "Circadian Error Analysis  (C3)",
             plot_circadian_errors,        (outputs,)),
        (8,  "Signal Difficulty vs F1",
             plot_signal_difficulty,       (outputs,)),
        (9,  "Grad-CAM on Worst Mis-predictions",
             plot_gradcam_worst,           (model, test_dl)),
        (10, "Reconstruction Quality  (C10)",
             plot_reconstruction_analysis, (model, test_dl, outputs)),
        (11, "PCGrad Gradient Conflict  (C10)",
             train_with_gradient_conflict_logging,
                                           (model, val_dl),
             {"n_batches": 80}),
        (12, "Multi-Scale Patch Ablation  (C4)",
             plot_multiscale_ablation,     (model, test_dl)),
    ]

    # ── run ───────────────────────────────────────────────────────────────
    results = {}   # sec → True/False
    t_total = _time.time()

    for entry in sections:
        sec, name, fn, pos = entry[0], entry[1], entry[2], entry[3]
        kw = entry[4] if len(entry) == 5 else {}

        if not _should_run(sec, args):
            print(f"  [skip] §{sec} {name}")
            continue

        ok = _run_section(sec, name, fn, *pos, **kw)
        results[sec] = ok

    # ── summary ───────────────────────────────────────────────────────────
    manifest = sorted(
        [f.name for f in INTERP_DIR.glob("*.pdf")] +
        [f.name for f in INTERP_DIR.glob("*.csv")] +
        [f.name for f in INTERP_DIR.glob("*.json")]
    )

    passed  = sum(v for v in results.values())
    failed  = sum(not v for v in results.values())
    elapsed = _time.time() - t_total

    print("\n" + "=" * 70)
    print(f"  Finished in {elapsed/60:.1f} min  |  "
          f"✓ {passed} passed  |  ✗ {failed} failed")
    print(f"  {len(manifest)} output files in {INTERP_DIR}:")
    for m in manifest:
        print(f"    {m}")

    if failed:
        failed_secs = [s for s, ok in results.items() if not ok]
        print(f"\n  To re-run failed sections only:")
        print(f"    python patchhar_interpretability.py "
              f"--only {' '.join(str(s) for s in failed_secs)} --no-collect")
    print("=" * 70)


if __name__ == "__main__":
    main()
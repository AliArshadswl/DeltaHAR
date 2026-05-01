"""
run_ablations.py
================
Ablation study orchestrator for PatchHAR v2.

Runs one training per contribution removal (C1..C10) plus the
all-removed baseline, across one or more random seeds. At the end,
collects all JSON result files and produces:

    ablation_results/
        ablation_table.csv          full numeric table
        ablation_table.tex          ready-to-paste LaTeX (Table IV)
        ablation_summary.json       raw numbers for further analysis

Usage examples
--------------
# Single seed (quick check):
    python run_ablations.py \\
        --script /path/to/your_training_script.py \\
        --seeds 42

# Three seeds (matches paper claim):
    python run_ablations.py \\
        --script /path/to/your_training_script.py \\
        --seeds 42 43 44

# Resume: skip conditions that already have results:
    python run_ablations.py \\
        --script /path/to/your_training_script.py \\
        --seeds 42 43 44 --resume

# Re-run only specific conditions:
    python run_ablations.py \\
        --script /path/to/your_training_script.py \\
        --seeds 42 43 44 --only C3 C4 ALL

Prerequisites
-------------
1. Apply the patch in training_script_patch.py to your training script.
2. The training script must accept --disable, --seed, --output_dir.
3. Your full model has already been trained (results exist for seed 42).
   The orchestrator will use those existing results rather than
   re-training the full model, unless --retrain_full is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ── Contribution definitions ───────────────────────────────────────────────────
CONTRIBUTIONS = {
    "C1":  ("C1_DUAL_DOMAIN_EMBEDDING",   "FFT branch in dual-domain embedding"),
    "C2":  ("C2_CALANET_SKIP_AGG",        "Hierarchical skip aggregation"),
    "C3":  ("C3_CIRCADIAN_BIAS",          "Circadian positional bias"),
    "C4":  ("C4_MULTISCALE_PATCHING",     "Multi-scale tokenisation (single ℓ=25)"),
    "C5":  ("C5_FREQ_AUGMENTATION",       "Frequency-aware augmentation"),
    "C6":  ("C6_LABEL_SMOOTH_TEMP",       "Label smoothing + temperature scaling"),
    "C7":  ("C7_PROTOTYPE_MEMORY",        "Class-prototype inference module"),
    "C8":  ("C8_STOCHASTIC_DEPTH",        "Stochastic depth"),
    "C9":  ("C9_MANIFOLD_MIXUP",          "Manifold Mixup (replaced with raw-input)"),
    "C10": ("C10_RECON_AUX_GRAD_SURGERY", "Reconstruction auxiliary loss"),
    "ALL": (None,                          "All removed (plain patch-Transformer)"),
}

# Metrics extracted from each result JSON
METRIC_KEYS = {
    "macro_f1": "Macro-F1",
    "kappa":    "κ",
    "mcc":      "MCC",
    "accuracy": "Acc",
}


# =============================================================================
# Argument parsing
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="PatchHAR v2 ablation study orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--script", required=True,
        help="Absolute path to your training Python script.",
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=[42],
        help="Random seeds to use (default: 42). Use e.g. --seeds 42 43 44.",
    )
    p.add_argument(
        "--base_dir", type=str,
        default="ablation_results",
        help="Root directory for all ablation outputs (default: ablation_results/).",
    )
    p.add_argument(
        "--full_results", type=str, default=None,
        help="Path to existing full-model patchhar_v2_results.json for seed 42. "
             "If provided, the full-model run is skipped for that seed.",
    )
    p.add_argument(
        "--only", nargs="+", default=None,
        metavar="Cn",
        help="Run only these conditions, e.g. --only C1 C3 ALL. "
             "By default all 11 conditions (C1..C10 + ALL) are run.",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Skip any seed/condition whose result JSON already exists.",
    )
    p.add_argument(
        "--retrain_full", action="store_true",
        help="Re-train the full model for each seed (default: reuse existing "
             "results for the full model if --full_results is given).",
    )
    p.add_argument(
        "--python", type=str, default=sys.executable,
        help="Python interpreter to use (default: current interpreter).",
    )
    p.add_argument(
        "--extra_args", nargs=argparse.REMAINDER, default=[],
        help="Extra arguments forwarded verbatim to the training script, "
             "e.g. -- --batch_size 64.",
    )
    return p.parse_args()


# =============================================================================
# Run a single training condition
# =============================================================================
def run_one(
    python: str,
    script: str,
    condition_id: str,
    seed: int,
    out_dir: Path,
    extra_args: list[str],
    resume: bool,
) -> Path | None:
    """
    Launch one training run via subprocess.
    Returns the path to the result JSON, or None if the run was skipped.
    """
    result_json = out_dir / "patchhar_v2_results.json"

    if resume and result_json.exists():
        print(f"    [SKIP] {condition_id} seed={seed}  (result exists)")
        return result_json

    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the command
    cmd = [python, script, "--seed", str(seed), "--output_dir", str(out_dir)]

    if condition_id == "ALL":
        # Disable all 10 contributions
        all_flags = [k for k in CONTRIBUTIONS if k != "ALL"]
        cmd += ["--disable"] + all_flags
    elif condition_id != "FULL":
        cmd += ["--disable", condition_id]

    cmd += extra_args

    log_path = out_dir / "train.log"

    print(f"\n  {'─'*60}")
    print(f"  Condition : {condition_id}")
    print(f"  Seed      : {seed}")
    print(f"  Output    : {out_dir}")
    print(f"  Log       : {log_path}")
    print(f"  Command   : {' '.join(cmd)}")
    print(f"  {'─'*60}")

    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        logf.write(proc.stdout)

    elapsed = time.time() - t0
    status  = "✓ done" if proc.returncode == 0 else "✗ FAILED"
    print(f"  {status} in {elapsed/60:.1f} min  (return code {proc.returncode})")

    if proc.returncode != 0:
        print(f"\n  Last 30 lines of log:\n")
        lines = proc.stdout.strip().split("\n")
        print("\n".join(lines[-30:]))
        return None

    if not result_json.exists():
        print(f"  [WARN] Result JSON not found at {result_json}")
        return None

    return result_json


# =============================================================================
# Result collection
# =============================================================================
def load_result(path: Path) -> dict:
    """Load and return the metrics dict from a result JSON."""
    data = json.loads(path.read_text())
    return data.get("metrics", {})


def collect_results(
    base_dir: Path,
    conditions: list[str],
    seeds: list[int],
) -> dict:
    """
    Returns a nested dict:
        results[condition][metric] = list of values (one per seed)
    """
    results = {}
    for cid in conditions:
        results[cid] = {k: [] for k in METRIC_KEYS}
        for seed in seeds:
            rpath = base_dir / cid / f"seed{seed}" / "patchhar_v2_results.json"
            if not rpath.exists():
                print(f"  [WARN] Missing: {rpath}")
                continue
            m = load_result(rpath)
            for k in METRIC_KEYS:
                if k in m:
                    results[cid][k].append(float(m[k]))
    return results


# =============================================================================
# Table generation
# =============================================================================
def compute_deltas(results: dict, full_key: str = "FULL") -> dict:
    """
    Compute mean ± std for each condition, then ΔF1 vs the full model.
    Returns a dict: condition → {metric: (mean, std), "delta_f1": Δ}
    """
    summary = {}
    for cid, metric_lists in results.items():
        summary[cid] = {}
        for k, vals in metric_lists.items():
            if vals:
                summary[cid][k] = (np.mean(vals), np.std(vals, ddof=0))
            else:
                summary[cid][k] = (float("nan"), float("nan"))

    # Compute ΔF1 relative to full model
    full_f1 = summary.get(full_key, {}).get("macro_f1", (float("nan"), 0))[0]
    for cid in summary:
        cid_f1 = summary[cid].get("macro_f1", (float("nan"), 0))[0]
        summary[cid]["delta_f1"] = (cid_f1 - full_f1) * 100  # in pp

    return summary, full_f1


def save_csv(summary: dict, conditions: list[str], out_path: Path):
    """Save the full numeric table as CSV."""
    rows = ["condition,label,macro_f1_mean,macro_f1_std,kappa_mean,kappa_std,"
            "mcc_mean,mcc_std,accuracy_mean,accuracy_std,delta_f1_pp"]
    for cid in conditions:
        label = CONTRIBUTIONS.get(cid, (None, cid))[1]
        s = summary.get(cid, {})
        f1m, f1s   = s.get("macro_f1", (float("nan"), float("nan")))
        km,  ks    = s.get("kappa",    (float("nan"), float("nan")))
        mm,  ms    = s.get("mcc",      (float("nan"), float("nan")))
        am,  as_   = s.get("accuracy", (float("nan"), float("nan")))
        df1        = s.get("delta_f1", float("nan"))
        rows.append(
            f"{cid},{label},"
            f"{f1m:.4f},{f1s:.4f},"
            f"{km:.4f},{ks:.4f},"
            f"{mm:.4f},{ms:.4f},"
            f"{am:.4f},{as_:.4f},"
            f"{df1:+.2f}"
        )
    out_path.write_text("\n".join(rows))
    print(f"\n  CSV  → {out_path}")


def save_latex(summary: dict, conditions_ablation: list[str], out_path: Path,
               full_key: str = "FULL", n_seeds: int = 1):
    """
    Save a ready-to-paste LaTeX table matching Table IV in the paper.
    Rows: C1..C10 individually disabled, then ALL removed.
    """
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Component ablation (CAPTURE-24, 80/20/51 split).",
        r"  $\Delta F_1^{\mathrm{macro}}$ is the change in macro-F1 (pp)",
        r"  relative to the full model; negative values indicate degradation.",
        (f"  Mean of {n_seeds} seed{'s' if n_seeds > 1 else ''}.}}" ),
        r"  \label{tab:ablation}",
        r"  \setlength{\tabcolsep}{3.5pt}",
        r"  \renewcommand{\arraystretch}{1.08}",
        r"  \begin{tabular}{clc}",
        r"    \toprule",
        r"    \textbf{ID} &",
        r"    \textbf{Component removed} &",
        r"    $\boldsymbol{\Delta F_1}$ \textbf{(pp)} \\",
        r"    \midrule",
    ]

    for cid in conditions_ablation:
        if cid in ("FULL",):
            continue
        label = CONTRIBUTIONS.get(cid, (None, cid))[1]
        s     = summary.get(cid, {})
        df1   = s.get("delta_f1", float("nan"))
        f1s   = s.get("macro_f1", (float("nan"), float("nan")))[1]

        if n_seeds > 1 and not (f1s != f1s):  # not nan
            delta_str = f"${df1:+.1f} \\pm {f1s * 100:.1f}$"
        else:
            delta_str = f"${df1:+.1f}$"

        id_str = cid if cid != "ALL" else r"\textemdash"
        row_start = r"    \rowcolor{gray!12}" if cid == "ALL" else "   "
        label_str = (r"\textit{All removed (plain patch-Transformer)}"
                     if cid == "ALL" else label)

        # Add separator line before ALL row
        if cid == "ALL":
            lines.append(r"    \midrule")

        lines.append(
            f"{row_start} {id_str} & {label_str} & {delta_str} \\\\"
        )

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    out_path.write_text("\n".join(lines))
    print(f"  LaTeX → {out_path}")


def print_summary_table(summary: dict, conditions: list[str]):
    """Print a readable summary table to stdout."""
    col_w = 48
    print(f"\n  {'─'*80}")
    print(f"  {'Condition':<10}  {'Component':<{col_w}}  {'ΔF1 (pp)':>10}  "
          f"{'F1 (%)':>9}  {'κ (%)':>9}")
    print(f"  {'─'*80}")
    for cid in conditions:
        label = CONTRIBUTIONS.get(cid, (None, cid))[1]
        s     = summary.get(cid, {})
        df1   = s.get("delta_f1", float("nan"))
        f1m   = s.get("macro_f1", (float("nan"),))[0]
        km    = s.get("kappa",    (float("nan"),))[0]
        label_str = (label[:col_w-3] + "...") if len(label) > col_w else label
        print(f"  {cid:<10}  {label_str:<{col_w}}  {df1:>+10.2f}  "
              f"{f1m*100:>9.3f}  {km*100:>9.3f}")
    print(f"  {'─'*80}\n")


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    script   = Path(args.script).resolve()

    if not script.exists():
        sys.exit(f"Training script not found: {script}")

    # Determine which conditions to run
    all_conditions = list(CONTRIBUTIONS.keys())   # C1..C10, ALL
    if args.only:
        run_conditions = [c.upper() for c in args.only]
        invalid = [c for c in run_conditions if c not in all_conditions]
        if invalid:
            sys.exit(f"Unknown conditions: {invalid}. Valid: {all_conditions}")
    else:
        run_conditions = all_conditions

    seeds = args.seeds

    print("=" * 70)
    print("  PatchHAR v2  —  Ablation Study Orchestrator")
    print(f"  Script    : {script}")
    print(f"  Base dir  : {base_dir}")
    print(f"  Seeds     : {seeds}")
    print(f"  Conditions: {run_conditions}")
    print(f"  Resume    : {args.resume}")
    print("=" * 70)

    # ── Training runs ─────────────────────────────────────────────────────────
    t_start = time.time()
    failed  = []

    for cid in run_conditions:
        for seed in seeds:
            out_dir = base_dir / cid / f"seed{seed}"

            # Handle --full_results shortcut for the FULL model
            if cid == "FULL" and args.full_results and not args.retrain_full:
                src = Path(args.full_results)
                if src.exists() and seed == seeds[0]:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    dst = out_dir / "patchhar_v2_results.json"
                    if not dst.exists():
                        import shutil
                        shutil.copy(src, dst)
                        print(f"  [COPY] Full-model results → {dst}")
                    else:
                        print(f"  [SKIP] Full model seed={seed} (already copied)")
                    continue

            rpath = run_one(
                python      = args.python,
                script      = str(script),
                condition_id = cid,
                seed        = seed,
                out_dir     = out_dir,
                extra_args  = args.extra_args,
                resume      = args.resume,
            )
            if rpath is None:
                failed.append((cid, seed))

    elapsed_total = time.time() - t_start
    print(f"\n  All training runs finished in {elapsed_total/3600:.2f} h")
    if failed:
        print(f"  FAILED runs: {failed}")

    # ── Result collection ─────────────────────────────────────────────────────
    print("\n  Collecting results …")
    collect_conditions = ["FULL"] + all_conditions  # full model first
    results = collect_results(base_dir, collect_conditions, seeds)

    summary, full_f1 = compute_deltas(results, full_key="FULL")

    # Output files
    base_dir.mkdir(parents=True, exist_ok=True)
    conditions_for_table = all_conditions  # C1..C10 + ALL

    save_csv(summary, ["FULL"] + conditions_for_table,
             base_dir / "ablation_table.csv")

    save_latex(summary, conditions_for_table,
               base_dir / "ablation_table.tex",
               n_seeds=len(seeds))

    # Save raw summary JSON
    raw = {
        cid: {
            k: (list(v) if isinstance(v, list) else v)
            for k, v in results.get(cid, {}).items()
        }
        for cid in ["FULL"] + conditions_for_table
    }
    (base_dir / "ablation_summary.json").write_text(
        json.dumps(raw, indent=2))

    # Print to terminal
    print_summary_table(summary, ["FULL"] + conditions_for_table)

    print("=" * 70)
    print(f"  Full model  macro-F1 = {full_f1*100:.3f}%")
    print(f"  Output files in: {base_dir}")
    if failed:
        print(f"  WARNING: {len(failed)} run(s) failed — "
              f"results for those seeds are missing.")
    print("=" * 70)


if __name__ == "__main__":
    main()
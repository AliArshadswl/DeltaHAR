"""
run_ablations.py
================
Ablation runner for PatchHAR v2.

Runs 12 configurations × 3 seeds = 36 subprocess calls, then collects
results and prints the LaTeX ablation table matching:

  \\begin{tabular}{|l|c|}
    Component removed   &  ΔF1^macro (pp)
  \\end{tabular}

Prerequisites
─────────────
1. Apply the two-edit patch described in patchhar_v2_patch.md to
   patchhar_v2.py.  The script must accept --run-id and --seed flags.
2. Place this file in the same directory as patchhar_v2.py  (or pass
   --script-path to point to it).

Usage
─────
  # Default: runs all ablations, 3 seeds each
  python run_ablations.py

  # Use more seeds, point at a custom script location
  python run_ablations.py --seeds 42 123 456 789 --script-path /path/to/patchhar_v2.py

  # Resume: skip configs whose result JSONs already exist
  python run_ablations.py --resume

  # Dry-run: print commands only, do not execute
  python run_ablations.py --dry-run

Output
──────
  ablation_summary.json  — raw per-seed metrics
  ablation_table.tex     — ready-to-paste LaTeX table
  ablation_table.txt     — plain-text table for quick inspection
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Ablation configurations
# Each entry:  (display_name, list_of_flags_to_DISABLE_for_this_ablation)
# The full model has nothing disabled.
# "all off" disables everything.
# ─────────────────────────────────────────────────────────────────────────────
ALL_CONTRIBS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"]

ABLATION_CONFIGS: list[tuple[str, list[str]]] = [
    # (run_id,                      disabled_flags)
    ("full",                        []),
    ("no_C1_fft_branch",            ["C1"]),
    ("no_C4_multiscale",            ["C4"]),
    ("no_C3_circadian",             ["C3"]),
    ("no_C6_label_smooth_temp",     ["C6"]),
    ("no_C5_freq_aug",              ["C5"]),
    ("no_C2_skip_agg",              ["C2"]),
    ("no_C9_manifold_mixup",        ["C9"]),
    ("no_C8_stoch_depth",           ["C8"]),
    ("no_C7_prototype",             ["C7"]),
    ("no_C10_recon_aux",            ["C10"]),
    ("all_off",                     ALL_CONTRIBS),
]

# Human-readable row labels (same order as ABLATION_CONFIGS, skip "full")
ROW_LABELS: dict[str, str] = {
    "no_C1_fft_branch":        "FFT branch in dual-domain embedding (C1)",
    "no_C4_multiscale":        "Multi-scale patching — single $\\ell\\!=\\!25$ (C4)",
    "no_C3_circadian":         "Circadian positional bias (C3)",
    "no_C6_label_smooth_temp": "Label smoothing and temperature scaling (C6)",
    "no_C5_freq_aug":          "Frequency-aware augmentation (C5)",
    "no_C2_skip_agg":          "Hierarchical skip aggregation (C2)",
    "no_C9_manifold_mixup":    "Manifold Mixup (replaced with raw input Mixup) (C9)",
    "no_C8_stoch_depth":       "Stochastic depth (C8)",
    "no_C7_prototype":         "Class-prototype inference module (C7)",
    "no_C10_recon_aux":        "Reconstruction auxiliary loss (C10)",
    "all_off":                 "All components removed (plain patch-Transformer)",
}

DEFAULT_SEEDS  = [42, 123, 456]
METRIC_KEY     = "macro_f1"        # key inside results JSON → metrics dict


# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PatchHAR v2 ablation runner")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                   help="Random seeds to average over")
    p.add_argument("--script-path", type=str, default="patchhar_v2.py",
                   help="Path to patchhar_v2.py")
    p.add_argument("--python", type=str, default=sys.executable,
                   help="Python interpreter to use")
    p.add_argument("--resume", action="store_true",
                   help="Skip configs whose result JSONs already exist for ALL seeds")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands only; do not run anything")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Override output directory (default: read from patchhar cfg)")
    p.add_argument("--timeout", type=int, default=None,
                   help="Timeout per run in seconds (default: no limit)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def result_path(output_dir: Path, run_id: str, seed: int) -> Path:
    """Must match the path used inside patchhar_v2.py after patching."""
    tag = f"{run_id}_s{seed}"
    return output_dir / f"results_{tag}.json"


def run_one(python: str, script: str,
            run_id: str, seed: int,
            disabled: list[str],
            timeout: int | None,
            dry: bool) -> bool:
    """Launch a single training run.  Returns True on success."""
    tag = f"{run_id}_s{seed}"
    cmd = [python, script,
           "--run-id", tag,
           "--seed",   str(seed)]
    if disabled:
        cmd += ["--disable"] + disabled

    print(f"\n{'─'*70}")
    print(f"  RUN : {tag}")
    print(f"  CMD : {' '.join(cmd)}")
    print(f"{'─'*70}")

    if dry:
        return True

    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            check=True,           # raises on non-zero exit
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Run failed with exit code {e.returncode}")
        return False
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Run timed out after {timeout}s")
        return False


# ─────────────────────────────────────────────────────────────────────────────
def load_metric(json_path: Path) -> float | None:
    """Read macro_f1 from a results JSON.  Returns None if unavailable."""
    try:
        data = json.loads(json_path.read_text())
        return float(data["metrics"][METRIC_KEY])
    except Exception as e:
        print(f"  [WARN] Could not read {json_path}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
def discover_output_dir(script_path: str, python: str) -> Path:
    """
    Import the script's Config object in a tiny subprocess to read OUTPUT_DIR,
    so we don't have to hard-code the path here.
    Falls back to the script's directory.
    """
    probe = (
        f"import sys; sys.argv=['x']; "
        f"import importlib.util as u; "
        f"sp = u.spec_from_file_location('m', r'{script_path}'); "
        f"m = u.module_from_spec(sp); sp.loader.exec_module(m); "
        f"print(str(m.cfg.OUTPUT_DIR))"
    )
    try:
        out = subprocess.check_output(
            [python, "-c", probe],
            stderr=subprocess.DEVNULL,
            timeout=30,
        ).decode().strip().splitlines()[-1]
        return Path(out)
    except Exception:
        fallback = Path(script_path).parent
        print(f"  [WARN] Could not probe OUTPUT_DIR; using {fallback}")
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
def build_table(summary: dict, seeds: list[int]) -> tuple[str, str]:
    """
    Returns (latex_table, plain_text_table).
    summary : { run_id: { seed: float | None } }
    """
    def mean_f1(run_id: str) -> float | None:
        vals = [summary[run_id].get(s) for s in seeds]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    full_f1 = mean_f1("full")
    if full_f1 is None:
        print("  [WARN] Full-model results missing — ΔF1 will be N/A")

    rows = []
    for run_id, _ in ABLATION_CONFIGS:
        if run_id == "full":
            continue
        label  = ROW_LABELS[run_id]
        f1     = mean_f1(run_id)
        if f1 is not None and full_f1 is not None:
            delta  = round((f1 - full_f1) * 100, 1)   # percentage points
            delta_str_tex  = f"${delta:+.1f}$"
            delta_str_plain = f"{delta:+.1f} pp"
        else:
            delta_str_tex   = "N/A"
            delta_str_plain = "N/A"
        rows.append((label, delta_str_tex, delta_str_plain))

    # ── LaTeX ──────────────────────────────────────────────────────────────
    n_seeds  = len(seeds)
    seed_str = ", ".join(str(s) for s in seeds)
    full_str = f"{full_f1*100:.1f}\\%" if full_f1 is not None else "N/A"

    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        (r"\caption{Ablation study (CAPTURE-24 80/20/51 split). "
         r"$\Delta F_1^{\rm macro}$ is the change in percentage points "
         r"relative to the full model (macro-F1\,=\," + full_str + r"); "
         r"negative values indicate degradation when the component is removed. "
         f"Mean of {n_seeds} seeds ({seed_str}).}}"),
        r"\label{tab:ablation}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{|l|c|}",
        r"\hline",
        r"\textbf{Component removed} & "
        r"$\boldsymbol{\Delta F_1^{\rm macro}}$ (pp) \\",
        r"\hline",
    ]
    # individual-component rows (all except "all_off")
    for label, delta_tex, _ in rows[:-1]:
        latex_lines.append(f"{label} & {delta_tex} \\\\")
    # separator before "all off"
    latex_lines.append(r"\hline")
    # "all off" row
    latex_lines.append(f"{rows[-1][0]} & {rows[-1][1]} \\\\")
    latex_lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    latex = "\n".join(latex_lines)

    # ── Plain text ─────────────────────────────────────────────────────────
    col1_w = max(len(r[0]) for r in rows) + 2
    header = f"{'Component removed':<{col1_w}}  ΔF1-macro"
    sep    = "─" * (col1_w + 14)
    plain_lines = [
        f"Full model macro-F1 = {full_f1*100:.2f}% "
        f"(mean of {n_seeds} seeds: {seed_str})"
        if full_f1 is not None else "Full model: N/A",
        "",
        header,
        sep,
    ]
    for label, _, delta_plain in rows[:-1]:
        plain_lines.append(f"{label:<{col1_w}}  {delta_plain}")
    plain_lines.append(sep)
    plain_lines.append(f"{rows[-1][0]:<{col1_w}}  {rows[-1][2]}")
    plain_lines.append(sep)
    plain = "\n".join(plain_lines)

    return latex, plain


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    seeds       = args.seeds
    script_path = args.script_path
    python      = args.python

    print("=" * 70)
    print("  PatchHAR v2 — Ablation Runner")
    print(f"  Script : {script_path}")
    print(f"  Seeds  : {seeds}")
    print(f"  Configs: {len(ABLATION_CONFIGS)}  "
          f"({len(ABLATION_CONFIGS) * len(seeds)} total runs)")
    print("=" * 70)

    # ── Determine output directory ────────────────────────────────────────
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = discover_output_dir(script_path, python)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output dir : {output_dir}")

    # ── Run all configurations ────────────────────────────────────────────
    summary: dict[str, dict[int, float | None]] = {
        run_id: {} for run_id, _ in ABLATION_CONFIGS
    }

    total = len(ABLATION_CONFIGS) * len(seeds)
    done  = 0

    for run_id, disabled in ABLATION_CONFIGS:
        for seed in seeds:
            done += 1
            tag      = f"{run_id}_s{seed}"
            res_path = result_path(output_dir, run_id, seed)

            # Resume: skip if result already exists
            if args.resume and res_path.exists():
                val = load_metric(res_path)
                if val is not None:
                    print(f"  [RESUME] {tag} → {METRIC_KEY}={val:.4f}")
                    summary[run_id][seed] = val
                    continue

            print(f"\n  [{done}/{total}] {tag}")
            ok = run_one(
                python, script_path,
                run_id, seed, disabled,
                args.timeout, args.dry_run,
            )

            if not args.dry_run:
                val = load_metric(res_path) if ok else None
                summary[run_id][seed] = val
                if val is not None:
                    print(f"  ✓  {METRIC_KEY} = {val:.4f}")
                else:
                    print(f"  ✗  result not found at {res_path}")

    # ── Aggregate and build tables ────────────────────────────────────────
    if args.dry_run:
        print("\n  [Dry-run] No results to aggregate.")
        return

    # Save raw summary
    summary_path = output_dir / "ablation_summary.json"
    # Convert int keys to str for JSON serialisation
    json_summary = {k: {str(s): v for s, v in sv.items()}
                    for k, sv in summary.items()}
    summary_path.write_text(json.dumps(json_summary, indent=2))
    print(f"\n  Raw summary saved: {summary_path}")

    latex, plain = build_table(summary, seeds)

    tex_path  = output_dir / "ablation_table.tex"
    txt_path  = output_dir / "ablation_table.txt"
    tex_path.write_text(latex)
    txt_path.write_text(plain)

    print(f"  LaTeX table saved : {tex_path}")
    print(f"  Plain table saved : {txt_path}")

    print("\n" + "=" * 70)
    print("  PLAIN-TEXT TABLE")
    print("=" * 70)
    print(plain)
    print("=" * 70)

    print("\n" + "=" * 70)
    print("  LaTeX TABLE")
    print("=" * 70)
    print(latex)
    print("=" * 70)


if __name__ == "__main__":
    main()
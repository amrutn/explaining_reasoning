#!/usr/bin/env python3
"""Standalone script to plot results from api_accuracy.py.

Only requires: numpy, matplotlib (no API keys or heavy dependencies).
Auto-discovers result files in the results/ directory and generates
one plot per dataset.

Usage:
    python code/plot_results.py [--results-dir results/]
"""

import argparse
import json
import os
from collections import defaultdict
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# Configuration (must match api_accuracy.py)
# ==========================================

PROMPT_KEYS = ["Direct", "L01", "L02", "L03", "L04", "L05"]
REPS = [0]

MODEL_PROVIDER = {
    "gemini-flash": "google",
    "deepseek-v3": "deepseek",
}

PROVIDER_COLORS = {
    "anthropic": "#2563eb",
    "openai": "#16a34a",
    "google": "#dc2626",
    "deepseek": "#7c3aed",
}

DATASET_TITLES = {
    "gsm8k": "GSM8K",
    "math": "MATH-500",
}


# ==========================================
# Plotting
# ==========================================


def plot_dataset(dataset_short, model_files):
    """Generate a plot for one dataset from its result files."""
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    has_any_data = False

    for model_key, filepath in sorted(model_files.items()):
        with open(filepath, "r") as f:
            data = json.load(f)
        if not data:
            continue

        plot_points = []

        for prompt_key in PROMPT_KEYS:
            rep_stats = {r: {"correct": 0, "total": 0} for r in REPS}
            all_tokens = []
            has_data = False

            for entry in data:
                if prompt_key not in entry.get("results", {}):
                    continue
                replicates = entry["results"][prompt_key]
                if not replicates:
                    continue

                has_data = True
                for r_entry in replicates:
                    r_idx = r_entry["replicate"]
                    if r_idx in rep_stats:
                        rep_stats[r_idx]["total"] += 1
                        if r_entry["is_correct"]:
                            rep_stats[r_idx]["correct"] += 1
                        all_tokens.append(r_entry["token_count"])

            if not has_data:
                continue

            replicate_error_rates = []
            for r in REPS:
                total = rep_stats[r]["total"]
                if total > 0:
                    error_rate = 1.0 - (rep_stats[r]["correct"] / total)
                    replicate_error_rates.append(error_rate)

            if not replicate_error_rates:
                continue

            mean_error_rate = np.mean(replicate_error_rates)
            if len(replicate_error_rates) > 1:
                sem_error_rate = np.std(replicate_error_rates, ddof=1) / np.sqrt(
                    len(replicate_error_rates)
                )
            else:
                sem_error_rate = 0.0

            avg_tokens = np.mean(all_tokens) if all_tokens else 0

            if prompt_key != "Direct":
                plot_points.append((avg_tokens, mean_error_rate, sem_error_rate))

        plot_points.sort(key=lambda p: p[0])

        if not plot_points:
            continue

        has_any_data = True
        x_val, y_val, y_err = zip(*plot_points)

        provider = MODEL_PROVIDER.get(model_key, "unknown")
        color = PROVIDER_COLORS.get(provider, "gray")

        ax.errorbar(
            x_val,
            y_val,
            yerr=y_err,
            fmt="-o",
            color=color,
            linewidth=2,
            label=model_key,
            capsize=3,
            markersize=6,
        )

    if not has_any_data:
        plt.close(fig)
        print(f"  No plottable data for {dataset_short}, skipping.")
        return

    title = DATASET_TITLES.get(dataset_short, dataset_short)
    ax.set_xlabel("Output Tokens")
    ax.set_ylabel("Test Error")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.6)

    out_path = f"{dataset_short}_api_comparison_error_vs_length.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


# ==========================================
# Main
# ==========================================


def main():
    parser = argparse.ArgumentParser(description="Plot results from api_accuracy.py")
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Directory containing result JSON files (default: results/)",
    )
    args = parser.parse_args()

    # Discover result files: pattern is {dataset_short}_{model_key}.json
    files = glob(os.path.join(args.results_dir, "*.json"))
    if not files:
        print(f"No JSON files found in {args.results_dir}/")
        return

    # Group by dataset
    datasets = defaultdict(dict)
    for filepath in files:
        basename = os.path.splitext(os.path.basename(filepath))[0]
        # Split on first underscore: "gsm8k_gemini-flash" -> ("gsm8k", "gemini-flash")
        parts = basename.split("_", 1)
        if len(parts) != 2:
            print(f"  Skipping unrecognized file: {basename}.json")
            continue
        dataset_short, model_key = parts
        datasets[dataset_short][model_key] = filepath

    print(f"Found {len(files)} result file(s) across {len(datasets)} dataset(s)\n")

    for dataset_short, model_files in sorted(datasets.items()):
        models_str = ", ".join(sorted(model_files.keys()))
        print(f"Plotting {dataset_short}: [{models_str}]")
        plot_dataset(dataset_short, model_files)

    print("\nDone.")


if __name__ == "__main__":
    main()

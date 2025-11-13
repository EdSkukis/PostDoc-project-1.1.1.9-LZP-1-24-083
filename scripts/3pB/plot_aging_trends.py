# plot_aging_trends_by_pH.py
"""
Plots the evolution of three mechanical parameters as a function of aging time,
with separate trends for each pH group:

    • Emod (GPa)
    • Flexural strength (MPa)
    • Flexural strain (-)

Input:
    summary_E_results.csv (same format as produced by process_3pb.py)

Output:
    Three PNG plots saved in the output directory:
        - Emod_vs_aging_by_pH.png
        - flexural_strength_vs_aging_by_pH.png
        - flexural_strain_vs_aging_by_pH.png

Each plot contains, for each pH group:
    - mean ± standard deviation vs aging time

Usage:
    python plot_aging_trends_by_pH.py summary_E_results.csv output_folder

If arguments are not provided:
    summary_E_results.csv is taken from current folder
    output_folder = ./outputs_aging
"""

from __future__ import annotations
import os
import sys
import re
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------- parsing helpers ----------

def extract_age_weeks(specimen_id: str) -> float:
    """
    Extract aging time in weeks from the specimen_id string.

    Expected patterns:
        "2weeks.xlsx - G_pH10_1"
        "4week - G_pH7_1"
    If the keyword "week" is not present, the specimen is treated as REF (0 weeks).
    """
    s = str(specimen_id).lower()
    m = re.search(r"(\d+)\s*week", s)
    if m:
        return float(m.group(1))
    return 0.0   # reference (no aging)


def extract_ph_group(specimen_id: str) -> str:
    """
    Extract pH group from specimen_id.

    Examples:
        "2weeks.xlsx - G_pH10_1" -> "pH10"
        "G_pH4_3"                -> "pH4"
        "GFRP_ref1"              -> "ref"

    If no pH is found but "ref" is present, returns "ref".
    Otherwise returns "unknown".
    """
    s = str(specimen_id).lower()

    # explicit reference group
    if "ref" in s:
        return "ref"

    # search for "ph" followed by digits (with optional underscore)
    m = re.search(r"p?h[_\-]?(\d+)", s)
    # safer: look specifically for "ph" + number
    m2 = re.search(r"ph[_\-]?(\d+)", s)
    if m2:
        return f"pH{m2.group(1)}"
    if m:
        return f"pH{m.group(1)}"

    return "unknown"


def ensure_output_folder(path: str) -> str:
    """Create the output directory if missing and return its path."""
    os.makedirs(path, exist_ok=True)
    return path


# ---------- plotting ----------

def plot_param_vs_age_by_ph(df: pd.DataFrame,
                            age_col: str,
                            ph_col: str,
                            param_col: str,
                            ylabel: str,
                            out_path: str) -> None:
    """
    Plot parameter vs aging time, one mean curve per pH group.
    No STD bands, no individual specimen points.
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    # Stable group sorting: "ref" first
    groups = sorted(df[ph_col].dropna().unique().tolist())
    if "ref" in groups:
        groups = ["ref"] + [g for g in groups if g != "ref"]

    for ph in groups:
        sub = df[df[ph_col] == ph]
        if sub.empty:
            continue

        # compute mean only
        stats = (
            sub.groupby(age_col)[param_col]
            .agg(["mean"])
            .reset_index()
            .sort_values(age_col)
        )

        # single line for mean values
        ax.plot(
            stats[age_col],
            stats["mean"],
            "-o",
            linewidth=2,
            markersize=6,
            label=ph
        )

    ax.set_xlabel("Aging time [weeks]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{param_col} vs aging time by pH group (mean only)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Group", loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)



# ---------- main pipeline ----------

def main() -> None:
    # Base default folder (where processed 3PB outputs are stored)
    default_base = "./mnt/data_epoxy/outputs"

    # 1) Read command-line arguments
    if len(sys.argv) >= 2:
        # User explicitly provided summary CSV path
        summary_path = sys.argv[1]
    else:
        # Default summary path in the GF folder
        summary_path = os.path.join(default_base, "summary_E_results.csv")

    # Derive default output folder from the summary file location
    default_out_dir = os.path.join(os.path.dirname(summary_path), "outputs_aging")

    if len(sys.argv) >= 3:
        out_dir = sys.argv[2]
    else:
        out_dir = default_out_dir

    # 2) Sanity checks and folder creation
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"Summary CSV not found: {os.path.abspath(summary_path)}")

    out_dir = ensure_output_folder(out_dir)

    # ---- Read summary data ----
    df = pd.read_csv(summary_path)

    # Extract aging time (weeks) and pH group from specimen ID
    df["age_weeks"] = df["specimen_id"].apply(extract_age_weeks)
    df["ph_group"] = df["specimen_id"].apply(extract_ph_group)

    # Validate required columns
    for col in ["Emod", "flexural_strength_MPa", "flexural_strain"]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in {summary_path}")

    # Drop rows with missing data
    df_clean = df.dropna(subset=[
        "age_weeks", "ph_group",
        "Emod", "flexural_strength_MPa", "flexural_strain"
    ])

    # ---- Three plots, separated by pH ----
    plot_param_vs_age_by_ph(
        df_clean,
        age_col="age_weeks",
        ph_col="ph_group",
        param_col="Emod",
        ylabel="Emod [GPa]",
        out_path=os.path.join(out_dir, "Emod_vs_aging_by_pH.png"),
    )

    plot_param_vs_age_by_ph(
        df_clean,
        age_col="age_weeks",
        ph_col="ph_group",
        param_col="flexural_strength_MPa",
        ylabel="Flexural strength [MPa]",
        out_path=os.path.join(out_dir, "flexural_strength_vs_aging_by_pH.png"),
    )

    plot_param_vs_age_by_ph(
        df_clean,
        age_col="age_weeks",
        ph_col="ph_group",
        param_col="flexural_strain",
        ylabel="Flexural strain [-]",
        out_path=os.path.join(out_dir, "flexural_strain_vs_aging_by_pH.png"),
    )

    print("Summary file:", os.path.abspath(summary_path))
    print("Plots saved to:", os.path.abspath(out_dir))



if __name__ == "__main__":
    main()

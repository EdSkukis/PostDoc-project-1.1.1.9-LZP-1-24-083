#!/usr/bin/env python3
"""
Universal plotting for Summary.csv

Examples:
  # Список доступных полей (включая производные)
  python generate_plot.py --list-cols

  # Rp0.2 vs Time (лог X), легенда по температурным группам
  python generate_plot.py --x log:Time_at_Rp0.2_s --y Rp0.2_MPa --legend Temp_group \
      --title "Rp0.2 vs Time (legend=temperature)"

  # E vs ln(speed), легенда по скорости
  python generate_plot.py --x log:TestSpeed_mm_min --y E_MPa --legend Speed_group

  # С трендовой линией по ln(X)
  python generate_plot.py --x log:Time_at_Rp0.2_s --y Rp0.2_MPa --legend Temp_group --trend-lnX
"""

import argparse
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DERIVED_HELP = [
    "Speed_group = int(TestSpeed_mm_min)",
    "Temp_group  = 22/40/50 by Temp_C_mean bins (<30→22, <45→40, else 50)",
    "log:<col>, ln:<col>, sqrt:<col> = on-the-fly transforms"
]

def load_and_enrich(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    # Derived groups
    if "TestSpeed_mm_min" in df.columns:
        df["Speed_group"] = df["TestSpeed_mm_min"].astype(int)
    if "Temp_C_mean" in df.columns:
        def _tgrp(t):
            if t < 30: return 22
            elif t < 45: return 40
            else: return 50
        df["Temp_group"] = df["Temp_C_mean"].apply(_tgrp).astype(int)
    return df

def list_columns(df: pd.DataFrame):
    cols = sorted(df.columns.tolist())
    print("Available columns:")
    for c in cols:
        print(" -", c)
    print("\nDerived options:")
    for h in DERIVED_HELP:
        print(" -", h)

def ensure_series(df: pd.DataFrame, expr: str) -> pd.Series:
    """
    Supports:
      - raw column: "Rp0.2_MPa"
      - transforms: "log:col", "ln:col", "sqrt:col"
    """
    if ":" in expr:
        op, col = expr.split(":", 1)
        col = col.strip()
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found for transform '{op}:'")
        x = df[col].astype(float)
        if op in ("log", "ln"):
            x = x.where(x > 0, np.nan)
            return np.log(x)
        elif op == "sqrt":
            x = x.where(x >= 0, np.nan)
            return np.sqrt(x)
        else:
            raise ValueError(f"Unknown transform '{op}:'; use log:, ln:, sqrt:")
    else:
        if expr not in df.columns:
            raise KeyError(f"Column '{expr}' not found")
        return df[expr]

def main():
    # По умолчанию искать рядом со скриптом
    script_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    default_csv = script_dir / "out" / "Summary.csv"

    ap = argparse.ArgumentParser(description="Universal scatter plotter for Summary.csv")
    ap.add_argument("--csv", default=str(default_csv), help="Path to Summary.csv")
    ap.add_argument("--list-cols", action="store_true", help="List available columns and exit")
    ap.add_argument("--x", help="X axis: column or transform (e.g., log:Time_at_Rp0.2_s)")
    ap.add_argument("--y", help="Y axis: column or transform (e.g., Rp0.2_MPa)")
    ap.add_argument("--legend", help="Legend group column (e.g., Temp_group, Speed_group)")
    ap.add_argument("--logx", action="store_true", help="Force log scale on X")
    ap.add_argument("--logy", action="store_true", help="Force log scale on Y")
    ap.add_argument("--title", default=None, help="Plot title")
    ap.add_argument("--xlabel", default=None, help="Custom X label")
    ap.add_argument("--ylabel", default=None, help="Custom Y label")
    ap.add_argument("--outfile", default="out/plots/Summary/out", help="Output PNG path")
    ap.add_argument("--outcsv", default=None, help="Optional CSV with plotted points")
    ap.add_argument("--alpha", type=float, default=0.9, help="Marker alpha")
    ap.add_argument("--size", type=float, default=40, help="Marker size")
    ap.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    ap.add_argument("--trend-lnX", action="store_true",
                    help="Add per-group linear trend on Y vs ln(X) (only for X>0)")
    args = ap.parse_args()

    df = load_and_enrich(args.csv)

    if args.list_cols:
        list_columns(df)
        return

    if not (args.x and args.y and args.legend):
        raise SystemExit("Provide --x, --y, --legend or use --list-cols")

    # Build series
    Xs = ensure_series(df, args.x)
    Ys = ensure_series(df, args.y)

    if args.legend not in df.columns:
        raise KeyError(f"Legend column '{args.legend}' not found. Use --list-cols to inspect options.")
    Ls = df[args.legend]

    # Assemble and clean
    dat = pd.DataFrame({"X": Xs, "Y": Ys, "L": Ls}).replace([np.inf, -np.inf], np.nan).dropna()
    if dat.empty:
        raise SystemExit("No data_epoxy to plot after cleaning. Check transforms and positivity.")

    # Plot
    plt.figure(figsize=(7, 5))
    for key, g in dat.groupby("L"):
        plt.scatter(g["X"], g["Y"], label=str(key), alpha=args.alpha, s=args.size)

    # Optional trend per group
    if args.trend_lnX:
        pos = dat["X"] > 0
        dat_pos = dat.loc[pos]
        for key, g in dat_pos.groupby("L"):
            x = np.log(g["X"].values)
            y = g["Y"].values
            if len(x) >= 2 and np.isfinite(x).all() and np.isfinite(y).all():
                A = np.vstack([np.ones_like(x), x]).T
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                a, b = coef
                x_line = np.linspace(x.min(), x.max(), 200)
                y_line = a + b * x_line
                plt.plot(np.exp(x_line), y_line, linewidth=1.5, alpha=0.9)

    # Scales
    if args.logx: plt.xscale("log")
    if args.logy: plt.yscale("log")

    # Labels
    xlabel = args.xlabel if args.xlabel else args.x
    ylabel = args.ylabel if args.ylabel else args.y
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    title = args.title if args.title else f"{args.y} vs {args.x} (legend: {args.legend})"
    plt.title(title)
    plt.grid(True, which="both")
    plt.legend(title=args.legend)
    plt.tight_layout()

    # Save
    out = Path(args.outfile) if args.outfile else Path(f"plot_{args.y}_vs_{args.x}_legend_{args.legend}.png")
    plt.savefig(out.as_posix(), dpi=args.dpi)
    print(f"Saved: {out.as_posix()}")

    if args.outcsv:
        dat_out = dat.copy()
        dat_out.columns = [args.x, args.y, args.legend]
        dat_out.to_csv(args.outcsv, index=False)
        print(f"Saved data_epoxy: {args.outcsv}")

if __name__ == "__main__":
    main()

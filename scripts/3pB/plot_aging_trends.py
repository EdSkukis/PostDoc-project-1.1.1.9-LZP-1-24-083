# plot_aging_trends_by_pH.py

from __future__ import annotations
import os
import sys
import re
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- parsing helpers ----------

def extract_ph_group(specimen_id: str) -> str:
    """Извлекает только pH-группу или ref."""
    s = str(specimen_id).lower()
    if "ref" in s:
        return "ref"
    m = re.search(r"ph[_\-]?(\d+)", s)
    if m:
        return f"pH{m.group(1)}"
    return "unknown"


def extract_age_label(specimen_id: str) -> str:
    """
    Извлекает срок старения и формирует текстовую метку.
    Добавляет '(dry)' если образец высушен.
    """
    s = str(specimen_id).lower()
    if "ref" in s:
        return "0 weeks"

    is_dry = any(kw in s for kw in ["dry", "dried", "сух"])

    # Ищем число и единицу измерения
    m = re.search(r"(\d+)\s*(w|week|m|month|нед|мес)", s)
    if m:
        val = m.group(1)
        unit = "weeks" if m.group(2).startswith(('w', 'нед')) else "months"
        label = f"{val} {unit}"
    else:
        label = "unknown time"

    if is_dry:
        label += " (dry)"

    return label


def age_sort_key(label: str) -> float:
    """
    Ключ для правильной сортировки сроков старения на графике и в легенде.
    Обеспечивает порядок: 0 weeks -> 2 weeks -> 12 weeks -> 12 weeks (dry)
    """
    if label == "0 weeks": return 0.0
    val = 0.0

    # Вытаскиваем числовое значение
    m = re.search(r"(\d+)", label)
    if m:
        val = float(m.group(1))

    # Если месяцы, переводим в недели для правильной сортировки
    if "month" in label:
        val *= 4.33

        # Сдвигаем сухие образцы чуть правее в рамках той же недели
    if "dry" in label:
        val += 0.1

    return val


def ensure_output_folder(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ---------- статистика ----------

def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Считает mean и std по каждой комбинации (ph_group, age_label)."""
    stats = (
        df
        .groupby(["ph_group", "age_label"])
        .agg(
            Emod_mean=("Emod", "mean"),
            Emod_std=("Emod", "std"),
            flexural_strength_mean=("flexural_strength_MPa", "mean"),
            flexural_strength_std=("flexural_strength_MPa", "std"),
            flexural_strain_mean=("flexural_strain", "mean"),
            flexural_strain_std=("flexural_strain", "std"),
            n=("Emod", "count"),
        )
        .reset_index()
    )
    return stats


# ---------- bar-плот по pH и сроку ----------

def ordered_ph_groups(groups: List[str]) -> List[str]:
    """Стабильная сортировка pH-групп."""
    groups = [g for g in groups if pd.notna(g)]
    unique = sorted(set(groups))
    priority = ["ref", "pH1", "pH4", "pH7", "pH10", "pH12"]

    ordered = [g for g in priority if g in unique]
    remaining = [g for g in unique if g not in priority]
    ordered.extend(remaining)
    return ordered


def bar_param_by_ph_and_age(
        stats: pd.DataFrame,
        param_prefix: str,
        ylabel: str,
        out_path: str,
) -> None:
    mean_col = f"{param_prefix}_mean"
    std_col = f"{param_prefix}_std"

    ph_groups = ordered_ph_groups(stats["ph_group"].unique().tolist())
    ages = sorted(stats["age_label"].unique().tolist(), key=age_sort_key)

    x = np.arange(len(ph_groups))
    width = 0.8 / max(len(ages), 1)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, age in enumerate(ages):
        offset = (i - (len(ages) - 1) / 2) * width
        positions = x + offset

        means = []
        stds = []
        for ph in ph_groups:
            row = stats[(stats["ph_group"] == ph) & (stats["age_label"] == age)]
            if row.empty:
                means.append(np.nan)
                stds.append(0.0)
            else:
                means.append(row[mean_col].iloc[0])
                stds.append(row[std_col].iloc[0] if not np.isnan(row[std_col].iloc[0]) else 0.0)

        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)

        ax.bar(positions, means, width=width, label=age)
        ax.errorbar(positions, means, yerr=stds, fmt="none", capsize=3, color="black", alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(ph_groups)
    ax.set_xlabel("Group")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{param_prefix} vs aging time by pH (mean ± std)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Condition")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------- отклонения от ref ----------

def add_delta_vs_ref(stats: pd.DataFrame, param_prefix: str) -> pd.DataFrame:
    """
    Добавляет столбец с отклонением от базового ref (0 weeks) в %.
    """
    mean_col = f"{param_prefix}_mean"

    # Находим базовое значение ref (0 weeks)
    ref_df = stats[stats["ph_group"] == "ref"]

    if ref_df.empty:
        stats[f"{param_prefix}_delta_pct"] = np.nan
        return stats

    global_ref_val = ref_df[mean_col].iloc[0]

    stats[f"{param_prefix}_delta_pct"] = (
            (stats[mean_col] - global_ref_val) / global_ref_val * 100.0
    )
    return stats


def bar_delta_vs_ref(
        delta_stats: pd.DataFrame,
        param_prefix: str,
        out_path: str,
) -> None:
    delta_col = f"{param_prefix}_delta_pct"

    df = delta_stats[delta_stats["ph_group"] != "ref"].copy()
    if df.empty:
        return

    ph_groups = ordered_ph_groups(df["ph_group"].unique().tolist())
    ph_groups = [g for g in ph_groups if g != "ref"]

    ages = sorted(df["age_label"].unique().tolist(), key=age_sort_key)
    x = np.arange(len(ph_groups))
    width = 0.8 / max(len(ages), 1)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, age in enumerate(ages):
        offset = (i - (len(ages) - 1) / 2) * width
        positions = x + offset

        values = []
        for ph in ph_groups:
            row = df[(df["ph_group"] == ph) & (df["age_label"] == age)]
            if row.empty:
                values.append(np.nan)
            else:
                values.append(row[delta_col].iloc[0])

        values = np.array(values, dtype=float)
        ax.bar(positions, values, width=width, label=age)

    ax.set_xticks(x)
    ax.set_xticklabels(ph_groups)
    ax.set_xlabel("Group")
    ax.set_ylabel("Δ vs ref [%]")
    ax.set_title(f"{param_prefix}: deviation from ref by pH and aging time")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Condition")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------- main ----------

def main() -> None:
    default_base = "./mnt/G/outputs"

    if len(sys.argv) >= 2:
        summary_path = sys.argv[1]
    else:
        summary_path = os.path.join(default_base, "summary_E_results.csv")

    default_out_dir = os.path.join(os.path.dirname(summary_path), "outputs_aging")

    if len(sys.argv) >= 3:
        out_dir = sys.argv[2]
    else:
        out_dir = default_out_dir

    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"Summary CSV not found: {os.path.abspath(summary_path)}")

    out_dir = ensure_output_folder(out_dir)

    df = pd.read_csv(summary_path)

    # Используем новые метки вместо голых цифр
    df["age_label"] = df["specimen_id"].apply(extract_age_label)
    df["ph_group"] = df["specimen_id"].apply(extract_ph_group)

    for col in ["Emod", "flexural_strength_MPa", "flexural_strain"]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in {summary_path}")

    df_clean = df.dropna(
        subset=["age_label", "ph_group", "Emod", "flexural_strength_MPa", "flexural_strain"]
    )

    stats = compute_stats(df_clean)

    # 1. Bar-графики mean ± std
    bar_param_by_ph_and_age(stats, "Emod", "Emod [GPa]", os.path.join(out_dir, "Emod_bar_by_pH_and_age.png"))
    bar_param_by_ph_and_age(stats, "flexural_strength", "Flexural strength [MPa]",
                            os.path.join(out_dir, "flexural_strength_bar_by_pH_and_age.png"))
    bar_param_by_ph_and_age(stats, "flexural_strain", "Flexural strain [-]",
                            os.path.join(out_dir, "flexural_strain_bar_by_pH_and_age.png"))

    # 2. Отклонения от ref
    for param_prefix in ["Emod", "flexural_strength", "flexural_strain"]:
        delta_stats = add_delta_vs_ref(stats, param_prefix)
        csv_name = f"{param_prefix}_delta_vs_ref.csv"
        delta_stats.to_csv(os.path.join(out_dir, csv_name), index=False)
        png_name = f"{param_prefix}_delta_bar_vs_ref.png"
        bar_delta_vs_ref(delta_stats, param_prefix, os.path.join(out_dir, png_name))

    print("Summary file:", os.path.abspath(summary_path))
    print("Plots and delta tables saved to:", os.path.abspath(out_dir))


if __name__ == "__main__":
    main()
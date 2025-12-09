# plot_aging_trends_by_pH.py
"""
Строит графики старения для epoxy по трём параметрам:
    • Emod (GPa)
    • Flexural strength (MPa)
    • Flexural strain (-)

Формат входного файла:
    summary_E_results.csv (как из process_3pb.py)

Выход:
    В папке out_dir (по умолчанию ./mnt/G/outputs/outputs_aging) создаются:
        - Emod_bar_by_pH_and_age.png
        - flexural_strength_bar_by_pH_and_age.png
        - flexural_strain_bar_by_pH_and_age.png

        - Emod_delta_vs_ref.csv
        - flexural_strength_delta_vs_ref.csv
        - flexural_strain_delta_vs_ref.csv

        - Emod_delta_bar_vs_ref.png
        - flexural_strength_delta_bar_vs_ref.png
        - flexural_strain_delta_bar_vs_ref.png

Графики:
    1) Для каждого параметра: столбчатые диаграммы mean ± std
       по группам pH, отдельные столбцы — разные сроки старения.
    2) Для каждого параметра: отклонение от ref для каждой недели, %.

Использование:
    python plot_aging_trends_by_pH.py summary_E_results.csv output_folder

Если аргументы не заданы:
    summary_E_results.csv берётся из ./mnt/G/outputs
    output_folder = ./mnt/G/outputs/outputs_aging
"""

from __future__ import annotations
import os
import sys
import re
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- parsing helpers ----------

def extract_age_weeks(specimen_id: str) -> float:
    """
    Извлекает срок старения в неделях из specimen_id.

    Ожидаемые варианты:
        "2weeks.xlsx - E_pH10_1"
        "4week - E_pH7_1"

    Если "week" не найдено — считаем, что это ref (0 недель).
    """
    s = str(specimen_id).lower()
    m = re.search(r"(\d+)\s*week", s)
    if m:
        return float(m.group(1))
    return 0.0


def extract_ph_group(specimen_id: str) -> str:
    """
    Извлекает pH-группу из specimen_id.

    Примеры:
        "2weeks.xlsx - E_pH10_1" -> "pH10"
        "E_pH4_3"                -> "pH4"
        "E_ref1"                 -> "ref"

    Если ничего не найдено — "unknown".
    """
    s = str(specimen_id).lower()

    if "ref" in s:
        return "ref"

    # Ищем именно "ph" + число
    m = re.search(r"ph[_\-]?(\d+)", s)
    if m:
        return f"pH{m.group(1)}"

    return "unknown"


def ensure_output_folder(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ---------- статистика ----------

def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Считает mean и std по каждой комбинации (ph_group, age_weeks).
    """
    stats = (
        df
        .groupby(["ph_group", "age_weeks"])
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
    """
    Стабильная сортировка pH-групп:
        ref, pH1, pH4, pH7, pH10, pH12, остальные по алфавиту.
    """
    groups = [g for g in groups if pd.notna(g)]
    unique = sorted(set(groups))

    # базовый порядок, который нас интересует
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
    """
    Строит bar-график mean ± std:
        ось X — pH-группы,
        по каждой группе — столбцы для разных сроков старения.
    """
    mean_col = f"{param_prefix}_mean"
    std_col = f"{param_prefix}_std"

    ph_groups = ordered_ph_groups(stats["ph_group"].unique().tolist())
    ages = sorted(stats["age_weeks"].unique().tolist())

    x = np.arange(len(ph_groups))
    width = 0.8 / max(len(ages), 1)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, age in enumerate(ages):
        # позиции столбцов для данного срока
        offset = (i - (len(ages) - 1) / 2) * width
        positions = x + offset

        means = []
        stds = []
        for ph in ph_groups:
            row = stats[(stats["ph_group"] == ph) & (stats["age_weeks"] == age)]
            if row.empty:
                means.append(np.nan)
                stds.append(0.0)
            else:
                means.append(row[mean_col].iloc[0])
                stds.append(row[std_col].iloc[0] if not np.isnan(row[std_col].iloc[0]) else 0.0)

        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)

        bars = ax.bar(positions, means, width=width, label=f"{age:g} weeks")
        ax.errorbar(
            positions,
            means,
            yerr=stds,
            fmt="none",
            capsize=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(ph_groups)
    ax.set_xlabel("Group")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{param_prefix} vs aging time by pH (mean ± std)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Aging time")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------- отклонения от ref ----------

def add_delta_vs_ref(stats: pd.DataFrame, param_prefix: str) -> pd.DataFrame:
    """
    Добавляет столбец с отклонением от ref в % для каждого срока старения.

    Δ% = (value - ref_same_week) / ref_same_week * 100
    """
    mean_col = f"{param_prefix}_mean"

    # ref по каждой неделе
    ref = (
        stats[stats["ph_group"] == "ref"]
        [["age_weeks", mean_col]]
        .rename(columns={mean_col: "ref_mean"})
    )

    merged = stats.merge(ref, on="age_weeks", how="left")
    merged[f"{param_prefix}_delta_pct"] = (
        (merged[mean_col] - merged["ref_mean"]) / merged["ref_mean"] * 100.0
    )
    return merged


def bar_delta_vs_ref(
    delta_stats: pd.DataFrame,
    param_prefix: str,
    out_path: str,
) -> None:
    """
    Строит bar-графики Δ% от ref:
        ось X — pH-группы (кроме ref),
        по каждой группе — столбцы для разных сроков старения.
    """
    delta_col = f"{param_prefix}_delta_pct"

    # исключаем сам ref, т.к. там всегда 0%
    df = delta_stats[delta_stats["ph_group"] != "ref"].copy()
    if df.empty:
        return

    ph_groups = ordered_ph_groups(df["ph_group"].unique().tolist())
    ph_groups = [g for g in ph_groups if g != "ref"]

    ages = sorted(df["age_weeks"].unique().tolist())
    x = np.arange(len(ph_groups))
    width = 0.8 / max(len(ages), 1)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, age in enumerate(ages):
        offset = (i - (len(ages) - 1) / 2) * width
        positions = x + offset

        values = []
        for ph in ph_groups:
            row = df[(df["ph_group"] == ph) & (df["age_weeks"] == age)]
            if row.empty:
                values.append(np.nan)
            else:
                values.append(row[delta_col].iloc[0])

        values = np.array(values, dtype=float)
        ax.bar(positions, values, width=width, label=f"{age:g} weeks")

    ax.set_xticks(x)
    ax.set_xticklabels(ph_groups)
    ax.set_xlabel("Group")
    ax.set_ylabel("Δ vs ref [%]")
    ax.set_title(f"{param_prefix}: deviation from ref by pH and aging time")
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Aging time")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------- main ----------

def main() -> None:
    default_base = "./mnt/E/outputs"

    # путь к summary
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

    # читаем данные
    df = pd.read_csv(summary_path)

    # извлекаем срок и pH
    df["age_weeks"] = df["specimen_id"].apply(extract_age_weeks)
    df["ph_group"] = df["specimen_id"].apply(extract_ph_group)

    # проверяем нужные колонки
    for col in ["Emod", "flexural_strength_MPa", "flexural_strain"]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in {summary_path}")

    df_clean = df.dropna(
        subset=["age_weeks", "ph_group", "Emod", "flexural_strength_MPa", "flexural_strain"]
    )

    # общая статистика
    stats = compute_stats(df_clean)

    # --- 1. bar-графики mean ± std ---
    bar_param_by_ph_and_age(
        stats,
        param_prefix="Emod",
        ylabel="Emod [GPa]",
        out_path=os.path.join(out_dir, "Emod_bar_by_pH_and_age.png"),
    )

    bar_param_by_ph_and_age(
        stats,
        param_prefix="flexural_strength",
        ylabel="Flexural strength [MPa]",
        out_path=os.path.join(out_dir, "flexural_strength_bar_by_pH_and_age.png"),
    )

    bar_param_by_ph_and_age(
        stats,
        param_prefix="flexural_strain",
        ylabel="Flexural strain [-]",
        out_path=os.path.join(out_dir, "flexural_strain_bar_by_pH_and_age.png"),
    )

    # --- 2. отклонение от ref в % ---
    for param_prefix in ["Emod", "flexural_strength", "flexural_strain"]:
        delta_stats = add_delta_vs_ref(stats, param_prefix)
        # сохраняем таблицу
        csv_name = f"{param_prefix}_delta_vs_ref.csv"
        delta_stats.to_csv(os.path.join(out_dir, csv_name), index=False)

        # график отклонений
        png_name = f"{param_prefix}_delta_bar_vs_ref.png"
        bar_delta_vs_ref(
            delta_stats,
            param_prefix=param_prefix,
            out_path=os.path.join(out_dir, png_name),
        )

    print("Summary file:", os.path.abspath(summary_path))
    print("Plots and delta tables saved to:", os.path.abspath(out_dir))


if __name__ == "__main__":
    main()

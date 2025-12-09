#!/usr/bin/env python3
"""
Full TGA Friedman analysis workflow:

1) Parse raw txt files (Mettler "Curve Values") -> CSV (data/).
2) Compute alpha, d(alpha)/dt.
3) Perform Friedman isoconversional analysis:
   ln(dalpha_dt) vs 1/T for alpha-grid.
4) Generate:
   - PNG plots
   - CSV with E(alpha)
   - Excel report with key tables.

Requirements:
    pip install numpy pandas matplotlib openpyxl
"""

import os
import math
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# User-configurable settings
# -----------------------------

# Where raw .txt files lie
RAW_DIR = "./raw_txt"
# Where intermediate CSVs and processed data go
DATA_DIR = "./data"
# Where results & figures go
OUT_DIR = "./output"
# Path for Excel report
REPORT_XLSX = os.path.join(OUT_DIR, "tga_friedman_report.xlsx")


# Alpha grid for Friedman analysis
ALPHA_START = 0.05
ALPHA_STOP = 0.95
ALPHA_STEP = 0.05

# Smoothing window for alpha(T) and T before differentiation
SMOOTH_WINDOW = 7

R_GAS = 8.314462618  # J/(mol*K)

def get_beta_from_filename(fname: str) -> float | None:
    """
    Извлекает скорость нагрева из имени файла.
    Ожидает подстроку вида '10min', '15min', '20min', '30min', …
    Возвращает beta [K/min] или None, если не найдено.
    """
    name = os.path.basename(fname).lower()
    m = re.search(r"(\d+)\s*min", name)
    if not m:
        return None
    return float(m.group(1))


def convert_all_txt_to_csv() -> list[str]:
    """
    Берёт все файлы в RAW_DIR (расширения .txt и .csv),
    пытается:
      1) вытащить beta из имени,
      2) распарсить в таблицу Index/Ts/Value,
      3) пересчитать в CSV для Фридмана.

    Возвращает список путей к созданным CSV.
    """
    ensure_dirs()
    created: list[str] = []

    for fname in sorted(os.listdir(RAW_DIR)):
        if not (fname.lower().endswith(".txt") or fname.lower().endswith(".csv")):
            continue

        in_path = os.path.join(RAW_DIR, fname)

        beta = get_beta_from_filename(fname)
        if beta is None:
            print(f"[SKIP] Cannot detect heating rate from filename: {fname}")
            continue

        try:
            print(f"[TXT->CSV] {fname}, beta={beta} K/min")
            df_txt = parse_curve_values_txt(in_path)
            df_csv = build_csv_for_friedman(df_txt, beta)
        except Exception as e:
            print(f"[ERROR] Failed to convert {fname}: {e}")
            continue

        out_name = os.path.splitext(fname)[0] + ".csv"
        out_path = os.path.join(DATA_DIR, out_name)
        df_csv.to_csv(out_path, index=False)
        created.append(out_path)
        print(f"  -> {out_path}")

    if not created:
        print("[WARN] No files converted in RAW_DIR.")

    return created

# -----------------------------
# Utilities
# -----------------------------

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Simple centered moving average, edge-padded."""
    if window < 3:
        return x
    k = int(window)
    if k % 2 == 0:
        k += 1
    pad = k // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(k) / k
    return np.convolve(xpad, kernel, mode="valid")


# -----------------------------
# Stage 1: TXT -> CSV
# -----------------------------

def _read_raw_table(path: str) -> pd.DataFrame:
    """
    Пробуем прочитать файл как CSV без заголовка с несколькими кодировками.
    Возвращаем "сырой" DataFrame (как есть).
    """
    encodings_to_try = ["utf-8", "utf-16", "latin1"]

    last_err = None
    for enc in encodings_to_try:
        try:
            raw = pd.read_csv(path, header=None, encoding=enc)
            return raw
        except UnicodeDecodeError as e:
            last_err = e
            continue

    # если все варианты упали – поднимаем последнюю ошибку
    raise UnicodeDecodeError(
        last_err.encoding,
        last_err.object,
        last_err.start,
        last_err.end,
        f"All encodings failed for {path}: {last_err.reason}",
    )


def parse_curve_values_txt(path: str) -> pd.DataFrame:
    """
    Парсер файлов вида:

    Values:,,,,,
    Index,Ts,Value,,,
    [°C],[%],,,,
    0,42.3473,100,,,
    5,42.7143,100.032,,,
    ...

    Возвращает DataFrame с колонками:
        Index, Ts_C, Value_percent
    """
    raw = _read_raw_table(path)

    # ищем строку, где первый столбец == 'Index'
    col0 = raw[0].astype(str).str.strip().str.lower()
    header_idx = np.where(col0 == "index")[0]
    if header_idx.size == 0:
        raise ValueError(
            f"Header row with 'Index' not found in {path}. "
            f"First 10 rows:\n{raw.head(10)}"
        )
    header_row = header_idx[0]

    # строка с единицами ([°C],[%]) сразу после заголовка -> пропускаем её
    data_start = header_row + 2

    # данные начинаются с data_start до конца файла
    data = raw.iloc[data_start:].copy()

    # берём первые три колонки: Index, Ts, Value
    data = data.iloc[:, :3]
    data.columns = ["Index", "Ts_C", "Value_percent"]

    # числа с запятой → с точкой
    data["Ts_C"] = data["Ts_C"].astype(str).str.replace(",", ".", regex=False)
    data["Value_percent"] = data["Value_percent"].astype(str).str.replace(",", ".", regex=False)

    # приводим типы
    data["Index"] = pd.to_numeric(data["Index"], errors="coerce")
    data["Ts_C"] = pd.to_numeric(data["Ts_C"], errors="coerce")
    data["Value_percent"] = pd.to_numeric(data["Value_percent"], errors="coerce")

    # убираем строки без чисел
    data = data.dropna(subset=["Index", "Ts_C", "Value_percent"])

    # сортируем по Index
    data = data.sort_values("Index").reset_index(drop=True)

    if data.empty:
        raise ValueError(f"No numeric curve values parsed from {path}")

    return data

def build_csv_for_friedman(df: pd.DataFrame,
                           beta_K_per_min: float) -> pd.DataFrame:
    """
    From Ts_C [°C] and Value_percent [%] to Friedman CSV:
        temperature_K, time_s, mass_fraction, beta_K_per_min
    - mass_fraction is normalized to initial value.
    - time_s is reconstructed from T and beta [K/min].
    """
    Ts_C = df["Ts_C"].to_numpy(dtype=float)
    val = df["Value_percent"].to_numpy(dtype=float)

    T_K = Ts_C + 273.15
    mass_fraction = val / val[0]  # normalize to initial

    beta_K_per_s = beta_K_per_min / 60.0
    T0 = T_K[0]
    # t = (T - T0)/beta
    time_s = (T_K - T0) / beta_K_per_s

    out = pd.DataFrame({
        "temperature_K": T_K,
        "time_s": time_s,
        "mass_fraction": mass_fraction,
        "beta_K_per_min": beta_K_per_min,
    })
    return out

# -----------------------------
# Stage 2: Friedman analysis
# -----------------------------

@dataclass
class RunData:
    name: str
    beta_K_per_min: float
    beta_K_per_s: float
    T_K: np.ndarray
    t_s: np.ndarray
    mass_fraction: np.ndarray
    alpha: np.ndarray
    dalpha_dt: np.ndarray


def compute_alpha_from_mass_fraction(mf: np.ndarray) -> np.ndarray:
    """
    Compute conversion alpha from mass_fraction m(t)/m0.
    Assume plateau at the end as mf_inf.
    """
    mf0 = mf[0]
    # plateau at the end: average of last 50 points (or all)
    if len(mf) >= 50:
        mf_inf = mf[-50:].mean()
    else:
        mf_inf = mf[-1]
    if mf_inf >= mf0:
        # fallback: use min as final
        mf_inf = mf.min()
    alpha = (mf0 - mf) / max(1e-12, (mf0 - mf_inf))
    return np.clip(alpha, 0.0, 1.0)


def load_runs_from_csv(pattern: str = "*.csv") -> List[RunData]:
    """
    Load CSV files from DATA_DIR and compute alpha, dalpha/dt.
    """
    ensure_dirs()
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    if len(csv_files) < 2:
        raise RuntimeError("Need at least 2 CSV runs in ./data for Friedman.")

    runs: List[RunData] = []

    for path in csv_files:
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}

        Tcol = cols.get("temperature_k")
        tcol = cols.get("time_s")
        mfcol = cols.get("mass_fraction")
        betacol = cols.get("beta_k_per_min")

        if Tcol is None or tcol is None or mfcol is None:
            print(f"[SKIP] Missing required columns in {path}")
            continue

        T_K = df[Tcol].to_numpy(dtype=float)
        t_s = df[tcol].to_numpy(dtype=float)
        mf = df[mfcol].to_numpy(dtype=float)

        if betacol is not None:
            beta_K_per_min = float(df[betacol].iloc[0])
        else:
            # estimate from dT/dt
            dTdt = np.gradient(T_K, t_s)
            beta_K_per_min = np.median(dTdt) * 60.0

        beta_K_per_s = beta_K_per_min / 60.0

        alpha_raw = compute_alpha_from_mass_fraction(mf)

        # Smooth T and alpha for derivative
        alpha_s = moving_average(alpha_raw, SMOOTH_WINDOW)
        T_s = moving_average(T_K, SMOOTH_WINDOW)

        # d(alpha)/dT then * beta
        dadt_dT = np.gradient(alpha_s, T_s)
        dalpha_dt = dadt_dT * beta_K_per_s

        run = RunData(
            name=os.path.basename(path),
            beta_K_per_min=beta_K_per_min,
            beta_K_per_s=beta_K_per_s,
            T_K=T_K,
            t_s=t_s,
            mass_fraction=mf,
            alpha=alpha_raw,
            dalpha_dt=dalpha_dt,
        )
        runs.append(run)
        print(f"[LOAD] {run.name}: beta={beta_K_per_min} K/min, N={len(T_K)}")

    if len(runs) < 2:
        raise RuntimeError("After loading, <2 runs are available. Check data.")

    return runs


def interp_at_alpha(run: RunData,
                    alpha_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For a single run, interpolate T and dalpha_dt at each alpha value in alpha_grid.
    Returns T(alpha), dalpha_dt(alpha).
    """
    # sort by alpha
    order = np.argsort(run.alpha)
    a = run.alpha[order]
    T = run.T_K[order]
    r = run.dalpha_dt[order]

    # remove duplicate alphas (keep first)
    uniq_a, idx = np.unique(a, return_index=True)
    a = uniq_a
    T = T[idx]
    r = r[idx]

    mask = (a >= 0.0) & (a <= 1.0)
    a = a[mask]
    T = T[mask]
    r = r[mask]

    T_of_alpha = np.full_like(alpha_grid, np.nan, dtype=float)
    r_of_alpha = np.full_like(alpha_grid, np.nan, dtype=float)

    if len(a) < 3:
        return T_of_alpha, r_of_alpha

    T_of_alpha = np.interp(alpha_grid, a, T, left=np.nan, right=np.nan)
    r_of_alpha = np.interp(alpha_grid, a, r, left=np.nan, right=np.nan)
    return T_of_alpha, r_of_alpha


def friedman_fit(alpha_grid: np.ndarray,
                 T_mat: np.ndarray,
                 rate_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each alpha, perform linear regression:
        ln(dalpha_dt) = b + m * (1/T)
    E(alpha) = -m * R_GAS, lnA_app = b
    """
    E = np.full_like(alpha_grid, np.nan, dtype=float)
    lnA_app = np.full_like(alpha_grid, np.nan, dtype=float)

    for j, a in enumerate(alpha_grid):
        Tvals = T_mat[:, j]
        rvals = rate_mat[:, j]

        mask = np.isfinite(Tvals) & np.isfinite(rvals) & (Tvals > 0) & (rvals > 0)
        if mask.sum() < 2:
            continue

        x = 1.0 / Tvals[mask]
        y = np.log(rvals[mask])

        A = np.vstack([np.ones_like(x), x]).T
        coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
        b, m = coeff  # y = b + m x

        E[j] = -m * R_GAS
        lnA_app[j] = b

    return E, lnA_app


def run_friedman_analysis(runs: List[RunData]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main Friedman pipeline:
    - Build alpha_grid
    - Interpolate T and dalpha_dt for each run at each alpha
    - Fit Friedman for each alpha

    Returns:
        friedman_points_df, E_alpha_df
    """
    alpha_grid = np.arange(ALPHA_START, ALPHA_STOP + 1e-12, ALPHA_STEP)

    T_rows = []
    rate_rows = []
    run_names = [r.name for r in runs]
    betas = [r.beta_K_per_min for r in runs]

    # collect interpolation data and also build "points" table
    records_points = []

    for run in runs:
        T_of_a, r_of_a = interp_at_alpha(run, alpha_grid)
        T_rows.append(T_of_a)
        rate_rows.append(r_of_a)

        for a, T_val, rate_val in zip(alpha_grid, T_of_a, r_of_a):
            if not (np.isfinite(T_val) and np.isfinite(rate_val) and rate_val > 0 and T_val > 0):
                continue
            records_points.append({
                "run": run.name,
                "beta_K_per_min": run.beta_K_per_min,
                "alpha": a,
                "T_K": T_val,
                "invT_1_per_K": 1.0 / T_val,
                "dalpha_dt_1_per_s": rate_val,
                "ln_dalpha_dt": math.log(rate_val),
            })

    T_mat = np.vstack(T_rows)
    rate_mat = np.vstack(rate_rows)

    # Friedman fits
    E_J, lnA = friedman_fit(alpha_grid, T_mat, rate_mat)

    records_E = []
    for a, E_j, lnA_j in zip(alpha_grid, E_J, lnA):
        if not np.isfinite(E_j):
            continue
        records_E.append({
            "alpha": a,
            "E_J_per_mol": E_j,
            "E_kJ_per_mol": E_j / 1000.0,
            "lnA_app": lnA_j,
        })

    friedman_points_df = pd.DataFrame(records_points)
    E_alpha_df = pd.DataFrame(records_E)

    return friedman_points_df, E_alpha_df


# -----------------------------
# Stage 3: Plots
# -----------------------------

def plot_tg_dtg(runs: List[RunData]):
    # TG: mass_fraction vs T
    plt.figure()
    for r in runs:
        plt.plot(r.T_K - 273.15, r.mass_fraction * 100.0,
                 label=f"{r.name} ({r.beta_K_per_min:.0f} K/min)")
    plt.xlabel("Temperature [°C]")
    plt.ylabel("Mass [% of initial]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "tg_all.png"), dpi=200)
    plt.close()

    # DTG: approximate dm/dT from mass_fraction
    plt.figure()
    for r in runs:
        mf_s = moving_average(r.mass_fraction, SMOOTH_WINDOW)
        T_s = moving_average(r.T_K, SMOOTH_WINDOW)
        dmf_dT = np.gradient(mf_s, T_s)
        plt.plot(T_s - 273.15, dmf_dT,
                 label=f"{r.name} ({r.beta_K_per_min:.0f} K/min)")
    plt.xlabel("Temperature [°C]")
    plt.ylabel("d(m/m0)/dT")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "dtg_all.png"), dpi=200)
    plt.close()


def plot_alpha_and_rates(runs: List[RunData]):
    # alpha vs T
    plt.figure()
    for r in runs:
        plt.plot(r.T_K - 273.15, r.alpha,
                 label=f"{r.name} ({r.beta_K_per_min:.0f} K/min)")
    plt.xlabel("Temperature [°C]")
    plt.ylabel("Conversion α [-]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "alpha_vs_T.png"), dpi=200)
    plt.close()

    # d(alpha)/dt vs T
    plt.figure()
    for r in runs:
        plt.plot(r.T_K - 273.15, r.dalpha_dt,
                 label=f"{r.name} ({r.beta_K_per_min:.0f} K/min)")
    plt.xlabel("Temperature [°C]")
    plt.ylabel("dα/dt [1/s]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "dalpha_dt_vs_T.png"), dpi=200)
    plt.close()


def plot_friedman_lines(friedman_points_df: pd.DataFrame,
                        n_alphas_to_plot: int = 6):
    """
    Scatter ln(dalpha_dt) vs 1/T for a few alpha levels.
    """
    if friedman_points_df.empty:
        return

    alphas = sorted(friedman_points_df["alpha"].unique())
    if len(alphas) <= n_alphas_to_plot:
        sel_alphas = alphas
    else:
        # pick evenly spaced alphas
        idx = np.linspace(0, len(alphas) - 1, n_alphas_to_plot).astype(int)
        sel_alphas = [alphas[i] for i in idx]

    plt.figure()
    for a in sel_alphas:
        df_a = friedman_points_df[ friedman_points_df["alpha"].round(4) == round(a, 4) ]
        x = df_a["invT_1_per_K"].to_numpy()
        y = df_a["ln_dalpha_dt"].to_numpy()
        plt.scatter(x, y, label=f"α={a:.2f}")

    plt.xlabel("1/T [1/K]")
    plt.ylabel("ln(dα/dt) [1/s]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "friedman_ln_dadt_vs_invT.png"), dpi=200)
    plt.close()


def plot_E_and_lnA(E_alpha_df: pd.DataFrame):
    if E_alpha_df.empty:
        return

    # E(alpha)
    plt.figure()
    plt.plot(E_alpha_df["alpha"], E_alpha_df["E_kJ_per_mol"])
    plt.xlabel("α [-]")
    plt.ylabel("E(α) [kJ/mol]")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "E_alpha.png"), dpi=200)
    plt.close()

    # lnA_app(alpha)
    plt.figure()
    plt.plot(E_alpha_df["alpha"], E_alpha_df["lnA_app"])
    plt.xlabel("α [-]")
    plt.ylabel("ln[A·f(α)] (apparent)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "lnA_alpha.png"), dpi=200)
    plt.close()


# -----------------------------
# Stage 4: Excel report
# -----------------------------

def write_excel_report(runs: List[RunData],
                       friedman_points_df: pd.DataFrame,
                       E_alpha_df: pd.DataFrame):
    """
    Create Excel report with sheets:
      - runs_alpha: T, alpha, dalpha_dt per run
      - friedman_points: points used for regression
      - E_alpha: E(alpha) & lnA_app
    """
    ensure_dirs()

    with pd.ExcelWriter(REPORT_XLSX, engine="openpyxl") as writer:
        # Sheet 1: per-run alpha & rate
        records_runs = []
        for r in runs:
            for T, t, mf, a, rate in zip(r.T_K, r.t_s, r.mass_fraction,
                                         r.alpha, r.dalpha_dt):
                records_runs.append({
                    "run": r.name,
                    "beta_K_per_min": r.beta_K_per_min,
                    "T_K": T,
                    "T_C": T - 273.15,
                    "time_s": t,
                    "mass_fraction": mf,
                    "alpha": a,
                    "dalpha_dt_1_per_s": rate,
                })
        df_runs = pd.DataFrame(records_runs)
        df_runs.to_excel(writer, sheet_name="runs_alpha", index=False)

        # Sheet 2: Friedman points
        if not friedman_points_df.empty:
            friedman_points_df.to_excel(writer, sheet_name="friedman_points",
                                        index=False)

        # Sheet 3: E(alpha)
        if not E_alpha_df.empty:
            E_alpha_df.to_excel(writer, sheet_name="E_alpha", index=False)

    print(f"[REPORT] Excel written to: {REPORT_XLSX}")


# -----------------------------
# Main
# -----------------------------

def main():
    ensure_dirs()

    print("=== Stage 1: TXT -> CSV ===")
    convert_all_txt_to_csv()

    print("=== Stage 2: Friedman analysis ===")
    runs = load_runs_from_csv(pattern="0_sist_*_air.csv")
    friedman_points_df, E_alpha_df = run_friedman_analysis(runs)

    friedman_points_path = os.path.join(OUT_DIR, "friedman_points.csv")
    E_alpha_path = os.path.join(OUT_DIR, "friedman_E_alpha.csv")
    friedman_points_df.to_csv(friedman_points_path, index=False)
    E_alpha_df.to_csv(E_alpha_path, index=False)
    print(f"[OUT] {friedman_points_path}")
    print(f"[OUT] {E_alpha_path}")

    print("=== Stage 3: Plots ===")
    plot_tg_dtg(runs)
    plot_alpha_and_rates(runs)
    plot_friedman_lines(friedman_points_df)
    plot_E_and_lnA(E_alpha_df)
    print(f"[OUT] Plots saved to {OUT_DIR}/")

    print("=== Stage 4: Excel report ===")
    write_excel_report(runs, friedman_points_df, E_alpha_df)

    print("Done.")


if __name__ == "__main__":
    main()

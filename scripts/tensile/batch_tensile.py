"""
Batch processor for tensile test workbooks:
- Reads the data sheet whose name equals the file stem.
- Reads metadata from the "Results" sheet (or the closest alternative).
- Removes unit/header helper rows.
- Computes engineering strain/stress, Young's modulus (on a strain window),
  failure time (at force maximum), Hollomon (n, K) on the true plastic strain
  segment, and 0.2% proof stress Rp0.2 via the offset method.
- Renders PNG stress–strain with σ_max and Rp0.2 markers (guides + offset line).
- Writes Combined/Summary CSV and an Excel report with a pivot.
- Builds an interactive Plotly overview grouped by temperature.

Each step is a separate function with short explanations.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# Optional, for the HTML overview plot
try:
    import plotly.express as px
except Exception:  # pragma: no cover
    px = None

# ---------------- Configuration ----------------
# Data-sheet column names (case-insensitive)
K_TIME   = "Test time"            # seconds
K_FORCE  = "Standard force"       # N
K_TRAVEL = "Standard travel"      # mm
K_STRAIN = "Fine strain"          # percent (%)
K_TEMP   = "Specimen temperature" # °C

# Metadata sheet name and fields
SHEET_META = "Results"
M_SPEED = "Test speed"    # mm/min
M_FMAX  = "Fmax"          # N
M_DLF   = "dL at Fmax"    # mm
M_A0    = "a0"            # mm
M_B0    = "b0"            # mm
M_S0    = "S0"            # mm^2

# Strain window to estimate E on engineering values (fractions, not %)
E_STRAIN_WINDOW: Tuple[float, float] = (0.0005, 0.010)
E_MIN_POINTS = 5
ADD_E_TO_PNG = True
DRAW_FORCE_DISP = False

# Window for Hollomon fit on true plastic strain
HOLL_WINDOW: Tuple[float, float] = (0.002, 0.05)

# Offset for Rp0.2
PROOF_OFFSET = 0.2

# File with optional manual overrides for test speed
OVERRIDE_CSV = "data/overrides.csv"

# I/O paths
DATA_DIR = Path("data")
OUT_DIR  = Path("out")
(OUT_DIR / "plots").mkdir(parents=True, exist_ok=True)

# Unit-row detectors
UNITS_DATA_RE = re.compile(r"(?:\bs\b|\bN\b|\bmm\b|%|°C)", re.IGNORECASE)
UNITS_META_RE = re.compile(r"(?:mm/min|mm²|\bmm\b|\bN\b|%)", re.IGNORECASE)
UNITS_DATA_SET = {"s", "N", "mm", "%", "°C"}
UNITS_META_SET = {"mm/min", "N", "mm", "mm²", "%"}

# ---------------- Small utilities ----------------

def norm_cols(df: pd.DataFrame) -> Dict[str, str]:
    """Map {lowercased_name: original_name} for case-insensitive access."""
    return {str(c).lower().strip(): c for c in df.columns}


def looks_like_units_row(row: pd.Series, units_set: Set[str]) -> bool:
    """True if all non-empty cells look like unit labels."""
    vals = {str(x).strip() for x in row.tolist() if pd.notna(x) and str(x).strip() != ""}
    return len(vals) > 0 and vals.issubset(units_set)


def safe_excel_reader(path: Path) -> pd.ExcelFile:
    """Open Excel with fallback engines: calamine → openpyxl/xlrd."""
    try:
        return pd.ExcelFile(path, engine="calamine")
    except Exception:
        try:
            return pd.ExcelFile(path, engine="openpyxl")
        except Exception:
            return pd.ExcelFile(path)


def get_number(meta_df: pd.DataFrame, key: str) -> float:
    """Read a single numeric value from META by column name; NaN if missing."""
    mc = norm_cols(meta_df)
    if key.lower() in mc:
        v = pd.to_numeric(meta_df[mc[key.lower()]].iloc[0], errors="coerce")
        return float(v) if pd.notna(v) else float("nan")
    return float("nan")


def get_area(meta_df: pd.DataFrame) -> Optional[float]:
    """Cross-sectional area: prefer S0, else a0*b0. Units: mm²."""
    mc = norm_cols(meta_df)
    if M_S0.lower() in mc:
        v = pd.to_numeric(meta_df[mc[M_S0.lower()]].iloc[0], errors="coerce")
        if pd.notna(v) and v > 0:
            return float(v)
    a = pd.to_numeric(meta_df[mc.get(M_A0.lower(), M_A0)].iloc[0], errors="coerce") if M_A0.lower() in mc else np.nan
    b = pd.to_numeric(meta_df[mc.get(M_B0.lower(), M_B0)].iloc[0], errors="coerce") if M_B0.lower() in mc else np.nan
    if pd.notna(a) and pd.notna(b) and a > 0 and b > 0:
        return float(a * b)
    return None


def load_overrides(path: Path) -> Dict[str, float]:
    """Load speed overrides by file name or stem."""
    ov: Dict[str, float] = {}
    if path.exists():
        df = pd.read_csv(path)
        required = {"File", "TestSpeed_mm_min"}
        if not required.issubset(set(df.columns)):
            raise ValueError("overrides.csv must contain columns: File,TestSpeed_mm_min")
        for _, r in df.iterrows():
            fname = str(r["File"]).strip()
            speed = float(r["TestSpeed_mm_min"])  # type: ignore[arg-type]
            ov[fname] = speed
            ov[Path(fname).stem] = speed
    return ov

# ---------------- Workbook parsing ----------------

def parse_data_sheet(xls: pd.ExcelFile, base: str) -> tuple[pd.DataFrame, str]:
    """Read the data sheet whose name equals the file stem."""
    sheet = next((s for s in xls.sheet_names if s.strip().lower() == base.strip().lower()), None)
    if not sheet:
        raise ValueError(f"Data sheet '{base}' not found")
    df = xls.parse(sheet_name=sheet, header=1)
    # Drop the units row if present
    if not df.empty and (looks_like_units_row(df.iloc[0], UNITS_DATA_SET) or df.iloc[0].astype(str).str.contains(UNITS_DATA_RE, na=False).any()):
        df = df.drop(index=0).reset_index(drop=True)
    return df, sheet


def parse_meta_sheet(xls: pd.ExcelFile, data_sheet_name: str) -> pd.DataFrame:
    """Read META sheet (Results) or the closest alternative."""
    if SHEET_META in xls.sheet_names:
        meta = xls.parse(sheet_name=SHEET_META, header=0)
    else:
        fallback = next((s for s in xls.sheet_names if s != data_sheet_name), None)
        if not fallback:
            raise ValueError("Meta sheet not found")
        meta = xls.parse(sheet_name=fallback, header=0)
    if not meta.empty and (looks_like_units_row(meta.iloc[0], UNITS_META_SET)
                           or meta.iloc[0].astype(str).str.contains(UNITS_META_RE, na=False).any()):
        meta = meta.drop(index=0).reset_index(drop=True)
    return meta

# ---------------- Row processing (clean-up) ----------------

def baseline_shift(series: pd.Series, method: str = "mean_first_n", n: int = 20) -> tuple[pd.Series, float]:
    """Zero-shift by subtracting first/min/mean of the first n samples."""
    if method == "first":
        shift = float(series.iloc[0])
    elif method == "min":
        shift = float(series.min())
    elif method == "mean_first_n":
        n = max(1, min(n, len(series)))
        shift = float(series.iloc[:n].mean())
    else:
        raise ValueError("Unknown baseline method")
    return series - shift, shift


def enforce_monotonic_strain(df: pd.DataFrame, strain_col: str) -> tuple[pd.DataFrame, List[int]]:
    """Enforce non-decreasing strain: drop sensor rollback points."""
    keep = [0]
    last = float(df[strain_col].iloc[0])
    for i in range(1, len(df)):
        s = float(df[strain_col].iloc[i])
        if s >= last:
            keep.append(i)
            last = s
    removed = [i for i in range(len(df)) if i not in keep]
    return df.iloc[keep].reset_index(drop=True), removed


def hampel_mask(x: pd.Series, window: int = 7, k: float = 4.0) -> pd.Series:
    """Hampel outlier mask (robust to isolated spikes)."""
    med = x.rolling(window=window, center=True, min_periods=1).median()
    resid = x - med
    mad = 1.4826 * np.median(np.abs(resid.dropna()))
    mad = float(mad if np.isfinite(mad) and mad > 0 else 1e-9)
    z = np.abs(resid) / mad
    return z > k


def remove_spikes(df: pd.DataFrame, y_col: str, window: int = 7, k: float = 6.0) -> tuple[pd.DataFrame, List[int]]:
    """Remove outliers using first differences with a rolling median."""
    dy = df[y_col].diff().fillna(0.0)
    med = dy.rolling(window, center=True, min_periods=3).median()
    resid = dy - med
    mad = np.median(np.abs(resid.dropna().values))
    thr = (mad if np.isfinite(mad) and mad > 0 else 1e-9) * k
    mask = np.abs(resid) > thr
    idx = df.index[mask].tolist()
    return df.drop(index=idx).reset_index(drop=True), idx


def remove_sigma_jumps(df: pd.DataFrame,
                       strain_col: str = "Strain",
                       stress_col: str = "Stress_MPa",
                       window: int = 9,
                       k: float = 8.0,
                       min_dstrain: float = 1e-4) -> tuple[pd.DataFrame, List[int]]:
    """
    Filter “needle” artifacts on σ–ε: abnormally large Δσ at tiny Δε.
    Uses MAD on Δσ and a |Δε| threshold.
    """
    d_eps = df[strain_col].diff().to_numpy()
    d_sig = df[stress_col].diff().to_numpy()

    vert_mask = (np.abs(d_eps) < min_dstrain) & (np.abs(d_sig) > 0)

    s = pd.Series(d_sig)
    med = s.rolling(window, center=True, min_periods=3).median()
    resid = s - med
    mad = np.median(np.abs(resid.dropna()))
    thr = (mad if np.isfinite(mad) and mad > 0 else 1e-9) * k

    jump_mask = np.abs(resid) > thr
    mask = vert_mask | jump_mask.to_numpy()

    idx = df.index[mask].tolist()
    if not idx:
        return df.reset_index(drop=True), []
    return df.drop(index=idx).reset_index(drop=True), idx


def trim_post_peak(df: pd.DataFrame, y_col: str) -> tuple[pd.DataFrame, int, int]:
    """Trim everything after the maximum of y_col (keep the peak)."""
    idx_peak = int(df[y_col].idxmax())
    before = len(df)
    out = df.loc[:idx_peak].copy()
    return out.reset_index(drop=True), idx_peak, before - len(out)

# ---------------- Transforms and metrics ----------------

def add_engineering(df: pd.DataFrame, area_mm2: float) -> pd.DataFrame:
    """Add engineering strain and stress (MPa)."""
    out = df.copy()
    out["Strain"] = pd.to_numeric(out["Strain"], errors="coerce")
    out["Stress_MPa"] = pd.to_numeric(out["Force_N"], errors="coerce") / float(area_mm2)
    return out.dropna(subset=["Stress_MPa", "Strain"]).reset_index(drop=True)


def add_true(df: pd.DataFrame) -> pd.DataFrame:
    """Add true strain/stress assuming volume constancy."""
    out = df.copy()
    out["TrueStrain"] = np.log(1.0 + out["Strain"].astype(float))
    out["TrueStress_MPa"] = out["Stress_MPa"].astype(float) * (1.0 + out["Strain"].astype(float))
    return out


def estimate_E(df: pd.DataFrame, window: Tuple[float, float] = E_STRAIN_WINDOW, min_pts: int = E_MIN_POINTS) -> float:
    """Linear fit σ–ε on small strains → Young's modulus, MPa."""
    lo, hi = window
    seg = df[(df["Strain"] >= lo) & (df["Strain"] <= hi)]
    if len(seg) < max(min_pts, 2):
        return float("nan")
    x = seg["Strain"].to_numpy()
    y = seg["Stress_MPa"].to_numpy()
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m)


def proof_stress_rp02(df: pd.DataFrame, E_MPa: float, offset: float = PROOF_OFFSET) -> tuple[float, float, float]:
    """
    Find Rp0.2 (MPa), strain at Rp0.2, and corresponding Time_s (s).
    """
    if not np.isfinite(E_MPa) or E_MPa <= 0:
        return float("nan"), float("nan"), float("nan")

    eps = df["Strain"].to_numpy(dtype=float)
    sig = df["Stress_MPa"].to_numpy(dtype=float)
    time = df["Time_s"].to_numpy(dtype=float) if "Time_s" in df.columns else np.full_like(eps, np.nan)

    g = sig - E_MPa * (eps - offset)

    for i in range(1, len(g)):
        if np.sign(g[i-1]) == np.sign(g[i]):
            continue
        # linear interpolation
        t = abs(g[i-1]) / (abs(g[i-1]) + abs(g[i]))
        e = (1 - t) * eps[i-1] + t * eps[i]
        s = (1 - t) * sig[i-1] + t * sig[i]
        ts = (1 - t) * time[i-1] + t * time[i]
        return float(s), float(e), float(ts)

    return float("nan"), float("nan"), float("nan")


def hollomon_true(df_true: pd.DataFrame, E_MPa: float, window: Tuple[float, float] = HOLL_WINDOW) -> tuple[float, float]:
    """Estimate Hollomon parameters from σ_true = K · (ε_pl_true)^n within a window."""
    if not np.isfinite(E_MPa) or E_MPa <= 0:
        return float("nan"), float("nan")
    eps_t = df_true["TrueStrain"].to_numpy(dtype=float)
    sig_t = df_true["TrueStress_MPa"].to_numpy(dtype=float)
    eps_pl = eps_t - sig_t / E_MPa  # истинная пластическая
    mask = (eps_pl >= window[0]) & (eps_pl <= window[1]) & (eps_pl > 0)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    X = np.log(eps_pl[mask])
    Y = np.log(sig_t[mask])
    A = np.vstack([X, np.ones_like(X)]).T
    n, lnK = np.linalg.lstsq(A, Y, rcond=None)[0]
    K = float(np.exp(lnK))
    return float(n), K


def peak_and_failure(df: pd.DataFrame) -> tuple[float, float, float]:
    """σ_max (MPa), ε at σ_max, and failure time (by F_max)."""
    i_sig = int(df["Stress_MPa"].idxmax())
    sig_max = float(df.loc[i_sig, "Stress_MPa"])
    eps_at = float(df.loc[i_sig, "Strain"])
    # time at maximum power
    i_f = int(df["Force_N"].idxmax())
    t_fail = float(df.loc[i_f, "Time_s"]) if "Time_s" in df.columns else float("nan")
    return sig_max, eps_at, t_fail

# ---------------- Visualization ----------------

def plot_sigma_epsilon(df: pd.DataFrame, out_png: Path, E_MPa: float | None = None, rp02: Tuple[float, float] | None = None,
                       title: str = "Engineering stress–strain") -> None:
    """PNG with σ–ε, dashed guides for σ_max and Rp0.2, and the offset line."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot(df["Strain"], df["Stress_MPa"], label="σ–ε")

    # peak marker
    sig_max, eps_at, _ = peak_and_failure(df)
    ax.axhline(sig_max, ls="--", alpha=0.5)
    ax.axvline(eps_at, ls="--", alpha=0.5)
    ax.annotate(f"σ_max={sig_max:.2f} MPa", xy=(eps_at, sig_max), xytext=(5,5), textcoords="offset points")

    # Rp0.2 and offset line
    if rp02 and all(np.isfinite(rp02)):
        s02, e02, _ = rp02
        ax.axhline(s02, ls="--", color="grey", alpha=0.6)
        ax.axvline(e02, ls="--", color="grey", alpha=0.6)
        ax.annotate(f"Rp0.2={s02:.2f} MPa", xy=(e02, s02), xytext=(5,-15), textcoords="offset points")
    if E_MPa and np.isfinite(E_MPa) and E_MPa > 0:
        xs = np.linspace(0, max(df["Strain"].max(), (rp02[1] if rp02 else 0.01)) * 1.05, 200)
        ys = E_MPa * (xs - PROOF_OFFSET)
        ax.plot(xs, ys, lw=1.0, alpha=0.7, label=f"offset line ({PROOF_OFFSET:.1f}%)")

    ax.set_xlabel("Engineering strain, ε")
    ax.set_ylabel("Engineering stress, MPa")
    ax.set_ylim(0, 1.2*max(df["Stress_MPa"]))
    # ax.set_xlim(0,  PROOF_OFFSET* 12)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

# ---------------- Main per-file pipeline ----------------

def process_file(path: Path, overrides: Dict[str, float]) -> tuple[pd.DataFrame, Dict]:
    """Processing a single file: cleanup, metrics, graphs. Returns (df, summary)."""
    xls = safe_excel_reader(path)
    base = path.stem

    data_df, data_sheet_name = parse_data_sheet(xls, base)
    meta_df = parse_meta_sheet(xls, data_sheet_name)

    # Column mapping
    dc = norm_cols(data_df)
    need = {K_TIME, K_FORCE, K_TRAVEL, K_STRAIN}
    missing = [n for n in need if n.lower() not in dc]
    if missing:
        raise KeyError(f"{path.name}: missing columns: {missing}")

    # To numeric series
    time   = pd.to_numeric(data_df[dc[K_TIME.lower()]],   errors="coerce")
    force  = pd.to_numeric(data_df[dc[K_FORCE.lower()]],  errors="coerce")
    disp   = pd.to_numeric(data_df[dc[K_TRAVEL.lower()]], errors="coerce")
    strain = pd.to_numeric(data_df[dc[K_STRAIN.lower()]], errors="coerce") / 100.0
    temp   = pd.to_numeric(data_df[dc[K_TEMP.lower()]],   errors="coerce") if K_TEMP.lower() in dc else pd.Series(np.nan, index=data_df.index)

    area = get_area(meta_df)
    if not area or area <= 0:
        raise ValueError("missing/invalid area (S0 or a0*b0)")

    # Crosshead speed (with overrides if present)
    test_speed = get_number(meta_df, M_SPEED)
    if path.name in overrides or base in overrides:
        test_speed = overrides.get(path.name, overrides.get(base, test_speed))

    # Assemble dataframe
    df = pd.DataFrame({
        "Experiment": base,
        "File": path.name,
        "Time_s": time,
        "Force_N": force,
        "Disp_mm": disp,
        "Strain": strain,
        "Temp_C": temp,
        "Area_mm2": area,
        "TestSpeed_mm_min": test_speed,
    }).dropna(subset=["Force_N", "Disp_mm", "Strain"]).sort_values("Time_s").reset_index(drop=True)

    # Cleaning steps
    df, _ = enforce_monotonic_strain(df, "Strain")
    df, _ = remove_spikes(df, "Force_N", window=7, k=6.0)

    # Recompute σ from force (after cleaning)
    df = add_engineering(df, area)

    # Remove “needle” jumps in dσ/dε before global Hampel and trimming
    df, _ = remove_sigma_jumps(df,
                               strain_col="Strain",
                               stress_col="Stress_MPa",
                               window=9, k=8.0, min_dstrain=1e-4)

    # Hampel on σ level (soft single-point spike filter)
    mask = hampel_mask(df["Stress_MPa"], window=7, k=4.0)
    if mask.any():
        df = df.loc[~mask].reset_index(drop=True)

    # Trim after peak
    df, idx_peak, _ = trim_post_peak(df, "Stress_MPa")

    # E and metrics
    E_MPa = estimate_E(df)
    rp02 = proof_stress_rp02(df, E_MPa, PROOF_OFFSET)
    sig_max, eps_at, t_fail = peak_and_failure(df)

    # Hollomon on true stress–strain
    df_true = add_true(df)
    n, K = hollomon_true(df_true, E_MPa, HOLL_WINDOW)

    # Plot
    out_png = OUT_DIR / "plots" / f"{base}.png"
    plot_sigma_epsilon(df, out_png, E_MPa if ADD_E_TO_PNG else None, rp02, title=f't={int((df["Temp_C"].mean())) if "Temp_C" in df else float("nan")}C & v={test_speed}mm/min')

    # Summary
    summary = {
        "Experiment": base,
        "File": path.name,
        "Area_mm2": area,
        "Temp_C_mean": float(df["Temp_C"].mean()) if "Temp_C" in df else float("nan"),
        "TestSpeed_mm_min": test_speed,
        "E_MPa": E_MPa,
        "Rp0.2_MPa": rp02[0] if rp02 and np.isfinite(rp02[0]) else float("nan"),
        "Strain_at_Rp0.2": rp02[1] if rp02 and np.isfinite(rp02[1]) else float("nan"),
        "Time_at_Rp0.2_s": rp02[2] if np.isfinite(rp02[2]) else float("nan"),
        "Sigma_max_MPa": sig_max,
        "Strain_at_Sigmax": eps_at,
        "FailureTime_s": t_fail,
        "Hollomon_n": n,
        "Hollomon_K_MPa": K,
        "Fmax_meta_N": get_number(meta_df, M_FMAX),
        "dL_at_Fmax_mm": get_number(meta_df, M_DLF),
    }

    return df, summary

# ---------------- Reporting ----------------

def write_reports(combined: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Сохраняет CSV, Excel с pivot и интерактивный HTML график (если plotly доступен)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_DIR / "Combined.csv", index=False)
    summary.to_csv(OUT_DIR / "Summary.csv", index=False)

    # Excel with pivot by temperature
    with pd.ExcelWriter(OUT_DIR / "Report.xlsx", engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="Summary", index=False)
        try:
            piv = pd.pivot_table(summary, index=pd.Series(np.round(summary["Temp_C_mean"], 1), name="Temp_C"),
                                 values=["E_MPa", "Rp0.2_MPa", "Sigma_max_MPa"], aggfunc="mean")
            piv.to_excel(xw, sheet_name="Pivot_by_Temp")
        except Exception:
            pass

    # HTML overview of all σ–ε (color = temperature)
    if px is not None and not combined.empty:
        df_plot = combined.copy()
        df_plot["Temp_tag"] = np.round(df_plot["Temp_C"], 1)
        fig = px.line(df_plot, x="Strain", y="Stress_MPa", color="Temp_tag", line_group="Experiment",
                      hover_data=["Experiment", "Temp_C"], title="σ–ε overview")
        fig.write_html(str(OUT_DIR / "interactive_summary.html"))



def plot_all_sigma_epsilon(all_df: pd.DataFrame,
                           summary_df: pd.DataFrame | None,
                           out_html: Path) -> Path:
    """
    all_df: ['Experiment','Strain','Stress_MPa','Temp_C','TestSpeed_mm_min']
    summary_df (опц.): ['Experiment','Rp02_MPa','Rp02_Strain', 'E_MPa'(опц.)]
    """

    def _add_rp02(fig, eps, sig, E_MPa=None, x_max=None, group=""):
        """Add Rp0.2: marker + guides + offset line (if E available)."""
        eps = float(eps);
        sig = float(sig)
        fig.add_trace(go.Scatter(
            x=[eps], y=[sig], mode='markers',
            name=f'Rp0.2 = {sig:.2f} MPa',
            marker=dict(size=11, symbol='x'),
            legendgroup=group, hovertemplate='ε=%{x:.4f}<br>σ=%{y:.2f} MPa<extra>Rp0.2</extra>'
        ))
        # # направляющие
        # fig.add_trace(go.Scatter(x=[eps, eps], y=[0, sig], mode='lines',
        #                          line=dict(dash='dot', width=1),
        #                          showlegend=False, hoverinfo='skip', legendgroup=group))
        # fig.add_trace(go.Scatter(x=[0, eps], y=[sig, sig], mode='lines',
        #                          line=dict(dash='dot', width=1),
        #                          showlegend=False, hoverinfo='skip', legendgroup=group))
        # # offset-линия
        # if E_MPa and not pd.isna(E_MPa):
        #     x2 = float(x_max) if x_max is not None else eps
        #     x_off = [0, x2]
        #     y_off = [E_MPa * (x - PROOF_OFFSET) for x in x_off]
        #     fig.add_trace(go.Scatter(
        #         x=x_off, y=y_off, mode='lines', line=dict(dash='dash'),
        #         name='offset line (0.2%)', legendgroup=group,
        #         hovertemplate='ε=%{x:.4f}<br>σ=%{y:.2f} MPa<extra>offset</extra>'
        #     ))


    def _prepare_rp(summary_df: pd.DataFrame) -> pd.DataFrame:
        if summary_df is None or summary_df.empty:
            return pd.DataFrame(columns=['Rp0.2_MPa', 'Rp0.2_Strain', 'E_MPa']).set_index(
                pd.Index([], name='Experiment'))

        cols = {c.lower().strip(): c for c in summary_df.columns}
        need = ('experiment', 'rp0.2_mpa', 'strain_at_rp0.2')
        if not all(n in cols for n in need):
            return pd.DataFrame(columns=['Rp0.2_MPa', 'Strain_at_Rp0.2', 'E_MPa']).set_index(
                pd.Index([], name='Experiment'))

        ren = {cols['experiment']: 'Experiment',
               cols['rp0.2_mpa']: 'Rp0.2_MPa',
               cols['strain_at_rp0.2']: 'Strain_at_Rp0.2'}
        if 'e_mpa' in cols:
            ren[cols['e_mpa']] = 'E_MPa'

        rp = (summary_df.rename(columns=ren)
              .drop_duplicates('Experiment', keep='last')
              .set_index('Experiment'))
        for c in ['Rp0.2_MPa', 'Strain_at_Rp0.2', 'E_MPa']:
            if c in rp.columns:
                rp[c] = pd.to_numeric(rp[c], errors='coerce')
        return rp

    fig = go.Figure()

    # Fast access to Rp0.2 (+ E if present)
    rp = _prepare_rp(summary_df)

    for exp, dfi in all_df.groupby('Experiment', sort=False):
        t = float(dfi['Temp_C'].mean()) if 'Temp_C' in dfi.columns else float('nan')
        v = (float(dfi['TestSpeed_mm_min'].dropna().iloc[0])
             if 'TestSpeed_mm_min' in dfi.columns and dfi['TestSpeed_mm_min'].notna().any()
             else float('nan'))
        name = f"{exp} | t={int(round(t))}°C | v={v:g} mm/min"

        fig.add_trace(go.Scatter(
            x=dfi['Strain'], y=dfi['Stress_MPa'],
            mode='lines', name=name, legendgroup=exp,
            customdata=(dfi['Strain']),
            hovertemplate="ε=%{customdata:.2f}%<br>σ=%{y:.2f} MPa<br>"+name+"<extra></extra>"
        ))

        # Rp0.2 (если есть)
        if rp is not None and exp in rp.index:
            row = rp.loc[exp]
            if pd.notna(row['Rp0.2_MPa']) and pd.notna(row['Strain_at_Rp0.2']):
                _add_rp02(fig,
                          eps=row['Strain_at_Rp0.2'],
                          sig=row['Rp0.2_MPa'],
                          E_MPa=(row['E_MPa'] if 'E_MPa' in row else None),
                          x_max=dfi['Strain'].max(),
                          group=exp)

    fig.update_layout(
        title="Click legend to hide selected curves",
        xaxis_title="Engineering strain, ε",
        yaxis_title="Engineering stress, MPa",
        template="plotly_white",
        hovermode="closest",
        legend=dict(itemclick='toggleothers', itemdoubleclick='toggle')
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs='cdn')
    return out_html

def main() -> None:
    if not DATA_DIR.exists():
        raise SystemExit("Put .xls/.xlsx files into ./data")
    files = sorted(DATA_DIR.glob("*.xls*"))
    if not files:
        raise SystemExit("No Excel files found in ./data")

    overrides = load_overrides(Path(OVERRIDE_CSV))

    all_rows: List[pd.DataFrame] = []
    summary_rows: List[Dict] = []

    for f in files:
        try:
            df, summ = process_file(f, overrides)
            all_rows.append(df)
            summary_rows.append(summ)
        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")
    if not all_rows:
        raise SystemExit("Nothing processed")

    combined_df = pd.concat(all_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    plot_all_sigma_epsilon(combined_df, summary_df, OUT_DIR / "plots" / "ALL_sigma_epsilon.html")

    write_reports(combined_df, summary_df)
    print(f"Done. Files saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()


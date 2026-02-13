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
K_TRAVEL = "Absolute crosshead travel"      # mm
K_STRAIN = "Strain"               # percent (%)
K_TEMP   = "Specimen temperature" # °C

# New flag to indicate if K_FORCE is actually Stress in MPa
K_FORCE_IS_STRESS_MPA = True

# Metadata sheet name and fields
SHEET_META = "Results"
M_SPEED = "Test speed"    # mm/min
M_FMAX  = "Fmax"          # N
M_DLF   = "dL at Fmax"    # mm
M_A0    = "a0"            # mm
M_B0    = "b0"            # mm
M_S0    = "S0"            # mm^2

# Strain window to estimate E on engineering values (fractions, not %)
E_STRAIN_WINDOW: Tuple[float, float] = (0.001, 0.015)
E_MIN_POINTS = 2
ADD_E_TO_PNG = True
DRAW_FORCE_DISP = False

# Window for Hollomon fit on true plastic strain
HOLL_WINDOW: Tuple[float, float] = (0.001, 0.025)

# Offset for Rp0.2
PROOF_OFFSET = 0.2

# Control whether to apply baseline shift
APPLY_BASELINE_SHIFT = False

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

def parse_data_sheet(xls: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    """Read a specific data sheet from the Excel file."""
    sheet = next((s for s in xls.sheet_names if s.strip().lower() == sheet_name.strip().lower()), None)
    if not sheet:
        raise ValueError(f"Data sheet '{sheet_name}' not found")
    df = xls.parse(sheet_name=sheet, header=1)
    # Drop the units row if present
    if not df.empty and (looks_like_units_row(df.iloc[0], UNITS_DATA_SET) or df.iloc[0].astype(str).str.contains(UNITS_DATA_RE, na=False).any()):
        df = df.drop(index=0).reset_index(drop=True)
    return df


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
    if K_FORCE_IS_STRESS_MPA: # Use the new flag
        out["Stress_MPa"] = pd.to_numeric(out["Force_N"], errors="coerce")
    else:
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

    x = seg["Strain"].to_numpy()
    y = seg["Stress_MPa"].to_numpy()
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m)


def proof_stress_rp02(df: pd.DataFrame, E_MPa: float, offset: float = PROOF_OFFSET) -> tuple[float, float]:
    """Find Rp0.2 (MPa) and ε at intersection of σ(ε) with the offset line."""
    if not np.isfinite(E_MPa) or E_MPa <= 0:
        return float("nan"), float("nan")
    # function f(ε) = σ(ε) - E*(ε - offset)
    eps = df["Strain"].to_numpy(dtype=float)
    sig = df["Stress_MPa"].to_numpy(dtype=float)
    g = sig - E_MPa * (eps - offset)
    # look for a change in sign and interpolate
    for i in range(1, len(g)):
        if np.sign(g[i-1]) == np.sign(g[i]):
            continue
        # linear interpolation
        t = abs(g[i-1]) / (abs(g[i-1]) + abs(g[i]))
        e = (1 - t) * eps[i-1] + t * eps[i]
        s = (1 - t) * sig[i-1] + t * sig[i]
        return float(s), float(e)
    return float("nan"), float("nan")


def hollomon_true(df_true: pd.DataFrame, E_MPa: float, window: Tuple[float, float] = HOLL_WINDOW) -> tuple[float, float]:
    """Estimate Hollomon parameters from σ_true = K · (ε_pl_true)^n within a window."""
    if not np.isfinite(E_MPa) or E_MPa <= 0:
        return float("nan"), float("nan")
    eps_t = df_true["TrueStrain"].to_numpy(dtype=float)
    sig_t = df_true["TrueStress_MPa"].to_numpy(dtype=float)
    eps_pl = eps_t - sig_t / E_MPa  # истинная пластическая
    mask = (eps_pl >= window[0]) & (eps_pl <= window[1]) & (eps_pl > 0)

    # Filter out non-positive stress values before taking logarithm
    mask = mask & (sig_t > 0) # Apply (sig_t > 0) to the full sig_t array and then combine with mask

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
                       title: str = "Engineering stress–strain", e_mod_seg: Optional[pd.DataFrame] = None) -> None:
    """PNG with σ–ε, dashed guides for σ_max and Rp0.2, and the offset line."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot(df["Strain"], df["Stress_MPa"], label="σ–ε")

    # Add E-modulus fitting points if provided
    if e_mod_seg is not None and not e_mod_seg.empty:
        ax.scatter(e_mod_seg["Strain"], e_mod_seg["Stress_MPa"], color='red', marker='o', s=50, label="E-modulus points")

        # Get min and max strain points from the E-modulus segment
        min_strain_point = e_mod_seg.iloc[0] # The segment is already sorted by strain
        max_strain_point = e_mod_seg.iloc[-1]

        # Annotate min strain point
        ax.annotate(f'E_start: ({min_strain_point["Strain"]:.3f}, {min_strain_point["Stress_MPa"]:.2f})',
                    xy=(min_strain_point["Strain"], min_strain_point["Stress_MPa"]),
                    xytext=(5, -15), textcoords="offset points", color='red')

        # Annotate max strain point
        ax.annotate(f'E_end: ({max_strain_point["Strain"]:.3f}, {max_strain_point["Stress_MPa"]:.2f})',
                    xy=(max_strain_point["Strain"], max_strain_point["Stress_MPa"]),
                    xytext=(5, 5), textcoords="offset points", color='red')


    # peak marker
    sig_max, eps_at, _ = peak_and_failure(df)
    ax.axhline(sig_max, ls="--", alpha=0.5)
    ax.axvline(eps_at, ls="--", alpha=0.5)
    ax.annotate(f"σ_max={sig_max:.2f} MPa", xy=(eps_at, sig_max), xytext=(5,5), textcoords="offset points")

    # Rp0.2 and offset line
    if rp02 and all(np.isfinite(rp02)):
        s02, e02 = rp02
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

def process_experiment_from_workbook(
    xls: pd.ExcelFile,
    exp_id: str,
    meta_row: pd.Series,
    overrides: Dict[str, float],
    excel_file_name: str
) -> tuple[pd.DataFrame, Dict]:
    """
    Processes a single experiment's data and metadata from an opened Excel workbook.
    """
    # Parse data sheet for the current experiment ID
    data_df = parse_data_sheet(xls, exp_id)

    # Column mapping (using the original logic from process_file)
    dc = norm_cols(data_df)
    need = {K_TIME, K_FORCE, K_TRAVEL, K_STRAIN}
    missing = [n for n in need if n.lower() not in dc]
    if missing:
        raise KeyError(f"Experiment {exp_id} in {excel_file_name}: missing columns: {missing}")

    # To numeric series
    time   = pd.to_numeric(data_df[dc[K_TIME.lower()]],   errors="coerce")
    force  = pd.to_numeric(data_df[dc[K_FORCE.lower()]],  errors="coerce")
    disp   = pd.to_numeric(data_df[dc[K_TRAVEL.lower()]], errors="coerce")
    strain = pd.to_numeric(data_df[dc[K_STRAIN.lower()]], errors="coerce") / 100.0
    # K_TEMP is optional, so check if it exists in data_df.columns after norm_cols
    temp_col_name = dc.get(K_TEMP.lower())
    temp   = pd.to_numeric(data_df[temp_col_name], errors="coerce") if temp_col_name else pd.Series(np.nan, index=data_df.index)


    # Extract metadata from meta_row (Series) for the current experiment
    # We need to map the column names from the meta_row.index to the M_ constants
    meta_cols = norm_cols(meta_row.to_frame().T) # Convert series to df for norm_cols
    
    # Area calculation
    area = None
    s0_col = meta_cols.get(M_S0.lower())
    if s0_col:
        s0_val = pd.to_numeric(meta_row[s0_col], errors='coerce')
        if pd.notna(s0_val) and s0_val > 0:
            area = float(s0_val)
    
    if area is None: # If S0 is not available or invalid, try a0 * b0
        a0_col = meta_cols.get(M_A0.lower())
        b0_col = meta_cols.get(M_B0.lower())
        a0_val = pd.to_numeric(meta_row[a0_col], errors='coerce') if a0_col else np.nan
        b0_val = pd.to_numeric(meta_row[b0_col], errors='coerce') if b0_col else np.nan
        if pd.notna(a0_val) and pd.notna(b0_val) and a0_val > 0 and b0_val > 0:
            area = float(a0_val * b0_val)

    if not area or area <= 0:
        raise ValueError(f"Experiment {exp_id} in {excel_file_name}: missing/invalid area (S0 or a0*b0) in metadata.")

    # Crosshead speed (with overrides if present)
    test_speed = None
    mspeed_col = meta_cols.get(M_SPEED.lower())
    if mspeed_col:
        speed_val = pd.to_numeric(meta_row[mspeed_col], errors='coerce')
        if pd.notna(speed_val):
            test_speed = float(speed_val)
    
    # Apply global overrides (if any)
    if exp_id in overrides: # Check if exp_id is in overrides keys
        test_speed = overrides[exp_id]
    elif excel_file_name in overrides: # Fallback to excel file name if experiment ID isn't specifically overridden
        test_speed = overrides[excel_file_name]

    if test_speed is None:
        test_speed = float('nan') # Default to NaN if no speed found or overridden

    # Assemble initial DataFrame
    df_initial = pd.DataFrame({
        "Experiment": exp_id,
        "File": excel_file_name, # The entire Excel file is now the 'file'
        "Time_s": time,
        "Force_N": force,
        "Disp_mm": disp,
        "Strain": strain,
        "Temp_C": temp,
        "Area_mm2": area,
        "TestSpeed_mm_min": test_speed,
    }).sort_values("Time_s").reset_index(drop=True)

    # --- Prepare df_display for plotting (minimal processing) ---
    df_display = df_initial.copy()

    # Apply baseline shift to df_display if enabled
    if APPLY_BASELINE_SHIFT:
        shifted_force, _ = baseline_shift(df_display["Force_N"], method="mean_first_n")
        df_display["Force_N"] = shifted_force

        shifted_strain, _ = baseline_shift(df_display["Strain"], method="mean_first_n")
        df_display["Strain"] = shifted_strain
    
    # Calculate Stress_MPa for df_display (from potentially shifted force)
    df_display = add_engineering(df_display, area)


    # --- Prepare df_calc for calculations (full cleaning) ---
    df_calc = df_initial.copy() # Start with a fresh copy of initial data

    # Apply baseline shift to df_calc if enabled (for consistency with display)
    if APPLY_BASELINE_SHIFT:
        shifted_force, _ = baseline_shift(df_calc["Force_N"], method="mean_first_n")
        df_calc["Force_N"] = shifted_force

        shifted_strain, _ = baseline_shift(df_calc["Strain"], method="mean_first_n")
        df_calc["Strain"] = shifted_strain
    
    # Cleaning steps for df_calc
    df_calc, _ = enforce_monotonic_strain(df_calc, "Strain")
    # df_calc, _ = remove_spikes(df_calc, "Force_N", window=5, k=120.0)

    # Recompute σ from force (after cleaning) for df_calc
    df_calc = add_engineering(df_calc, area)

    # Remove “needle” jumps in dσ/dε before global Hampel and trimming
    # df_calc, _ = remove_sigma_jumps(df_calc,
    #                            strain_col="Strain",
    #                            stress_col="Stress_MPa",
    #                            window=5, k=120.0, min_dstrain=1e-2)

    # Hampel on σ level (soft single-point spike filter)
    # mask = hampel_mask(df_calc["Stress_MPa"], window=5, k=120.0)
    # if mask.any():
    #     df_calc = df_calc.loc[~mask].reset_index(drop=True)

    # Trim after peak
    # df_calc, idx_peak, _ = trim_post_peak(df_calc, "Stress_MPa")

    # E and metrics
    E_MPa = estimate_E(df_calc)

    # Re-create seg for plotting the E-modulus region from df_calc
    lo, hi = E_STRAIN_WINDOW
    e_mod_seg_for_plot = df_calc[(df_calc["Strain"] >= lo) & (df_calc["Strain"] <= hi)]

    rp02 = proof_stress_rp02(df_calc, E_MPa, PROOF_OFFSET)
    sig_max, eps_at, t_fail = peak_and_failure(df_calc) # This now works on df_calc

    # Hollomon on true stress–strain
    df_true = add_true(df_calc)
    n, K = hollomon_true(df_true, E_MPa, HOLL_WINDOW)

    # Plot
    out_png = OUT_DIR / "plots" / f"{exp_id}.png"
    plot_sigma_epsilon(df_display, out_png,
                       E_MPa if ADD_E_TO_PNG else None,
                       rp02,
                       title=f'Experiment {exp_id} — t={int(df_calc["Temp_C"].mean()) if "Temp_C" in df_calc and df_calc["Temp_C"].notna().any() else float("nan")}C & v={test_speed}mm/min',
                       e_mod_seg=e_mod_seg_for_plot) # New parameter being passed

    # Summary
    # Need to extract Fmax_meta_N and dL_at_Fmax_mm from meta_row
    fmax_meta_col = meta_cols.get(M_FMAX.lower())
    fmax_meta_val = pd.to_numeric(meta_row[fmax_meta_col], errors='coerce') if fmax_meta_col else np.nan

    dlf_meta_col = meta_cols.get(M_DLF.lower())
    dlf_meta_val = pd.to_numeric(meta_row[dlf_meta_col], errors='coerce') if dlf_meta_col else np.nan

    summary = {
        "Experiment": exp_id,
        "File": excel_file_name,
        "Area_mm2": area,
        "Temp_C_mean": float(df_calc["Temp_C"].mean()) if "Temp_C" in df_calc and df_calc["Temp_C"].notna().any() else float("nan"),
        "TestSpeed_mm_min": test_speed,
        "E_MPa": E_MPa,
        "Rp0.2_MPa": rp02[0] if rp02 and np.isfinite(rp02[0]) else float("nan"),
        "Strain_at_Rp0.2": rp02[1] if rp02 and np.isfinite(rp02[1]) else float("nan"),
        "Sigma_max_MPa": sig_max,
        "Strain_at_Sigmax": eps_at,
        "FailureTime_s": t_fail,
        "Hollomon_n": n,
        "Hollomon_K_MPa": K,
        "Fmax_meta_N": float(fmax_meta_val) if pd.notna(fmax_meta_val) else float("nan"),
        "dL_at_Fmax_mm": float(dlf_meta_val) if pd.notna(dlf_meta_val) else float("nan"),
    }

    return df_calc, summary


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
        t = float(dfi['Temp_C'].mean()) if 'Temp_C' in dfi.columns and dfi['Temp_C'].notna().any() else float('nan')
        v = (float(dfi['TestSpeed_mm_min'].dropna().iloc[0])
             if 'TestSpeed_mm_min' in dfi.columns and dfi['TestSpeed_mm_min'].notna().any()
             else float('nan'))
        
        temp_str = f"{int(round(t))}°C" if np.isfinite(t) else "N/A"
        speed_str = f"{v:g} mm/min" if np.isfinite(v) else "N/A"

        name = f"{exp} | t={temp_str} | v={speed_str}"

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
    # Filter out temporary Excel files starting with ~$
    all_files = DATA_DIR.glob("*.xls*")
    files = sorted([f for f in all_files if not f.name.startswith("~$")])

    if not files:
        raise SystemExit("No Excel files found in ./data (excluding temporary files).")
    if len(files) > 1:
        raise SystemExit("Only one Excel file is expected in the data/ directory for this script when processing multiple experiments within a single file.")

    excel_file_path = files[0]
    xls = safe_excel_reader(excel_file_path)
    
    # Load overrides from a separate CSV file if it exists (as before)
    overrides = load_overrides(Path(OVERRIDE_CSV))

    # Parse the metadata sheet to get all experiment IDs and their metadata
    # The second argument (data_sheet_name) is just a placeholder here, as we are reading the whole meta sheet.
    full_meta_df = parse_meta_sheet(xls, "dummy_data_sheet_name") 

    if full_meta_df.empty:
        raise SystemExit(f"Metadata sheet '{SHEET_META}' in '{excel_file_path.name}' is empty or not found.")

    # Assuming the first column of the metadata sheet contains the experiment IDs
    # and the metadata for each experiment is in the corresponding row.
    
    # Assuming first column is the experiment ID
    experiment_ids_series = full_meta_df.iloc[:, 0]
    
    # Filter out rows where experiment_id is NaN
    valid_experiment_rows = full_meta_df[experiment_ids_series.notna()].copy()
    valid_experiment_rows.iloc[:, 0] = valid_experiment_rows.iloc[:, 0].astype(str) # Ensure IDs are strings

    if valid_experiment_rows.empty:
        raise SystemExit(f"No valid experiment IDs found in the first column of the metadata sheet '{SHEET_META}'.")

    all_rows: List[pd.DataFrame] = []
    summary_rows: List[Dict] = []

    # Iterate through each valid experiment row
    for index, meta_row in valid_experiment_rows.iterrows():
        exp_id = meta_row.iloc[0] # The first column is the experiment ID
        
        try:
            df, summ = process_experiment_from_workbook(xls, exp_id, meta_row, overrides, excel_file_path.name)
            all_rows.append(df)
            summary_rows.append(summ)
        except Exception as e:
            print(f"[ERROR] Experiment {exp_id} in {excel_file_path.name}: {e}")
            
    if not all_rows:
        raise SystemExit("Nothing processed")

    combined_df = pd.concat(all_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    plot_all_sigma_epsilon(combined_df, summary_df, OUT_DIR / "plots" / "ALL_sigma_epsilon.html")

    write_reports(combined_df, summary_df)
    print(f"Done. Files saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()


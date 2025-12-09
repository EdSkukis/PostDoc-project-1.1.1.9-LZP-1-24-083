# process_3pb.py
"""
3-point bending batch processor — optimized.

Key features
------------
- Robust CSV parser for EU decimals and quoted numbers.
- Geometry + modulus window read from an "info" CSV.
- WINDOW_MODE selects interpretation of the E-window:
    "strain": e_start_pct/e_end_pct are absolute strain values (e.g., 0.005..0.025)
    "force" : e_start_pct/e_end_pct are fractions of Fmax (e.g., 0.05..0.25)
- Stress–strain computation; Young’s modulus via single fast regression in the window.
- Per-specimen CSV, optional per-specimen PNG with E red line from origin.
- Combined overlay PNG and interactive Plotly HTML with group toggles.
- Summary CSV including Emod (GPa), flexural strength/strain.
- Aggregated all-points CSV.
- Rendering speed-up via decimation for plots; no redundant CSV re-reads.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# ----------------------------- Settings ------------------------------------

# Interpret e_start_pct / e_end_pct from info.csv as:
#   "strain": absolute strain values (e.g., 0.005..0.025)
#   "force" : fractions of Fmax      (e.g., 0.05..0.25)
WINDOW_MODE: Literal["strain", "force"] = "strain"

# Draw individual PNGs per specimen
MAKE_INDIVIDUAL_PNG = True

# Max points per trace in plots (decimation). CSV outputs remain full-fidelity.
MAX_PLOT_POINTS = 2000


# ----------------------------- Data Structures -----------------------------

@dataclass
class Geometry:
    """Specimen geometry and modulus window settings."""
    span_mm: float          # L
    width_b_mm: float       # b0
    thickness_h_mm: float   # a0
    e_start_pct: float      # per WINDOW_MODE: strain value OR fraction of Fmax
    e_end_pct: float        # per WINDOW_MODE: strain value OR fraction of Fmax


@dataclass
class SpecimenResult:
    """Computed outputs for a single specimen."""
    specimen_id: str
    E_GPa: float
    Fmax: float
    dL_at_Fmax: float
    FBreak: float
    dL_at_break: float
    start_pct: float
    end_pct: float
    n_points: int
    flexural_strength_MPa: float
    flexural_strain: float
    curve_csv_path: str
    figure_png_path: str
    group: str               # ref, pH-1, pH-4, pH-7, pH-10, pH-12, unknown


# ----------------------------- Parsing helpers -----------------------------

DEFLECTION_PATTERNS = r"(deformation|deflection|absolute\s*crosshead\s*travel|midspan\s*deflection|прогиб|деформация)"
FORCE_PATTERNS      = r"(standard\s*force|force|load|сила|нагрузка)"
TIME_PATTERNS       = r"(test\s*time|time|время)"


def read_info_table(info_csv_path: str) -> pd.DataFrame:
    """Read geometry and modulus-window table.
    Expect first row to contain units. Actual data starts from row 1.
    EU decimal commas are converted to dots. Numeric columns are coerced.
    """
    raw = pd.read_csv(info_csv_path)
    df = raw.iloc[1:].copy().reset_index(drop=True)
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(",", ".", regex=False).str.strip()
    num_cols = ['Span', 'b0', 'a0', 'S0',
                "Begin of Young's modulus determination",
                "End of Young's modulus determination"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Specimen ID'] = df['Specimen ID'].astype(str).str.strip()
    return df


def build_geometry_index(info_df: pd.DataFrame) -> Dict[str, Geometry]:
    """Convert info table to an index by specimen id."""
    idx: Dict[str, Geometry] = {}
    for _, row in info_df.iterrows():
        sid = str(row['Specimen ID']).strip()
        if not sid:
            continue

        # e_start = float(row["Begin of Young's modulus determination"])
        # e_end   = float(row["End of Young's modulus determination"])
        e_start = float(0.005)
        e_end   = float(0.025)
        # WINDOW_MODE == "strain": values are absolute strain; keep as-is

        idx[sid] = Geometry(
            span_mm=float(row['Span']),
            width_b_mm=float(row['b0']),
            thickness_h_mm=float(row['a0']),
            e_start_pct=e_start,
            e_end_pct=e_end,
        )
    return idx


def read_experiment_csv(path: str) -> pd.DataFrame:
    """Read experiment CSV (Format A):
        row0: specimen id repeated
        row1: labels
        row2: units
        row3..: data
    Returns DataFrame with float columns: time_s, deformation_mm, force_N.
    """
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    if len(raw) < 4:
        raise ValueError(f"{path} too short")

    labels = raw.iloc[1].astype(str).str.strip().tolist()
    data = raw.iloc[3:].reset_index(drop=True).copy()

    # Normalize cells to floats
    for c in data.columns:
        data[c] = (
            data[c]
            .astype(str)
            .str.replace('"', '', regex=False)
            .str.replace(",", ".", regex=False)
            .str.replace(" ", "", regex=False)
        )
        data[c] = pd.to_numeric(data[c], errors="coerce")

    # Map columns by regex
    col_map = {}
    for idx_, label in enumerate(labels):
        lab = label.lower()
        if re.search(TIME_PATTERNS, lab): col_map["time_s"] = idx_
        if re.search(DEFLECTION_PATTERNS, lab): col_map["deformation_mm"] = idx_
        if re.search(FORCE_PATTERNS, lab): col_map["force_N"] = idx_

    if "deformation_mm" not in col_map or "force_N" not in col_map:
        raise ValueError(f"Required columns missing in {os.path.basename(path)}. Found labels: {labels}")

    out = pd.DataFrame({
        "time_s": data[col_map.get("time_s")] if "time_s" in col_map else np.arange(len(data)),
        "deformation_mm": data[col_map["deformation_mm"]],
        "force_N": data[col_map["force_N"]],
    })
    # Ensure monotonic deformation for interpolation searchsorted
    out = out.sort_values(by="deformation_mm").reset_index(drop=True)
    return out


# ----------------------------- Mechanics -----------------------------------

def compute_stress_strain(exp_df: pd.DataFrame, L_mm: float, b_mm: float, h_mm: float) -> pd.DataFrame:
    """Compute strain ε and stress σ for 3PB.
    ε = 6 D h / L^2
    σ = 3 P L / (2 b h^2), returned in MPa
    """
    L = L_mm / 1000.0
    b = b_mm / 1000.0
    h = h_mm / 1000.0
    D_m = exp_df['deformation_mm'].values / 1000.0
    P = exp_df['force_N'].values

    eps = 6.0 * D_m * h / (L * L)
    sigma_Pa = 3.0 * P * L / (2.0 * b * h * h)
    sigma_MPa = sigma_Pa / 1e6

    out = pd.DataFrame({
        'time_s': exp_df['time_s'].values,
        'deformation_mm': exp_df['deformation_mm'].values,
        'force_N': exp_df['force_N'].values,
        'strain': eps,
        'stress_MPa': sigma_MPa
    }).dropna(subset=['strain','stress_MPa']).reset_index(drop=True)

    return out


# ----------------------------- Modulus (fast) ------------------------------
def fit_modulus_fast(stress_MPa: np.ndarray,
                     strain: np.ndarray,
                     force_N: np.ndarray,
                     start_val: float,
                     end_val: float,
                     window_mode: str = "strain") -> Tuple[float, float, Tuple[Optional[int], Optional[int]]]:
    """Fast linear fit inside the requested window.
    Returns (E_GPa, intercept_MPa, (i_start, i_end)).
    """
    if window_mode == "force":
        Fmax = float(np.nanmax(force_N))
        lo, hi = start_val * Fmax, end_val * Fmax
        mask = (force_N >= lo) & (force_N <= hi)
    else:
        lo, hi = start_val, end_val
        mask = (strain >= lo) & (strain <= hi)

    idx = np.where(mask)[0]
    if len(idx) < 2:
        return float('nan'), float('nan'), (None, None)

    x = strain[idx]
    y = stress_MPa[idx]
    a, b = np.polyfit(x, y, 1)  # σ = a*ε + b
    E_GPa = a / 1000.0
    return E_GPa, b, (int(idx[0]), int(idx[-1]))


def fit_modulus_astm(strain: np.ndarray,
                     deflection_mm: np.ndarray,
                     force_N: np.ndarray,
                     L_mm: float,
                     b_mm: float,
                     h_mm: float,
                     start_val: float,
                     end_val: float,
                     window_mode: str = "force"
                     ) -> Tuple[float, float, Tuple[Optional[int], Optional[int]]]:
    """
    ASTM D790: E_B = L^3 * m / (4 b d^3)

    strain         – массив ε (для выбора окна, если window_mode='strain')
    deflection_mm  – прогиб D, мм (кривая load–deflection)
    force_N        – нагрузка P, Н
    L_mm, b_mm, h_mm – геометрия образца
    start_val, end_val – границы окна:
        - если window_mode='force': доли Fmax (0.05..0.25)
        - если window_mode='strain': абсолютные значения ε (0.005..0.025)

    Возвращает:
        E_GPa         – модуль в GPa
        intercept_MPa – свободный член для σ–ε (для красной линии на графике)
        (i_start, i_end) – индексы окна в массиве
    """
    P = force_N
    D = deflection_mm

    # выбор окна
    if window_mode == "force":
        Fmax = float(np.nanmax(P))
        lo, hi = start_val * Fmax, end_val * Fmax
        mask = (P >= lo) & (P <= hi)
    else:  # window_mode == "strain"
        eps = strain
        lo, hi = start_val, end_val
        mask = (eps >= lo) & (eps <= hi)

    idx = np.where(mask)[0]
    if len(idx) < 2:
        return float('nan'), float('nan'), (None, None)

    # линейная аппроксимация P(D): P = m*D + c
    x = D[idx]
    y = P[idx]
    m_N_per_mm, c_N = np.polyfit(x, y, 1)

    # ASTM D790: модуль изгиба, MPa
    EB_MPa = (L_mm**3 * m_N_per_mm) / (4.0 * b_mm * (h_mm**3))
    E_GPa = EB_MPa / 1000.0

    # пересчёт свободного члена в σ–ε для красивой красной линии
    # σ = 3PL / (2 b h^2)  → σ = a*ε + b_sigma
    # b_sigma = 3 L c / (2 b h^2)
    intercept_MPa = (3.0 * L_mm * c_N) / (2.0 * b_mm * (h_mm**2))

    return float(E_GPa), float(intercept_MPa), (int(idx[0]), int(idx[-1]))


# ----------------------------- Utilities -----------------------------------

def decimate(df: pd.DataFrame, max_pts: int = 2000) -> pd.DataFrame:
    """Return at most max_pts points evenly spaced by index for plotting speed."""
    n = len(df)
    if n <= max_pts:
        return df
    idx = np.linspace(0, n - 1, max_pts).astype(int)
    return df.iloc[idx].reset_index(drop=True)


# ----------------------------- Grouping ------------------------------------

def infer_group(name: str) -> str:
    """Infer group label from specimen id or filename stem."""
    n = name.lower()
    if "ref" in n:
        return "ref"
    m = re.search(r"ph[-_]?(\d+)", n, flags=re.I)
    if m:
        val = m.group(1)
        if val in {"1","4","7","10","12"}:
            return f"pH-{val}"
    return "unknown"


# ----------------------------- Plotting ------------------------------------

def plot_individual_png(ss: pd.DataFrame, specimen_id: str,
                        E_GPa: float, intercept_MPa: float,
                        x_v1: Optional[float], x_v2: Optional[float],
                        out_path: str) -> None:
    """Save a PNG with σ–ε curve, window markers at given x positions, and red E-line from origin to σ_max."""
    fig, ax = plt.subplots(figsize=(7,5))
    ssp = ss  # already sorted by deformation

    ax.plot(ssp['strain'].values, ssp['stress_MPa'].values, label=specimen_id)

    # Window markers at x positions
    if x_v1 is not None:
        ax.axvline(x_v1, linestyle='--', color='k')
    if x_v2 is not None:
        ax.axvline(x_v2, linestyle='--', color='k')

    # E line from origin to sigma_max
    if np.isfinite(E_GPa):
        sigma_max = float(np.nanmax(ssp['stress_MPa'].values))
        E_MPa = E_GPa * 1000.0
        sigma_max = float(np.nanmax(ssp['stress_MPa'].values))
        # Линия σ = E*ε + b
        x_range = ss['strain'].values
        y_line = E_MPa * x_range + intercept_MPa
        ax.plot(x_range, y_line, color='r', linewidth=1.5)

    ax.set_xlabel("Strain [-]")
    ax.set_ylabel("Stress [MPa]")
    ax.set_title(f"{specimen_id} — 3PB stress–strain (E≈{E_GPa:.3f} GPa)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_overlay_png(curves: List[Tuple[str, pd.DataFrame]], out_path: str) -> None:
    """Save an overlay PNG of all σ–ε curves."""
    fig, ax = plt.subplots(figsize=(8,6))
    for sid, df in curves:
        dfp = decimate(df, MAX_PLOT_POINTS)
        ax.plot(dfp['strain'].values, dfp['stress_MPa'].values, label=sid)
    ax.set_xlabel("Strain [-]")
    ax.set_ylabel("Stress [MPa]")
    ax.set_title("3PB Stress–Strain: all specimens")
    ax.grid(True)
    if curves:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_overlay_html_grouped(curves: List[Tuple[str, str, pd.DataFrame]], out_path: str) -> None:
    """Interactive Plotly overlay with group toggles.
    curves: list of (group_label, specimen_id, df)
    - One legend item per group (ref, pH-1, pH-4, pH-7, pH-10, pH-12, unknown).
    - Clicking a group toggles all traces in that group.
    - Buttons: Show all / Hide all.
    """
    fig = go.Figure()

    groups = ["ref", "pH-1", "pH-4", "pH-7", "pH-10", "pH-12", "unknown"]
    for g in groups:
        # Dummy trace to show legend entry per group
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            name=g,
            legendgroup=g,
            showlegend=True,
            line=dict(width=2),
        ))

    # Add specimen traces with legendgroup, hidden in legend
    for group_label, sid, df in curves:
        dfp = decimate(df, MAX_PLOT_POINTS)
        hover = f"group={group_label}<br>specimen={sid}<br>strain=%{{x:.6f}}<br>stress=%{{y:.3f}} MPa<extra></extra>"
        fig.add_trace(go.Scatter(
            x=dfp["strain"],
            y=dfp["stress_MPa"],
            mode="lines",
            name=sid,
            legendgroup=group_label,
            showlegend=False,
            hovertemplate=hover
        ))

    n_traces = len(fig.data)
    buttons = [
        dict(label="Show all", method="update", args=[{"visible": [True] * n_traces}]),
        dict(label="Hide all", method="update", args=[{"visible": [False] * n_traces}]),
    ]
    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="right", x=0.0, y=1.15,
                          xanchor="left", yanchor="top", buttons=buttons)],
        title="3PB Stress–Strain (interactive, grouped)",
        xaxis_title="Strain [-]",
        yaxis_title="Stress [MPa]",
        template="simple_white",
        legend_title="Group",
        legend=dict(groupclick="togglegroup"),
        width=1100,
        height=700,
        margin=dict(t=90, r=20, b=60, l=70),
    )
    fig.write_html(out_path, include_plotlyjs="cdn")


# ----------------------------- Pipeline ------------------------------------

def match_geometry(geometry_idx: Dict[str, Geometry], filename_stem: str) -> Tuple[str, Geometry]:
    """Match a geometry row by checking if specimen id is a substring of filename stem."""
    for sid, geom in geometry_idx.items():
        if sid and sid in filename_stem:
            return sid, geom
    # Fallback: use filename stem and first geometry
    first_sid = next(iter(geometry_idx.keys()))
    return filename_stem, geometry_idx[first_sid]


def process_specimen_file(path: str, geometry_idx: Dict[str, Geometry], out_dir: str,
                          make_individual: bool = False) -> Optional[Tuple[SpecimenResult, pd.DataFrame]]:
    """Process a single experiment CSV into metrics and outputs.
    Returns (SpecimenResult, processed_curve_df) or None if parsing failed.
    """
    try:
        dfraw = read_experiment_csv(path)
    except Exception as e:
        print(f"Skip {os.path.basename(path)}: {e}")
        return None

    fname = os.path.splitext(os.path.basename(path))[0]
    specimen_id, geom = match_geometry(geometry_idx, fname)
    group_label = infer_group(specimen_id)

    ss = compute_stress_strain(dfraw, geom.span_mm, geom.width_b_mm, geom.thickness_h_mm)

    # flexural strength and strain at stress peak
    idx_strength = int(np.nanargmax(ss['stress_MPa'].values))
    flex_strength_MPa = float(ss['stress_MPa'].iloc[idx_strength])
    flex_strain = float(ss['strain'].iloc[idx_strength])

    # Modulus in requested window (fast)
    # E_GPa, intercept_MPa, (i_start, i_end) = fit_modulus_fast(
    #     ss['stress_MPa'].values,
    #     ss['strain'].values,
    #     ss['force_N'].values,
    #     start_val=geom.e_start_pct,
    #     end_val=geom.e_end_pct,
    #     window_mode=WINDOW_MODE,
    # )

    # Modulus in requested window (ASTM)
    E_GPa, intercept_MPa, (i_start, i_end) = fit_modulus_astm(
        strain=ss['strain'].values,
        deflection_mm=ss['deformation_mm'].values,
        force_N=ss['force_N'].values,
        L_mm=geom.span_mm,
        b_mm=geom.width_b_mm,
        h_mm=geom.thickness_h_mm,
        start_val=geom.e_start_pct,
        end_val=geom.e_end_pct,
        window_mode=WINDOW_MODE,
    )

    # Metrics
    Fmax = float(np.nanmax(ss['force_N'].values))
    idx_Fmax = int(np.nanargmax(ss['force_N'].values))
    dL_at_Fmax = float(ss['deformation_mm'].iloc[idx_Fmax])
    idx_break = int(len(ss) - 1)
    FBreak = float(ss['force_N'].iloc[idx_break])
    dL_at_break = float(ss['deformation_mm'].iloc[idx_break])

    # Outputs
    os.makedirs(out_dir, exist_ok=True)
    curve_csv_path = os.path.join(out_dir, f"{specimen_id}_stress_strain.csv")
    fig_png_path = os.path.join(out_dir, f"{specimen_id}_stress_strain.png")
    ss.to_csv(curve_csv_path, index=False)

    # Prepare window marker x positions for plotting
    if WINDOW_MODE == "strain":
        x_v1 = geom.e_start_pct
        x_v2 = geom.e_end_pct
    else:
        x_v1 = ss['strain'].iloc[i_start] if i_start is not None else None
        x_v2 = ss['strain'].iloc[i_end]   if i_end   is not None else None

    # Individual PNG (optional)
    if make_individual:
        plot_individual_png(ss, specimen_id, E_GPa, intercept_MPa, x_v1, x_v2, fig_png_path)
    else:
        fig_png_path = ""

    res = SpecimenResult(
        specimen_id=specimen_id,
        E_GPa=float(E_GPa),
        Fmax=Fmax,
        dL_at_Fmax=dL_at_Fmax,
        FBreak=FBreak,
        dL_at_break=dL_at_break,
        start_pct=float(geom.e_start_pct),
        end_pct=float(geom.e_end_pct),
        n_points=int(len(ss)),
        flexural_strength_MPa=flex_strength_MPa,
        flexural_strain=flex_strain,
        curve_csv_path=curve_csv_path,
        figure_png_path=fig_png_path,
        group=group_label
    )
    return res, ss


def process_folder(folder: str, info_csv_name_substr: str = "info") -> dict:
    """Process all experiments in a folder and export figures, CSVs, HTML.
    Returns a dict with paths to outputs.
    """
    # Load geometry
    candidates = [f for f in os.listdir(folder) if f.lower().endswith(".csv") and info_csv_name_substr in f.lower()]
    if not candidates:
        raise FileNotFoundError("Info CSV not found in folder.")
    info_path = os.path.join(folder, candidates[0])
    info_df = read_info_table(info_path)
    geom_idx = build_geometry_index(info_df)

    # Collect experiment files
    exp_files = [f for f in os.listdir(folder) if f.lower().endswith(".csv") and f != os.path.basename(info_path)]
    selected: List[str] = []
    for f in exp_files:
        base = os.path.splitext(os.path.basename(f))[0]
        if any(sid and sid in base for sid in geom_idx.keys()):
            selected.append(f)
    if not selected:
        selected = exp_files

    out_dir = os.path.join(folder, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    results: List[SpecimenResult] = []
    overlay_curves_png: List[Tuple[str, pd.DataFrame]] = []
    overlay_curves_html: List[Tuple[str, str, pd.DataFrame]] = []
    all_points: List[pd.DataFrame] = []

    for f in sorted(selected):
        res_pair = process_specimen_file(
            os.path.join(folder, f),
            geom_idx,
            out_dir,
            make_individual=MAKE_INDIVIDUAL_PNG
        )
        if res_pair is None:
            continue
        res, df_curve = res_pair
        overlay_curves_png.append((res.specimen_id, df_curve))
        overlay_curves_html.append((res.group, res.specimen_id, df_curve))
        df_curve2 = df_curve.copy()
        df_curve2.insert(0, "specimen_id", res.specimen_id)
        all_points.append(df_curve2)
        results.append(res)

    # Combined PNG
    overlay_png = os.path.join(out_dir, "overlay_stress_strain.png")
    plot_overlay_png(overlay_curves_png, overlay_png)

    # Combined HTML with grouped toggles
    overlay_html = os.path.join(out_dir, "overlay_stress_strain.html")
    plot_overlay_html_grouped(overlay_curves_html, overlay_html)

    # Summary CSV (Emod in GPa)
    summary_path = os.path.join(out_dir, "summary_E_results.csv")
    if results:
        summary_df = pd.DataFrame([{
            "specimen_id": r.specimen_id,
            "Fmax": r.Fmax,
            "dL at Fmax": r.dL_at_Fmax,
            "FBreak": r.FBreak,
            "dL at break": r.dL_at_break,
            "Emod": r.E_GPa,  # GPa
            "start_pct": r.start_pct,
            "end_pct": r.end_pct,
            "n_points": r.n_points,
            "flexural_strength_MPa": r.flexural_strength_MPa,
            "flexural_strain": r.flexural_strain
        } for r in results])
        summary_df.to_csv(summary_path, index=False)
    else:
        pd.DataFrame(columns=[
            "specimen_id","Fmax","dL at Fmax","FBreak","dL at break","Emod",
            "start_pct","end_pct","n_points","flexural_strength_MPa","flexural_strain"
        ]).to_csv(summary_path, index=False)

    # Aggregated points CSV
    if all_points:
        all_df = pd.concat(all_points, ignore_index=True)
        all_points_path = os.path.join(out_dir, "all_processed_points.csv")
        all_df.to_csv(all_points_path, index=False)
    else:
        all_points_path = None

    return {
        "info_path": info_path,
        "selected_files": selected,
        "outputs": {
            "overlay_png": overlay_png,
            "overlay_html": overlay_html,
            "summary_csv": summary_path,
            "all_points_csv": all_points_path,
            "folder": out_dir
        }
    }


def main() -> None:
    # Per your request: fixed relative folder
    folder = ("./mnt/G")
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Data folder not found: {os.path.abspath(folder)}")
    result = process_folder(folder)
    print(result)


if __name__ == "__main__":
    main()

# process_3pb.py
"""
3-point bending batch processor.

Features
--------
- Robust CSV parser for experiment files with EU decimals and quoted numbers.
- Reads geometry and E-window from an "info" CSV.
- Computes stress–strain, Young's modulus in a force-percentage window.
- Exports:
  * per-specimen processed CSV (σ–ε points),
  * optional per-specimen PNG,
  * combined PNG overlay,
  * interactive Plotly HTML overlay with group toggles (ref, pH-1, pH-4, pH-7, pH-10, pH-12),
  * summary CSV with required columns,
  * aggregated all-points CSV.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# ----------------------------- Data Structures -----------------------------

@dataclass
class Geometry:
    """Specimen geometry and modulus window settings."""
    span_mm: float          # L
    width_b_mm: float       # b0
    thickness_h_mm: float   # a0
    e_start_pct: float      # fraction, e.g. 0.05
    e_end_pct: float        # fraction, e.g. 0.25


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
    curve_csv_path: str
    figure_png_path: str
    group: str               # ref, pH-1, pH-4, pH-7, pH-10, pH-12, unknown


# ----------------------------- Parsing helpers -----------------------------

DEFLECTION_PATTERNS = r"(deformation|deflection|absolute\s*crosshead\s*travel|midspan\s*deflection|прогиб|деформация)"
FORCE_PATTERNS      = r"(standard\s*force|force|load|сила|нагрузка)"
TIME_PATTERNS       = r"(test\s*time|time|время)"


def read_info_table(info_csv_path: str) -> pd.DataFrame:
    """Read geometry and modulus-window table.
    Expect first row to contain units. Actual data_epoxy starts from row 1.
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
        idx[sid] = Geometry(
            span_mm=float(row['Span']),
            width_b_mm=float(row['b0']),
            thickness_h_mm=float(row['a0']),
            e_start_pct=float(row["Begin of Young's modulus determination"]),
            e_end_pct=float(row["End of Young's modulus determination"]),
        )
    return idx


def read_experiment_csv(path: str) -> pd.DataFrame:
    """Read experiment CSV (Format A):
        row0: specimen id repeated
        row1: labels
        row2: units
        row3..: data_epoxy
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
    for idx, label in enumerate(labels):
        lab = label.lower()
        if re.search(TIME_PATTERNS, lab): col_map["time_s"] = idx
        if re.search(DEFLECTION_PATTERNS, lab): col_map["deformation_mm"] = idx
        if re.search(FORCE_PATTERNS, lab): col_map["force_N"] = idx

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


def fit_modulus(stress_MPa: np.ndarray, strain: np.ndarray, force_N: np.ndarray,
                start_pct: float, end_pct: float) -> Tuple[float, Tuple[Optional[int], Optional[int]], np.ndarray]:
    """Linear fit of σ vs ε within a force window [start_pct..end_pct]*Fmax.
    Returns (E_MPa, (i_start, i_end), mask).
    """
    Fmax = float(np.nanmax(force_N))
    lo = start_pct * Fmax
    hi = end_pct * Fmax
    mask = (force_N >= lo) & (force_N <= hi)
    idxs = np.where(mask)[0]
    if len(idxs) < 2:
        return float('nan'), (None, None), mask

    x = strain[idxs]
    y = stress_MPa[idxs]
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a) / 1000, (int(idxs[0]), int(idxs[-1])), mask


def interp_force_at_deflection(df: pd.DataFrame, D_target_mm: float) -> float:
    """Linear interpolate force at a given deflection in mm.
    Assumes df is sorted by deformation_mm. Returns NaN if D_target outside data_epoxy range.
    """
    D = df['deformation_mm'].values
    F = df['force_N'].values
    if not np.isfinite(D_target_mm):
        return float('nan')
    if len(D) < 2 or np.nanmin(D) > D_target_mm or np.nanmax(D) < D_target_mm:
        return float('nan')
    idx = int(np.searchsorted(D, D_target_mm))
    if idx == 0 or idx >= len(D):
        return float('nan')
    x0, x1 = D[idx-1], D[idx]
    y0, y1 = F[idx-1], F[idx]
    if not (np.isfinite(x0) and np.isfinite(x1) and x1 != x0):
        return float('nan')
    t = (D_target_mm - x0) / (x1 - x0)
    return float(y0 + t * (y1 - y0))


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

def plot_individual_png(ss: pd.DataFrame, specimen_id: str, E_GPa: float,
                        i_start: Optional[int], i_end: Optional[int],
                        out_path: str) -> None:
    """Save a PNG with σ–ε curve, dashed window markers, and red E-line from origin to σ_max."""
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(ss['strain'].values, ss['stress_MPa'].values, label=specimen_id)

    # Window markers (if available)
    if i_start is not None:
        ax.axvline(ss['strain'].iloc[i_start], linestyle='--', color='k')
    if i_end is not None:
        ax.axvline(ss['strain'].iloc[i_end], linestyle='--', color='k')

    # E line from origin to sigma_max
    if np.isfinite(E_GPa):
        sigma_max = float(np.nanmax(ss['stress_MPa'].values))
        if E_GPa > 0 and np.isfinite(sigma_max):
            x_end = sigma_max / E_GPa
            # Clip to observed strain range to keep inside plot
            x_end = min(x_end, float(np.nanmax(ss['strain'].values)))
            x_line = np.array([0.0, x_end])
            y_line = E_GPa * x_line
            ax.plot(x_line, y_line, color='r')

    ax.set_xlabel("Strain [-]")
    ax.set_ylabel("Stress [MPa]")
    ax.set_title(f"{specimen_id} — 3PB stress–strain (E≈{E_GPa:.0f} MGa)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_overlay_png(curves: List[Tuple[str, pd.DataFrame]], out_path: str) -> None:
    """Save an overlay PNG of all σ–ε curves."""
    fig, ax = plt.subplots(figsize=(7,5))
    for sid, df in curves:
        ax.plot(df['strain'].values, df['stress_MPa'].values, label=sid)
    ax.set_xlabel("Strain [-]")
    ax.set_ylabel("Stress [MPa]")
    ax.set_title("3PB Stress–Strain: all specimens")
    ax.grid(True)
    if curves:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_overlay_html_grouped(curves: List[Tuple[str, str, pd.DataFrame]], out_path: str) -> None:
    """Interactive Plotly overlay with group toggles.
    curves: list of (group_label, specimen_id, df)
    - One legend item per group (ref, pH-1, pH-4, pH-7, pH-10, pH-12, unknown).
    - Clicking a group toggles all traces in that group.
    - Buttons: Show all / Hide all.
    """
    fig = go.Figure()

    # Create one legend entry per group as a dummy trace with legendgroup.
    groups = ["ref", "pH-1", "pH-4", "pH-7", "pH-10", "pH-12", "unknown"]
    for g in groups:
        # Dummy invisible trace for legend control
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            name=g,
            legendgroup=g,
            showlegend=True,
            line=dict(width=2),
        ))

    # Add specimen traces with legendgroup, but do not show them in legend
    for group_label, sid, df in curves:
        hover = f"group={group_label}<br>specimen={sid}<br>strain=%{{x:.6f}}<br>stress=%{{y:.3f}} MPa<extra></extra>"
        fig.add_trace(go.Scatter(
            x=df["strain"],
            y=df["stress_MPa"],
            mode="lines",
            name=sid,
            legendgroup=group_label,
            showlegend=False,   # controlled via group dummy trace
            hovertemplate=hover
        ))

    # Build visibility masks for groups (dummy + members)
    n_traces = len(fig.data)
    # Map from group -> indices of traces in fig.data_epoxy
    group_to_indices: Dict[str, List[int]] = {g: [] for g in groups}
    # Identify indices: first len(groups) are dummies; the rest are specimen traces
    for idx_g, g in enumerate(groups):
        group_to_indices[g].append(idx_g)  # include dummy trace

    # Assign specimen traces to group
    for i in range(len(groups), n_traces):
        g = fig.data[i].legendgroup or "unknown"
        if g not in group_to_indices:
            group_to_indices[g] = []
        group_to_indices[g].append(i)

    # Start visible=True for all groups
    visible = [True] * n_traces

    # Add updatemenus (buttons)
    buttons = [
        dict(label="Show all", method="update", args=[{"visible": [True] * n_traces}]),
        dict(label="Hide all", method="update", args=[{"visible": [False] * n_traces}]),
    ]
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.0, y=1.15, xanchor="left", yanchor="top",
                buttons=buttons
            )
        ],
        title="3pB Stress–Strain",
        xaxis_title="Strain [-]",
        yaxis_title="Stress [MPa]",
        template="simple_white",
        legend_title="Group",
        width=1100,
        height=700,
        margin=dict(t=90, r=20, b=60, l=70),
    )

    # Native Plotly behavior: click on a legend item (group dummy) toggles all traces in that legendgroup.
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
                          make_individual: bool = False) -> Optional[SpecimenResult]:
    """Process a single experiment CSV into metrics and outputs.
    Returns SpecimenResult or None if parsing failed.
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
    E_GPa, (i_start, i_end), mask = fit_modulus(
        ss['stress_MPa'].values, ss['strain'].values, ss['force_N'].values,
        geom.e_start_pct, geom.e_end_pct
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

    # Individual PNG (optional)
    if make_individual:
        plot_individual_png(ss, specimen_id, E_GPa, i_start, i_end, fig_png_path)
    else:
        fig_png_path = ""

    return SpecimenResult(
        specimen_id=specimen_id,
        E_GPa=float(E_GPa),
        Fmax=Fmax,
        dL_at_Fmax=dL_at_Fmax,
        FBreak=FBreak,
        dL_at_break=dL_at_break,
        start_pct=float(geom.e_start_pct),
        end_pct=float(geom.e_end_pct),
        n_points=int(len(ss)),
        curve_csv_path=curve_csv_path,
        figure_png_path=fig_png_path,
        group=group_label
    )


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
    # Prefer files containing known specimen ids
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
        res = process_specimen_file(
            os.path.join(folder, f),
            geom_idx,
            out_dir,
            make_individual=True  # only combined graphs required
        )
        if res is None:
            continue
        # Read processed curve for overlays and aggregation
        df_curve = pd.read_csv(res.curve_csv_path)
        overlay_curves_png.append((res.specimen_id, df_curve))
        overlay_curves_html.append((res.group, res.specimen_id, df_curve))
        df_curve.insert(0, "specimen_id", res.specimen_id)
        all_points.append(df_curve)
        results.append(res)

    # Combined PNG
    overlay_png = os.path.join(out_dir, "overlay_stress_strain.png")
    plot_overlay_png(overlay_curves_png, overlay_png)

    # Combined HTML with grouped toggles
    overlay_html = os.path.join(out_dir, "overlay_stress_strain.html")
    plot_overlay_html_grouped(overlay_curves_html, overlay_html)

    # Summary CSV (exact required columns and names)
    summary_path = os.path.join(out_dir, "summary_E_results.csv")
    if results:
        summary_df = pd.DataFrame([{
            "specimen_id": r.specimen_id,
            "Fmax": r.Fmax,
            "dL at Fmax": r.dL_at_Fmax,
            "FBreak": r.FBreak,
            "dL at break": r.dL_at_break,
            "Emod": r.E_GPa,
            "start_pct": r.start_pct,
            "end_pct": r.end_pct,
            "n_points": r.n_points
        } for r in results])
        summary_df.to_csv(summary_path, index=False)
    else:
        pd.DataFrame(columns=["specimen_id","Fmax","dL at Fmax","FBreak","dL at break","Emod","start_pct","end_pct","n_points"]).to_csv(summary_path, index=False)

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
    folder = "./mnt/data_gf"
    result = process_folder(folder)
    print(result)


if __name__ == "__main__":
    main()

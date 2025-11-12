#!/usr/bin/env python3
"""
TGA Friedman isoconversional analysis.
Usage:
  python tga_friedman.py --data-dir ./data --pattern "example_*Kmin.csv" --alpha-grid 0.05 0.95 0.01
"""
import argparse, glob, os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
R = 8.314462618

def read_run(path):
    df = pd.read_csv(path, comment="#")
    cols = {c.lower(): c for c in df.columns}
    Tcol = cols.get("temperature_k") or cols.get("t")
    tcol = cols.get("time_s") or cols.get("time")
    mcol = cols.get("mass"); mfcol = cols.get("mass_fraction")
    betacol = cols.get("beta_k_per_min")
    if Tcol is None or tcol is None or (mcol is None and mfcol is None):
        raise ValueError(f"Missing required columns in {path}")
    df = df.sort_values(Tcol).reset_index(drop=True)
    T = df[Tcol].to_numpy(float); t = df[tcol].to_numpy(float)
    if mfcol is not None:
        mf = df[mfcol].to_numpy(float)
        m0, minf = mf[0], min(mf[-50:].mean(), mf[-1])
        if minf >= m0: m0, minf = mf.max(), mf.min()
        alpha = (m0 - mf) / max(1e-12, (m0 - minf))
    else:
        m = df[mcol].to_numpy(float); m0 = m[0]
        minf = min(m[-50:].mean(), m[-1]); 
        if minf >= m0: minf = m.min()
        alpha = (m0 - m) / max(1e-12, (m0 - minf))
    alpha = np.clip(alpha, 0, 1)
    if betacol is not None:
        beta = float(df[betacol].iloc[0]) / 60.0
    else:
        beta = np.median(np.gradient(T, t))
    return {"path": path, "T": T, "t": t, "alpha": alpha, "beta": beta}

def smooth(x, window=7):
    if window < 3: return x
    k = int(window) + (int(window) % 2 == 0)
    pad = k//2; xpad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xpad, np.ones(k)/k, mode="valid")

def interp_at_alpha(T, alpha, dalpha_dt, alpha_grid):
    order = np.argsort(alpha)
    a = alpha[order]; Tm = T[order]; r = dalpha_dt[order]
    ua, idx = np.unique(a, return_index=True)
    a, Tm, r = a[idx], Tm[idx], r[idx]
    mask = (a >= 0) & (a <= 1)
    a, Tm, r = a[mask], Tm[mask], r[mask]
    if len(a) < 3:
        return np.full_like(alpha_grid, np.nan, float), np.full_like(alpha_grid, np.nan, float)
    T_of_a = np.interp(alpha_grid, a, Tm, left=np.nan, right=np.nan)
    r_of_a = np.interp(alpha_grid, a, r, left=np.nan, right=np.nan)
    return T_of_a, r_of_a

def fit_friedman(alpha_grid, T_mat, rate_mat):
    E = np.full_like(alpha_grid, np.nan, float)
    b0 = np.full_like(alpha_grid, np.nan, float)
    for j in range(len(alpha_grid)):
        Tj = T_mat[:, j]; rj = rate_mat[:, j]
        mask = np.isfinite(Tj) & np.isfinite(rj) & (Tj > 0) & (rj > 0)
        if mask.sum() >= 2:
            x = 1.0 / Tj[mask]; y = np.log(rj[mask])
            A = np.vstack([np.ones_like(x), x]).T
            b, m = np.linalg.lstsq(A, y, rcond=None)[0]
            E[j] = -m * R; b0[j] = b
    return E, b0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--pattern", default="*.csv")
    ap.add_argument("--alpha-grid", nargs=3, type=float, default=[0.05, 0.95, 0.05])
    ap.add_argument("--out-dir", default="./output")
    ap.add_argument("--smooth-window", type=int, default=7)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.data_dir, a.pattern)))
    if len(files) < 2: raise SystemExit("Need ≥2 runs with different heating rates.")
    runs = [read_run(p) for p in files]
    for r in runs:
        alpha_s = smooth(r["alpha"], a.smooth_window)
        T_s = smooth(r["T"], a.smooth_window)
        dadt_dT = np.gradient(alpha_s, T_s)
        r["dalpha_dt"] = dadt_dT * r["beta"]
    a0, a1, da = a.alpha_grid
    alpha_grid = np.arange(a0, a1 + 1e-12, da)
    T_rows, R_rows = [], []
    for r in runs:
        T_of_a, r_of_a = interp_at_alpha(r["T"], r["alpha"], r["dalpha_dt"], alpha_grid)
        T_rows.append(T_of_a); R_rows.append(r_of_a)
    T_mat = np.vstack(T_rows); rate_mat = np.vstack(R_rows)
    E, b = fit_friedman(alpha_grid, T_mat, rate_mat)
    os.makedirs(a.out_dir, exist_ok=True)
    pd.DataFrame({"alpha": alpha_grid, "E_J_per_mol": E, "lnA_app": b}).to_csv(
        os.path.join(a.out_dir, "friedman_E_alpha.csv"), index=False)
    # Plots
    plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])
    plt.figure()
    for r in runs: plt.plot(r["T"], r["alpha"], label=os.path.basename(r["path"]))
    plt.xlabel("Temperature [K]"); plt.ylabel("Conversion α"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(a.out_dir, "alpha_vs_T.png"), dpi=200); plt.close()
    plt.figure()
    sel = np.linspace(alpha_grid.min(), alpha_grid.max(), 6)
    for val in sel:
        j = int(np.argmin(np.abs(alpha_grid - val)))
        x = 1.0 / T_mat[:, j]; y = np.log(np.clip(rate_mat[:, j], 1e-300, None))
        m = np.isfinite(x) & np.isfinite(y)
        plt.scatter(x[m], y[m], label=f"α={alpha_grid[j]:.2f}")
    plt.xlabel("1/T [1/K]"); plt.ylabel("ln(dα/dt) [1/s]"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(a.out_dir, "dalpha_dt_vs_invT_selected_alphas.png"), dpi=200); plt.close()
    plt.figure(); plt.plot(alpha_grid, E); plt.xlabel("α"); plt.ylabel("E(α) [J/mol]"); plt.tight_layout()
    plt.savefig(os.path.join(a.out_dir, "E_alpha.png"), dpi=200); plt.close()
    print("Done.")

if __name__ == "__main__":
    main()

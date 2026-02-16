import os
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import re

R = 8.314462618

def extract_beta_from_filename(filename):
    match = re.search(r'_(\d+)min_', filename)
    if match:
        return int(match.group(1))
    return None

def run_kissinger_analysis(
    input_dir='data_modified',
    output_dir='kinetics_results',
    da_dt_column='dalpha_dt',
    save_plots=True,
    save_csv=True
):
    try:
        os.makedirs(output_dir, exist_ok=True)
        data = {}
        sample_name = None

        for filename in os.listdir(input_dir):
            if not filename.endswith('_processed.csv'):
                continue
            filepath = os.path.join(input_dir, filename)
            beta = extract_beta_from_filename(filename)
            if beta is None:
                continue

            name_match = re.match(r'(.*?)_\d+min_', filename)
            if name_match:
                current_name = name_match.group(1)
                if sample_name is None:
                    sample_name = current_name

            df = pd.read_csv(filepath)
            if not all(col in df.columns for col in ['T_K', da_dt_column]):
                continue

            df['abs_da_dt'] = np.abs(df[da_dt_column])
            idx_max = df['abs_da_dt'].idxmax()
            T_p = df.loc[idx_max, 'T_K']

            data[beta] = T_p

        if len(data) < 2:
            raise ValueError("Нужно минимум 2 кривых")

        betas = sorted(data.keys())
        T_p = [data[b] for b in betas]

        x = 1 / np.array(T_p)
        y = np.log(np.array(betas) / (np.array(T_p) ** 2))

        res = linregress(x, y)
        slope = res.slope
        Ea = -slope * R / 1000
        R2 = res.rvalue ** 2

        points_df = pd.DataFrame({'beta_min': betas, 'T_p_K': T_p, 'T_p_C': np.array(T_p) - 273.15})

        files_saved = {}
        if save_csv:
            ea_path = os.path.join(output_dir, f'{sample_name}_kissinger_Ea.csv')
            pd.DataFrame([{'Ea_kJ_mol': Ea, 'R2': R2, 'slope': slope, 'intercept': res.intercept}]).to_csv(ea_path, index=False, float_format='%.6f')
            files_saved['ea_csv'] = ea_path

            points_path = os.path.join(output_dir, f'{sample_name}_kissinger_points.csv')
            points_df.to_csv(points_path, index=False, float_format='%.6f')
            files_saved['points_csv'] = points_path

        if save_plots:
            plt.figure(figsize=(8, 6))
            plt.scatter(x, y, color='blue')
            plt.plot(x, res.slope * x + res.intercept, '--', color='red')
            plt.xlabel('1/T_p (K⁻¹)')
            plt.ylabel('ln(β / T_p²)')
            plt.title(f'Kissinger Plot — {sample_name}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'{sample_name}_kissinger_plot.png')
            plt.savefig(plot_path, dpi=300)
            plt.close()
            files_saved['plot'] = plot_path

        return {
            'status': 'success',
            'sample_name': sample_name,
            'Ea_kJ_mol': Ea,
            'R2': R2,
            'points_df': points_df,
            'files': files_saved,
            'message': 'Kissinger analysis completed'
        }

    except Exception as e:
        return {'status': 'error', 'message': str(e)}

if __name__ == '__main__':
    result = run_kissinger_analysis(
        input_dir='../data_modified',
        output_dir='../kinetics_results',
        da_dt_column='dalpha_dt',
        save_plots=True,
        save_csv=True
    )
    print(result['message'])
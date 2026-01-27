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

def run_ozawa_analysis(
    input_dir='data_modified',
    output_dir='kinetics_results',
    alpha_start=0.05,
    alpha_end=0.96,
    alpha_step=0.05,
    alpha_tolerance=0.03,
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
            if not all(col in df.columns for col in ['T_K', 'inv_T_K', 'alpha_percent']):
                continue

            data[beta] = df

        if len(data) < 2:
            raise ValueError("Нужно минимум 2 кривых")

        alpha_levels = np.arange(alpha_start, alpha_end, alpha_step)

        ea_results = {'alpha': alpha_levels.tolist(), 'Ea_kJ_mol': [], 'R2': [], 'slope': [], 'intercept': []}
        points_list = []
        plot_data = {}

        for alpha_target in alpha_levels:
            rows = []
            for beta, df in sorted(data.items()):
                df['alpha'] = df['alpha_percent'] / 100
                idx = (df['alpha'] - alpha_target).abs().idxmin()
                row = df.loc[idx]
                if abs(row['alpha'] - alpha_target) > alpha_tolerance:
                    continue
                rows.append({
                    'beta_min': beta,
                    'alpha_actual': row['alpha'],
                    'T_K': row['T_K'],
                    '1/T_K': row['inv_T_K']
                })

            if len(rows) < 2:
                ea_results['Ea_kJ_mol'].append(np.nan)
                ea_results['R2'].append(np.nan)
                ea_results['slope'].append(np.nan)
                ea_results['intercept'].append(np.nan)
                continue

            x = np.array([r['1/T_K'] for r in rows])
            y = np.log(np.array([r['beta_min'] for r in rows]))

            res = linregress(x, y)
            slope = res.slope
            intercept = res.intercept
            r_value = res.rvalue
            Ea = -slope * R / 1000

            ea_results['Ea_kJ_mol'].append(Ea)
            ea_results['R2'].append(r_value**2)
            ea_results['slope'].append(slope)
            ea_results['intercept'].append(intercept)

            plot_data[alpha_target] = {
                '1/T': x.tolist(),
                'ln_beta': y.tolist(),
                'labels': [f'{r["beta_min"]} min' for r in rows]
            }

            for r in rows:
                r.update({
                    'alpha_target': alpha_target,
                    'Ea_kJ_mol': Ea,
                    'R2': r_value**2,
                    'slope': slope,
                    'intercept': intercept
                })
                points_list.append(r)

        ea_df = pd.DataFrame(ea_results)
        points_df = pd.DataFrame(points_list)
        if not points_df.empty:
            points_df = points_df.sort_values(['alpha_target', 'beta_min']).reset_index(drop=True)

        files_saved = {}
        if save_csv:
            ea_path = os.path.join(output_dir, f'{sample_name}_ozawa_Ea.csv')
            ea_df.to_csv(ea_path, index=False, float_format='%.6f')
            files_saved['ea_csv'] = ea_path

            points_path = os.path.join(output_dir, f'{sample_name}_ozawa_points.csv')
            points_df.to_csv(points_path, index=False, float_format='%.6f')
            files_saved['points_csv'] = points_path

        if save_plots:
            # График линий
            plt.figure(figsize=(12, 8))
            colors = plt.cm.viridis(np.linspace(0, 1, len(plot_data)))
            for i, (alpha, pts) in enumerate(sorted(plot_data.items())):
                if len(pts['1/T']) < 2:
                    continue
                color = colors[i]
                plt.scatter(pts['1/T'], pts['ln_beta'], color=color, label=f'α = {alpha:.2f}')
                res = linregress(pts['1/T'], pts['ln_beta'])
                plt.plot(pts['1/T'], res.slope * np.array(pts['1/T']) + res.intercept, '--', color=color, alpha=0.7)
            plt.xlabel('1/T (K⁻¹)')
            plt.ylabel('ln(β)')
            plt.title(f'Ozawa Plot — {sample_name}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            lines_path = os.path.join(output_dir, 'ozawa_lines.png')
            plt.savefig(lines_path, dpi=300)
            plt.close()
            files_saved['lines_plot'] = lines_path

            # График Ea vs α
            plt.figure(figsize=(10, 6))
            plt.plot(ea_df['alpha'], ea_df['Ea_kJ_mol'], 'o-', color='green')
            plt.xlabel('Conversion degree α')
            plt.ylabel('Ea (kJ/mol)')
            plt.title(f'Activation Energy vs Conversion — {sample_name} (Ozawa)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            ea_plot_path = os.path.join(output_dir, 'ozawa_Ea_vs_alpha.png')
            plt.savefig(ea_plot_path, dpi=300)
            plt.close()
            files_saved['ea_plot'] = ea_plot_path

        return {
            'status': 'success',
            'sample_name': sample_name,
            'ea_df': ea_df,
            'points_df': points_df,
            'files': files_saved,
            'message': f'Ozawa analysis completed with {len(alpha_levels)} levels'
        }

    except Exception as e:
        return {'status': 'error', 'message': str(e)}

if __name__ == '__main__':
    result = run_ozawa_analysis(
        input_dir='../data_modified',
        output_dir='../kinetics_results',
        alpha_start=0.05,
        alpha_end=0.75,
        alpha_step=0.025,
        alpha_tolerance=0.03,
        save_plots=True,
        save_csv=True
    )
    print(result['message'])
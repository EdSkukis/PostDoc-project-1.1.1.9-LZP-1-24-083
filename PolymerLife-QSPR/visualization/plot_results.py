import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from utils.logger import logger


class ModelEvaluator:
    def __init__(self, output_dir="results/figures/"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

    def plot_parity(self, y_true: np.ndarray, y_pred: np.ndarray, target_name: str, units: str):
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, color='#2c7bb6')

        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (1:1)')

        plt.title(f"Prediction accuracy: {target_name}", fontsize=14, pad=15)
        plt.xlabel(f"Experimental values ({units})", fontsize=12)
        plt.ylabel(f"Predicted values ({units})", fontsize=12)

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        textstr = f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f} {units}\nMAE = {mae:.2f} {units}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
                       verticalalignment='top', bbox=props)

        plt.legend(loc='lower right')
        plt.tight_layout()

        filepath = f"{self.output_dir}/parity_{target_name.replace(' ', '_')}.png"
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Parity Plot saved: {filepath}")
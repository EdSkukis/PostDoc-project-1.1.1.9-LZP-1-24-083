import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from utils.logger import logger


class SHAPAnalyzer:
    def __init__(self, model, X_train: pd.DataFrame, output_dir="results/figures/"):
        self.model = model
        self.X_train = X_train
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info("Calculating SHAP values...")
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer(self.X_train)

    def plot_summary(self, target_name="Durability"):
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, self.X_train, max_display=15, show=False)

        plt.title(f"SHAP Summary: The influence of signs on {target_name}", fontsize=14, pad=20)
        plt.tight_layout()

        filepath = f"{self.output_dir}/shap_summary_{target_name}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"SHAP Summary saved: {filepath}")

    def plot_dependence(self, feature_x: str, target_name="Durability"):
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(feature_x, self.shap_values.values, self.X_train, show=False)

        plt.title(f"SHAP Dependence: {feature_x}", fontsize=14, pad=20)
        plt.tight_layout()

        filepath = f"{self.output_dir}/shap_dep_{feature_x}_{target_name}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    def log_top_features(self, top_n=10, target_name="Target"):
        mean_abs_shap = np.abs(self.shap_values.values).mean(axis=0)

        importance_df = pd.DataFrame({
            'Feature': self.X_train.columns,
            'SHAP_Importance': mean_abs_shap
        })

        top_features = importance_df.sort_values(by='SHAP_Importance', ascending=False).head(top_n)

        logger.info(f"--- Top-{top_n} for target: {target_name} ---")
        for idx, row in top_features.iterrows():
            logger.info(f"{row['Feature']:>15} | Contribution (SHAP): {row['SHAP_Importance']:.6f}")

        return top_features
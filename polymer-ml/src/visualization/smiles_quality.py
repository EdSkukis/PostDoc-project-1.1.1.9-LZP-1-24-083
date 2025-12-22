import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_smiles_quality_stats(df: pd.DataFrame, output_dir: str = "./smiles_check"):
    os.makedirs(output_dir, exist_ok=True)

    # 1) Бар-чарт по Quality_raw
    plt.figure()
    df["Quality_raw"].value_counts().sort_index().plot(kind="bar")
    plt.title("SMILES quality (raw)")
    plt.xlabel("Quality_raw")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quality_raw_counts.png"))
    plt.close()

    # 2) Бар-чарт по Quality_final
    plt.figure()
    df["Quality_final"].value_counts().sort_index().plot(kind="bar")
    plt.title("SMILES quality (final, after auto-fix)")
    plt.xlabel("Quality_final")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quality_final_counts.png"))
    plt.close()

    # 3) Топ-10 причин ошибок
    reason_counts = df["Quality_reason"].value_counts().head(10)
    plt.figure()
    reason_counts.plot(kind="bar")
    plt.title("Top-10 Quality_reason")
    plt.xlabel("Reason")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quality_reason_top10.png"))
    plt.close()

    print(f"Saved plots to {output_dir}:")
    print("  quality_raw_counts.png")
    print("  quality_final_counts.png")
    print("  quality_reason_top10.png")

from src.data_loader import load_polymers_dataset, process_and_save_smiles
from src.visualization.smiles_quality import plot_smiles_quality_stats


def analyze_data():
    """
    Loads the polymer dataset, processes SMILES, and generates quality plots.
    """
    try:
        df = load_polymers_dataset(debug=False)
        df_full, _, _, _ = process_and_save_smiles(df)
        plot_smiles_quality_stats(df_full)
    except Exception as e:
        print(f"An error occurred during data analysis: {e}")


if __name__ == "__main__":
    analyze_data()

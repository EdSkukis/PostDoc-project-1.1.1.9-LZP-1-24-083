import os
import pandas as pd
import kagglehub


def load_polymers_dataset():
    """
    Скачивает датасет с KaggleHub и возвращает очищенный DataFrame
    с колонками: SMILES, Tg, PolymerClass.
    """
    dataset_path = kagglehub.dataset_download(
        "linyeping/extra-dataset-with-smilestgpidpolimers-class"
    )
    files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError("CSV file not found in downloaded dataset folder.")
    csv_file = os.path.join(dataset_path, files[0])

    df = pd.read_csv(csv_file)
    if "polymer class" in df.columns:
        df = df.rename(columns={"polymer class": "PolymerClass"})

    for col in ["SMILES", "Tg", "PolymerClass"]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing in dataset.")

    df = df.dropna(subset=["SMILES", "Tg", "PolymerClass"])
    df["Tg"] = pd.to_numeric(df["Tg"], errors="coerce")
    df = df.dropna(subset=["Tg"])

    # Логически: X = SMILES, y = [Tg, PolymerClass]
    return df[["SMILES", "Tg", "PolymerClass"]]

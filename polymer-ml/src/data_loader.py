import os
import pandas as pd
import kagglehub
from rdkit import Chem, RDLogger


def _is_valid_smiles(s: str, remove_asterisk: bool = True) -> bool:
    if s is None:
        return False
    s = str(s)
    if remove_asterisk:
        s = s.replace("*", "")
    mol = Chem.MolFromSmiles(s)
    return mol is not None


def load_polymers_dataset(min_samples_per_class: int = 2, debug: bool = False):
    """
    Загружает датасет с KaggleHub, нормализует названия колонок,
    чистит данные и (опционально) удаляет классы PolymerClass,
    где количество образцов < min_samples_per_class.
    """
    if debug:
        print("[DEBUG] Downloading dataset from KaggleHub...")

    dataset_path = kagglehub.dataset_download(
        "linyeping/extra-dataset-with-smilestgpidpolimers-class"
    )

    csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
    if not csv_files:
        raise RuntimeError("CSV not found in KaggleHub dataset folder")

    csv_file = os.path.join(dataset_path, csv_files[0])

    if debug:
        print(f"[DEBUG] Using CSV file: {csv_file}")

    df = pd.read_csv(csv_file)

    # 1) Нормализация имён колонок
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    if debug:
        print("[DEBUG] Columns after normalization:", df.columns.tolist())

    # 2) Переименование в целевые имена
    rename_map = {
        "smiles": "SMILES",
        "tg": "Tg",
        "polymer_class": "PolymerClass",
    }
    df = df.rename(columns=rename_map)

    if debug:
        print("[DEBUG] Columns after rename:", df.columns.tolist())

    # 3) Удаляем строку, где SMILES == 'SMILES' (повтор заголовка)
    if "SMILES" not in df.columns or "Tg" not in df.columns or "PolymerClass" not in df.columns:
        raise RuntimeError("Required columns ['SMILES', 'Tg', 'PolymerClass'] not found after rename")

    df = df[df["SMILES"].astype(str).str.upper() != "SMILES"]

    # 4) Чистка по NaN и приведение Tg к числу
    df = df.dropna(subset=["SMILES", "Tg", "PolymerClass"])
    df["Tg"] = pd.to_numeric(df["Tg"], errors="coerce")
    df = df.dropna(subset=["Tg"])

    if debug:
        print("[DEBUG] Shape after basic cleaning:", df.shape)

    # ФИЛЬТРАЦИЯ НЕВАЛИДНЫХ SMILES
    if debug:
        total_before = len(df)
    mask_valid = df["SMILES"].apply(_is_valid_smiles)
    df = df[mask_valid].reset_index(drop=True)
    if debug:
        print(f"[DEBUG] Removed {total_before - len(df)} rows with invalid SMILES")

    # 5) Фильтрация редких классов по min_samples_per_class
    class_counts = df["PolymerClass"].value_counts()
    if debug:
        print("\n[DEBUG] Class counts before filtering:")
        print(class_counts)

    if min_samples_per_class is not None and min_samples_per_class > 1:
        valid_classes = class_counts[class_counts >= min_samples_per_class].index
        dropped = class_counts[class_counts < min_samples_per_class]

        if debug and len(dropped) > 0:
            print(f"\n[DEBUG] Dropping classes with < {min_samples_per_class} samples:")
            print(dropped)

        df = df[df["PolymerClass"].isin(valid_classes)].reset_index(drop=True)

        if debug:
            print("\n[DEBUG] Class counts after filtering:")
            print(df["PolymerClass"].value_counts())

    if debug:
        print("\n[DEBUG] Final shape:", df.shape)
        print(df.head())

    return df[["SMILES", "Tg", "PolymerClass"]]

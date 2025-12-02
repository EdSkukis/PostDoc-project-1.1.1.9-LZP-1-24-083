import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors
from rdkit import RDLogger

from src.config import FP_N_BITS, FP_RADIUS, MODEL_DIR

RDLogger.DisableLog("rdApp.*")


def explain_top_fingerprint_bits(
    pipeline,
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    top_n: int = 10,
    out_dir: str = f"{MODEL_DIR}/fp_bit_explanations",
    max_examples_per_bit: int = 5,
):
    """
    Анализирует топ-N важных fingerprint-битов (как в show_feature_importances)
    и сохраняет для них примеры фрагментов.

    Делает:
      - PNG-файлы с фрагментами для каждого бита;
      - CSV с описанием (bit_id, importance, example_smiles, fragment_smiles, image_path).

    Параметры
    ----------
    pipeline : sklearn.Pipeline
        Твой pipeline, содержащий шаг "multi" с полем .regressor (RandomForest).
    df : pd.DataFrame
        Датафрейм с колонкой SMILES (желательно очищенной, например "SMILES_clean").
    smiles_col : str
        Имя колонки со SMILES.
    top_n : int
        Сколько битов анализировать.
    out_dir : str
        Каталог для сохранения результатов.
    max_examples_per_bit : int
        Максимум примеров молекул на один бит.
    """
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # 1) Берём важности из того же регрессора, что используется в show_feature_importances
    regressor = pipeline.named_steps["multi"].regressor
    importances = regressor.feature_importances_
    fp_importances = importances[:FP_N_BITS]

    # сортировка битов по убыванию важности
    bit_indices_sorted = np.argsort(fp_importances)[::-1]
    top_bits = bit_indices_sorted[:top_n]

    # 2) Список SMILES для поиска примеров
    smiles_list = df[smiles_col].astype(str).tolist()

    records = []

    print("\n=== Explaining top fingerprint bits (Tg regressor) ===")
    print(f"FP_N_BITS={FP_N_BITS}, FP_RADIUS={FP_RADIUS}")
    print(f"Using smiles column: {smiles_col}")
    print(f"Will analyze top-{top_n} bits\n")

    for rank, bit_id in enumerate(top_bits, start=1):
        importance = float(fp_importances[bit_id])
        print(f"--- Bit {bit_id} (rank {rank}, importance={importance:.5f}) ---")

        examples_found = 0

        for idx, smi in enumerate(smiles_list):
            if examples_found >= max_examples_per_bit:
                break

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            # вычисляем Morgan fingerprint с bitInfo для данной молекулы
            bit_info = {}
            _ = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol,
                FP_RADIUS,
                nBits=FP_N_BITS,
                bitInfo=bit_info,
            )

            if bit_id not in bit_info:
                continue

            # для этого бита есть одна или несколько пар (atom_idx, radius_used)
            atom_idx, radius_used = bit_info[bit_id][0]

            # окружение радиуса radius_used вокруг atom_idx
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius_used, atom_idx)
            if not env:
                # если RDKit не нашёл окружение — пропускаем
                continue

            submol = Chem.PathToSubmol(mol, env)
            frag_smiles = Chem.MolToSmiles(submol)

            img_path = os.path.join(
                img_dir,
                f"bit_{bit_id}_example_{examples_found+1}.png",
            )
            Draw.MolToFile(submol, img_path)

            print(f"  Example {examples_found+1}:")
            print(f"    SMILES:           {smi}")
            print(f"    fragment_SMILES:  {frag_smiles}")
            print(f"    atom_idx:         {atom_idx}, radius_used: {radius_used}")
            print(f"    image:            {img_path}")

            records.append(
                {
                    "rank": rank,
                    "bit_id": bit_id,
                    "importance": importance,
                    "example_index": idx,
                    "example_smiles": smi,
                    "fragment_radius_used": int(radius_used),
                    "fragment_smiles": frag_smiles,
                    "image_path": img_path,
                }
            )

            examples_found += 1

        if examples_found == 0:
            print("  No molecules found that activate this bit (in provided df).")
            records.append(
                {
                    "rank": rank,
                    "bit_id": bit_id,
                    "importance": importance,
                    "example_index": None,
                    "example_smiles": None,
                    "fragment_radius_used": None,
                    "fragment_smiles": None,
                    "image_path": None,
                }
            )

    # 3) Сохраняем CSV с итогами
    df_bits = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, "top_tg_bits_fragments.csv")
    df_bits.to_csv(csv_path, index=False)

    print(f"\nSaved fragment images to: {img_dir}")
    print(f"Saved CSV summary to: {csv_path}")

    return df_bits
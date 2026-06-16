import os
import sys
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdFingerprintGenerator

# Защита путей
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.logger import logger
from utils.config import PIPELINE_CONFIG


def decode_and_count_fingerprints(bit_ids: list, output_dir="results/mfpgen/"):
    logger.info(f"Begin visualizing chemical fragments for bits: {bit_ids}")

    data_path = os.path.join(project_root, "data", "extended_polymer_dataset.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(project_root, "data", "raw", "extended_polymer_dataset.csv")
        if not os.path.exists(data_path):
            logger.error(f"Data file not found!")
            return

    df = pd.read_csv(data_path)
    x_col = PIPELINE_CONFIG["x"]["col_name"]
    df = df.dropna(subset=[x_col])
    total_molecules = len(df)

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)

    # Создаем папку (используем абсолютный путь от корня проекта)
    full_output_dir = os.path.join(project_root, output_dir)
    os.makedirs(full_output_dir, exist_ok=True)

    counts = {bit_id: 0 for bit_id in bit_ids}
    drawn = {bit_id: False for bit_id in bit_ids}

    logger.info(f"Let's scan {total_molecules} molecules from the database...")

    for _, row in df.iterrows():
        smiles = str(row[x_col]).replace('[*]', 'C').replace('*', 'C')
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        _ = mfpgen.GetFingerprint(mol, additionalOutput=ao)
        bit_info = ao.GetBitInfoMap()

        for bit_id in bit_ids:
            if bit_id in bit_info:
                counts[bit_id] += 1

                if not drawn[bit_id]:
                    svg_string = Draw.DrawMorganBit(mol, bit_id, bit_info)
                    img_path = os.path.join(full_output_dir, f"fragment_{bit_id}.svg")
                    with open(img_path, "w", encoding="utf-8") as f:
                        f.write(svg_string)
                    drawn[bit_id] = True


    logger.info("=== RESULTS OF STATISTICAL ANALYSIS ===")
    for bit_id in bit_ids:
        percentage = (counts[bit_id] / total_molecules) * 100
        logger.info(
            f"Fragment fp_{bit_id} occurs in {counts[bit_id]} from {total_molecules} polymers ({percentage:.1f}%)")


if __name__ == "__main__":
    top_bits = [392, 429, 64]
    decode_and_count_fingerprints(top_bits)

import os
import pandas as pd
import kagglehub
from typing import Optional, Tuple
from rdkit import Chem, RDLogger
from src.config import SMILES_CHECK_DIR

# RDLogger.DisableLog("rdApp.*")  # глушим спам RDKit


def _try_parse_smiles(s: str):
    """Утилита: безопасный парсинг SMILES, возвращает (mol, sanitize_ok: bool)."""
    if s is None:
        return None, False
    s = str(s).strip()
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None, False
    try:
        Chem.SanitizeMol(mol)
        return mol, True
    except Exception:
        return mol, False


def classify_smiles_with_reason(smi: str) -> tuple[str, str]:
    """
    Возвращает (quality, reason), где quality ∈ {'valid', 'fixable', 'invalid'}.
    reason — короткое текстовое описание причины / состояния.
    """
    if smi is None or str(smi).strip() == "":
        return "invalid", "empty_or_none"

    s = str(smi).strip()

    # 1) Прямая попытка парсинга
    mol, ok = _try_parse_smiles(s)
    if ok:
        return "valid", "parsed_and_sanitized"

    # Если молекула вообще не парсится — возможен fixable, проверяем шаблоны
    # 2) Признаки типичных "исправимых" ошибок
    if "()" in s:
        return "fixable", "empty_branch"
    if "N(=O)=O" in s:
        return "fixable", "raw_nitro_group"
    if " " in s:
        return "fixable", "contains_spaces"
    if "*" in s:
        return "fixable", "contains_asterisk"
    if "." in s:
        return "fixable", "multi_fragment_maybe_salt"
    for bad in ["?", "@@@" ]:
        if bad in s:
            return "fixable", f"contains_{bad}"

    # 3) Если ничего из этого — считаем по умолчанию invalid
    if mol is not None and not ok:
        return "fixable", "sanitization_error"
    return "invalid", "unparsable"


def fix_smiles(smi: str) -> Tuple[Optional[str], str]:

    """
    Пытается автоматически исправить SMILES.
    Возвращает (fixed_smiles | None, fix_info).
    Если вернуть валидный SMILES не удалось — fixed_smiles = None.
    """
    if smi is None:
        return None, "none_input"

    original = str(smi)
    s = original.strip()
    applied_fixes = []

    # 1) Удаляем пробелы
    if " " in s:
        s = s.replace(" ", "")
        applied_fixes.append("remove_spaces")

    # 2) Заменяем "сырые" нитрогруппы N(=O)=O → [N+](=O)[O-]
    if "N(=O)=O" in s:
        s = s.replace("N(=O)=O", "[N+](=O)[O-]")
        applied_fixes.append("normalize_nitro")

    # 3) Убираем пустые скобки
    if "()" in s:
        while "()" in s:
            s = s.replace("()", "")
        applied_fixes.append("remove_empty_branches")

    # 4) Убираем звёздочки (часто метки фрагментов)
    if "*" in s:
        s = s.replace("*", "")
        applied_fixes.append("remove_asterisks")

    # 5) Удаляем соли/мелкие фрагменты — оставляем самый длинный фрагмент
    if "." in s:
        fragments = s.split(".")
        # Берём фрагмент с максимальной длиной
        main_frag = max(fragments, key=len)
        if main_frag != s:
            s = main_frag
            applied_fixes.append("keep_largest_fragment")

    # 6) Финальная проверка исправленного SMILES
    mol, ok = _try_parse_smiles(s)
    if ok:
        if not applied_fixes:
            return s, "no_fix_needed_but_was_called"
        return s, "applied:" + ",".join(applied_fixes)

    # 7) Если всё ещё невалидно — возвращаем None
    return None, "cannot_fix:" + ",".join(applied_fixes) if applied_fixes else "cannot_fix_no_pattern"

def process_and_save_smiles(df: pd.DataFrame, output_dir: str = SMILES_CHECK_DIR):
    os.makedirs(output_dir, exist_ok=True)

    # 1) Классификация исходных SMILES
    qualities = []
    reasons = []
    for smi in df["SMILES"]:
        q, r = classify_smiles_with_reason(smi)
        qualities.append(q)
        reasons.append(r)

    df["Quality_raw"] = qualities
    df["Quality_reason"] = reasons

    print(df.head())
    # 2) Сохраняем "сырые" три файла (как было изначально)
    raw_valid = df[df["Quality_raw"] == "valid"].copy()
    raw_fixable = df[df["Quality_raw"] == "fixable"].copy()
    raw_invalid = df[df["Quality_raw"] == "invalid"].copy()

    raw_valid.to_csv(os.path.join(output_dir, "valid_raw.csv"), index=False)
    raw_fixable.to_csv(os.path.join(output_dir, "fixable_raw.csv"), index=False)
    raw_invalid.to_csv(os.path.join(output_dir, "invalid_raw.csv"), index=False)

    # 3) Пытаемся исправить fixable и invalid
    fixed_smiles = []
    fix_infos = []
    for smi in df["SMILES"]:
        fixed, info = fix_smiles(smi)
        fixed_smiles.append(fixed)
        fix_infos.append(info)

    df["SMILES_fixed"] = fixed_smiles
    df["Fix_info"] = fix_infos

    # 4) Проверяем, какие fixed стали валидными
    final_quality = []
    for orig_q, smi_fixed in zip(df["Quality_raw"], df["SMILES_fixed"]):
        if smi_fixed is None:
            # остались как есть
            final_quality.append(orig_q)
        else:
            # перепроверяем уже fixed
            _, ok = _try_parse_smiles(smi_fixed)
            if ok:
                # теперь точно valid
                final_quality.append("valid_after_fix")
            else:
                final_quality.append(orig_q)

    df["Quality_final"] = final_quality

    # 5) Формируем три итоговых файла для работы
    #    a) Полностью валидные (исходные + исправленные)
    final_valid = df[df["Quality_final"].isin(["valid", "valid_after_fix"])].copy()
    # SMILES_clean — удобная колонка для обучения (исправленный, если есть)
    final_valid["SMILES_clean"] = final_valid.apply(
        lambda row: row["SMILES_fixed"] if pd.notna(row["SMILES_fixed"]) and row["SMILES_fixed"] is not None else row["SMILES"],
        axis=1
    )

    #    b) Всё ещё проблемные после попыток исправления
    final_invalid = df[~df["Quality_final"].isin(["valid", "valid_after_fix"])].copy()

    #    c) Отдельно «исправленные / восстановленные»
    recovered = final_valid[final_valid["Quality_raw"] != "valid"].copy()

    final_valid.to_csv(os.path.join(output_dir, "valid_final.csv"), index=False)
    final_invalid.to_csv(os.path.join(output_dir, "invalid_final.csv"), index=False)
    recovered.to_csv(os.path.join(output_dir, "recovered_from_fixable_or_invalid.csv"), index=False)

    print("\n=== SMILES PROCESSING REPORT ===")
    print("Raw counts:")
    print(df["Quality_raw"].value_counts())
    print("\nFinal counts (after auto-fix):")
    print(df["Quality_final"].value_counts())

    print(f"\nFiles written to: {output_dir}")
    print("  valid_raw.csv, fixable_raw.csv, invalid_raw.csv")
    print("  valid_final.csv, invalid_final.csv, recovered_from_fixable_or_invalid.csv")

    return df, final_valid, final_invalid, recovered


def load_polymers_dataset(debug: bool = False):
    """
    Загружает датасет с KaggleHub, нормализует названия колонок,
    чистит данные.
    """
    if debug:
        print("[DEBUG] Downloading dataset from KaggleHub...")
    
    try:
        dataset_path = kagglehub.dataset_download(
            "linyeping/extra-dataset-with-smilestgpidpolimers-class"
        )
    except Exception as e:
        print(f"Error downloading dataset from KaggleHub: {e}")
        return pd.DataFrame() # Return empty dataframe on error

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

    return df[["SMILES", "Tg", "PolymerClass"]]

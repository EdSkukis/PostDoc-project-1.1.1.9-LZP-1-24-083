import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from .parser import parse_tga_txt_to_csv

# Настройка логирования
logger = logging.getLogger("Preprocessing")


def parse_and_convert(file_path: Path):
    """
    Устойчивый парсинг CSV файла с расчетом кинетических параметров (T_K, alpha, dalpha_dt).
    """
    try:
        if file_path.stat().st_size == 0:
            logger.warning(f"Файл пуст: {file_path.name}")
            return pd.DataFrame()

        # Читаем CSV, который подготовил наш парсер
        df = pd.read_csv(file_path)

        if df.empty:
            logger.warning(f"Данные не найдены в {file_path.name}")
            return pd.DataFrame()

        df.columns = df.columns.str.strip()

        # Проверка обязательных колонок (теперь они называются как в приборе)
        required = {"Ts", "Value", "Index"}
        if not required.issubset(df.columns):
            logger.error(f"В {file_path.name} отсутствуют колонки: {required - set(df.columns)}")
            return pd.DataFrame()

        # Переименование для внутреннего использования
        df = df.rename(columns={
            "Ts": "T_C",
            "Value": "mass_percent",
            "Index": "time_s"
        })

        # Конвертация в числа
        for col in ["T_C", "mass_percent", "time_s"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=["T_C", "mass_percent", "time_s"])

        if len(df) < 10:
            logger.warning(f"Слишком мало валидных строк в {file_path.name}")
            return pd.DataFrame()

        # --- Расчеты ---
        df["T_K"] = df["T_C"] + 273.15
        df["inv_T_K"] = 1 / df["T_K"]

        initial_mass = df['mass_percent'].iloc[0]
        final_mass = df['mass_percent'].iloc[-1]
        mass_range = initial_mass - final_mass

        if abs(mass_range) < 1e-6:
            logger.error(f"Нулевое изменение массы в {file_path.name}, расчет alpha невозможен.")
            return pd.DataFrame()

        # Расчет степени превращения (alpha)
        df['alpha'] = abs(initial_mass - df['mass_percent']) / mass_range
        df['alpha_percent'] = df['alpha'] * 100

        # Расчет скорости превращения dalpha/dt
        dt = np.gradient(df['time_s'])
        dt[dt <= 0] = np.nan  # Защита от деления на ноль

        df['dalpha_dt'] = np.gradient(df['alpha_percent']) / dt

        # Логарифм скорости (нужен для метода Фридмана)
        df['ln_dalpha_dt'] = np.where(df['dalpha_dt'] > 0, np.log(df['dalpha_dt']), np.nan)

        return df

    except Exception as e:
        logger.exception(f"Критическая ошибка при обработке {file_path.name}: {str(e)}")
        return pd.DataFrame()


def process_all_raw_files(input_dir='data_csv', output_dir='data_modified'):
    """
    Основной цикл обработки: TXT -> CSV -> Кинетические параметры.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processed_list = []

    # Ищем и TXT (сырые с прибора) и CSV (если загружены сразу в CSV)
    raw_files = list(input_path.glob("*.txt")) + list(input_path.glob("*.csv"))
    logger.info(f"Найдено файлов для обработки: {len(raw_files)}")

    for file_path in raw_files:
        try:
            # 1. Конвертация если это TXT
            if file_path.suffix.lower() == '.txt':
                temp_csv = output_path / file_path.with_suffix('.csv').name
                if not parse_tga_txt_to_csv(file_path, temp_csv):
                    continue
                target_file = temp_csv
            else:
                target_file = file_path

            # 2. Математическая обработка
            df_processed = parse_and_convert(target_file)

            if not df_processed.empty:
                out_name = f"{file_path.stem}_processed.csv"
                df_processed.to_csv(output_path / out_name, index=False)
                processed_list.append(out_name)
                logger.info(f"Успешно обработан: {out_name}")

        except Exception as e:
            logger.error(f"Не удалось обработать файл {file_path.name}: {e}")

    return processed_list
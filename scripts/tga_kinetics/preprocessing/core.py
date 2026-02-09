import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Preprocessing")


def parse_and_convert(file_path: Path):
    """
    Устойчивый парсинг файла с детальным логированием ошибок.
    """
    try:
        # Проверка размера файла перед чтением
        if file_path.stat().st_size == 0:
            logger.warning(f"Файл пуст: {file_path.name}")
            return pd.DataFrame()

        df = pd.read_csv(file_path)

        if df.empty:
            logger.warning(f"Данные не найдены в {file_path.name}")
            return pd.DataFrame()

        # Очистка заголовков (убираем лишние пробелы)
        df.columns = df.columns.str.strip()

        # Валидация колонок
        required = {"Ts", "Value", "Index"}
        if not required.issubset(df.columns):
            logger.error(f"В {file_path.name} отсутствуют колонки: {required - set(df.columns)}")
            return pd.DataFrame()

        # Пропускаем строку с единицами измерения, если она не числовая
        df = df.iloc[1:].reset_index(drop=True)

        # Переименование
        df = df.rename(columns={
            "Ts": "T_C",
            "Value": "mass_percent",
            "Index": "time_s"
        })

        # Конвертация в числа с отловом некорректных значений
        for col in ["T_C", "mass_percent", "time_s"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=["T_C", "mass_percent", "time_s"])

        if len(df) < 10:
            logger.warning(f"Слишком мало валидных строк в {file_path.name} ({len(df)})")
            return pd.DataFrame()

        # Расчеты
        df["T_K"] = df["T_C"] + 273.15
        df["inv_T_K"] = 1 / df["T_K"]

        initial_mass = df['mass_percent'].iloc[0]
        final_mass = df['mass_percent'].iloc[-1]

        mass_range = initial_mass - final_mass
        if abs(mass_range) < 1e-6:
            logger.error(f"Нулевое изменение массы в {file_path.name}, расчет alpha невозможен.")
            return pd.DataFrame()

        df['alpha'] = abs(initial_mass - df['mass_percent']) / mass_range
        df['alpha_percent'] = df['alpha'] * 100

        # dalpha/dt с обработкой деления на ноль в dt
        dt = np.gradient(df['time_s'])
        dt[dt == 0] = np.nan  # Избегаем inf
        df['dalpha_dt'] = np.gradient(df['alpha_percent']) / dt

        # Логарифм скорости
        df['ln_dalpha_dt'] = np.where(df['dalpha_dt'] > 0, np.log(df['dalpha_dt']), np.nan)

        return df

    except Exception as e:
        logger.exception(f"Критическая ошибка при обработке {file_path.name}: {str(e)}")
        return pd.DataFrame()


def process_all_raw_files(input_dir='data_csv', output_dir='data_modified'):
    """
    Основной цикл с Pathlib для безопасной работы на сервере.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processed_list = []

    if not input_path.exists():
        logger.error(f"Входная директория не найдена: {input_dir}")
        return []

    files = list(input_path.glob("*.csv"))
    logger.info(f"Найдено файлов для обработки: {len(files)}")

    for file_path in files:
        df = parse_and_convert(file_path)

        if not df.empty:
            out_name = f"{file_path.stem}_processed.csv"
            df.to_csv(output_path / out_name, index=False)
            processed_list.append(out_name)
            logger.info(f"Успешно обработан: {out_name}")

    return processed_list
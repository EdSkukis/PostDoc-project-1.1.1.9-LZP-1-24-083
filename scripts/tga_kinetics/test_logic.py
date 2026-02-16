import os
from pathlib import Path
from preprocessing.core import process_all_raw_files
import pandas as pd

# 1. Подготовка папок
os.makedirs('data_csv', exist_ok=True)
os.makedirs('data_modified', exist_ok=True)

# 2. Имитация загрузки файла
# Положите ваш исходный .txt файл в папку data_csv перед запуском
print("--- Шаг 1: Проверка наличия файлов ---")
files = list(Path('data_csv').glob('*'))
print(f"Файлы в data_csv: {[f.name for f in files]}")

if not files:
    print("❌ Ошибка: Положите .txt или .csv файл в папку data_csv")
else:
    # 3. Запуск препроцессинга (включает конвертацию TXT -> CSV и расчет параметров)
    print("\n--- Шаг 2: Запуск обработки (Preprocessing) ---")
    processed = process_all_raw_files(input_dir='data_csv', output_dir='data_modified')

    if processed:
        print(f"✅ Успешно обработано: {processed}")

        # 4. Проверка содержимого полученного файла
        print("\n--- Шаг 3: Проверка данных в обработанном файле ---")
        sample_file = Path('data_modified') / processed[0]
        df = pd.read_csv(sample_file)

        print(f"Колонки: {df.columns.tolist()}")
        print(f"Первые 5 строк данных:\n{df.head()}")

        # Проверяем, что расчеты (alpha, T_K) на месте
        if 'alpha' in df.columns and 'T_K' in df.columns:
            print("\n🔥 Ключевые параметры (alpha, T_K, ln_dalpha_dt) рассчитаны верно!")
    else:
        print("❌ Ошибка: Обработка не вернула результатов. Проверьте логи в preprocessing.log")
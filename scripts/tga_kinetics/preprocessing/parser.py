import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger("API")


def parse_tga_txt_to_csv(input_path: Path, output_path: Path):
    """
    Parses a TGA TXT file.
    Columns: Index, Ts, Value
    Stop: at the first empty line after the start of the data.
    """
    data_lines = []
    start_reading = False

    try:
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                clean_line = line.strip()

                # 1. Точный поиск заголовка (Index Ts Value)
                if "Index" in clean_line and "Ts" in clean_line and "Value" in clean_line:
                    start_reading = True
                    continue

                if start_reading:
                    # 2. Остановка на пустой строке
                    if not clean_line:
                        break

                    # 3. Разделение строки
                    parts = clean_line.split()

                    # 4. Валидация структуры (минимум 3 колонки)
                    if len(parts) >= 3:
                        try:
                            # Пробуем перевести в числа для проверки валидности
                            idx = int(parts[0])
                            ts = float(parts[1])
                            value = float(parts[2])

                            # Сохраняем строку
                            data_lines.append([idx, ts, value])
                        except ValueError:
                            # Если в строке мусор (например, "---" или текст) — пропускаем
                            logger.warning(f"Skipped invalid line in {input_path.name}: {clean_line}")
                            continue

        if not data_lines:
            logger.error(f"No valid data found in {input_path.name}")
            return False

        # 5. Создание DataFrame с оригинальными именами колонок
        df = pd.DataFrame(data_lines, columns=['Index', 'Ts', 'Value'])

        # 6. Финальная валидация данных перед сохранением
        # Удаляем дубликаты индексов и строки с NaN
        df = df.drop_duplicates(subset=['Index']).dropna()

        # Сохраняем в CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully converted: {input_path.name} (Rows: {len(df)})")
        return True

    except Exception as e:
        logger.error(f"Error parsing {input_path.name}: {e}")
        return False
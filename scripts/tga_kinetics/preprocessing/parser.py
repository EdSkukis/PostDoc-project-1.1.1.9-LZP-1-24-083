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
    encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1251']
    for enc in encodings:
        data_lines = []
        start_reading = False
        try:
            with open(input_path, 'r', encoding=enc, errors='ignore') as f:
                for line in f:
                    clean_line = line.strip()

                    if "Index" in clean_line and "Ts" in clean_line and "Value" in clean_line:
                        start_reading = True
                        continue

                    if start_reading:
                        if not clean_line:
                            break

                        parts = clean_line.split()

                        if len(parts) >= 3:
                            try:
                                idx = int(parts[0])
                                ts = float(parts[1])
                                value = float(parts[2])

                                data_lines.append([idx, ts, value])
                            except ValueError:
                                logger.warning(f"Skipped invalid line in {input_path.name}: {clean_line}")
                                continue

            if data_lines:
                df = pd.DataFrame(data_lines, columns=['Index', 'Ts', 'Value'])
                df = df.drop_duplicates(subset=['Index']).dropna()

                output_path.parent.mkdir(parents=True, exist_ok=True)

                df.to_csv(output_path, index=False)
                logger.info(f"Successfully converted {input_path.name} using {enc} ({len(df)} rows)")
                return True

        except Exception as e:
            logger.debug(f"Failed to read {input_path.name} with {enc}: {e}")
            continue

    logger.error(f"Could not parse {input_path.name} with any provided encoding.")
    return False
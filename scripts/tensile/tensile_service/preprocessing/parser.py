import pandas as pd
import io
import logging

logger = logging.getLogger(__name__)


def split_raw_txt_by_test(file_path: str, encoding: str = "utf-8"):
    """
    Parses combined TXT file. Detects 'Strain (%)' headers to split experiments.
    """
    experiments = []
    current_block = []

    with open(file_path, 'r', encoding=encoding) as f:
        for line in f:
            if "Strain (%)" in line:
                if current_block:
                    df = pd.read_csv(io.StringIO("".join(current_block)))
                    if not df.empty: experiments.append(df)
                    current_block = []
            current_block.append(line)

        if current_block:
            df = pd.read_csv(io.StringIO("".join(current_block)))
            if not df.empty: experiments.append(df)

    return experiments
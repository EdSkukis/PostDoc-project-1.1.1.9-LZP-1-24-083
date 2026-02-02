import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

def parse_and_convert(file_path):
    """
    Parse and process a CSV file from TGA/DSC experiment.
    ... file structure example:
    Index, Ts, Value
    [C], [%]
    0, 42.3473, 100
    ...
    - Reads CSV, renames columns, converts to numeric.
    - Calculates temperature in K, inverse T, conversion alpha, rate dalpha/dt, ln(rate).
    - Handles errors gracefully, returns empty DF on failure.
    """
    df = None  # Initialize to None for error handling
    try:
        # Read CSV; assume first row is data or skip if needed
        df = pd.read_csv(file_path)

        # Check if DF is empty
        if df.empty:
            raise ValueError("Empty DataFrame after reading CSV")

        # Drop first row if it's metadata (e.g., units); adjust if not needed
        df = df.iloc[1:].reset_index(drop=True)

        # Check required columns exist
        required_cols = ["Ts", "Value", "Index"]
        if not all(col in df.columns for col in required_cols):
            raise KeyError(f"Missing required columns: {set(required_cols) - set(df.columns)}")

        # Rename columns for clarity
        df = df.rename(columns={
            "Ts": "T_C",  # Temperature in °C
            "Value": "mass_percent",  # Mass in %
            "Index": "time_s",  # Time in seconds
        })

        # Convert all columns to numeric, coerce errors to NaN
        df = df.apply(pd.to_numeric, errors='coerce')

        # Drop rows with all NaN (if any)
        df = df.dropna(how='all')

        # If after cleaning DF is too small, skip calculations
        if len(df) < 10:
            raise ValueError(f"DataFrame too small for calculations {len(df)}")

        # Basic temperature calculation
        df["T_K"] = df["T_C"] + 273.15  # Temperature in Kelvin
        df["inv_T_K"] = 1 / df["T_K"]  # Inverse temperature in 1/K

        # Time in minutes
        df['time_min'] = df['time_s'] / 60

        # Alpha calculation
        # Conversion alpha: Normalize to max mass to handle initial buoyancy effect
        initial_mass = df['mass_percent'].iloc[0] # ~100%
        final_mass = df['mass_percent'].iloc[-1]

        if initial_mass == final_mass:
            raise ZeroDivisionError("Initial and final mass are equal; cannot compute alpha")

        df['alpha'] = abs(initial_mass - df['mass_percent']) / (initial_mass - final_mass)  # 0 to 1
        df['alpha_percent'] = df['alpha'] * 100


        df['dalpha_dt'] = np.gradient(df['alpha_percent'], df['time_s'])


        # window_length = 3
        # polyorder = 3
        # if window_length % 2 == 0:
        #     window_length += 1
        # if window_length > len(df):
        #     window_length = len(df) if len(df) % 2 else len(df) - 1
        #
        # alpha_smooth = savgol_filter(df['alpha_percent'], window_length=window_length, polyorder=polyorder)
        #
        # # da/dt
        # df['dalpha_dt'] = np.gradient(alpha_smooth, df['time_s'])


        # ln(dalpha/dt) - only positive values (common in Friedman method)
        df['ln_dalpha_dt'] = np.where(df['dalpha_dt'] > 0, np.log(df['dalpha_dt']), np.nan)

        print(f"Successfully parsed {file_path}")
        # print(df.head())  # Print head for debugging

    except Exception as e:  # Catch all exceptions for robustness
        print(f"Could not parse or process {file_path}: {str(e)}")
        df = pd.DataFrame()  # Return empty DF on error

    # Return only existing columns safely
    return_cols = ['time_s', 'T_C', 'T_K', 'inv_T_K', 'mass_percent',
                   'alpha', 'alpha_percent', 'dalpha_dt', 'ln_dalpha_dt']
    if not df.empty:
        return df[[col for col in return_cols if col in df.columns]]
    else:
        return df


def main():
    """
    Main function: Process all CSV files in input_dir and save to output_dir.
    """
    input_dir = '../data_csv'
    output_dir = '../data_modified'
    os.makedirs(output_dir, exist_ok=True)  # Create output dir if not exists

    for filename in os.listdir(input_dir):
        if not filename.endswith('.csv'):
            print(f"Skipping non-CSV file: {filename}")
            continue

        file_path = os.path.join(input_dir, filename)
        try:
            df = parse_and_convert(file_path)
            if df.empty:
                print(f"Skipping empty result for {filename}")
                continue

            output_filename = os.path.splitext(filename)[0] + '_processed.csv'
            output_path = os.path.join(output_dir, output_filename)
            df.to_csv(output_path, index=False)
            print(f"Successfully converted {filename} to {output_filename}")
        except Exception as e:
            print(f"Could not convert {filename}: {str(e)}")


if __name__ == '__main__':
    main()
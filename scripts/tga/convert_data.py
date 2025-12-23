import os
import pandas as pd
import numpy as np


def parse_and_convert(file_path):
    """
    Parse and process a CSV file from TGA/DSC experiment.
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
        if len(df) < 2:
            raise ValueError("DataFrame too small for calculations")

        # Calculations
        df["T_K"] = df["T_C"] + 273.15  # Temperature in Kelvin
        df["inv_T_K"] = 1 / df["T_K"]  # Inverse temperature in 1/K

        # Conversion alpha: Normalize to max mass to handle initial buoyancy effect
        initial_mass = df['mass_percent'].max()  # Use max to handle initial mass increase
        final_mass = df['mass_percent'].min()  # Assume min is residue

        if initial_mass == final_mass:
            raise ZeroDivisionError("Initial and final mass are equal; cannot compute alpha")

        df['alpha'] = (initial_mass - df['mass_percent']) / (initial_mass - final_mass)  # alpha from 0 to 1
        # No abs or *100; alpha should be 0-1; if negative, data issue

        df['time_min'] = df['time_s'] / 60  # Time in minutes

        # Rate of conversion dalpha/dt (in 1/min)
        df['dalpha_dt'] = np.gradient(df['alpha'], df['time_min'])

        # Ln of rate; handle <=0 by setting NaN
        df['ln_dalpha_dt'] = np.log(df['dalpha_dt'].where(df['dalpha_dt'] > 0, np.nan))

        print(f"Successfully parsed {file_path}")
        print(df.head())  # Print head for debugging

    except Exception as e:  # Catch all exceptions for robustness
        print(f"Could not parse or process {file_path}: {str(e)}")
        df = pd.DataFrame()  # Return empty DF on error

    return df


def main():
    """
    Main function: Process all CSV files in input_dir and save to output_dir.
    """
    input_dir = 'data_csv'
    output_dir = 'data_modified'
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

            output_filename = os.path.splitext(filename)[0] + '_modified.csv'  # Add suffix for clarity
            output_path = os.path.join(output_dir, output_filename)
            df.to_csv(output_path, index=False)
            print(f"Successfully converted {filename} to {output_filename}")
        except Exception as e:
            print(f"Could not convert {filename}: {str(e)}")


if __name__ == '__main__':
    main()
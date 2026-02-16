import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Define output directory for plots
PLOT_DIR = Path("out/summary_plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Loads and merges combined and summary data."""
    try:
        combined_df = pd.read_csv("out/Combined.csv")
        summary_df = pd.read_csv("out/Summary.csv")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure 'out/Combined.csv' and 'out/Summary.csv' exist.")
        return None

    # Extract condition from experiment name
    summary_df['Condition'] = summary_df['Experiment'].apply(lambda x: 'W' if x.startswith('W') else x.split('_')[0])

    # Merge data
    # Select only necessary columns from summary_df to avoid conflicts
    summary_subset = summary_df[['Experiment', 'Condition', 'Temp_C_mean']]
    merged_df = pd.merge(combined_df, summary_subset, on='Experiment')
    return merged_df

def plot_stress_strain_grouped(df, group_by_col, group_by_values, curve_col, output_dir, file_prefix, plot_title_prefix, palette=None):
    """
    Generates and saves a series of stress-strain plots.
    Each plot corresponds to a value from group_by_values.
    Curves in each plot are colored based on curve_col.
    """
    if df.empty:
        print(f"Skipping plot generation for '{plot_title_prefix}' because the dataframe is empty.")
        return
        
    for value in group_by_values:
        plt.figure(figsize=(10, 8))
        subset_df = df[df[group_by_col] == value]

        if subset_df.empty:
            continue

        ax = sns.lineplot(data=subset_df, x='Strain', y='Stress_MPa', hue=curve_col, errorbar=None, palette=palette)

        ax.set_title(f'{plot_title_prefix}{value}')
        ax.set_xlabel('Engineering Strain (ε)')
        ax.set_ylabel('Engineering Stress (MPa)')
        ax.grid(True)
        
        # Get the legend from the axes and set its title
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, title=curve_col)


        sanitized_value = str(value).replace('.', '_')
        plt.savefig(output_dir / f"{file_prefix}_{sanitized_value}.png")
        plt.close()

def main():
    df = load_data()
    if df is None:
        return

    # Define the custom color palette for speeds
    speed_palette = {
        6.6: 'blue',
        66.0: 'red',
        660.0: 'black'
    }

    # --- Plot Set 1: New data (RH and Speed) ---
    new_data_df = df[~df['Experiment'].str.startswith('S')].copy()
    humidity_conditions = ['Ref', 'W', 'RH75', 'RH98']
    new_data_df = new_data_df[new_data_df['Condition'].isin(humidity_conditions)]
    
    # Get speeds present in new data
    speeds_new = sorted(new_data_df['TestSpeed_mm_min'].unique())

    print("--- Generating plots for new data (Humidity vs. Speed) ---")
    # 4 plots, one for each humidity, curves are speeds
    plot_stress_strain_grouped(new_data_df,
                               group_by_col='Condition',
                               group_by_values=humidity_conditions,
                               curve_col='TestSpeed_mm_min',
                               output_dir=PLOT_DIR,
                               file_prefix='new_data_humidity',
                               plot_title_prefix='Stress-Strain for Condition: ',
                               palette=speed_palette)

    # 3 plots, one for each speed, curves are humidities
    plot_stress_strain_grouped(new_data_df,
                               group_by_col='TestSpeed_mm_min',
                               group_by_values=speeds_new,
                               curve_col='Condition',
                               output_dir=PLOT_DIR,
                               file_prefix='new_data_speed',
                               plot_title_prefix='Stress-Strain for Speed: ')

    # --- Plot Set 2: Old data (Temperature and Speed) ---
    old_data_df = df[df['Experiment'].str.startswith('S')].copy()
    
    # Group temperatures into bins to get 3 groups
    bins = [20, 30, 41, 55] # Adjusted bins
    labels = ['20-30 C', '31-41 C', '42-55 C']
    old_data_df['Temp_Group'] = pd.cut(old_data_df['Temp_C_mean'], bins=bins, labels=labels, right=False)
    
    temperatures_old_groups = sorted(old_data_df['Temp_Group'].dropna().unique())
    speeds_old = sorted(old_data_df['TestSpeed_mm_min'].unique())
    
    print("\n--- Generating plots for old data (Temperature vs. Speed) ---")
    # Plots for old data
    plot_stress_strain_grouped(old_data_df,
                               group_by_col='Temp_Group',
                               group_by_values=temperatures_old_groups,
                               curve_col='TestSpeed_mm_min',
                               output_dir=PLOT_DIR,
                               file_prefix='old_data_temp',
                               plot_title_prefix='Stress-Strain for Temperature: ',
                               palette=speed_palette)

    plot_stress_strain_grouped(old_data_df,
                               group_by_col='TestSpeed_mm_min',
                               group_by_values=speeds_old,
                               curve_col='Temp_Group',
                               output_dir=PLOT_DIR,
                               file_prefix='old_data_speed',
                               plot_title_prefix='Stress-Strain for Speed: ')
    
    print(f"\nAll plots saved to {PLOT_DIR.resolve()}")


if __name__ == "__main__":
    main()

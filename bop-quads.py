import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import re

# --- 1. Core Data Loading and Structuring Function ---

def create_structured_bop_table(countries_dict: dict):
    """
    Loads raw quarterly BOP data for multiple countries, calculates key aggregates,
    and structures the data into a standard IMF presentation format.

    Args:
        countries_dict (dict): A dictionary mapping 3-letter codes to full names.
                               Example: {'ARG': 'Argentina', 'BRA': 'Brazil'}
    Returns:
        A pandas DataFrame with the structured BOP data, or None if data is not found.
    """
    country_codes = list(countries_dict.keys())
    print(f"--- Loading and processing BOP data for: {', '.join(countries_dict.values())} ---")

    try:
        df_raw = pd.read_csv('balanza_latam.csv')
    except FileNotFoundError:
        print("Error: Make sure 'balanza_latam.csv' is in your script's directory.")
        return None

    series_endings = {
        #Componentes current account
        'Goods Balance': 'NETCD_T.G.USD.Q',
        'Services Balance': 'NETCD_T.S.USD.Q',
        'Primary Income': 'NETCD_T.IN1.USD.Q',
        'Secondary Income': 'NETCD_T.IN2.USD.Q',
        'Current Account': 'NETCD_T.CAB.USD.Q',

        'Capital Account': 'NETCD_T.KAB.USD.Q',

        #Componentes Financial Account
        'Direct Investment': 'NNAFANIL_T.D_F.USD.Q',
        'Portfolio Investment': 'NNAFANIL_T.P_F.USD.Q',
        'Financial Derivatives': 'NNAFANIL_T.F_F7.USD.Q',
        'Other Investment': 'NNAFANIL_T.O_F.USD.Q',
        'Reserve Assets': 'A_T.R_F.USD.Q',
        'Financial Account': 'NNAFANIL_T.FAB.USD.Q',
        
        'Net Errors & Omissions': 'NETCD_T.EO.USD.Q'
    }

    series_codes_to_keep = [f"{code}.{ending}" for code in country_codes for ending in series_endings.values()]
    df_filtered = df_raw[df_raw['SERIES_CODE'].isin(series_codes_to_keep)].copy()
    print(df_filtered.head())  # Debugging line to check the filtered Data

    if df_filtered.empty:
        print(f"Warning: No quarterly data found for the specified country codes: {country_codes}.")
        return None

    df_filtered['Country_Code'] = df_filtered['SERIES_CODE'].apply(lambda x: x.split('.')[0])
    series_code_to_indicator_map = {f"{code}.{ending}": name for code in country_codes for name, ending in series_endings.items()}
    df_filtered['Indicator_Short'] = df_filtered['SERIES_CODE'].map(series_code_to_indicator_map)
    
    quarterly_cols_pattern = re.compile(r'^\d{4}-Q[1-4]$')
    quarterly_cols = [col for col in df_raw.columns if quarterly_cols_pattern.match(col)]
    
    df_melted = df_filtered.melt(
        id_vars=['Country_Code', 'Indicator_Short'],
        value_vars=quarterly_cols,
        var_name='Quarter',
        value_name='Value'
    )
    
    df_pivot = df_melted.pivot_table(
        index=['Country_Code', 'Quarter'],
        columns='Indicator_Short',
        values='Value'
    ).reset_index()

    df_pivot['Quarter'] = pd.to_datetime(df_pivot['Quarter']).dt.to_period('Q')

    df_pivot.sort_values(by=['Country_Code', 'Quarter'], inplace=True)
    
    df_bop = df_pivot.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.name not in ['Country_Code', 'Quarter'] else x)

    df_bop = df_bop.fillna(0)
    # Calculate Aggregates
    current_account_components = ['Goods Balance', 'Services Balance', 'Primary Income', 'Secondary Income']
    financial_account_components = ['Direct Investment', 'Portfolio Investment', 'Financial Derivatives', 'Other Investment', 'Reserve Assets']
    
    # With this improved section:
    df_bop['Current Account_Calculated'] = df_bop[current_account_components].sum(axis=1)

    # Create a 'Validation' column to check for discrepancies
    # We round to 2 decimal places to avoid floating point precision issues
    df_bop['CA_Validation_Diff'] = (df_bop['Current Account'].round(2) - df_bop['Current Account_Calculated'].round(2))

    # Print a warning for any rows where the check fails
    mismatches = df_bop[~np.isclose(df_bop['Current Account'], df_bop['Current Account_Calculated'], atol=0.01)]

    if not mismatches.empty:
        print("\n--- WARNING: Current Account Mismatches Found! ---")
        print("The sum of components does not match the reported Current Account for:")
        print(df_bop['CA_Validation_Diff'].round(2))
        print(mismatches[['Country_Code', 'Quarter', 'Current Account', 'Current Account_Calculated']])
    else:
        print("\n✓ Current Account validation successful. All components add up correctly.")

    # You can drop the check columns before exporting if you wish
    df_bop.drop(columns=['Current Account_Calculated', 'CA_Validation_Diff'], inplace=True)



    df_bop['Financial Account_Calculated'] = df_bop[financial_account_components].sum(axis=1)

    # Create a 'Validation' column to check for discrepancies
    # We round to 2 decimal places to avoid floating point precision issues
    df_bop['FA_Validation_Diff'] = (df_bop['Financial Account'].round(2) - df_bop['Financial Account_Calculated'].round(2))

    # Print a warning for any rows where the check fails
    mismatches = df_bop[~np.isclose(df_bop['Financial Account'], df_bop['Financial Account_Calculated'], atol=0.01)]

    if not mismatches.empty:
        print("\n--- WARNING: Financial Account Mismatches Found! ---")
        print("The sum of components does not match the reported Financial Account for:")
        print(mismatches[['Country_Code', 'Quarter', 'Financial Account', 'Financial Account_Calculated']])
        print("\nDetailed differences:")
        for idx, row in mismatches.iterrows():
            print(f"Row {idx}: {row['Country_Code']} {row['Quarter']} - Diff: {row['FA_Validation_Diff']:.10f}")
    else:
        print("\n✓ Financial Account validation successful. All components add up correctly.")

    # You can drop the check columns before exporting if you wish
    df_bop.drop(columns=['Financial Account_Calculated', 'FA_Validation_Diff'], inplace=True)





    df_bop['Financial Account'] = df_bop[financial_account_components].sum(axis=1)
    
    above_the_line_components = ['Current Account', 'Capital Account', 'Financial Account', 'Net Errors & Omissions']
    df_bop['Overall Balance'] = df_bop[above_the_line_components].sum(axis=1)
    
    df_bop['Country_Name'] = df_bop['Country_Code'].map(countries_dict)
    
    print("✓ Data processing complete.")
    return df_bop

# --- 2. Plotting Functions ---

def plot_detailed_bop(df_country_data, country_name):
    """
    Generates a detailed quarterly stacked bar chart for a single country.
    """
    print(f"\n--- Generating Detailed BOP Graph for {country_name} ---")
    df_plot = df_country_data.copy()
    df_plot['Financing (-Δ Reserves)'] = -1 * df_plot['Reserve Assets']
    df_plot['QuarterStr'] = df_plot['Quarter'].astype(str) # Use string representation for plotting
    df_plot.set_index('QuarterStr', inplace=True)
    
    # --- Component Lists and Colors ---
    current_account_components = ['Goods Balance', 'Services Balance', 'Primary Income', 'Secondary Income']
    financial_account_components = ['Direct Investment', 'Portfolio Investment', 'Financial Derivatives', 'Other Investment']
    other_components = ['Capital Account', 'Net Errors & Omissions']
    plot_order = current_account_components + other_components + financial_account_components
    
    colors = {
        'Goods Balance': '#e74c3c', 'Services Balance': '#d35400', 'Primary Income': '#c0392b', 'Secondary Income': '#f1c40f',
        'Capital Account': '#f39c12', 'Net Errors & Omissions': '#7f8c8d',
        'Direct Investment': '#27ae60', 'Portfolio Investment': '#2ecc71', 'Financial Derivatives': '#1abc9c', 'Other Investment': '#3498db'
    }

    # --- Visualization ---
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(20, 10))
    
    df_plot[plot_order].plot(kind='bar', stacked=True, ax=ax, color=[colors.get(c) for c in plot_order], alpha=0.85, width=0.8)
    
    ax.plot(df_plot.index, df_plot['Overall Balance'], color='#34495e', marker='o', linestyle='-', linewidth=3, markersize=8, label='Overall Balance (Sum of Bars)', zorder=10)
    ax.scatter(df_plot.index, df_plot['Financing (-Δ Reserves)'], color='#8e44ad', marker='X', s=150, linewidth=2, label='Financing (-Δ in Reserves)', zorder=11, edgecolors='w')
    
    # --- Formatting ---
    ax.axhline(0, color='black', linewidth=1.5, linestyle='--')
    ax.set_title(f"{country_name}: Detailed Quarterly Balance of Payments", fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel("Quarter", fontsize=14)
    ax.set_ylabel("Millions of USD", fontsize=14)
    formatter = mticker.FuncFormatter(lambda x, p: f'{int(x):,}')
    ax.yaxis.set_major_formatter(formatter)
    
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=11)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=11, loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.show()

def plot_multi_country_overall_balance(df_all_data):
    """
    Generates a line chart comparing the Overall Balance for multiple countries.
    """
    print("\n--- Generating Multi-Country Overall Balance Comparison ---")
    
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    df_all_data['QuarterStr'] = df_all_data['Quarter'].astype(str)
    
    sns.lineplot(
        data=df_all_data,
        x='QuarterStr',
        y='Overall Balance',
        hue='Country_Name',
        palette='colorblind',
        marker='o',
        linewidth=2.5,
        ax=ax
    )

    # --- Formatting ---
    ax.axhline(0, color='black', linewidth=1, linestyle='--')
    ax.set_title("Quarterly BOP Overall Balance Comparison", fontsize=18, fontweight='bold')
    ax.set_xlabel("Quarter", fontsize=12)
    ax.set_ylabel("Millions of USD", fontsize=12)
    formatter = mticker.FuncFormatter(lambda x, p: f'{int(x):,}')
    ax.yaxis.set_major_formatter(formatter)
    
    # Make x-axis labels more readable
    tick_labels = df_all_data['QuarterStr'].unique()
    ax.set_xticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=90, ha='center')
    # Show fewer labels to avoid clutter
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=16))
    
    ax.legend(title="Country", fontsize=10)
    
    plt.tight_layout()
    plt.show()

# --- 3. Main Execution Block ---

if __name__ == "__main__":
    
    # Define the countries you want to process.
    # IMPORTANT: Your 'balanza_latam.csv' file MUST contain the data for all countries listed here.
    countries_to_analyze = {
        'ARG': 'Argentina',
        'BRA': 'Brazil',  # Example: Uncomment if you add Brazil data
        'CHL': 'Chile',    # Example: Uncomment if you add Chile data
        'COL': 'Colombia', # Example: Uncomment if you add Colombia data
        'URY': 'Uruguay',   # Example: Uncomment if you add Uruguay data
        'PER': 'Peru',       # Example: Uncomment if you add Peru data
        'MEX': 'Mexico',
        'BOL': 'Bolivia',
        'PRY': 'Paraguay'
    }
    
    # Load and process the data for all specified countries
    bop_data = create_structured_bop_table(countries_to_analyze)
    print('Data!!!!', bop_data)

    if bop_data is not None:
        # --- Export the structured data to a CSV file ---
        output_csv_filename = "BOP_quad_analysis.csv"

        final_column_order = [
            # Identifiers
            'Country_Name',
            'Country_Code',
            'Quarter',
            # Current Account & Components
            'Goods Balance',
            'Services Balance',
            'Primary Income',
            'Secondary Income',
            'Current Account', # Aggregate
            # Capital Account
            'Capital Account',
            # Financial Account & Components
            'Direct Investment',
            'Portfolio Investment',
            'Financial Derivatives',
            'Other Investment',
            'Financial Account', # Aggregate
            # Balancing Items
            'Net Errors & Omissions',
            'Overall Balance', # Aggregate
            'Reserve Assets' # Below-the-line financing
        ]

        # Reorder the DataFrame columns before exporting
        bop_data_ordered = bop_data[final_column_order]
        bop_data_ordered.fillna(0, inplace=True)
        bop_data_ordered.to_csv(output_csv_filename, index=False)
        print(f"\n✓ Full structured data exported to: {output_csv_filename}")

        # --- Generate Plots ---
        
        # 1. Detailed plot for each country individually
        for code, name in countries_to_analyze.items():
            single_country_df = bop_data[bop_data['Country_Code'] == code]
            plot_detailed_bop(single_country_df, name)
            
        # 2. Comparison plot if more than one country is specified
        if len(countries_to_analyze) > 1:
            plot_multi_country_overall_balance(bop_data)
        else:
            print("\nSkipping multi-country comparison plot as only one country was specified.")

        print("\nAll tasks completed successfully.")
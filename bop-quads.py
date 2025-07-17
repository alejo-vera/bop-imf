import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import re

# --- 1. Core Data Loading and Structuring Function ---

def create_structured_bop_table(countries_dict: dict):
    """
    Carga datos trimestrales de BOP para múltiples países, calcula agregados clave,
    y estructura los datos en un formato de presentación estándar del FMI.

    Args:
        countries_dict (dict): Un mapeo de códigos de 3 letras a nombres completos.
                               Ejemplo: {'ARG': 'Argentina', 'BRA': 'Brazil'}
    Returns:
        Un DataFrame de pandas con los datos de BOP estructurados, o None si no se encuentra datos.
    """
    country_codes = list(countries_dict.keys())
    print(f"--- Cargando y procesando datos de BOP para: {', '.join(countries_dict.values())} ---")

    try:
        df_raw = pd.read_csv('balanza_latam.csv')
    except FileNotFoundError:
        print("Error: Asegúrate de que 'balanza_latam.csv' esté en el directorio de tu script.")
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
        'Reserve Assets': 'NNAFANIL_T.RUE.USD.Q',
        'Financial Account': 'NNAFANIL_T.FAB.USD.Q',
        
        'Net Errors & Omissions': 'NETCD_T.EO.USD.Q'
    }

    series_codes_to_keep = [f"{code}.{ending}" for code in country_codes for ending in series_endings.values()]
    df_filtered = df_raw[df_raw['SERIES_CODE'].isin(series_codes_to_keep)].copy()

    if df_filtered.empty:
        print(f"Advertencia: No se encontraron datos trimestrales para los códigos de país especificados: {country_codes}.")
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

    
    above_the_line_components = ['Current Account', 'Capital Account'] 
    df_bop['Net lending/borrowing'] = df_bop[above_the_line_components].sum(axis=1)
    
    df_bop['Country_Name'] = df_bop['Country_Code'].map(countries_dict)
    
    print("✓ Procesamiento de Datos completado.")
    return df_bop

# --- 2. Ploteo de Funciones ---

def plot_detailed_bop(df_country_data, country_name):
    """
    Genera un gráfico de barras apiladas trimestral detallado para un solo país.
    """
    print(f"\n--- Generando Gráfico Detallado de BOP para {country_name} ---")
    df_plot = df_country_data.copy()
    df_plot['(Δ Reserves)'] = -1 * df_plot['Reserve Assets']
    df_plot['QuarterStr'] = df_plot['Quarter'].astype(str) # Use string representation for plotting
    df_plot.set_index('QuarterStr', inplace=True)
    
    current_account_components = ['Goods Balance', 'Services Balance', 'Primary Income', 'Secondary Income']
    financial_account_components = ['Direct Investment', 'Portfolio Investment', 'Financial Derivatives', 'Other Investment']
    other_components = ['Capital Account', 'Net Errors & Omissions']
    plot_order = current_account_components + other_components + financial_account_components
    
    colors = {
        'Goods Balance': '#e74c3c', 'Services Balance': '#d35400', 'Primary Income': '#c0392b', 'Secondary Income': '#f1c40f',
        'Capital Account': '#f39c12', 'Net Errors & Omissions': '#7f8c8d',
        'Direct Investment': '#27ae60', 'Portfolio Investment': '#2ecc71', 'Financial Derivatives': '#1abc9c', 'Other Investment': '#3498db'
    }

    # --- Visualización ---
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(20, 10))
    
    df_plot[plot_order].plot(kind='bar', stacked=True, ax=ax, color=[colors.get(c) for c in plot_order], alpha=0.85, width=0.8)
    
    ax.plot(df_plot.index, df_plot['Net lending/borrowing'], color='#34495e', marker='o', linestyle='-', linewidth=3, markersize=8, label='Overall Balance', zorder=10)
    ax.scatter(df_plot.index, df_plot['(Δ Reserves)'], color='#8e44ad', marker='X', s=150, linewidth=2, label='(Δ in Reserves)', zorder=11, edgecolors='w')
    
    # --- Formateo ---
    ax.axhline(0, color='black', linewidth=1.5, linestyle='--')
    ax.set_title(f"{country_name}: Gráfico Detallado de la Balanza de Pagos Trimestral", fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel("Trimestre", fontsize=14)
    ax.set_ylabel("Billones de USD", fontsize=14)
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
    Genera un gráfico de líneas que compara la Cuenta Corriente para múltiples países.
    """
    print("\n--- Generando Comparación de la Cuenta Corriente entre Múltiples Países ---")

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    df_all_data['QuarterStr'] = df_all_data['Quarter'].astype(str)
    
    sns.lineplot(
        data=df_all_data,
        x='QuarterStr',
        y='Net lending/borrowing',
        hue='Country_Name',
        palette='colorblind',
        marker='o',
        linewidth=2.5,
        ax=ax
    )

    # --- Formateo ---
    ax.axhline(0, color='black', linewidth=1, linestyle='--')
    ax.set_title("Comparación Trimestral de la Balanza de Pagos", fontsize=18, fontweight='bold')
    ax.set_xlabel("Trimestre", fontsize=12)
    ax.set_ylabel("Billones de USD", fontsize=12)
    formatter = mticker.FuncFormatter(lambda x, p: f'{int(x):,}')
    ax.yaxis.set_major_formatter(formatter)
    
    tick_labels = df_all_data['QuarterStr'].unique()
    ax.set_xticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=90, ha='center')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=16))

    ax.legend(title="País", fontsize=10)

    plt.tight_layout()
    plt.show()

# --- 3. Ejecución Principal de código ---

if __name__ == "__main__":

    # Define los países que quieres procesar.
    # IMPORTANTE:  Tu archivo 'balanza_latam.csv' DEBE contener los datos de todos los países enumerados aquí.
    countries_to_analyze = {
        'ARG': 'Argentina',
        'BRA': 'Brazil',  
        'CHL': 'Chile',    
        'COL': 'Colombia', 
        'URY': 'Uruguay',   
        'PER': 'Peru',       
        'MEX': 'Mexico',
        'BOL': 'Bolivia',
        'PRY': 'Paraguay'
    }

    # Carga y procesa los datos para todos los países especificados
    bop_data = create_structured_bop_table(countries_to_analyze)

    if bop_data is not None:
        # --- Exporta la data estructurada a un archivo CSV ---
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
            'Net lending/borrowing', # Aggregate
            'Reserve Assets' # Below-the-line financing
        ]

        # Reordena las columnas del DataFrame antes de exportar
        bop_data_ordered = bop_data[final_column_order]
        bop_data_ordered.fillna(0, inplace=True)
        bop_data_ordered.to_csv(output_csv_filename, index=False)
        print(f"\n✓ Data completa estructurada exportada a: {output_csv_filename}")

        # --- Genera Plots ---

        # 1. Plot detallado para cada país individualmente
        for code, name in countries_to_analyze.items():
            single_country_df = bop_data[bop_data['Country_Code'] == code]
            plot_detailed_bop(single_country_df, name)

        # 2. Plot de comparación si se especifican más de un país
        if len(countries_to_analyze) > 1:
            plot_multi_country_overall_balance(bop_data)
        else:
            print("\nSaltea el plot de comparación entre países ya que solo se especificó un país.")

        print("\nTodas las tareas se completaron con éxito.")
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- 1. Carga de Data ---
# Esta función carga los datos preestructurados.
# El decorador @st.cache_data hace que la aplicación sea rápida al cargar los datos solo una vez.
# @st.cache_data
def load_bop_data(filepath="BOP_quad_analysis.csv"):
    """Carga los datos preestructurados de la BOP desde el archivo CSV."""
    try:
        df = pd.read_csv(filepath)
        # Asegurarse de que la columna 'Quarter' se trate como una cadena para la estabilidad de la representación gráfica
        df['Quarter'] = df['Quarter'].astype(str)
        return df
    except FileNotFoundError:
        return None

# --- 2. Gráfico para una Vista Detallada ---
def create_detailed_bop_figure(df_single_country: pd.DataFrame, country_name: str):
    """
    Crea una figura interactiva de Plotly para la BOP detallada de un solo país.

    Args:
        df_single_country (pd.DataFrame): El DataFrame que contiene datos para SOLO un país.
        country_name (str): El nombre completo del país para el título.

    Returns:
        Figura Plotly.
    """
    df_plot = df_single_country.copy()
    print(df_plot.columns)
    df_plot['(Δ Reserves)'] = df_plot['Reserve Assets']  # Invertir el saldo neto para mostrarlo como financiamiento

    components_to_plot = [
        'Goods Balance', 'Services Balance', 'Primary Income', 'Secondary Income',
        'Capital Account', 'Net Errors & Omissions',
        'Direct Investment', 'Portfolio Investment', 'Financial Derivatives', 'Other Investment'
    ]

    legend_names = [name.strip() for name in components_to_plot]

    colors = {
        'Goods Balance': '#e74c3c', 'Services Balance': '#d35400', 'Primary Income': '#c0392b', 'Secondary Income': '#f1c40f',
        'Capital Account': '#f39c12', 'Net Errors & Omissions': '#7f8c8d',
        'Direct Investment': '#27ae60', 'Portfolio Investment': '#2ecc71', 'Financial Derivatives': '#1abc9c', 'Other Investment': '#3498db'
    }

    fig = go.Figure()

    for component, legend_name in zip(components_to_plot, legend_names):
        if component in df_plot.columns:
            fig.add_trace(go.Bar(
                x=df_plot['Quarter'],
                y=df_plot[component],
                name=legend_name,
                marker_color=colors.get(component)
            ))

    fig.add_trace(go.Scatter(
        x=df_plot['Quarter'],
        y=df_plot['Net lending/borrowing'],
        mode='lines+markers',
        name='Net lending/borrowing',
        line=dict(color='#34495e', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_plot['Quarter'],
        y=df_plot['(Δ Reserves)'],
        mode='markers',
        name='(Δ Reserves)',
        marker=dict(symbol='x-thin', size=12, color='#8e44ad', line=dict(width=2))
    ))

    fig.update_layout(
        barmode='relative', 
        title_text=f"Desglose de Balanza de Pagos: {country_name}",
        xaxis_title="Cuatrimestre. Fuente: Balanza de Pagos según metodología FMI",
        yaxis_title="Expresado en Billones de USD",
        legend_title="Componente BOP",
        template="plotly_white",
        height=700,
        xaxis_tickangle=-45,
        legend=dict(traceorder='normal'),
        yaxis=dict(
            tickformat="$",
            ticksuffix="B"
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
    
    return fig


st.set_page_config(layout="wide")
st.title("Análisis Interactivo de la Balanza de Pagos")

bop_data = load_bop_data()

if bop_data is None:
    st.error("Error: No pude encontrar el archivo 'BOP_Structured_Quarterly_Analysis.csv'.")
    st.write("Por favor, ejecuta primero tu script de procesamiento de datos para crear este archivo y colócalo en el mismo directorio que esta aplicación Streamlit.")
else:
    all_countries = sorted(bop_data['Country_Name'].unique())

    # --- Part 1: Vista detallada por País ---
    st.header("Desglose Detallado por País")

    country_to_show = st.selectbox(
        'Selecciona un país para ver el desglose detallado:',
        options=all_countries
    )

    if country_to_show:
        # Filtra el DataFrame principal para el país seleccionado
        df_single = bop_data[bop_data['Country_Name'] == country_to_show]

        # Genera y muestra el gráfico utilizando nuestra función dedicada
        detailed_fig = create_detailed_bop_figure(df_single, country_to_show)
        st.plotly_chart(detailed_fig, use_container_width=True)

    # --- Part 2: Comparación entre Países ---
    if len(all_countries) > 1:
        st.header("Comparación del Saldo General")
        
        selected_countries = st.multiselect(
            'Selecciona los países a comparar:',
            options=all_countries,
            default=all_countries
        )

        if selected_countries:
            df_multi_filtered = bop_data[bop_data['Country_Name'].isin(selected_countries)]
            
            fig_multi = px.line(
                df_multi_filtered,
                x='Quarter',
                y='Net lending/borrowing',
                color='Country_Name',
                markers=True,
                labels={'Net lending/borrowing': 'Balanza de Pagos (Expresado en Billones de USD)', 'Quarter': 'Cuatrimestre. Fuente: Balanza de Pagos según metodología FMI'},
                title='Comparación del Saldo de Balanza de Pagos Trimestral',
            )
            fig_multi.update_yaxes(
                tickformat="$", 
                ticksuffix="B"
            )
            fig_multi.add_hline(y=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig_multi, use_container_width=True)

    # --- Parte 3: Tabla de Datos y Descarga ---
    st.header("Mira la tabla y descárgala")
    st.write("La tabla a continuación muestra los datos estructurados utilizados para generar estos gráficos.")

    # Muestra el DataFrame completo en la aplicación
    st.dataframe(bop_data)

    # Proporciona un botón de descarga para el CSV
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv_to_download = convert_df_to_csv(bop_data)

    st.download_button(
       label="Descarga la tabla completa de BOP como un CSV",
       data=csv_to_download,
       file_name="BOP_Structured_Quarterly_Analysis.csv",
       mime="text/csv",
    )
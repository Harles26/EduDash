import streamlit as st
import pandas as pd
from utils import (load_data, perform_eda, clustering_analysis,
                   regression_analysis, correlation_analysis, predictive_analysis)

def main():
    st.title('Data Science Dashboard')

    st.sidebar.header('Sube tu archivo')
    uploaded_file = st.sidebar.file_uploader('Elige un archivo CSV o Excel', type=['csv', 'xlsx'])

    if uploaded_file is not None:
        delimiter = st.sidebar.radio("Selecciona el delimitador del archivo CSV", (",", ";"))
        df = load_data(uploaded_file, delimiter)
        st.write('### Vista previa de los datos')
        st.write(df.head())
        
        if st.sidebar.checkbox('Análisis Exploratorio de Datos (EDA)'):
            perform_eda(df)
        
        if st.sidebar.checkbox('Análisis de Clustering'):
            clustering_analysis(df)
        
        if st.sidebar.checkbox('Análisis de Regresión'):
            regression_analysis(df)
        
        if st.sidebar.checkbox('Visualización de Correlaciones'):
            correlation_analysis(df)
        
        if st.sidebar.checkbox('Análisis Predictivo'):
            predictive_analysis(df)
    else:
        st.write('Por favor sube un archivo CSV o Excel para comenzar.')

if __name__ == "__main__":
    main()

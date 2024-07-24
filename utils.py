import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

def load_data(uploaded_file):
    """Carga los datos desde un archivo CSV o Excel subido."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

def perform_eda(df):
    """Realiza un análisis exploratorio de datos (EDA)."""
    st.write('### Descripción de los datos')
    st.write(df.describe())
    
    st.write('### Gráficos de distribución')
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        st.write(f'Distribución de {col}')
        fig = px.histogram(df, x=col, nbins=30, marginal='box', title=f'Distribución de {col}')
        st.plotly_chart(fig)

def clustering_analysis(df):
    """Realiza un análisis de clustering en los datos."""
    st.write('### Análisis de Clustering')
    
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    n_clusters = st.sidebar.slider('Número de clusters', 2, 10, 3)
    selected_columns = st.sidebar.multiselect('Selecciona columnas para clustering', numeric_columns)

    if len(selected_columns) > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        clusters = kmeans.fit_predict(df[selected_columns])
        df['Cluster'] = clusters
        centroids = kmeans.cluster_centers_

        st.write('### Resultados del Clustering')
        st.write(df)

        st.write('### Visualización de Clusters')
        plot_clusters(df, selected_columns, centroids)
    else:
        st.write('Selecciona al menos dos columnas numéricas para realizar clustering.')

def plot_clusters(df, selected_columns, centroids):
    """Genera gráficos de visualización de clusters."""
    fig = px.scatter(df, x=selected_columns[0], y=selected_columns[1], color='Cluster', title='Visualización de Clusters')
    fig.add_scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers', marker=dict(color='red', size=12, symbol='x'), name='Centroides')
    st.plotly_chart(fig)

    if len(selected_columns) > 2:
        col1, col2 = st.sidebar.selectbox('Selecciona dos columnas para visualizar', [(col1, col2) for i, col1 in enumerate(selected_columns) for col2 in selected_columns[i+1:]])
        fig = px.scatter(df, x=col1, y=col2, color='Cluster', title=f'Visualización de Clusters: {col1} vs {col2}')
        col1_idx = selected_columns.index(col1)
        col2_idx = selected_columns.index(col2)
        fig.add_scatter(x=centroids[:, col1_idx], y=centroids[:, col2_idx], mode='markers', marker=dict(color='red', size=12, symbol='x'), name='Centroides')
        st.plotly_chart(fig)

def regression_analysis(df):
    """Realiza un análisis de regresión en los datos."""
    st.write('### Análisis de Regresión')
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    target_column = st.sidebar.selectbox('Selecciona la variable objetivo', numeric_columns)
    feature_columns = st.sidebar.multiselect('Selecciona las variables predictoras', numeric_columns)

    if len(feature_columns) > 0 and target_column:
        X = df[feature_columns]
        y = df[target_column]
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)
        st.write(f'Error Cuadrático Medio: {mse}')

        st.write('### Gráficos de Regresión')
        for col in feature_columns:
            st.write(f'Regresión de {target_column} vs {col}')
            fig = px.scatter(df, x=col, y=target_column, title=f'Regresión de {target_column} vs {col}')
            
            # Añadir línea de tendencia manualmente
            trendline = go.Scatter(x=df[col], y=predictions, mode='lines', name='Línea de tendencia', line=dict(color='red'))
            fig.add_trace(trendline)
            
            st.plotly_chart(fig)

def correlation_analysis(df):
    """Genera un heatmap de las correlaciones entre variables."""
    st.write('### Visualización de Correlaciones')
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    fig = px.imshow(df[numeric_columns].corr(), text_auto=True, aspect="auto", title='Matriz de Correlaciones')
    st.plotly_chart(fig)

def predictive_analysis(df):
    """Realiza un análisis predictivo utilizando un modelo de Random Forest."""
    st.write('### Análisis Predictivo')
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    target_column = st.sidebar.selectbox('Selecciona la variable objetivo para predicción', numeric_columns)
    feature_columns = st.sidebar.multiselect('Selecciona las variables predictoras para predicción', numeric_columns)

    if len(feature_columns) > 0 and target_column:
        X = df[feature_columns]
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        st.write(f'Error Cuadrático Medio en el set de prueba: {mse}')

        st.write('### Importancia de las Variables')
        feature_importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
        st.bar_chart(feature_importances)

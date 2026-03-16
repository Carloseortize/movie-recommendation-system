"""
🎬 Movie Recommender System Dashboard
Dashboard interactivo para sistema de recomendación de películas
"""
import os
import urllib.request
import zipfile
import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configuración de página
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Título
st.title("🎬 Sistema de Recomendación de Películas")
st.markdown("---")

# Cargar datos
@st.cache_data
def load_data():
    """Carga los datos de MovieLens, descargándolos si es necesario"""
    
    # Definir rutas
    data_dir = 'data'
    ml_dir = os.path.join(data_dir, 'ml-latest-small')
    movies_path = os.path.join(ml_dir, 'movies.csv')
    ratings_path = os.path.join(ml_dir, 'ratings.csv')
    
    # Verificar si los datos existen, si no, descargarlos
    if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
        with st.spinner('📥 Descargando dataset MovieLens... (solo la primera vez)'):
            # Crear directorio si no existe
            os.makedirs(data_dir, exist_ok=True)
            
            # URL del dataset
            url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
            zip_path = os.path.join(data_dir, 'ml-latest-small.zip')
            
            # Descargar archivo
            urllib.request.urlretrieve(url, zip_path)
            
            # Extraer archivos
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            # Limpiar archivo zip
            os.remove(zip_path)
            
            st.success('✅ Dataset descargado y extraído correctamente!')
    
    # Cargar datos con Polars
    movies = pl.read_csv(movies_path)
    ratings = pl.read_csv(ratings_path)
    
    # Extraer año (opcional pero útil)
    movies = movies.with_columns(
        pl.col('title')
        .str.extract(r'\((\d{4})\)$', 1)
        .alias('year')
        .cast(pl.Int32, strict=False)
    )
    
    return movies, ratings

@st.cache_resource
def build_similarity_matrix(_ratings, _movies):
    """Construye matriz de similitud"""
    # Filtrar películas con suficientes ratings
    movie_stats = _ratings.group_by('movieId').agg(
        pl.count().alias('rating_count')
    ).filter(pl.col('rating_count') >= 20)
    
    popular_movies = movie_stats['movieId'].to_list()
    ratings_filtered = _ratings.filter(pl.col('movieId').is_in(popular_movies))
    
    # Crear matriz
    ratings_pd = ratings_filtered.to_pandas()
    user_movie_matrix = ratings_pd.pivot_table(
        index='userId', 
        columns='movieId', 
        values='rating'
    ).fillna(0)
    
    # Calcular similitud
    item_similarity = cosine_similarity(user_movie_matrix.T)
    similarity_df = pd.DataFrame(
        item_similarity,
        index=user_movie_matrix.columns,
        columns=user_movie_matrix.columns
    )
    
    return similarity_df, popular_movies

# Cargar datos
with st.spinner('Cargando datos...'):
    movies, ratings = load_data()
    similarity_matrix, popular_movies = build_similarity_matrix(ratings, movies)

# Sidebar - Estadísticas rápidas
with st.sidebar:
    st.header("📊 Estadísticas Rápidas")
    st.metric("Total Películas", f"{movies.height:,}")
    st.metric("Total Calificaciones", f"{ratings.height:,}")
    st.metric("Usuarios Activos", f"{ratings['userId'].n_unique():,}")
    st.metric("Películas en Modelo", f"{len(popular_movies):,}")
    
    st.markdown("---")
    st.header("🔍 Acerca de")
    st.info(
        "Este sistema usa **filtrado colaborativo** "
        "basado en similitud coseno para recomendar "
        "películas similares a partir de tus favoritas."
    )

# Main content - Dos columnas
col1, col2 = st.columns([1, 1])

with col1:
    st.header("🎯 Generar Recomendaciones")
    
    # Selector de película
    movie_options = movies.filter(
        pl.col('movieId').is_in(popular_movies)
    )['title'].to_list()
    
    selected_movie = st.selectbox(
        "Elige una película que te guste:",
        options=sorted(movie_options),
        index=movie_options.index('Toy Story (1995)') if 'Toy Story (1995)' in movie_options else 0
    )
    
    # Número de recomendaciones
    n_recs = st.slider("Número de recomendaciones:", 5, 20, 10)
    
    # Botón de recomendar
    if st.button("🔮 Recomendar Películas", type="primary"):
        # Encontrar movieId
        movie_row = movies.filter(pl.col('title') == selected_movie)
        if movie_row.height > 0:
            movie_id = movie_row['movieId'][0]
            
            if movie_id in similarity_matrix.columns:
                # Obtener similares
                similar_scores = similarity_matrix[movie_id].sort_values(ascending=False)[1:n_recs+1]
                
                # Mostrar recomendaciones
                st.subheader(f"📽️ Porque viste: **{selected_movie}**")
                
                recommendations_data = []
                for idx, (sim_movie_id, score) in enumerate(similar_scores.items(), 1):
                    movie_info = movies.filter(pl.col('movieId') == sim_movie_id)
                    if movie_info.height > 0:
                        title = movie_info['title'][0]
                        year = movie_info['year'][0] if movie_info['year'][0] else "N/A"
                        recommendations_data.append({
                            "Rank": idx,
                            "Película": title,
                            "Año": year,
                            "Similitud": f"{score:.2%}"
                        })
                
                rec_df = pd.DataFrame(recommendations_data)
                st.dataframe(rec_df, use_container_width=True, hide_index=True)
                
                # Visualización de similitud
                fig = px.bar(
                    rec_df.head(5),
                    x='Similitud',
                    y='Película',
                    orientation='h',
                    title='Top 5 Recomendaciones - Nivel de Similitud',
                    color='Similitud',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Esta película no tiene suficientes datos para generar recomendaciones.")
        else:
            st.error("Película no encontrada")

with col2:
    st.header("📈 Análisis de Datos")
    
    # Tabs para diferentes visualizaciones
    tab1, tab2, tab3 = st.tabs(["Calificaciones", "Películas Populares", "Géneros"])
    
    with tab1:
        # Distribución de calificaciones
        rating_dist = ratings.group_by('rating').agg(
            pl.count().alias('count')
        ).sort('rating').to_pandas()
        
        fig = px.bar(
            rating_dist,
            x='rating',
            y='count',
            title='Distribución de Calificaciones',
            labels={'rating': 'Calificación', 'count': 'Frecuencia'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Rating Promedio", f"{ratings['rating'].mean():.2f}")
        with col_b:
            st.metric("Mediana", f"{ratings['rating'].median():.2f}")
        with col_c:
            st.metric("Moda", f"{ratings['rating'].mode()[0]:.1f}")
    
    with tab2:
        # Películas más calificadas
        top_movies = ratings.group_by('movieId').agg([
            pl.count().alias('rating_count'),
            pl.mean('rating').alias('avg_rating')
        ]).join(movies, on='movieId').sort('rating_count', descending=True).head(10).to_pandas()
        
        fig = px.bar(
            top_movies,
            x='title',
            y='rating_count',
            color='avg_rating',
            title='Top 10 Películas Más Calificadas',
            labels={'title': 'Película', 'rating_count': 'N° Calificaciones', 'avg_rating': 'Rating Prom.'},
            color_continuous_scale='Viridis'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Análisis por género usando DuckDB
        import duckdb
        conn = duckdb.connect()
        conn.register('movies', movies.to_pandas())
        conn.register('ratings', ratings.to_pandas())
        
        genre_query = """
        WITH genre_exploded AS (
            SELECT 
                unnest(string_split(m.genres, '|')) as genre,
                r.rating
            FROM movies m
            JOIN ratings r ON m.movieId = r.movieId
        )
        SELECT 
            genre,
            COUNT(*) as total_ratings,
            AVG(rating) as avg_rating
        FROM genre_exploded
        GROUP BY genre
        ORDER BY total_ratings DESC;
        """
        
        genre_df = conn.execute(genre_query).fetchdf()
        
        fig = px.scatter(
            genre_df,
            x='total_ratings',
            y='avg_rating',
            text='genre',
            title='Géneros: Popularidad vs Calidad',
            labels={'total_ratings': 'Total Ratings', 'avg_rating': 'Rating Promedio'}
        )
        fig.update_traces(textposition='top center')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>"
    "📊 Proyecto de Portafolio - Data Analyst | "
    "Dataset: MovieLens | "
    "Tecnologías: Polars, DuckDB, Scikit-learn, Streamlit"
    "</div>", 
    unsafe_allow_html=True
)

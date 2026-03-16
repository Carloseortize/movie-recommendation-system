"""
📦 Módulo de procesamiento de datos para el sistema de recomendación de películas.

Este módulo contiene funciones para cargar, limpiar y transformar los datos
de MovieLens utilizando Polars (alternativa moderna y rápida a Pandas).
"""

import polars as pl
import numpy as np
from typing import Tuple, Optional, List
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MovieLensDataProcessor:
    """
    Clase para procesar el dataset MovieLens.
    
    Esta clase encapsula toda la lógica de procesamiento de datos:
    - Carga de archivos CSV
    - Limpieza y transformación
    - Feature engineering
    - Filtrado de datos para el modelo
    """
    
    def __init__(self, data_path: str = 'data/ml-latest-small/'):
        """
        Inicializa el procesador con la ruta de los datos.
        
        Args:
            data_path: Ruta donde se encuentran los archivos CSV
        """
        self.data_path = data_path
        self.movies: Optional[pl.DataFrame] = None
        self.ratings: Optional[pl.DataFrame] = None
        self.processed_movies: Optional[pl.DataFrame] = None
        self.processed_ratings: Optional[pl.DataFrame] = None
        
    def load_data(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Carga los archivos movies.csv y ratings.csv usando Polars.
        
        Returns:
            Tuple con (movies_df, ratings_df)
        """
        logger.info("📥 Cargando datos desde %s", self.data_path)
        
        try:
            self.movies = pl.read_csv(f'{self.data_path}movies.csv')
            self.ratings = pl.read_csv(f'{self.data_path}ratings.csv')
            
            logger.info(f"✅ Películas cargadas: {self.movies.height:,}")
            logger.info(f"✅ Calificaciones cargadas: {self.ratings.height:,}")
            logger.info(f"✅ Usuarios únicos: {self.ratings['userId'].n_unique():,}")
            
            return self.movies, self.ratings
            
        except Exception as e:
            logger.error(f"❌ Error cargando datos: {e}")
            raise
    
    def extract_year_from_title(self) -> pl.DataFrame:
        """
        Extrae el año de lanzamiento del título de la película.
        
        Formato típico: "Toy Story (1995)"
        
        Returns:
            DataFrame con columna 'year' añadida
        """
        logger.info("📅 Extrayendo año de lanzamiento...")
        
        if self.movies is None:
            raise ValueError("Primero debe cargar los datos con load_data()")
        
        self.processed_movies = self.movies.with_columns(
            pl.col('title')
            .str.extract(r'\((\d{4})\)$', 1)  # Regex para capturar 4 dígitos entre paréntesis al final
            .alias('year')
            .cast(pl.Int32, strict=False)  # Convertir a entero (null si no hay match)
        )
        
        # Estadísticas
        movies_with_year = self.processed_movies.filter(pl.col('year').is_not_null()).height
        logger.info(f"✅ Año extraído para {movies_with_year:,} películas ({movies_with_year/self.movies.height*100:.1f}%)")
        
        return self.processed_movies
    
    def process_genres(self) -> pl.DataFrame:
        """
        Procesa el campo de géneros, creando una lista y flags one-hot.
        
        Returns:
            DataFrame con géneros procesados
        """
        logger.info("🎭 Procesando géneros...")
        
        if self.processed_movies is None:
            self.processed_movies = self.movies
        
        # Crear lista de géneros
        self.processed_movies = self.processed_movies.with_columns(
            pl.col('genres').str.split('|').alias('genres_list')
        )
        
        # Obtener todos los géneros únicos
        all_genres = set()
        for genres in self.processed_movies['genres_list'].to_list():
            all_genres.update(genres)
        
        # Eliminar '(no genres listed)' si existe
        all_genres.discard('(no genres listed)')
        
        logger.info(f"📌 Géneros encontrados: {', '.join(sorted(all_genres))}")
        
        # Crear columnas one-hot para cada género
        for genre in sorted(all_genres):
            self.processed_movies = self.processed_movies.with_columns(
                pl.col('genres_list')
                .list.contains(genre)
                .alias(f'genre_{genre.lower()}')
            )
        
        return self.processed_movies
    
    def calculate_movie_stats(self) -> pl.DataFrame:
        """
        Calcula estadísticas agregadas por película.
        
        Returns:
            DataFrame con estadísticas (rating_count, avg_rating, etc.)
        """
        logger.info("📊 Calculando estadísticas por película...")
        
        if self.ratings is None:
            raise ValueError("Primero debe cargar los datos con load_data()")
        
        # Usar lazy execution para eficiencia
        lazy_ratings = self.ratings.lazy()
        
        movie_stats = (
            lazy_ratings
            .group_by('movieId')
            .agg([
                pl.count().alias('rating_count'),
                pl.mean('rating').alias('avg_rating'),
                pl.std('rating').alias('std_rating'),
                pl.min('rating').alias('min_rating'),
                pl.max('rating').alias('max_rating'),
                pl.quantile('rating', 0.25).alias('q25_rating'),
                pl.quantile('rating', 0.5).alias('median_rating'),
                pl.quantile('rating', 0.75).alias('q75_rating')
            ])
            .collect()
        )
        
        logger.info(f"✅ Estadísticas calculadas para {movie_stats.height:,} películas")
        
        return movie_stats
    
    def calculate_user_stats(self) -> pl.DataFrame:
        """
        Calcula estadísticas por usuario.
        
        Returns:
            DataFrame con estadísticas de usuarios
        """
        logger.info("👤 Calculando estadísticas por usuario...")
        
        if self.ratings is None:
            raise ValueError("Primero debe cargar los datos con load_data()")
        
        user_stats = (
            self.ratings.lazy()
            .group_by('userId')
            .agg([
                pl.count().alias('user_rating_count'),
                pl.mean('rating').alias('user_avg_rating'),
                pl.std('rating').alias('user_std_rating')
            ])
            .collect()
        )
        
        logger.info(f"✅ Estadísticas calculadas para {user_stats.height:,} usuarios")
        
        return user_stats
    
    def filter_movies_by_popularity(self, min_ratings: int = 20) -> List[int]:
        """
        Filtra películas que tienen al menos min_ratings calificaciones.
        
        Args:
            min_ratings: Número mínimo de calificaciones requerido
            
        Returns:
            Lista de movieIds que cumplen el criterio
        """
        logger.info(f"🎬 Filtrando películas con al menos {min_ratings} calificaciones...")
        
        if self.ratings is None:
            raise ValueError("Primero debe cargar los datos con load_data()")
        
        movie_counts = (
            self.ratings
            .group_by('movieId')
            .agg(pl.count().alias('count'))
            .filter(pl.col('count') >= min_ratings)
        )
        
        popular_movies = movie_counts['movieId'].to_list()
        logger.info(f"✅ {len(popular_movies):,} películas cumplen el criterio")
        
        return popular_movies
    
    def prepare_data_for_model(self, min_ratings: int = 20) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Prepara los datos para el modelo de recomendación.
        
        Args:
            min_ratings: Mínimo de calificaciones para incluir una película
            
        Returns:
            Tuple con (movies_filtered, ratings_filtered)
        """
        logger.info("🔄 Preparando datos para el modelo...")
        
        # Obtener películas populares
        popular_movies = self.filter_movies_by_popularity(min_ratings)
        
        # Filtrar ratings
        self.processed_ratings = self.ratings.filter(
            pl.col('movieId').is_in(popular_movies)
        )
        
        # Filtrar movies
        if self.processed_movies is None:
            self.processed_movies = self.movies
        
        self.processed_movies = self.processed_movies.filter(
            pl.col('movieId').is_in(popular_movies)
        )
        
        logger.info(f"✅ Datos preparados:")
        logger.info(f"   - Películas: {self.processed_movies.height:,}")
        logger.info(f"   - Calificaciones: {self.processed_ratings.height:,}")
        logger.info(f"   - Usuarios: {self.processed_ratings['userId'].n_unique():,}")
        
        return self.processed_movies, self.processed_ratings
    
    def get_movie_title_by_id(self, movie_id: int) -> str:
        """Obtiene el título de una película por su ID."""
        if self.movies is None:
            return "Unknown"
        
        result = self.movies.filter(pl.col('movieId') == movie_id)
        if result.height > 0:
            return result['title'][0]
        return "Unknown"
    
    def get_movie_id_by_title(self, title: str, exact_match: bool = True) -> Optional[int]:
        """
        Obtiene el ID de una película por su título.
        
        Args:
            title: Título a buscar
            exact_match: Si es True, busca coincidencia exacta; si es False, búsqueda parcial
            
        Returns:
            movieId o None si no se encuentra
        """
        if self.movies is None:
            return None
        
        if exact_match:
            result = self.movies.filter(pl.col('title') == title)
        else:
            result = self.movies.filter(pl.col('title').str.contains(title, literal=True))
        
        if result.height > 0:
            return result['movieId'][0]
        return None
    
    def get_top_movies(self, n: int = 10, min_ratings: int = 50) -> pl.DataFrame:
        """
        Obtiene las películas mejor calificadas (con mínimo de ratings).
        
        Args:
            n: Número de películas a retornar
            min_ratings: Mínimo de calificaciones requerido
            
        Returns:
            DataFrame con top películas
        """
        if self.ratings is None or self.movies is None:
            raise ValueError("Primero debe cargar los datos")
        
        movie_stats = (
            self.ratings
            .group_by('movieId')
            .agg([
                pl.count().alias('rating_count'),
                pl.mean('rating').alias('avg_rating')
            ])
            .filter(pl.col('rating_count') >= min_ratings)
            .join(self.movies, on='movieId')
            .sort('avg_rating', descending=True)
            .head(n)
        )
        
        return movie_stats


# Funciones de utilidad para usar fuera de la clase

def create_user_movie_pivot(ratings_df: pl.DataFrame, 
                           movies_df: pl.DataFrame, 
                           n_movies: int = 1000) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Crea una matriz usuario-película para el modelo.
    
    Args:
        ratings_df: DataFrame de calificaciones
        movies_df: DataFrame de películas
        n_movies: Número de películas más populares a incluir
        
    Returns:
        Tuple con (matriz, lista de userIds, lista de movieIds)
    """
    logger.info("📊 Creando matriz usuario-película...")
    
    # Obtener películas más populares
    top_movies = (
        ratings_df
        .group_by('movieId')
        .agg(pl.count().alias('count'))
        .sort('count', descending=True)
        .head(n_movies)
    )['movieId'].to_list()
    
    # Filtrar ratings
    ratings_filtered = ratings_df.filter(pl.col('movieId').is_in(top_movies))
    
    # Obtener usuarios únicos
    users = ratings_filtered['userId'].unique().to_list()
    
    # Crear matriz (versión simple con pandas para scikit-learn)
    import pandas as pd
    ratings_pd = ratings_filtered.to_pandas()
    
    pivot = ratings_pd.pivot_table(
        index='userId',
        columns='movieId',
        values='rating',
        fill_value=0
    )
    
    logger.info(f"✅ Matriz creada: {pivot.shape[0]} usuarios x {pivot.shape[1]} películas")
    
    return pivot.values, list(pivot.index), list(pivot.columns)


if __name__ == "__main__":
    # Ejemplo de uso
    processor = MovieLensDataProcessor()
    
    # Cargar datos
    movies, ratings = processor.load_data()
    
    # Procesar
    processor.extract_year_from_title()
    processor.process_genres()
    
    # Calcular estadísticas
    movie_stats = processor.calculate_movie_stats()
    user_stats = processor.calculate_user_stats()
    
    # Preparar para modelo
    movies_model, ratings_model = processor.prepare_data_for_model(min_ratings=20)
    
    print("\n🎬 Primeras 5 películas procesadas:")
    print(processor.processed_movies.select(['title', 'year', 'genres_list']).head(5))

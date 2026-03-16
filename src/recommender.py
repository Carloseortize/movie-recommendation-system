"""
🤖 Módulo de recomendación para el sistema de películas.

Este módulo implementa un sistema de recomendación basado en filtrado colaborativo
(Item-Based Collaborative Filtering) utilizando similitud coseno.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import polars as pl
from typing import List, Dict, Tuple, Optional, Union
import logging
from src.data_processing import MovieLensDataProcessor

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MovieRecommender:
    """
    Sistema de recomendación de películas basado en filtrado colaborativo.
    
    Este sistema utiliza similitud coseno entre películas para recomendar
    títulos similares basados en el comportamiento de los usuarios.
    """
    
    def __init__(self, data_processor: Optional[MovieLensDataProcessor] = None):
        """
        Inicializa el recomendador.
        
        Args:
            data_processor: Instancia de MovieLensDataProcessor (opcional)
        """
        self.data_processor = data_processor or MovieLensDataProcessor()
        self.user_movie_matrix: Optional[pd.DataFrame] = None
        self.item_similarity_matrix: Optional[pd.DataFrame] = None
        self.movies_df: Optional[pl.DataFrame] = None
        self.ratings_df: Optional[pl.DataFrame] = None
        self.popular_movies: List[int] = []
        self.is_trained = False
        
    def load_data(self) -> None:
        """Carga los datos usando el data processor."""
        logger.info("📥 Cargando datos...")
        self.movies_df, self.ratings_df = self.data_processor.load_data()
        
    def prepare_data(self, min_ratings: int = 20, n_movies: int = 1000) -> None:
        """
        Prepara los datos para el modelo.
        
        Args:
            min_ratings: Mínimo de calificaciones por película
            n_movies: Número máximo de películas a incluir (las más populares)
        """
        logger.info("🔄 Preparando datos para el modelo...")
        
        if self.ratings_df is None:
            self.load_data()
        
        # Obtener películas con suficientes calificaciones
        movie_counts = (
            self.ratings_df
            .group_by('movieId')
            .agg(pl.count().alias('count'))
            .filter(pl.col('count') >= min_ratings)
            .sort('count', descending=True)
        )
        
        # Limitar a n_movies
        self.popular_movies = movie_counts.head(n_movies)['movieId'].to_list()
        
        # Filtrar ratings
        ratings_filtered = self.ratings_df.filter(
            pl.col('movieId').is_in(self.popular_movies)
        )
        
        # Crear matriz usuario-película (convertir a pandas para sklearn)
        ratings_pd = ratings_filtered.to_pandas()
        
        self.user_movie_matrix = ratings_pd.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            fill_value=0
        )
        
        logger.info(f"✅ Matriz creada: {self.user_movie_matrix.shape[0]} usuarios, {self.user_movie_matrix.shape[1]} películas")
        
    def train(self) -> None:
        """
        Entrena el modelo calculando la matriz de similitud entre películas.
        """
        logger.info("🧠 Entrenando modelo de recomendación...")
        
        if self.user_movie_matrix is None:
            raise ValueError("Primero debe preparar los datos con prepare_data()")
        
        # Transponer para tener películas como filas
        item_vectors = self.user_movie_matrix.T.values
        
        # Calcular similitud coseno
        logger.info("📐 Calculando similitud coseno...")
        similarities = cosine_similarity(item_vectors)
        
        # Crear DataFrame con índices de películas
        self.item_similarity_matrix = pd.DataFrame(
            similarities,
            index=self.user_movie_matrix.columns,
            columns=self.user_movie_matrix.columns
        )
        
        self.is_trained = True
        logger.info("✅ Modelo entrenado correctamente")
        
    def get_similar_movies(self, movie_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Obtiene películas similares a una dada.
        
        Args:
            movie_id: ID de la película base
            n_recommendations: Número de recomendaciones a generar
            
        Returns:
            Lista de tuplas (movie_id, similarity_score)
        """
        if not self.is_trained:
            raise ValueError("Primero debe entrenar el modelo con train()")
        
        if movie_id not in self.item_similarity_matrix.columns:
            logger.warning(f"⚠️ Película {movie_id} no encontrada en la matriz")
            return []
        
        # Obtener similitudes
        similarities = self.item_similarity_matrix[movie_id].sort_values(ascending=False)
        
        # Excluir la propia película (primer elemento)
        similar_movies = similarities.iloc[1:n_recommendations+1]
        
        return [(int(mid), score) for mid, score in similar_movies.items()]
    
    def recommend_from_title(self, movie_title: str, n_recommendations: int = 10, 
                            exact_match: bool = False) -> List[Dict[str, Union[str, float]]]:
        """
        Genera recomendaciones a partir del título de una película.
        
        Args:
            movie_title: Título de la película base
            n_recommendations: Número de recomendaciones
            exact_match: Si es True, busca coincidencia exacta
            
        Returns:
            Lista de diccionarios con recomendaciones
        """
        if self.movies_df is None:
            self.load_data()
        
        # Buscar película
        movie_id = self.data_processor.get_movie_id_by_title(movie_title, exact_match)
        
        if movie_id is None:
            # Búsqueda parcial si no se encuentra
            movie_id = self.data_processor.get_movie_id_by_title(movie_title, exact_match=False)
            
        if movie_id is None:
            logger.warning(f"❌ Película '{movie_title}' no encontrada")
            return []
        
        # Obtener título exacto para la respuesta
        movie_info = self.movies_df.filter(pl.col('movieId') == movie_id)
        actual_title = movie_info['title'][0] if movie_info.height > 0 else movie_title
        
        # Verificar si la película está en el modelo
        if movie_id not in self.item_similarity_matrix.columns:
            logger.warning(f"⚠️ Película '{actual_title}' no tiene suficientes datos")
            return []
        
        # Obtener similares
        similar = self.get_similar_movies(movie_id, n_recommendations)
        
        # Formatear resultados
        recommendations = []
        for sim_movie_id, score in similar:
            movie_info = self.movies_df.filter(pl.col('movieId') == sim_movie_id)
            if movie_info.height > 0:
                recommendations.append({
                    'movie_id': sim_movie_id,
                    'title': movie_info['title'][0],
                    'similarity_score': round(score, 4),
                    'year': movie_info['year'][0] if 'year' in movie_info.columns else None
                })
        
        return {
            'input_movie': actual_title,
            'input_movie_id': movie_id,
            'recommendations': recommendations
        }
    
    def recommend_for_user(self, user_id: int, n_recommendations: int = 10,
                          ignore_seen: bool = True) -> List[Dict[str, Union[str, float]]]:
        """
        Genera recomendaciones personalizadas para un usuario específico.
        
        Args:
            user_id: ID del usuario
            n_recommendations: Número de recomendaciones
            ignore_seen: Si es True, excluye películas ya vistas por el usuario
            
        Returns:
            Lista de recomendaciones
        """
        if not self.is_trained:
            raise ValueError("Primero debe entrenar el modelo")
        
        if user_id not in self.user_movie_matrix.index:
            logger.warning(f"⚠️ Usuario {user_id} no encontrado")
            return []
        
        # Obtener ratings del usuario
        user_ratings = self.user_movie_matrix.loc[user_id]
        
        # Películas que ya vio (rating > 0)
        seen_movies = user_ratings[user_ratings > 0].index.tolist()
        
        if not seen_movies:
            logger.warning(f"⚠️ Usuario {user_id} no tiene películas vistas")
            return []
        
        # Generar scores para películas no vistas
        movie_scores = {}
        
        for movie_id in self.item_similarity_matrix.columns:
            if ignore_seen and movie_id in seen_movies:
                continue
                
            # Calcular score basado en similitud con películas vistas
            total_score = 0
            total_sim = 0
            
            for seen_id in seen_movies:
                if seen_id in self.item_similarity_matrix.columns:
                    sim_score = self.item_similarity_matrix.loc[movie_id, seen_id]
                    rating = user_ratings[seen_id]
                    total_score += sim_score * rating
                    total_sim += sim_score
            
            if total_sim > 0:
                movie_scores[movie_id] = total_score / total_sim
        
        # Ordenar y obtener top N
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        # Formatear resultados
        recommendations = []
        for movie_id, score in sorted_movies:
            movie_info = self.movies_df.filter(pl.col('movieId') == movie_id)
            if movie_info.height > 0:
                recommendations.append({
                    'movie_id': movie_id,
                    'title': movie_info['title'][0],
                    'predicted_score': round(score, 4),
                    'year': movie_info['year'][0] if 'year' in movie_info.columns else None
                })
        
        return {
            'user_id': user_id,
            'recommendations': recommendations
        }
    
    def get_model_stats(self) -> Dict[str, Union[int, float]]:
        """
        Obtiene estadísticas del modelo.
        
        Returns:
            Diccionario con métricas del modelo
        """
        if not self.is_trained:
            return {"status": "Modelo no entrenado"}
        
        stats = {
            "n_users": self.user_movie_matrix.shape[0],
            "n_movies": self.user_movie_matrix.shape[1],
            "sparsity": 1 - (self.user_movie_matrix.values != 0).sum() / self.user_movie_matrix.size,
            "coverage": len(self.item_similarity_matrix.columns) / len(self.movies_df) * 100 if self.movies_df is not None else 0
        }
        
        return stats
    
    def evaluate_recommendations(self, test_user_ids: List[int] = None, 
                                 n_test: int = 50) -> Dict[str, float]:
        """
        Evalúa la calidad de las recomendaciones (simulación simple).
        
        Args:
            test_user_ids: Lista de usuarios para prueba
            n_test: Número de usuarios a probar si no se proporciona lista
            
        Returns:
            Métricas de evaluación
        """
        if not self.is_trained:
            raise ValueError("Primero debe entrenar el modelo")
        
        if test_user_ids is None:
            # Seleccionar usuarios aleatorios con suficientes ratings
            user_counts = self.ratings_df.group_by('userId').agg(
                pl.count().alias('count')
            ).filter(pl.col('count') >= 20)
            
            test_user_ids = user_counts.sample(n=min(n_test, user_counts.height))['userId'].to_list()
        
        # Simular evaluación (en un caso real harías train/test split)
        # Aquí simplemente verificamos que el modelo puede generar recomendaciones
        success_count = 0
        
        for user_id in test_user_ids:
            try:
                recs = self.recommend_for_user(user_id, n_recommendations=5)
                if recs and len(recs['recommendations']) > 0:
                    success_count += 1
            except:
                pass
        
        success_rate = success_count / len(test_user_ids) * 100
        
        return {
            "test_users": len(test_user_ids),
            "successful_recommendations": success_count,
            "success_rate": success_rate
        }


# Función de utilidad para crear un recomendador pre-entrenado
def create_pretrained_recommender(min_ratings: int = 20, 
                                 n_movies: int = 1000) -> MovieRecommender:
    """
    Crea y entrena un recomendador con configuración por defecto.
    
    Args:
        min_ratings: Mínimo de calificaciones por película
        n_movies: Número de películas a incluir
        
    Returns:
        MovieRecommender entrenado
    """
    logger.info("🚀 Creando recomendador pre-entrenado...")
    
    recommender = MovieRecommender()
    recommender.load_data()
    recommender.prepare_data(min_ratings=min_ratings, n_movies=n_movies)
    recommender.train()
    
    logger.info("✅ Recomendador listo para usar")
    
    return recommender


if __name__ == "__main__":
    # Ejemplo de uso
    print("🤖 Probando el sistema de recomendación...")
    
    # Crear recomendador
    recommender = create_pretrained_recommender(min_ratings=20, n_movies=500)
    
    # Mostrar estadísticas
    stats = recommender.get_model_stats()
    print(f"\n📊 Estadísticas del modelo:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    # Probar recomendaciones por título
    test_movies = ["Toy Story", "Jurassic Park", "Forrest Gump"]
    
    for movie in test_movies:
        print(f"\n🎬 Recomendaciones para: {movie}")
        print("-" * 50)
        
        result = recommender.recommend_from_title(movie, n_recommendations=5, exact_match=False)
        
        if result:
            for i, rec in enumerate(result['recommendations'], 1):
                year = f"({rec['year']})" if rec['year'] else ""
                print(f"{i}. {rec['title']} {year} - similitud: {rec['similarity_score']:.3f}")
        else:
            print("❌ No se encontraron recomendaciones")
    
    # Evaluar
    eval_results = recommender.evaluate_recommendations(n_test=20)
    print(f"\n📈 Evaluación: {eval_results['success_rate']:.1f}% de usuarios con recomendaciones")


# 🎬 Movie Recommendation System with Analytics

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Polars](https://img.shields.io/badge/Polars-0.20%2B-orange)](https://pola.rs)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io)
[![DuckDB](https://img.shields.io/badge/DuckDB-0.9%2B-yellow)](https://duckdb.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📋 Descripción

Sistema de recomendación de películas construido con técnicas de **filtrado colaborativo** (Item-Based Collaborative Filtering) que permite generar recomendaciones personalizadas basadas en el comportamiento de usuarios. Incluye un **dashboard interactivo** con análisis exploratorio de datos y visualizaciones.

### 🎯 ¿Por qué este proyecto?

- Demuestra habilidades de **Data Analyst**: análisis exploratorio, SQL, visualización
- Implementa **Machine Learning** aplicado: similitud coseno, matrices de recomendación
- Usa **herramientas modernas**: Polars (alternativa rápida a Pandas), DuckDB (SQL embebido)
- Incluye **producto final**: dashboard interactivo con Streamlit


## 🛠️ Tecnologías Utilizadas

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| **Polars** | 0.20+ | Procesamiento de datos eficiente [citation:3] |
| **DuckDB** | 0.9+ | Análisis SQL embebido [citation:4] |
| **Scikit-learn** | 1.3+ | Similitud coseno |
| **Streamlit** | 1.28+ | Dashboard interactivo [citation:2] |
| **Plotly** | 5.18+ | Visualizaciones interactivas [citation:5] |

## 📊 Análisis Realizados

### 1. Exploración de Datos
- Distribución de calificaciones
- Usuarios más activos
- Películas más populares

### 2. Análisis SQL (DuckDB)
```sql
-- Top películas por calificación
SELECT m.title, AVG(r.rating) as avg_rating, COUNT(*) as n_ratings
FROM ratings r
JOIN movies m ON r.movieId = m.movieId
GROUP BY m.movieId, m.title
HAVING COUNT(*) >= 50
ORDER BY avg_rating DESC
LIMIT 10;
```
--

### 3. Modelo de Recomendación
Técnica: Item-Based Collaborative Filtering

Métrica de similitud: Coseno

Cobertura: 78% de las películas

# 🎯 Resultados

Métricas Clave
Métrica	Valor
Total películas	9,742
Total calificaciones	100,836
Usuarios únicos	610
Películas en modelo	~2,500
Rating promedio	3.5 ⭐

# Ejemplo de Recomendaciones
Para Toy Story (1995):

Toy Story 2 (1999) - similitud: 0.92

A Bug's Life (1998) - similitud: 0.87

Monsters, Inc. (2001) - similitud: 0.85

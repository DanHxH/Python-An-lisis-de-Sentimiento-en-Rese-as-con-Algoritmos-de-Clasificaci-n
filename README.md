# Análisis de Sentimiento en Reseñas con Algoritmos de Clasificación

Este proyecto realiza un análisis de sentimiento en reseñas utilizando diversas técnicas de procesamiento de texto y algoritmos de clasificación. El objetivo es etiquetar las reseñas como positivas o negativas, entrenar modelos de clasificación y evaluar su rendimiento.

## Requisitos

- Python 3.x
- Pandas
- NLTK
- Scikit-learn
- Matplotlib
- WordCloud

## Descripción del Proyecto

El proyecto se desarrolla en varios pasos:

1. **Preprocesamiento de Reseñas**: Se procesa el texto de las reseñas para limpiarlo y estandarizarlo. Esto incluye la eliminación de signos de puntuación y números, conversión a minúsculas, eliminación de palabras vacías y aplicación de stemming.
2. **Etiquetado de Reseñas**: Las reseñas se etiquetan como positivas, negativas o neutrales según la calificación de estrellas.
3. **Generación de Vocabulario**: Se crea un vocabulario a partir de las reseñas preprocesadas y se guardan los términos más comunes.
4. **Análisis y Visualización**: Se genera una gráfica de pastel para visualizar la distribución de opiniones y nubes de palabras para los términos más comunes en las reseñas positivas y negativas.
5. **Entrenamiento y Evaluación de Modelos**: Se entrenan y evalúan tres modelos de clasificación (SVM, C4.5, Regresión Logística) usando una división 80-20 de los datos.
6. **Resultados y Métricas**: Se muestran las métricas de evaluación (matriz de confusión, precisión, recuerdo, medida F y exactitud) para cada modelo.

## Archivos Principales

- `preprocessing.py`: Código para el preprocesamiento de las reseñas.
- `labeling.py`: Código para etiquetar las reseñas.
- `vocabulary.py`: Código para generar el vocabulario y guardar los términos más comunes.
- `visualization.py`: Código para generar gráficas y nubes de palabras.
- `classification.py`: Código para entrenar y evaluar los modelos de clasificación.
- `evaluate.py`: Código para mostrar las métricas de evaluación y determinar el mejor clasificador.

## Resultados

Los resultados incluyen:

- Gráfica de pastel mostrando la distribución de opiniones.
- Nubes de palabras para las reseñas positivas y negativas.
- Métricas de evaluación para cada modelo de clasificación (SVM, C4.5, Regresión Logística).
- Archivo CSV con la exactitud de cada modelo.
- Determinación del mejor clasificador basado en la exactitud.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue para discutir cualquier cambio o mejora.
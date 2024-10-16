# Aprendizaje_autom-tico_para_textos
Customer Segmentation and Sentiment Analysis Project

//English

Data Science Bootcamp / Project applying machine learning to insurance for customer segmentation, benefit prediction, and data masking.

Customer segmentation: Develop a machine learning model to identify similar customers and improve marketing strategies (scikit-learn, pandas).
Benefit prediction: Implement predictive models to assess the likelihood of new customers receiving benefits and the expected amount of those benefits (scikit-learn, numpy).
This project involved developing a system to filter and categorize movie review sentiment using a machine learning model.

The objective was to train a model to automatically detect negative reviews from a dataset of IMDB movie reviews with sentiment labels (positive/negative).
The model needed to achieve an F1 score of at least 0.85.
Here's a breakdown of the steps involved:

Data Acquisition and Preprocessing:

Downloaded a dataset of IMDB movie reviews with sentiment labels.
Cleaned the text data by removing special characters, converting to lowercase, and removing stop words.
Lemmatized the text to reduce words to their base form.
Used batch processing for efficiency when dealing with large datasets.
Vectorized the text data using TF-IDF to represent the reviews numerically.
Explored the distribution of sentiment labels and movie release years.
Model Training and Evaluation:

Defined and trained several machine learning models, including Logistic Regression and LightGBM.
Implemented functions to evaluate model performance using metrics like F1 score, accuracy, precision, recall, ROC AUC, and PR AUC.
Compared the performance of different models and hyperparameter configurations.
Advanced Techniques (Optional):

Explored the use of n-grams (considering sequences of words) for text vectorization.
Investigated the application of pre-trained language models like BERT for sentiment analysis.


// Español
Bootcamp de Ciencia de Datos / Proyecto de aplicación de aprendizaje automático en seguros para segmentación de clientes, predicción de beneficios y enmascaramiento de datos.

Segmentación de clientes: Desarrollar un modelo de aprendizaje automático para identificar clientes similares y mejorar las estrategias de marketing (scikit-learn, pandas).
Predicción de beneficios: Implementar modelos predictivos para evaluar la probabilidad de que nuevos clientes reciban beneficios y el monto esperado de esos beneficios (scikit-learn, numpy).
Este proyecto consistió en desarrollar un sistema para filtrar y clasificar el sentimiento de las reseñas de películas mediante un modelo de aprendizaje automático.

El objetivo era entrenar un modelo para detectar automáticamente las reseñas negativas a partir de un conjunto de datos de reseñas de películas de IMDB con etiquetas de sentimiento (positivo/negativo).
El modelo debía alcanzar un puntaje F1 de al menos 0.85.
A continuación se detalla el proceso:

Adquisición y preprocesamiento de datos:

Se descargó un conjunto de datos de reseñas de películas de IMDB con etiquetas de sentimiento.
Se limpiaron los datos de texto eliminando caracteres especiales, convirtiéndolos a minúsculas y eliminando las palabras vacías (stop words).
Se lematizaron los textos para reducir las palabras a su forma base.
Se utilizó el procesamiento por lotes para la eficiencia al trabajar con conjuntos de datos grandes.
Se vectorizó el texto con TF-IDF para representar las reseñas numéricamente.
Se exploró la distribución de las etiquetas de sentimiento y los años de estreno de las películas.
Entrenamiento y evaluación del modelo:

Se definieron y entrenaron varios modelos de aprendizaje automático, incluyendo Regresión Logística y LightGBM.
Se implementaron funciones para evaluar el rendimiento del modelo utilizando métricas como el puntaje F1, la precisión, la recuperación, el AUC de ROC y el AUC de PR.
Se comparó el rendimiento de diferentes modelos y configuraciones de hiperparámetros.
Técnicas avanzadas (Opcional):

Se exploró el uso de n-gramas (considerando secuencias de palabras) para la vectorización de texto.
Se investigó la aplicación de modelos de lenguaje preentrenados como BERT para el análisis de sentimiento.

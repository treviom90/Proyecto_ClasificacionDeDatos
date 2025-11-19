# Proyecto_ClasificacionDeDatos
 
Este proyecto busca demostrar cómo, a partir únicamente de la imagen de un póster de película, es posible predecir automáticamente tanto su score aproximado tipo IMDb como los géneros cinematográficos a los que pertenece. Todo esto se construye mediante un modelo de visión computacional entrenado con FastAI y un pipeline completamente automatizado en Airflow.

# Explicación del modelo y metodología

Este proyecto combina visión computacional, aprendizaje profundo y orquestación con Airflow para construir dos modelos basados en FastAI + PyTorch usando únicamente imágenes de posters de películas: un modelo para predecir el “score” (calificación) y otro modelo para clasificar automáticamente el género (single-label o multi-label). El propósito principal es evaluar si la información visual contenida en un poster es suficiente para: (1) estimar la calidad percibida de una película (score) y (2) predecir sus géneros cinematográficos. Ambos modelos funcionan exclusivamente a partir de la imagen, sin usar texto adicional, demostrando cómo los atributos gráficos pueden servir para enriquecer metadatos, construir sistemas de recomendación y realizar análisis visuales avanzados.

El objetivo del proyecto es construir un pipeline completo que, dado un CSV con metadatos, imágenes locales y URLs de posters, pueda descargar imágenes, normalizarlas, entrenar modelos profundos, evaluar desempeño, generar predicciones masivas y automatizar todo mediante Airflow. FastAI se utilizó como marco principal para el entrenamiento de los modelos debido a su facilidad para crear DataBlocks, realizar data augmentation, aplicar normalización estándar, entrenar modelos con transfer learning y exportarlos de forma sencilla para producción. En ambos modelos se emplea ResNet50 preentrenada en ImageNet, lo que permite aprender patrones visuales de forma rápida y eficiente incluso en CPU dentro del entorno de Airflow.

FastAI aporta ventajas clave: 
(1) los DataBlocks generan automáticamente la canalización de procesamiento (“imagen → transformaciones → entrenamiento → inferencia”), 
(2) el transfer learning facilita el entrenamiento con pocos datos y 
(3) el método fit_one_cycle optimiza la tasa de aprendizaje para mejorar la convergencia. Esto hace posible entrenar modelos sólidos con muy poco código.

# Modelo 1: Predicción del SCORE

La etiqueta de score no proviene del CSV, sino que se extrae del nombre del archivo del poster mediante una expresión regular. Por ejemplo, un archivo como `Inception_8.8.jpg` contiene el score `8.8`. Dado que estos valores son discretos (5.1, 6.2, 7.8, etc.) y no continuos, se modela el score como un problema de **clasificación multiclase**, no como regresión. Esto mejora la estabilidad del entrenamiento.

FastAI construye los DataLoaders usando un DataBlock con: `ImageBlock` como entrada, `CategoryBlock` como etiqueta, un `RandomSplitter` de 80/20 para entrenamiento/validación, transformaciones de data augmentation y normalización con estadísticas de ImageNet. Las imágenes se redimensionan a 300×180.

El entrenamiento del modelo se realiza con ResNet50 preentrenada, usando `CrossEntropyLossFlat` como función de pérdida y `accuracy` como métrica. El entrenamiento utiliza `fit_one_cycle(epochs=5)` para ajustar la tasa de aprendizaje. Primero se entrena la cabeza del modelo y luego se realiza fine-tuning sobre todas las capas. Un callback personalizado `SaveEveryEpoch` guarda checkpoints por epoch. El modelo final se exporta como `model_classification_score_final.pkl`.

Este modelo aprende patrones visuales asociados a distintos rangos de score, como estilos de diseño, iluminación, presencia de actores, composición gráfica e incluso estética general del poster.

# Modelo 2: Clasificación de GÉNERO (Multi-Label)

Las etiquetas de género provienen del CSV en una columna con formato como `Action|Adventure|Sci-Fi`. El pipeline detecta automáticamente si existe más de un género y convierte el problema en una **clasificación multilabel**, donde una imagen puede pertenecer a múltiples categorías simultáneamente.

FastAI construye el DataLoader usando `ImageBlock` como entrada y `MultiCategoryBlock` para las etiquetas. Se realiza un split 80/20, data augmentation y normalización. Se filtran imágenes que no tengan etiqueta correspondiente en el CSV, y se genera un vocabulario global de géneros.

En el entrenamiento, la arquitectura utilizada también es ResNet50. Para clasificación multilabel se emplea `BCEWithLogitsLossFlat` como función de pérdida y `accuracy_multi(thresh=0.5)` como métrica, que evalúa cuántas etiquetas por imagen se predicen correctamente. El entrenamiento usa `fit_one_cycle(epochs=5)`. El modelo aprende rasgos visuales como: tonos oscuros para drama, explosiones para acción, colores brillantes para comedia, iluminación futurista para sci-fi, etc. El modelo final se exporta como `model_genre_final.pkl`.

# Validación, entrenamiento y ausencia de folds

El proyecto utiliza una sola validación holdout del 20% con semilla fija (42). Esto garantiza reproducibilidad y simplicidad en el pipeline. No se implementó k-fold cross-validation debido a los tiempos de entrenamiento dentro de Airflow, pero el sistema puede extenderse fácilmente para soportarlo si se requiere más robustez estadística en la evaluación. Ambos modelos usan transfer learning, lo que acelera el entrenamiento y reduce el riesgo de overfitting en datasets moderados.

# Predicción y archivos generados

Tras el entrenamiento, ambos modelos producen archivos CSV con predicciones. Para el modelo de score, el resultado es:

file, label
poster1.jpg, 7.8
poster2.jpg, 5.4

Para el modelo de género, el resultado es:
file, predicted_labels, raw_probs
poster1.jpg, Action|Adventure, {"Action":0.91,"Adventure":0.88,...}


Ambos archivos se combinan en uno final llamado `predicciones_finales.csv`, que contiene score y géneros predichos para cada imagen. También se genera un archivo especial `resumen_tests_score_genero.csv` para imágenes de prueba llamadas test1..test10, útil para validación rápida.

# Conclusión técnica

Este proyecto demuestra cómo FastAI permite entrenar fácilmente modelos de visión profunda para resolver tanto problemas de clasificación multiclase (score) como multilabel (género). La integración con Airflow permite convertir este flujo en un pipeline automatizado, reproducible, escalable y listo para producción. El resultado es un sistema completo capaz de transformar datos visuales en metadatos estructurados útiles para análisis, recomendación y enriquecimiento de información cinematográfica.


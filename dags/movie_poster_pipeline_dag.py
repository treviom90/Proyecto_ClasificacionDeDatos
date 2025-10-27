from airflow import DAG
from airflow.decorators import dag, task
from datetime import datetime, timedelta
from utils.xcoms_utils.xcoms import push, pull
from utils.movie_poster.movie_poster_pipeline import (
    read_movies_genre, pred_regresion, pred_clasificacion,
    leaner, carga_imagenes
)

# --- Argumentos por defecto ---
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# --- Definición del DAG usando decorador ---
@dag(
    dag_id='prediccion_score_peliculas',
    description='Dprediccion',
    default_args=default_args,
    schedule= None, #'0 9 * * *',  # todos los días a las 9 AM
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['Procesamiento', 'Imagenes', 'Prediction']
)
def prediccion_peliculas():

    path = '/opt/airflow/ClasificacionDeDatos'

    @task(task_id = "lectura_df")
    def leer_csv(path, ti):
        path_lecture = path + '/MovieGenre.csv'
        df = read_movies_genre(path_lecture)
        push(ti,"df",df)

    @task(task_id = "modelo_regresion")
    def modelos_reg_pred(path, ti):
        path_img = path + '/poster_downloads'
        dls = pred_regresion(path_img)
        push(ti,"dsl", dls)

    @task(task_id = "modelo_clasificacion")
    def modelos_class_pred(path, ti):
        path_img = path + '/poster_downloads'
        dls_class = pred_clasificacion(path_img)
        push(ti,"dls_class", dls_class)

    @task(task_id = "entrenamiento")
    def aprendizaje(path, ti):
        dls = pull(ti, "modelo_regresion",dls)
        dls_class = pull(ti, "modelo_clasificacion",dls_class)
        learn_reg, learn_class = leaner(dls,dls_class)
        push(ti,"learn_reg", learn_reg)
        push(ti,"learn_class", learn_class)

    @task(task_id = "prediccion")
    def prediccion_peliculas(path, ti):
        learn_reg = pull(ti, "entrenamiento",learn_reg)
        learn_class = pull(ti, "entrenamiento",learn_class)
        results = carga_imagenes(path,learn_reg,learn_class)

    #Tasks
    lectura_df = leer_csv(path)
    modelo_regresion = modelos_reg_pred(path)
    modelo_clasificacion = modelos_class_pred(path)
    aprendizaje_entretamiento = aprendizaje(path)
    resultados_pred = prediccion_peliculas(path)

    #Dependencias
    lectura_df >> [modelo_regresion, modelo_clasificacion] >> aprendizaje_entretamiento >> resultados_pred

dag = prediccion_peliculas()
from airflow.decorators import dag, task
from datetime import datetime, timedelta
from utils.xcoms_utils.xcoms import push, pull
from utils.movie_poster.movie_poster_pipeline import (
    read_movies_genre,         # lectura del CSV
    learner_clases_train,      # ENTRENAMIENTO -> guarda .pkl y regresa ruta
    carga_imagenes_a_csv       # PREDICCIÓN -> escribe CSV
)

default_args = {
    'owner': 'airflow',
    'retries': 0, # 0 mientras depuras para evitar "reinicios"
    'retry_delay': timedelta(minutes=5),
}

@dag(
    dag_id='prediccion_score_peliculas',
    description='Predicción de score de películas por clasificación (FastAI) sin gráficas, salida CSV',
    default_args=default_args,
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['Procesamiento', 'Imagenes', 'Prediction']
)
def prediccion_peliculas():

    base_path = '/opt/airflow/ClasificacionDeDatos'
    images_subdir = 'poster_downloads'     # carpeta con imágenes para train e inferencia
    models_subdir = 'models'               # carpeta para guardado de .pkl
    outputs_subdir = 'out'                 # carpeta de CSVs

    @task(task_id="lectura_df")
    def leer_csv(path: str, ti=None):
        path_lecture = f"{path}/MovieGenre.csv"
        df = read_movies_genre(path_lecture)
        print(f"Columnas CSV: {df.columns.tolist()}")
        # Puedes empujar alguna metainfo ligera si la ocupas
        push(ti, "n_rows", len(df))
        push(ti, "csv_path", path_lecture)

    @task(task_id="modelo_clasificacion_cfg")
    def modelos_class_pred_cfg(path: str, ti=None):
        """
        En vez de empujar DataLoaders (objeto pesado no serializable),
        empujamos rutas y config mínima.
        """
        path_img = f"{path}/{images_subdir}"
        model_dir = f"{path}/{models_subdir}"
        out_dir = f"{path}/{outputs_subdir}"
        cfg = {
            "path_img": path_img,
            "model_dir": model_dir,
            "out_dir": out_dir,
            "epochs": 5,
            "seed": 42
        }
        push(ti, "cfg_class", cfg)

    @task(task_id="entrenamiento_clasificacion")
    def entrenamiento_clas(ti=None):
        """
        Toma config de XCom, entrena y devuelve la RUTA al .pkl final.
        """
        cfg = pull(ti, task_id="modelo_clasificacion_cfg", key="cfg_class")
        model_pkl_path = learner_clases_train(
            path_img=cfg["path_img"],
            out_dir=cfg["model_dir"],
            epochs=cfg["epochs"],
            lr=None,
            seed=cfg["seed"]
        )
        push(ti, "model_pkl_path", model_pkl_path)

    @task(task_id="prediccion")
    def hacer_prediccion(path: str, ti=None):
        """
        Carga el .pkl final y genera un CSV con predicciones.
        """
        model_pkl_path = pull(ti, task_id="entrenamiento_clasificacion", key="model_pkl_path")
        images_dir = f"{path}/{images_subdir}"   # puedes separar train/test si lo requieres
        out_csv = f"{path}/{outputs_subdir}/predicciones_score_peliculas.csv"

        csv_path = carga_imagenes_a_csv(
            images_dir=images_dir,
            model_pkl_path=model_pkl_path,
            out_csv=out_csv,
            seed=42,
            force_cpu=True
        )
        print(f"CSV de predicciones: {csv_path}")

    # Wiring
    lectura_df = leer_csv(base_path)
    modelo_clas_cfg = modelos_class_pred_cfg(base_path)
    training_cls = entrenamiento_clas()
    resultados_pred = hacer_prediccion(base_path)

    # Dependencias
    lectura_df >> modelo_clas_cfg >> training_cls >> resultados_pred

dag = prediccion_peliculas()

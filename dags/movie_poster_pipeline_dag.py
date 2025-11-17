from airflow.decorators import dag, task
from datetime import datetime, timedelta
import os
import pandas as pd
import requests
import re
from urllib.parse import urlsplit, urlunsplit
from airflow.exceptions import AirflowSkipException
from utils.xcoms_utils.xcoms import push, pull

from utils.movie_poster.movie_poster_pipeline import (
    read_movies_genre,
    learner_clases_train,   # SCORE
    carga_imagenes_a_csv,   # PRED SCORE -> CSV
    learner_genre_train,    # GÉNERO
    predict_genre_to_csv,   # PRED GÉNERO -> CSV
    _canon_basename_from_url_or_path, # <-- para normalizar nombres de descarga
    resumen_tests_score_genre          # <-- función nueva que ya tienes en el módulo
)

default_args = {
    "owner": "airflow",
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

@dag(
    dag_id="prediccion_score_y_genero",
    description="Entrena y predice Score (local) y Género (descargando URLs del CSV) con límites y robustez de descarga.",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["Procesamiento", "Imagenes", "Prediction", "Score", "Genero"]
)
def prediccion_score_y_genero():

    # === Paths base ===
    base_path = "/opt/airflow/ClasificacionDeDatos"
    images_subdir = "poster_downloads"     # Score usa esta carpeta
    models_subdir = "models"
    outputs_subdir = "out"

    # === Columnas CSV ===
    image_col = "Poster"   # aquí vienen las URLs
    label_col = "Genre"
    label_delim = "|"

    # ========= 1. Verificar insumos =========
    @task(task_id="verificar_insumos", execution_timeout=timedelta(minutes=10))
    def verificar_insumos(path: str, ti=None):
        csv_path = f"{path}/MovieGenre.csv"
        images_dir_score = f"{path}/{images_subdir}"

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No existe el CSV esperado: {csv_path}")

        df = read_movies_genre(csv_path)
        print(f"Columnas CSV: {df.columns.tolist()}")
        if image_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"El CSV debe contener columnas '{image_col}' y '{label_col}'.")

        if not os.path.isdir(images_dir_score):
            raise FileNotFoundError(f"No existe la carpeta de imágenes para Score: {images_dir_score}")

        push(ti, "csv_path", csv_path)
        push(ti, "images_dir_score", images_dir_score)
        push(ti, "n_rows_csv", len(df))

    # ========= 2. Preparar cfg (incluye límites) =========
    @task(task_id="preparar_cfg")
    def preparar_cfg(path: str, ti=None):
        model_dir = f"{path}/{models_subdir}"
        out_dir = f"{path}/{outputs_subdir}"
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        # Puedes sobreescribir vía Airflow Variables
        try:
            from airflow.models import Variable
            max_downloads = int(Variable.get("max_downloads", default_var=2000))
            min_success   = int(Variable.get("min_success",   default_var=300))
            sample_mode   = Variable.get("sample_mode",       default_var="per_label")  # "first" | "random" | "per_label"
            per_label_k   = int(Variable.get("per_label_k",   default_var=50))
        except Exception:
            max_downloads, min_success, sample_mode, per_label_k = 2000, 300, "per_label", 50

        cfg = {
            "model_dir": model_dir,
            "out_dir": out_dir,
            "epochs": 5,
            "seed": 42,
            "label_delim": label_delim,
            "image_col": image_col,
            "label_col": label_col,
            # límites descarga/entrenamiento de GÉNERO
            "max_downloads": max_downloads,
            "min_success":   min_success,
            "sample_mode":   sample_mode,
            "per_label_k":   per_label_k,
        }
        push(ti, "cfg", cfg)

    # ========= 3. Descargar imágenes del CSV (con límites y variantes) =========
    @task(task_id="descargar_imagenes_csv", execution_timeout=timedelta(hours=1))
    def descargar_imagenes_csv(path: str, ti=None):
        csv_path = pull(ti, task_id="verificar_insumos", key="csv_path")
        cfg = pull(ti, task_id="preparar_cfg", key="cfg")
        image_col, label_col = cfg["image_col"], cfg["label_col"]
        max_downloads = int(cfg["max_downloads"])
        min_success   = int(cfg["min_success"])
        sample_mode   = cfg["sample_mode"]
        per_label_k   = int(cfg["per_label_k"])

        df = read_movies_genre(csv_path)
        df = df[df[image_col].astype(str).str.startswith("http")].copy()

        # --- Muestreo: per_label | random | first ---
        if sample_mode == "per_label" and label_col in df.columns:
            parts = []
            for g, dfg in df.groupby(label_col):
                parts.append(dfg.sample(n=min(per_label_k, len(dfg)), random_state=42))
            df = pd.concat(parts, ignore_index=True)
        elif sample_mode == "random":
            df = df.sample(n=min(max_downloads, len(df)), random_state=42)
        else:  # "first"
            df = df.head(max_downloads)

        if len(df) > max_downloads:
            df = df.sample(n=max_downloads, random_state=42)

        dest_dir = f"{path}/poster_from_csv"
        os.makedirs(dest_dir, exist_ok=True)

        # Sesión con headers tipo navegador
        sess = requests.Session()
        sess.headers.update({
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0.0.0 Safari/537.36"),
            "Referer": "https://www.imdb.com/",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        })

        def candidate_urls(original_url: str) -> list[str]:
            parts = list(urlsplit(original_url))
            parts[3] = ""; parts[4] = ""  # sin query ni fragment
            u = urlunsplit(parts)
            u2 = u.replace("images-na.ssl-images-amazon.com", "m.media-amazon.com")
            m = re.search(r"(.*?_V1_).*(\.(?:jpg|jpeg|png))$", u2, flags=re.IGNORECASE)
            cands = []
            if m:
                head, ext = m.group(1), m.group(2)
                cands.extend([
                    f"{head}{ext}",      # _V1_.jpg
                    f"{head}SX300{ext}", # _V1_SX300.jpg
                    f"{head}SY400{ext}", # _V1_SY400.jpg
                    f"{head}UX600{ext}", # _V1_UX600.jpg
                    u2                   # original sin query
                ])
            else:
                cands.extend([u2, u])
            # quitar duplicados manteniendo orden
            seen, uniq = set(), []
            for cu in cands:
                if cu not in seen:
                    uniq.append(cu); seen.add(cu)
            return uniq

        def norm_name_from_url(url: str) -> str:
            # usa la misma lógica que el módulo (FIX clave)
            b = _canon_basename_from_url_or_path(url)
            if not (b.endswith(".jpg") or b.endswith(".jpeg") or b.endswith(".png")):
                b += ".jpg"
            return b

        created, failed, seen_names = 0, 0, set()

        for _, row in df.iterrows():
            url = str(row[image_col])
            fname = norm_name_from_url(url)
            if fname in seen_names:
                continue
            seen_names.add(fname)

            dst = os.path.join(dest_dir, fname)
            if os.path.exists(dst):
                created += 1
            else:
                ok = False
                for cand in candidate_urls(url):
                    try:
                        resp = sess.get(cand, timeout=25, allow_redirects=True, stream=True)
                        if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith("image/"):
                            with open(dst, "wb") as f:
                                for chunk in resp.iter_content(chunk_size=8192):
                                    if chunk:
                                        f.write(chunk)
                            ok = True
                            created += 1
                            break
                    except requests.RequestException:
                        continue
                if not ok:
                    failed += 1

            # Early stop al alcanzar el tope
            if created >= max_downloads:
                break

        print(f"[DOWNLOAD] OK: {created} | FAIL: {failed} | destino: {dest_dir} | limite: {max_downloads}")
        if created < min_success:
            # No entrenar género si no llegamos al mínimo
            raise AirflowSkipException(f"Solo {created} imágenes útiles (< {min_success}); se omite GÉNERO.")
        push(ti, "images_dir_genero", dest_dir)

    # ========= 4. Entrenar modelos =========
    @task(task_id="entrenar_score_cls", execution_timeout=timedelta(hours=4))
    def entrenar_score_cls(path: str, ti=None):
        cfg = pull(ti, task_id="preparar_cfg", key="cfg")
        images_dir = pull(ti, task_id="verificar_insumos", key="images_dir_score")
        pkl_path = learner_clases_train(
            path_img=images_dir,
            out_dir=cfg["model_dir"],
            epochs=cfg["epochs"],
            lr=None,
            seed=cfg["seed"]
        )
        push(ti, "model_score_pkl", pkl_path)

    @task(task_id="entrenar_genero", execution_timeout=timedelta(hours=4))
    def entrenar_genero(path: str, ti=None):
        # --- Config & paths ---
        cfg = pull(ti, task_id="preparar_cfg", key="cfg")
        csv_path = pull(ti, task_id="verificar_insumos", key="csv_path")
        images_dir_csv = pull(ti, task_id="descargar_imagenes_csv", key="images_dir_genero")

        # --- Airflow Variables (2.x/3.x) ---
        try:
            from airflow.models import Variable as _Var   # Airflow 2.x
        except Exception:
            from airflow.sdk import Variable as _Var      # Airflow 3.x

        min_label_freq = int(_Var.get("MIN_LABEL_FREQ", default_var=3))
        poster_max_pixels = int(_Var.get("POSTER_MAX_PIXELS", default_var=80000000))

        # Propaga a ENV para que las tome el módulo (independiente de la firma)
        os.environ["MIN_LABEL_FREQ"] = str(min_label_freq)
        os.environ["POSTER_MAX_PIXELS"] = str(poster_max_pixels)

        # --- Diagnóstico de cobertura antes de entrenar ---
        from utils.movie_poster.movie_poster_pipeline import validate_label_coverage
        stats = validate_label_coverage(
            images_dir=images_dir_csv,
            csv_path=csv_path,
            image_col=cfg["image_col"],
            label_col=cfg["label_col"],
            label_delim=cfg["label_delim"],
        )
        print(f"[COVERAGE pre-train] {stats}")

        if stats["with_label"] == 0:
            raise AirflowSkipException(
                "Cobertura 0 (ninguna imagen tiene etiqueta del CSV). "
                "Borra 'poster_from_csv' y vuelve a descargar; revisa normalización."
            )

        # --- Entrenamiento género con fallback de firma ---
        from utils.movie_poster.movie_poster_pipeline import learner_genre_train

        try:
            # Intento preferido (módulo actualizado)
            pkl_path = learner_genre_train(
                images_dir=images_dir_csv,
                csv_path=csv_path,
                out_dir=cfg["model_dir"],
                image_col=cfg["image_col"],
                label_col=cfg["label_col"],
                label_delim=cfg["label_delim"],
                epochs=cfg["epochs"],
                lr=None,
                seed=cfg["seed"],
                min_label_freq=min_label_freq,  # <-- si la firma lo acepta, perfecto
            )
        except TypeError as e:
            # Firma antigua: vuelve a intentar sin el argumento extra
            if "unexpected keyword argument 'min_label_freq'" in str(e):
                print("[WARN] learner_genre_train no acepta 'min_label_freq'; reintentando sin el parámetro.")
                pkl_path = learner_genre_train(
                    images_dir=images_dir_csv,
                    csv_path=csv_path,
                    out_dir=cfg["model_dir"],
                    image_col=cfg["image_col"],
                    label_col=cfg["label_col"],
                    label_delim=cfg["label_delim"],
                    epochs=cfg["epochs"],
                    lr=None,
                    seed=cfg["seed"],
                )
            else:
                raise
        except KeyError as e:
            # Caso típico: etiqueta quedó fuera del vocab en train.
            raise AirflowSkipException(
                f"Etiqueta fuera del vocabulario de entrenamiento: {e}. "
                "Aumenta MIN_LABEL_FREQ o reintenta (el split reintenta varias semillas en el módulo actualizado)."
            )

        push(ti, "model_genre_pkl", pkl_path)

    # ========= 5. Predicciones =========
    @task(task_id="predecir_score", execution_timeout=timedelta(hours=1))
    def predecir_score(path: str, ti=None):
        cfg = pull(ti, task_id="preparar_cfg", key="cfg")
        images_dir = pull(ti, task_id="verificar_insumos", key="images_dir_score")
        model_pkl = pull(ti, task_id="entrenar_score_cls", key="model_score_pkl")

        out_csv = f"{path}/{outputs_subdir}/predicciones_score.csv"
        csv_path = carga_imagenes_a_csv(
            images_dir=images_dir,
            model_pkl_path=model_pkl,
            out_csv=out_csv,
            seed=cfg["seed"],
            force_cpu=True
        )
        push(ti, "csv_score", csv_path)

    @task(task_id="predecir_genero", execution_timeout=timedelta(hours=1))
    def predecir_genero(path: str, ti=None):
        cfg = pull(ti, task_id="preparar_cfg", key="cfg")
        images_dir_csv = pull(ti, task_id="descargar_imagenes_csv", key="images_dir_genero")
        model_pkl = pull(ti, task_id="entrenar_genero", key="model_genre_pkl")

        out_csv = f"{path}/{outputs_subdir}/predicciones_genero.csv"
        csv_path = predict_genre_to_csv(
            images_dir=images_dir_csv,
            model_pkl_path=model_pkl,
            out_csv=out_csv,
            seed=cfg["seed"],
            force_cpu=True,
            thresh=0.5
        )
        push(ti, "csv_genero", csv_path)

    # ========= 6. Fusionar resultados =========
    @task(task_id="fusionar_salidas")
    def fusionar_salidas(path: str, ti=None):
        score_csv = pull(ti, task_id="predecir_score", key="csv_score")
        genero_csv = pull(ti, task_id="predecir_genero", key="csv_genero")

        df_score = pd.read_csv(score_csv)
        df_genero = pd.read_csv(genero_csv)

        df_score["file_key"] = df_score["file"].apply(lambda x: os.path.basename(str(x)))
        df_genero["file_key"] = df_genero["file"].apply(lambda x: os.path.basename(str(x)))

        df = df_score.merge(df_genero, on="file_key", how="outer", suffixes=("_score", "_genre"))
        out_csv = f"{path}/{outputs_subdir}/predicciones_finales.csv"
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"CSV final: {out_csv}")

    # ========= 7. Resumen TEST1..TEST10 (score + género) =========
    @task(task_id="resumen_tests_score_genero", execution_timeout=timedelta(minutes=30))
    def resumen_tests_score_genero_task(path: str, ti=None):
        cfg = pull(ti, task_id="preparar_cfg", key="cfg")
        # Las imágenes test1..test10 están directamente en `path`
        images_dir = path + "/prediccion1"
        model_score_pkl = pull(ti, task_id="entrenar_score_cls", key="model_score_pkl")
        model_genre_pkl = pull(ti, task_id="entrenar_genero", key="model_genre_pkl")

        out_csv = f"{path}/{outputs_subdir}/resumen_tests_score_genero.csv"
        csv_path = resumen_tests_score_genre(
            images_dir=images_dir,
            score_model_pkl=model_score_pkl,
            genre_model_pkl=model_genre_pkl,
            out_csv=out_csv,
            seed=cfg["seed"],
            force_cpu=True,
            thresh=0.5
        )
        print(f"Resumen TEST1..TEST10 (score + género): {csv_path}")

    # ========= Wiring =========
    insumos = verificar_insumos(base_path)
    cfg = preparar_cfg(base_path)
    descarga = descargar_imagenes_csv(base_path)
    train_score = entrenar_score_cls(base_path)
    train_genero = entrenar_genero(base_path)
    pred_score = predecir_score(base_path)
    pred_genero = predecir_genero(base_path)
    fusion = fusionar_salidas(base_path)
    resumen_tests = resumen_tests_score_genero_task(base_path)

    # Dependencias
    insumos >> cfg >> descarga
    [insumos, cfg] >> train_score >> pred_score 
    [train_score, pred_score, descarga] >> train_genero >> pred_genero
    [pred_score, pred_genero] >> fusion

    # El resumen requiere que YA estén entrenados ambos modelos
    [train_score, train_genero] >> resumen_tests

dag = prediccion_score_y_genero()


# from airflow.decorators import dag, task
# from datetime import datetime, timedelta
# import os
# import pandas as pd
# import requests
# import re
# from urllib.parse import urlsplit, urlunsplit
# from airflow.exceptions import AirflowSkipException
# from utils.xcoms_utils.xcoms import push, pull

# from utils.movie_poster.movie_poster_pipeline import (
#     read_movies_genre,
#     learner_clases_train,   # SCORE
#     carga_imagenes_a_csv,   # PRED SCORE -> CSV
#     learner_genre_train,    # GÉNERO
#     predict_genre_to_csv,   # PRED GÉNERO -> CSV
#     _canon_basename_from_url_or_path, # <-- para normalizar nombres de descarga
#     resumen_tests_score_genre
# )

# default_args = {
#     "owner": "airflow",
#     "retries": 0,
#     "retry_delay": timedelta(minutes=5),
# }

# @dag(
#     dag_id="prediccion_score_y_genero",
#     description="Entrena y predice Score (local) y Género (descargando URLs del CSV) con límites y robustez de descarga.",
#     default_args=default_args,
#     schedule=None,
#     start_date=datetime(2025, 1, 1),
#     catchup=False,
#     max_active_runs=1,
#     tags=["Procesamiento", "Imagenes", "Prediction", "Score", "Genero"]
# )
# def prediccion_score_y_genero():

#     # === Paths base ===
#     base_path = "/opt/airflow/ClasificacionDeDatos"
#     images_subdir = "poster_downloads"     # Score usa esta carpeta
#     models_subdir = "models"
#     outputs_subdir = "out"

#     # === Columnas CSV ===
#     image_col = "Poster"   # aquí vienen las URLs
#     label_col = "Genre"
#     label_delim = "|"

#     # ========= 1. Verificar insumos =========
#     @task(task_id="verificar_insumos", execution_timeout=timedelta(minutes=10))
#     def verificar_insumos(path: str, ti=None):
#         csv_path = f"{path}/MovieGenre.csv"
#         images_dir_score = f"{path}/{images_subdir}"

#         if not os.path.exists(csv_path):
#             raise FileNotFoundError(f"No existe el CSV esperado: {csv_path}")

#         df = read_movies_genre(csv_path)
#         print(f"Columnas CSV: {df.columns.tolist()}")
#         if image_col not in df.columns or label_col not in df.columns:
#             raise ValueError(f"El CSV debe contener columnas '{image_col}' y '{label_col}'.")

#         if not os.path.isdir(images_dir_score):
#             raise FileNotFoundError(f"No existe la carpeta de imágenes para Score: {images_dir_score}")

#         push(ti, "csv_path", csv_path)
#         push(ti, "images_dir_score", images_dir_score)
#         push(ti, "n_rows_csv", len(df))

#     # ========= 2. Preparar cfg (incluye límites) =========
#     @task(task_id="preparar_cfg")
#     def preparar_cfg(path: str, ti=None):
#         model_dir = f"{path}/{models_subdir}"
#         out_dir = f"{path}/{outputs_subdir}"
#         os.makedirs(model_dir, exist_ok=True)
#         os.makedirs(out_dir, exist_ok=True)

#         # Puedes sobreescribir vía Airflow Variables
#         try:
#             from airflow.models import Variable
#             max_downloads = int(Variable.get("max_downloads", default_var=2000))
#             min_success   = int(Variable.get("min_success",   default_var=300))
#             sample_mode   = Variable.get("sample_mode",       default_var="per_label")  # "first" | "random" | "per_label"
#             per_label_k   = int(Variable.get("per_label_k",   default_var=50))
#         except Exception:
#             max_downloads, min_success, sample_mode, per_label_k = 2000, 300, "per_label", 50

#         cfg = {
#             "model_dir": model_dir,
#             "out_dir": out_dir,
#             "epochs": 5,
#             "seed": 42,
#             "label_delim": label_delim,
#             "image_col": image_col,
#             "label_col": label_col,
#             # límites descarga/entrenamiento de GÉNERO
#             "max_downloads": max_downloads,
#             "min_success":   min_success,
#             "sample_mode":   sample_mode,
#             "per_label_k":   per_label_k,
#         }
#         push(ti, "cfg", cfg)

#     # ========= 3. Descargar imágenes del CSV (con límites y variantes) =========
#     @task(task_id="descargar_imagenes_csv", execution_timeout=timedelta(hours=1))
#     def descargar_imagenes_csv(path: str, ti=None):
#         csv_path = pull(ti, task_id="verificar_insumos", key="csv_path")
#         cfg = pull(ti, task_id="preparar_cfg", key="cfg")
#         image_col, label_col = cfg["image_col"], cfg["label_col"]
#         max_downloads = int(cfg["max_downloads"])
#         min_success   = int(cfg["min_success"])
#         sample_mode   = cfg["sample_mode"]
#         per_label_k   = int(cfg["per_label_k"])

#         df = read_movies_genre(csv_path)
#         df = df[df[image_col].astype(str).str.startswith("http")].copy()

#         # --- Muestreo: per_label | random | first ---
#         if sample_mode == "per_label" and label_col in df.columns:
#             parts = []
#             for g, dfg in df.groupby(label_col):
#                 parts.append(dfg.sample(n=min(per_label_k, len(dfg)), random_state=42))
#             df = pd.concat(parts, ignore_index=True)
#         elif sample_mode == "random":
#             df = df.sample(n=min(max_downloads, len(df)), random_state=42)
#         else:  # "first"
#             df = df.head(max_downloads)

#         if len(df) > max_downloads:
#             df = df.sample(n=max_downloads, random_state=42)

#         dest_dir = f"{path}/poster_from_csv"
#         os.makedirs(dest_dir, exist_ok=True)

#         # Sesión con headers tipo navegador
#         sess = requests.Session()
#         sess.headers.update({
#             "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#                            "AppleWebKit/537.36 (KHTML, like Gecko) "
#                            "Chrome/120.0.0.0 Safari/537.36"),
#             "Referer": "https://www.imdb.com/",
#             "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
#             "Accept-Language": "en-US,en;q=0.9",
#             "Connection": "keep-alive",
#         })

#         def candidate_urls(original_url: str) -> list[str]:
#             parts = list(urlsplit(original_url))
#             parts[3] = ""; parts[4] = ""  # sin query ni fragment
#             u = urlunsplit(parts)
#             u2 = u.replace("images-na.ssl-images-amazon.com", "m.media-amazon.com")
#             m = re.search(r"(.*?_V1_).*(\.(?:jpg|jpeg|png))$", u2, flags=re.IGNORECASE)
#             cands = []
#             if m:
#                 head, ext = m.group(1), m.group(2)
#                 cands.extend([
#                     f"{head}{ext}",      # _V1_.jpg
#                     f"{head}SX300{ext}", # _V1_SX300.jpg
#                     f"{head}SY400{ext}", # _V1_SY400.jpg
#                     f"{head}UX600{ext}", # _V1_UX600.jpg
#                     u2                   # original sin query
#                 ])
#             else:
#                 cands.extend([u2, u])
#             # quitar duplicados manteniendo orden
#             seen, uniq = set(), []
#             for cu in cands:
#                 if cu not in seen:
#                     uniq.append(cu); seen.add(cu)
#             return uniq

#         def norm_name_from_url(url: str) -> str:
#             # usa la misma lógica que el módulo (FIX clave)
#             b = _canon_basename_from_url_or_path(url)
#             if not (b.endswith(".jpg") or b.endswith(".jpeg") or b.endswith(".png")):
#                 b += ".jpg"
#             return b

#         created, failed, seen_names = 0, 0, set()

#         for _, row in df.iterrows():
#             url = str(row[image_col])
#             fname = norm_name_from_url(url)
#             if fname in seen_names:
#                 continue
#             seen_names.add(fname)

#             dst = os.path.join(dest_dir, fname)
#             if os.path.exists(dst):
#                 created += 1
#             else:
#                 ok = False
#                 for cand in candidate_urls(url):
#                     try:
#                         resp = sess.get(cand, timeout=25, allow_redirects=True, stream=True)
#                         if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith("image/"):
#                             with open(dst, "wb") as f:
#                                 for chunk in resp.iter_content(chunk_size=8192):
#                                     if chunk:
#                                         f.write(chunk)
#                             ok = True
#                             created += 1
#                             break
#                     except requests.RequestException:
#                         continue
#                 if not ok:
#                     failed += 1

#             # Early stop al alcanzar el tope
#             if created >= max_downloads:
#                 break

#         print(f"[DOWNLOAD] OK: {created} | FAIL: {failed} | destino: {dest_dir} | limite: {max_downloads}")
#         if created < min_success:
#             # No entrenar género si no llegamos al mínimo
#             raise AirflowSkipException(f"Solo {created} imágenes útiles (< {min_success}); se omite GÉNERO.")
#         push(ti, "images_dir_genero", dest_dir)

#     # ========= 4. Entrenar modelos =========
#     @task(task_id="entrenar_score_cls", execution_timeout=timedelta(hours=4))
#     def entrenar_score_cls(path: str, ti=None):
#         cfg = pull(ti, task_id="preparar_cfg", key="cfg")
#         images_dir = pull(ti, task_id="verificar_insumos", key="images_dir_score")
#         pkl_path = learner_clases_train(
#             path_img=images_dir,
#             out_dir=cfg["model_dir"],
#             epochs=cfg["epochs"],
#             lr=None,
#             seed=cfg["seed"]
#         )
#         push(ti, "model_score_pkl", pkl_path)

#     @task(task_id="entrenar_genero", execution_timeout=timedelta(hours=4))
#     def entrenar_genero(path: str, ti=None):
#         # --- Config & paths ---
#         cfg = pull(ti, task_id="preparar_cfg", key="cfg")
#         csv_path = pull(ti, task_id="verificar_insumos", key="csv_path")
#         images_dir_csv = pull(ti, task_id="descargar_imagenes_csv", key="images_dir_genero")

#         # --- Airflow Variables (2.x/3.x) ---
#         try:
#             from airflow.models import Variable as _Var   # Airflow 2.x
#         except Exception:
#             from airflow.sdk import Variable as _Var      # Airflow 3.x

#         min_label_freq = int(_Var.get("MIN_LABEL_FREQ", default_var=3))
#         poster_max_pixels = int(_Var.get("POSTER_MAX_PIXELS", default_var=80000000))

#         # Propaga a ENV para que las tome el módulo (independiente de la firma)
#         os.environ["MIN_LABEL_FREQ"] = str(min_label_freq)
#         os.environ["POSTER_MAX_PIXELS"] = str(poster_max_pixels)

#         # --- Diagnóstico de cobertura antes de entrenar ---
#         from utils.movie_poster.movie_poster_pipeline import validate_label_coverage
#         stats = validate_label_coverage(
#             images_dir=images_dir_csv,
#             csv_path=csv_path,
#             image_col=cfg["image_col"],
#             label_col=cfg["label_col"],
#             label_delim=cfg["label_delim"],
#         )
#         print(f"[COVERAGE pre-train] {stats}")

#         if stats["with_label"] == 0:
#             raise AirflowSkipException(
#                 "Cobertura 0 (ninguna imagen tiene etiqueta del CSV). "
#                 "Borra 'poster_from_csv' y vuelve a descargar; revisa normalización."
#             )

#         # --- Entrenamiento género con fallback de firma ---
#         from utils.movie_poster.movie_poster_pipeline import learner_genre_train

#         try:
#             # Intento preferido (módulo actualizado)
#             pkl_path = learner_genre_train(
#                 images_dir=images_dir_csv,
#                 csv_path=csv_path,
#                 out_dir=cfg["model_dir"],
#                 image_col=cfg["image_col"],
#                 label_col=cfg["label_col"],
#                 label_delim=cfg["label_delim"],
#                 epochs=cfg["epochs"],
#                 lr=None,
#                 seed=cfg["seed"],
#                 min_label_freq=min_label_freq,  # <-- si la firma lo acepta, perfecto
#             )
#         except TypeError as e:
#             # Firma antigua: vuelve a intentar sin el argumento extra
#             if "unexpected keyword argument 'min_label_freq'" in str(e):
#                 print("[WARN] learner_genre_train no acepta 'min_label_freq'; reintentando sin el parámetro.")
#                 pkl_path = learner_genre_train(
#                     images_dir=images_dir_csv,
#                     csv_path=csv_path,
#                     out_dir=cfg["model_dir"],
#                     image_col=cfg["image_col"],
#                     label_col=cfg["label_col"],
#                     label_delim=cfg["label_delim"],
#                     epochs=cfg["epochs"],
#                     lr=None,
#                     seed=cfg["seed"],
#                 )
#             else:
#                 raise
#         except KeyError as e:
#             # Caso típico: etiqueta quedó fuera del vocab en train.
#             raise AirflowSkipException(
#                 f"Etiqueta fuera del vocabulario de entrenamiento: {e}. "
#                 "Aumenta MIN_LABEL_FREQ o reintenta (el split reintenta varias semillas en el módulo actualizado)."
#             )

#         push(ti, "model_genre_pkl", pkl_path)

#     # ========= 5. Predicciones =========
#     @task(task_id="predecir_score", execution_timeout=timedelta(hours=1))
#     def predecir_score(path: str, ti=None):
#         cfg = pull(ti, task_id="preparar_cfg", key="cfg")
#         images_dir = pull(ti, task_id="verificar_insumos", key="images_dir_score")
#         model_pkl = pull(ti, task_id="entrenar_score_cls", key="model_score_pkl")

#         out_csv = f"{path}/{outputs_subdir}/predicciones_score.csv"
#         csv_path = carga_imagenes_a_csv(
#             images_dir=images_dir,
#             model_pkl_path=model_pkl,
#             out_csv=out_csv,
#             seed=cfg["seed"],
#             force_cpu=True
#         )
#         push(ti, "csv_score", csv_path)

#     @task(task_id="predecir_genero", execution_timeout=timedelta(hours=1))
#     def predecir_genero(path: str, ti=None):
#         cfg = pull(ti, task_id="preparar_cfg", key="cfg")
#         images_dir_csv = pull(ti, task_id="descargar_imagenes_csv", key="images_dir_genero")
#         model_pkl = pull(ti, task_id="entrenar_genero", key="model_genre_pkl")

#         out_csv = f"{path}/{outputs_subdir}/predicciones_genero.csv"
#         csv_path = predict_genre_to_csv(
#             images_dir=images_dir_csv,
#             model_pkl_path=model_pkl,
#             out_csv=out_csv,
#             seed=cfg["seed"],
#             force_cpu=True,
#             thresh=0.5
#         )
#         push(ti, "csv_genero", csv_path)

#     # ========= 6. Fusionar resultados =========
#     @task(task_id="fusionar_salidas")
#     def fusionar_salidas(path: str, ti=None):
#         score_csv = pull(ti, task_id="predecir_score", key="csv_score")
#         genero_csv = pull(ti, task_id="predecir_genero", key="csv_genero")

#         df_score = pd.read_csv(score_csv)
#         df_genero = pd.read_csv(genero_csv)

#         df_score["file_key"] = df_score["file"].apply(lambda x: os.path.basename(str(x)))
#         df_genero["file_key"] = df_genero["file"].apply(lambda x: os.path.basename(str(x)))

#         df = df_score.merge(df_genero, on="file_key", how="outer", suffixes=("_score", "_genre"))
#         out_csv = f"{path}/{outputs_subdir}/predicciones_finales.csv"
#         os.makedirs(os.path.dirname(out_csv), exist_ok=True)
#         df.to_csv(out_csv, index=False, encoding="utf-8")
#         print(f"CSV final: {out_csv}")

#     # ========= Wiring =========
#     insumos = verificar_insumos(base_path)
#     cfg = preparar_cfg(base_path)
#     descarga = descargar_imagenes_csv(base_path)
#     train_score = entrenar_score_cls(base_path)
#     train_genero = entrenar_genero(base_path)
#     pred_score = predecir_score(base_path)
#     pred_genero = predecir_genero(base_path)
#     fusion = fusionar_salidas(base_path)

#     # Dependencias
#     insumos >> cfg >> descarga
#     [insumos, cfg] >> train_score >> pred_score 
#     [train_score, pred_score, descarga]>> train_genero >> pred_genero
#     [pred_score, pred_genero] >> fusion

# dag = prediccion_score_y_genero()

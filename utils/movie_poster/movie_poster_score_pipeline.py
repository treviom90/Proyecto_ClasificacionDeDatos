# utils/movie_poster/movie_poster_pipeline.py
from fastai.vision.all import *
from pathlib import Path
import pandas as pd
import re
import csv
import os
import torch
from typing import Union, List, Optional

# =========================
# Runtime seguro para Airflow
# =========================
def _configure_runtime(seed: int = 42, deterministic: bool = True):
    set_seed(seed, reproducible=deterministic)
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
    except Exception:
        pass

# =========================
# Lectura de CSV (si lo necesitas en otra parte del pipeline)
# =========================
def read_movies_genre(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        delimiter=";",
        encoding="latin1",
        on_bad_lines="skip",
        engine="python"
    )
    df.dropna(inplace=True)
    return df

# =========================
# Etiquetas desde nombre archivo
# =========================
_float_pat = re.compile(r'(\d+(?:\.\d+)?)')

def get_score_labels(file_name: Union[str, Path]) -> str:
    s = str(file_name)
    m = _float_pat.search(s)
    if not m:
        raise ValueError(f"No se encontró etiqueta numérica en: {s}")
    return m.group(1)

# =========================
# DataLoaders (solo dentro de las funciones; no viajan por XCom)
# =========================
def _dls_classification(path_img: Union[str, Path], img_size=(300,180), bs: int = 16) -> DataLoaders:
    data_class = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y=get_score_labels,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        item_tfms=Resize(img_size),
        batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)]
    )
    return data_class.dataloaders(
        path_img, bs=bs, num_workers=0, shuffle=True, persistent_workers=False
    )

# =========================
# Callback: save checkpoints por epoch
# =========================
class SaveEveryEpoch(Callback):
    def __init__(self, dirpath: Union[str, Path], fname_prefix: str = "cls"):
        self.dir = Path(dirpath)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prefix = fname_prefix
    def after_epoch(self):
        ep = self.epoch
        try:
            self.learn.save(self.dir/f"{self.prefix}_{ep}")
        except Exception as e:
            print(f"[WARN] No se pudo guardar .pth en epoch {ep}: {e}")
        try:
            self.learn.export(self.dir/f"{self.prefix}_{ep}.pkl")
        except Exception as e:
            print(f"[WARN] No se pudo exportar .pkl en epoch {ep}: {e}")

# =========================
# ENTRENAMIENTO CLASIFICACIÓN
# =========================
def learner_clases_train(
    path_img: Union[str, Path],
    out_dir: Union[str, Path],
    epochs: int = 5,
    lr: Optional[float] = None,
    seed: int = 42
) -> str:
    """
    Entrena clasificación (resnet50) y exporta:
      - checkpoints por epoch
      - modelo final: out_dir / 'model_classification_score_final.pkl'
    Retorna la ruta del .pkl final.
    """
    _configure_runtime(seed=seed)
    dls_class = _dls_classification(path_img)

    learn_class = cnn_learner(
        dls=dls_class,
        arch=resnet50,
        loss_func=CrossEntropyLossFlat(),
        metrics=accuracy,
        cbs=[CSVLogger(), SaveEveryEpoch(out_dir, "cls")]
    )

    print("Inicio Entrenamiento clasificacion")
    learn_class.fit_one_cycle(epochs, lr_max=lr)
    print("Fin Entrenamiento clasificacion")

    # --- NUEVO: limpiar callbacks y handles antes de exportar ---
    # 1) Cerrar posibles file handles (CSVLogger, etc.)
    for cb in list(learn_class.cbs):
        # algunos loggers tienen atributos tipo "file", "f", "fh", "writer"
        for attr in ("file", "f", "fh", "writer"):
            fh = getattr(cb, attr, None)
            try:
                # CSVLogger.writer puede ser csv.writer (no tiene close)
                if hasattr(fh, "close"):
                    fh.close()
            except Exception:
                pass

    # 2) Quitar callbacks problemáticos del Learner antes de exportar
    #    (CSVLogger y el SaveEveryEpoch personalizado)
    to_remove = []
    for cb in list(learn_class.cbs):
        if isinstance(cb, (CSVLogger, SaveEveryEpoch)):
            to_remove.append(cb)
    for cb in to_remove:
        try:
            learn_class.remove_cb(cb)
        except Exception:
            pass

    # 3) Exportar ya "limpio"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_pkl = out_dir / "model_classification_score_final.pkl"
    learn_class.export(final_pkl)

    return str(final_pkl)

# =========================
# PREDICCIÓN A CSV (SIN GRÁFICAS)
# =========================
def _write_csv(rows: List[dict], out_csv: Union[str, Path]) -> str:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = set()
    for r in rows: keys.update(r.keys())
    keys = sorted(keys)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows: w.writerow(r)
    return str(out_csv)

def carga_imagenes_a_csv(
    images_dir: Union[str, Path],
    model_pkl_path: Union[str, Path],
    out_csv: Union[str, Path],
    seed: int = 42,
    force_cpu: bool = True
) -> str:
    """
    Carga el .pkl y predice todas las imágenes de images_dir.
    Guarda CSV con columnas: file, label, prob (y error si aplica).
    """
    _configure_runtime(seed=seed)
    learn = load_learner(model_pkl_path, cpu=force_cpu)

    files = list(get_image_files(images_dir))
    if not files:
        raise RuntimeError(f"No hay imágenes en {images_dir}")

    rows = []
    for p in files:
        try:
            pred_class, pred_idx, probs = learn.predict(p)
            #prob = float(probs[pred_idx]) if probs is not None else None
            rows.append({"file": str(p), "label": str(pred_class)}) #, "prob": prob
        except Exception as e:
            rows.append({"file": str(p), "label": None}) #, "prob": None, "error": str(e)

    return _write_csv(rows, out_csv)

from fastai.vision.all import *
from pathlib import Path
import pandas as pd
import torch
import os
import re
import csv
import json
import warnings
import unicodedata
from urllib.parse import urlsplit
from typing import Union, List, Optional
from functools import partial

# =============================================
# CONFIGURACIÓN SEGURA PARA AIRFLOW
# =============================================
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

# =============================================
# UTILIDADES GENERALES
# =============================================
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

# patrón regex para etiquetas de score (IMDB) embebidas en filename
_float_pat = re.compile(r'(\d+(?:\.\d+)?)')

def get_score_labels(file_name: Union[str, Path]) -> str:
    s = str(file_name)
    m = _float_pat.search(s)
    if not m:
        raise ValueError(f"No se encontró etiqueta numérica en: {s}")
    return m.group(1)

# =============================================
# NORMALIZACIÓN UNIFICADA (CSV y archivos locales)
# =============================================
def _canon_basename_from_url_or_path(s: Union[str, Path]) -> str:
    """
    Toma URL o path, quita query/fragment, obtiene basename con extensión.
    Reemplaza '@' y ',' por '_', normaliza unicode, lowercase y sin espacios.
    """
    s = str(s)
    try:
        parts = urlsplit(s)
        if parts.scheme and parts.netloc:
            s = os.path.basename(parts.path)
        else:
            s = os.path.basename(s)
    except Exception:
        s = os.path.basename(s)

    s = s.replace('@', '_').replace(',', '_')
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    s = s.strip().lower().replace(' ', '')
    return s  # mantiene extensión si la tenía

_v1_tail = re.compile(r'(.*?)(_v1_.*)$', flags=re.IGNORECASE)

def _canon_key_variants(basename_with_ext: str) -> list[str]:
    """
    Variantes canónicas para maximizar el match:
      - stem sin extensión
      - basename completo (con extensión)
      - versión sin sufijo `_V1_...` (con y sin extensión)
    """
    base = basename_with_ext
    stem, ext = os.path.splitext(base)
    variants = {stem, base}

    m = _v1_tail.match(stem)
    if m:
        core = m.group(1)
        variants.update({core, f"{core}{ext}"})

    return list(variants)

def _norm_key(s: Union[str, Path]) -> str:
    """
    Llave principal: stem canónico (sin extensión), usando la normalización unificada.
    """
    b = _canon_basename_from_url_or_path(s)
    return os.path.splitext(b)[0]

# =============================================
# DATALOADERS SCORE (clasificación por filename)
# =============================================
def _dls_classification(path_img: Union[str, Path], img_size=(300,180), bs: int = 16) -> DataLoaders:
    data_class = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y=get_score_labels,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        item_tfms=Resize(img_size),
        batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)]
    )
    return data_class.dataloaders(path_img, bs=bs, num_workers=0, shuffle=True, persistent_workers=False)

# =============================================
# CALLBACK GUARDADO POR EPOCH
# =============================================
class SaveEveryEpoch(Callback):
    def __init__(self, dirpath: Union[str, Path], fname_prefix: str = "cls"):
        self.dir = Path(dirpath)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prefix = fname_prefix
    def after_epoch(self):
        ep = self.epoch
        try: self.learn.save(self.dir/f"{self.prefix}_{ep}")
        except Exception as e: print(f"[WARN] No se pudo guardar .pth: {e}")
        try: self.learn.export(self.dir/f"{self.prefix}_{ep}.pkl")
        except Exception as e: print(f"[WARN] No se pudo exportar .pkl: {e}")

# =============================================
# ENTRENAMIENTO SCORE CLASIFICACIÓN
# =============================================
def learner_clases_train(
    path_img: Union[str, Path],
    out_dir: Union[str, Path],
    epochs: int = 5,
    lr: Optional[float] = None,
    seed: int = 42
) -> str:
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

    # limpiar callbacks/handles antes de exportar (evita PicklingError)
    for cb in list(learn_class.cbs):
        for attr in ("file","f","fh","writer"):
            fh = getattr(cb, attr, None)
            try:
                if hasattr(fh, "close"): fh.close()
            except: pass
    for cb in list(learn_class.cbs):
        if isinstance(cb, (CSVLogger, SaveEveryEpoch)):
            try: learn_class.remove_cb(cb)
            except: pass

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "model_classification_final.pkl"
    learn_class.export(p)
    return str(p)

# =============================================
# PREDICCIÓN SCORE → CSV
# =============================================
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
    _configure_runtime(seed=seed)
    learn = load_learner(model_pkl_path, cpu=force_cpu)
    files = list(get_image_files(images_dir))
    if not files:
        raise RuntimeError(f"No hay imágenes en {images_dir}")

    rows = []
    for p in files:
        try:
            pred_class, pred_idx, probs = learn.predict(p)
            prob = float(probs[pred_idx]) if probs is not None else None
            rows.append({"file": str(p), "label": str(pred_class), "prob": prob})
        except Exception as e:
            rows.append({"file": str(p), "label": None, "prob": None, "error": str(e)})

    return _write_csv(rows, out_csv)

# =============================================
# FUNCIONES PARA GÉNERO (single o multi-label)
# =============================================
def read_genre_label_map(
    csv_path: Union[str, Path],
    image_col: str="Poster",
    label_col: str="Genre",
    label_delim: Optional[str]="|"
) -> tuple[dict,bool]:
    """
    Devuelve (label_map, multilabel) con llaves robustas:
    - Llave principal: stem canónico (sin extensión)
    - Se agregan variantes que eliminan el sufijo `_V1_...` de IMDb
    """
    df = pd.read_csv(csv_path, delimiter=";", encoding="latin1", on_bad_lines="skip", engine="python")
    if image_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV debe contener columnas '{image_col}' y '{label_col}'")

    label_map = {}
    multilabel = label_delim is not None

    for _, row in df.iterrows():
        raw_img = str(row[image_col])

        base = _canon_basename_from_url_or_path(raw_img)
        variants = _canon_key_variants(base)

        if multilabel:
            labels = [g.strip() for g in str(row[label_col]).split(label_delim) if g and g.strip()]
        else:
            labels = str(row[label_col]).strip()

        for v in variants:
            k = os.path.splitext(v)[0]  # stem
            label_map[k] = labels

    return label_map, multilabel

def make_label_getter(label_map: dict):
    def _get_y(p: Union[str, Path]):
        key = _norm_key(p)
        if key not in label_map:
            raise KeyError(f"No hay etiqueta de género para '{os.path.basename(str(p))}' (clave normalizada='{key}')")
        return label_map[key]
    return _get_y

def dls_genre(
    images_dir: Union[str, Path],
    label_map: dict,
    multilabel: bool,
    img_size=(300,180),
    bs: int = 16,
    strict: bool = False
) -> DataLoaders:
    """
    Filtra imágenes sin etiqueta y falla pronto si el keep queda vacío.
    """
    get_y = make_label_getter(label_map)
    block = MultiCategoryBlock if multilabel else CategoryBlock

    all_imgs = list(get_image_files(images_dir))
    keep, drop = [], []

    for p in all_imgs:
        k = _norm_key(p)
        (keep if k in label_map else drop).append(p)

    if not keep:
        ejemplos = [os.path.basename(str(p)) for p in drop[:10]]
        raise RuntimeError(
            "No se encontró NINGUNA imagen con etiqueta en el CSV. "
            f"Dir: {images_dir}. Ejemplos que no machean: {ejemplos}."
        )

    if drop:
        warnings.warn(f"[WARN] {len(drop)} imágenes sin etiqueta en CSV (p.ej: {os.path.basename(str(drop[0]))}).")

    data = DataBlock(
        blocks=(ImageBlock, block),
        get_items=lambda _: keep,
        get_y=get_y,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        item_tfms=Resize(img_size),
        batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)]
    )
    return data.dataloaders(images_dir, bs=bs, num_workers=0, shuffle=True, persistent_workers=False)

# =============================================
# ENTRENAMIENTO GÉNERO
# =============================================
def learner_genre_train(
    images_dir: Union[str, Path],
    csv_path: Union[str, Path],
    out_dir: Union[str, Path],
    image_col: str="Poster",
    label_col: str="Genre",
    label_delim: Optional[str]="|",
    epochs: int=5,
    lr: Optional[float]=None,
    seed: int=42
) -> str:
    _configure_runtime(seed=seed)
    label_map, multilabel = read_genre_label_map(csv_path, image_col, label_col, label_delim)
    dls = dls_genre(images_dir, label_map, multilabel)

    if multilabel:
        loss = BCEWithLogitsLossFlat()
        metrics = [partial(accuracy_multi, thresh=0.5)]
    else:
        loss = CrossEntropyLossFlat()
        metrics = [accuracy]

    learn = cnn_learner(
        dls=dls,
        arch=resnet50,
        loss_func=loss,
        metrics=metrics,
        cbs=[CSVLogger(), SaveEveryEpoch(out_dir, "genre")]
    )

    print("Inicio Entrenamiento género")
    learn.fit_one_cycle(epochs, lr_max=lr)
    print("Fin Entrenamiento género")

    # limpiar callbacks/handles antes de exportar
    for cb in list(learn.cbs):
        for attr in ("file","f","fh","writer"):
            fh = getattr(cb, attr, None)
            try:
                if hasattr(fh, "close"): fh.close()
            except: pass
    for cb in list(learn.cbs):
        if isinstance(cb, (CSVLogger, SaveEveryEpoch)):
            try: learn.remove_cb(cb)
            except: pass

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "model_genre_final.pkl"
    learn.export(p)
    return str(p)

# =============================================
# PREDICCIÓN GÉNERO → CSV
# =============================================
def predict_genre_to_csv(
    images_dir: Union[str, Path],
    model_pkl_path: Union[str, Path],
    out_csv: Union[str, Path],
    seed: int=42,
    force_cpu: bool=True,
    thresh: float=0.5
) -> str:
    _configure_runtime(seed=seed)
    learn = load_learner(model_pkl_path, cpu=force_cpu)
    files = list(get_image_files(images_dir))
    if not files:
        raise RuntimeError(f"No hay imágenes en {images_dir}")

    try:
        classes = list(learn.dls.vocab)
    except Exception:
        classes = [str(c) for c in getattr(learn.dls, 'vocab', [])]
    is_multi = not hasattr(learn.dls.vocab, 'o2i')

    rows = []
    for p in files:
        try:
            pred = learn.predict(p)
            if is_multi:
                probs = (torch.sigmoid(pred[2]).detach().cpu().numpy().tolist()
                         if isinstance(pred[2], Tensor) else list(map(float, pred[2])))
                chosen = [classes[i] for i, pr in enumerate(probs) if pr >= thresh]
                rows.append({
                    "file": str(p),
                    "predicted_labels": "|".join(chosen) if chosen else "",
                    "raw_probs": json.dumps({cls: float(pr) for cls, pr in zip(classes, probs)})
                })
            else:
                pred_class, pred_idx, probs = pred
                prob_vec = (probs.detach().cpu().numpy().tolist()
                            if isinstance(probs, Tensor) else list(map(float, probs)))
                rows.append({
                    "file": str(p),
                    "predicted_labels": str(pred_class),
                    "raw_probs": json.dumps({cls: float(pr) for cls, pr in zip(classes, prob_vec)})
                })
        except Exception as e:
            rows.append({"file": str(p), "predicted_labels": "", "raw_probs": "", "error": str(e)})

    return _write_csv(rows, out_csv)

# =============================================
# (Opcional) VALIDACIÓN DE COBERTURA
# =============================================
def validate_label_coverage(
    images_dir: Union[str, Path],
    csv_path: Union[str, Path],
    image_col: str="Poster",
    label_col: str="Genre",
    label_delim: Optional[str]="|"
) -> dict:
    label_map, _ = read_genre_label_map(csv_path, image_col, label_col, label_delim)
    imgs = list(get_image_files(images_dir))
    imgs_norm = {_norm_key(p) for p in imgs}
    keys_csv = set(label_map.keys())
    missing = [os.path.basename(str(p)) for p in imgs if _norm_key(p) not in keys_csv]
    return {
        "num_images": len(imgs),
        "with_label": len(imgs_norm & keys_csv),
        "without_label": len(missing),
        "some_missing_examples": missing[:5]
    }

# =============================================
# RESUMEN: TEST1..TEST10 → SCORE + GÉNERO
# =============================================
def resumen_tests_score_genre(
    images_dir: Union[str, Path],
    score_model_pkl: Union[str, Path],
    genre_model_pkl: Union[str, Path],
    out_csv: Union[str, Path],
    seed: int = 42,
    force_cpu: bool = True,
    thresh: float = 0.5
) -> str:
    """
    Busca imágenes llamadas test1..test10 (cualquier extensión) en images_dir
    y genera un CSV con:
      - file          : ruta completa del archivo
      - name          : nombre base (test1, test2, ...)
      - score_label   : score predicho por el modelo de score
      - genre_labels  : géneros predichos por el modelo de género
    No agrega columnas de error.
    """
    _configure_runtime(seed=seed)

    # Carga de modelos
    learn_score = load_learner(score_model_pkl, cpu=force_cpu)
    learn_genre = load_learner(genre_model_pkl, cpu=force_cpu)

    images_dir = Path(images_dir)
    wanted_names = {f"test{i}" for i in range(1, 11)}

    # Filtrar solo test1..test10 por "stem" (nombre sin extensión)
    all_imgs = get_image_files(images_dir)
    files = [p for p in all_imgs if p.stem in wanted_names]

    if not files:
        raise RuntimeError(
            f"No se encontraron imágenes test1..test10 en {images_dir}. "
            "Asegúrate de que los archivos se llamen, por ejemplo, test1.jpg, test2.png, etc."
        )

    # Info del modelo de género
    try:
        classes = list(learn_genre.dls.vocab)
    except Exception:
        classes = [str(c) for c in getattr(learn_genre.dls, 'vocab', [])]
    is_multi = not hasattr(learn_genre.dls.vocab, 'o2i')

    rows = []
    for p in sorted(files, key=lambda x: x.stem):
        row = {
            "file": str(p),
            "name": p.stem,
            "score_label": "",
            "genre_labels": ""
        }

        # --- Predicción de SCORE ---
        try:
            pred_class_s, pred_idx_s, probs_s = learn_score.predict(p)
            row["score_label"] = str(pred_class_s)
        except Exception:
            # Si falla, dejamos score_label vacío
            pass

        # --- Predicción de GÉNERO ---
        try:
            pred_g = learn_genre.predict(p)
            if is_multi:
                probs_tensor = pred_g[2]
                if isinstance(probs_tensor, Tensor):
                    probs = torch.sigmoid(probs_tensor).detach().cpu().numpy().tolist()
                else:
                    probs = list(map(float, probs_tensor))
                chosen = [classes[i] for i, pr in enumerate(probs) if pr >= thresh]
                row["genre_labels"] = "|".join(chosen) if chosen else ""
            else:
                pred_class_g = pred_g[0]
                row["genre_labels"] = str(pred_class_g)
        except Exception:
            # Si falla, dejamos genre_labels vacío
            pass

        rows.append(row)

    return _write_csv(rows, out_csv)




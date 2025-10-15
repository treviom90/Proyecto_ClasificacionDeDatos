import pickle
import fsspec
import time
from airflow.providers.common.io.xcom.backend import XComObjectStorageBackend
from airflow.configuration import conf

class PickleXComBackend(XComObjectStorageBackend):
    """
    Backend que guarda en disco (XCOM_OBJECTSTORAGE_PATH) los bytes
    que tú le pases por pickle.dumps(), y al hacer pull() te devuelve
    esos mismos bytes para que hagas pickle.loads().
    """
@staticmethod
def serialize_value(value, key=None, **kwargs):
    # 1) Serializar con pickle si no es ya bytes
    payload = value if isinstance(value, (bytes, bytearray)) else pickle.dumps(value)

    # 2) Leer la ruta base desde la configuración de Airflow
    base_path = conf.get('common_io', 'xcom_objectstorage_path').rstrip('/')

    # 3) Construir un filename único
    run_id = kwargs.get("run_id", "no_runid")
    ts = int(time.time() * 1e6)
    filename = f"{key}{run_id}{ts}.bin"
    path = f"{base_path}/{filename}"

    # 4) Escribir el payload en disco (o S3, etc. según URI)
    fs, _, _ = fsspec.get_fs_token_paths(path)
    with fs.open(path, "wb") as f:
        f.write(payload)

        # 5) Devolver un JSON-safe con la ruta
        return {"path": path}

    @staticmethod
    def deserialize_value(result):
    # 1) Leer los bytes del disco
        path = result.value["path"]
        fs, _, _ = fsspec.get_fs_token_paths(path)
        with fs.open(path, "rb") as f:
            payload = f.read()
        # 2) Devolver los bytes crudos para tu pickle.loads()
    return payload
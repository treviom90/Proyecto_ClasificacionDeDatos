import pickle
from airflow.models.taskinstance import TaskInstance

def push(ti: TaskInstance, key: str, df):
    """
    Recibe cualquier objeto (DataFrame, dict, lista…) y hace:
    ti.xcom_push(key, pickle.dumps(obj))
     """
    ti.xcom_push(key=key, value=pickle.dumps(df))

def pull(ti: TaskInstance, task_id: str, key: str):
    """
    Recupera los bytes con ti.xcom_pull() y hace pickle.loads()
    para devolver el objeto original.
    """
    raw = ti.xcom_pull(task_ids=task_id, key=key)
    if raw is None:
        raise ValueError(f"No se encontró XCom para key={key}")
    return pickle.loads(raw)
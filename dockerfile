FROM apache/airflow:3.1.0

# Actualiza apt y instala git
USER root
RUN apt-get update && apt-get install -y git

# Instala el driver ODBC 17 para SQL Server
RUN apt-get update && \
     apt-get install -y curl gnupg && \
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql17 unixodbc-dev

# Cambia al usuario airflow para ejecutar pip
USER airflow

# Copia el archivo requirements.txt y luego instala las dependencias
COPY requirements.txt /opt/airflow/requirements.txt
RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt
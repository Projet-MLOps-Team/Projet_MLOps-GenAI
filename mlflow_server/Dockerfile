FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer MLflow + dépendances nécessaires
RUN pip install --no-cache-dir mlflow boto3 pymysql psycopg2-binary

# Variables d'environnement par défaut (modifiables dans ECS)
ENV MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db
ENV MLFLOW_ARTIFACT_ROOT=s3://mlflow-artifacts-patricia
ENV PORT=5000
ENV AWS_REGION=eu-west-3

# Exposer le port MLflow
EXPOSE 5000

# Lancer Mlflow Server
CMD mlflow server \
    --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
    --default-artifact-root ${MLFLOW_ARTIFACT_ROOT} \
    --host 0.0.0.0 \
    --port ${PORT} \
    --serve-artifacts

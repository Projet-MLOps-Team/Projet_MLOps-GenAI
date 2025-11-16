FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

# OS deps (si besoin de lib GL/ML, ajoute-les ici)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8080

# Très important : 0.0.0.0 et port 8080
CMD ["streamlit","run","app.py","--server.address","0.0.0.0","--server.port","8080"]

# -----------------------------
# Dockerfile pour Space (Streamlit)
# -----------------------------

# ⚙️ Image Python légère et récente
FROM python:3.11-slim

# 🔧 Variables d'environnement pour réduire la taille
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 🗂️ Dossier de travail
WORKDIR /app

# 📦 Copie des dépendances
COPY requirements.txt /app/requirements.txt

# 📦 Installation des dépendances système légères (si besoin de build)
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# 📦 Installation des packages Python
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# 📁 Copie du code de l'application
COPY . /app

# 🌐 Exposer le port Streamlit (Hugging Face utilise 7860)
EXPOSE 7860

# ▶️ Commande de lancement Streamlit (adressage 0.0.0.0 + port 7860)
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860"]

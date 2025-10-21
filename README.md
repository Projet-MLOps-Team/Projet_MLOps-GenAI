---
title: Loan Default Prediction
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
---

# 🧠 MLOps × GenAI — Loan Default Assistant

## 🔎 Introduction

Ce projet end-to-end combine **Machine Learning** et **Génération augmentée par la recherche (RAG)** pour aider une banque de détail à prédire le risque de défaut sur des prêts et interroger un assistant intelligent capable de :

- 📄 **Répondre via RAG** à partir d'un document interne (`ragdoc.pdf`)
- 🤖 **Faire des prédictions ML** sur le risque de défaut
- 🧮 **Réaliser des calculs** arithmétiques simples
- 🌐 **Lancer des recherches web** 

## 🎯 Objectifs du Projet

1. Construire un **modèle supervisé de classification** qui estime la probabilité de défaut
2. Exposer un **agent LangChain multi-outils** dans une application Streamlit
3. **Dockeriser** l'application pour un déploiement reproductible
4. Automatiser le **déploiement via CI/CD** (GitHub Actions → Hugging Face Spaces)

## 📊 Jeu de Données

Le fichier `data.csv` contient des informations de demandeurs de prêts et leur statut de défaut.

### Variables

- **Features**: `credit_lines_outstanding`, `loan_amt_outstanding`, `total_debt_outstanding`, `income`, `years_employed`, `fico_score`
- **Target**: `default` ∈ {0, 1}
- **Feature Engineering**: `debt_ratio = total_debt_outstanding / income` (créée par `eda.py`)

⚠️ **Note**: Le dataset peut être déséquilibré. Le modèle gère cela via `class_weight='balanced'` et un seuil ajustable.

## 🗂️ Structure du Projet
```
loan-default-assistant/
│
├── data.csv                    # Données brutes
├── eda.py                      # EDA + nettoyage → datafinal.csv
├── datafinal.csv               # Données préparées
│
├── train.py                    # Entraînement + export best_model.joblib
├── models/
│   ├── best_model.joblib       # Modèle sélectionné
│   └── meta.json               # Métadonnées
│
├── langchainagent.py           # Agent LangChain (4 outils)
├── app.py                      # Interface Streamlit
├── ragdoc.pdf                  # Document interne pour RAG
│
├── requirements.txt            # Dépendances Python
├── .github/
│   └── workflows/
│       └── deploy-huggingface.yml  # Pipeline CI/CD
│
└── README.md                   # Cette documentation
```

## 🚀 Installation et Utilisation

### Prérequis
```bash
Python 3.9+
OpenAI API Key ou autre LLM compatible
```

### Installation Locale
```bash
# 1. Cloner le repository
git clone https://github.com/Projet-MLOps-Team/Projet_MLOps-GenAI.git
cd Projet_MLOps-GenAI

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configurer les clés API
export OPENAI_API_KEY="sk-..."
```

### Pipeline Complet

#### 1️⃣ Préparation des Données
```bash
python eda.py
# → Génère datafinal.csv
```

**Opérations effectuées:**
- Nettoyage des valeurs manquantes
- Détection et traitement des outliers
- Feature engineering (`debt_ratio`)
- Normalisation des variables

#### 2️⃣ Entraînement du Modèle
```bash
python train.py
# → models/best_model.joblib + models/meta.json
```

**Modèles testés:**
- Logistic Regression
- Decision Tree
- Random Forest ⭐

**Sélection:** Basée sur **PR AUC** (Precision-Recall Area Under Curve)

#### 3️⃣ Lancer l'Application
```bash
streamlit run app.py
# → http://localhost:8501
```

## 🧩 Agent LangChain - 4 Outils

### 1. 📄 RAG Tool
Interroge le document `ragdoc.pdf` pour répondre aux questions internes.

### 2. 🤖 ML Predict Tool
Prédiction de défaut basée sur `best_model.joblib`.

### 3. 🧮 Calculator Tool
Calculs arithmétiques sécurisés.

### 4. 🌐 Web Search Tool
Recherche DuckDuckGo pour informations externes.

## 💬 Interface Streamlit

### Fonctionnalités

- **Chat unifié** : Questions RAG, ML, calculs, recherche web
- **Prédictions ML** : Évaluation du risque de défaut
- **Ingestion RAG** : Réponses depuis documents internes
- **Interface intuitive** : Conversation naturelle

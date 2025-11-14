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

# Classification supervisée avec suivi MLflow

## 1. Présentation du projet

Ce projet implémente une comparaison de modèles de classification supervisée à l’aide de **trois algorithmes** :

* Régression Logistique
* Arbre de Décision
* Random Forest

L’objectif est de proposer une approche complète intégrant la **gestion du déséquilibre des classes**, la **recherche d’hyperparamètres**, l’**optimisation du seuil de décision** et le **suivi expérimental avec MLflow**.

---

## 2. Objectifs principaux

* Entraîner **trois modèles** sur un jeu de données CSV (features + colonne cible).
* Gérer le **déséquilibre de classes** via `class_weight` et/ou `sample_weight`.
* (Optionnel) Effectuer une **recherche d’hyperparamètres** avec `RandomizedSearchCV`.
* Comparer les performances à l’aide des métriques suivantes :

  * AUC (ROC)
  * AUPRC (PR-AUC)
  * Accuracy
  * F1-score
  * Precision
  * Recall
  * Log Score

* Trouver le **seuil de décision optimal** maximisant le F1-score de la **classe minoritaire**.
* Sauvegarder le **meilleur pipeline** dans le répertoire :

  ```
  mlflow_artifacts/nom_modele
  best_model_local.pkl

  ```
* Enregistrer l’ensemble des **métriques, figures et résultats** dans **MLflow**, incluant :

  * Matrice de confusion
  * Courbes ROC et Precision-Recall

---

## 3. Fonctionnalités principales

| Fonctionnalité           | Description                                             |
| ------------------------ | ------------------------------------------------------- |
| Modèles évalués          | Régression Logistique, Arbre de Décision, Random Forest |
| Gestion du déséquilibre  | `class_weight` et/ou `sample_weight`                    |
| Tuning d’hyperparamètres | `RandomizedSearchCV` (optionnel)                        |
| Optimisation du seuil    | Recherche du seuil maximisant le F1-score minoritaire   |
| Suivi expérimental       | Logging complet avec MLflow                             |
| Export final             | `artifacts/best_model.joblib`                           |

---

## 4. Métriques et visualisations

Les performances sont évaluées selon plusieurs indicateurs :

* AUC (ROC)
* AUPRC (PR-AUC)
* Accuracy
* F1-score (global et minoritaire)
* Log loss

Les visualisations enregistrées dans MLflow comprennent :

* Matrice de confusion
* Courbe ROC
* Courbe Precision-Recall
* Comparatif global des modèles

---

## 5. Structure du projet (exemple)

```
.
├── data/
│   └── dataset.csv
├── mlflow_artifacts/
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── mlruns/
│   └── ...
├── src/
│   ├── data_artefacts.py
│   ├── data_processing.py
│   ├── metrics.py
│   ├── models.py
│   ├── save_best_model.py
│   ├── train_experiment.py
│   └── ...
├── best_model_local.pkl
|
├── main.py
│   
├── requirements.txt
└── README.md
```

---

## 6. Lancement rapide

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Lancement de l’interface MLflow

```bash
mlflow ui --backend-store-uri mlruns --default-artifact-root mlruns --host 127.0.0.1 --port 5000 
```

### Entraînement des modèles

```bash
python src/train.py --data data/dataset.csv
```

### Consultation des résultats

* Interface MLflow : [http://localhost:5000](http://localhost:5000)
* Modèle final enregistré : `artifacts/best_model.joblib`

---

## 7. Environnement et dépendances

* Python 3.x
* scikit-learn
* MLflow
* pandas
* numpy
* matplotlib
* seaborn
* joblib
---
title: Loan Default Prediction
emoji: 🏦
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
---

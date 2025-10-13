# 🧠 MLOps × GenAI  — Loan Default Assistant

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)]()
[![LangChain](https://img.shields.io/badge/LangChain-Agents-green)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)]()
[![Docker](https://img.shields.io/badge/Docker-Container-blue)]()
[![CI/CD](https://img.shields.io/badge/GitHub%20Actions-Automation-black)]()

## 🔎 Introduction
Ce projet end-to-end combine **Machine Learning** et **Génération augmentée par la recherche (RAG)** pour aider une banque de détail à **prédire le risque de défaut** sur des prêts et **interroger un assistant** capable de :
- répondre via RAG à partir d’un document interne,
- faire des **prédictions ML**,
- réaliser des **calculs** simples,
- lancer une **recherche web**.

### 🧰 Objectifs
- Construire un modèle supervisé (classification) qui estime la probabilité de défaut.
- Exposer un **agent LangChain (4 outils)** dans une **app Streamlit**.
- Dockeriser et automatiser le déploiement (CI/CD).

---

## 📦 Jeu de données
Le fichier **`data.csv`** contient des informations de demandeurs de prêts et leur statut de défaut (`default` ∈ {0,1}).  
Variables typiques :
- `credit_lines_outstanding`, `loan_amt_outstanding`, `total_debt_outstanding`
- `income`, `years_employed`, `fico_score`
- **Target** : `default`  
- **Feature ingénierée** : `debt_ratio = total_debt_outstanding / income` (créée par `eda.py`)

> ⚠️ Le dataset peut être déséquilibré. Le modèle gère cela via `class_weight='balanced'` et un **seuil** ajustable.

---

## 🗂️ Arborescence
```
data.csv              # Données brutes
eda.py                # EDA + nettoyage -> datafinal.csv
datafinal.csv         # Données prêtes pour l’entraînement
train.py              # Entraînement + export best_model.joblib (+ meta.json)
langchainagent.py     # Agent LangChain (RAG, MLPredict, Calculator, WebSearch)
app.py                # UI Streamlit (chat)
ragdoc.pdf            # Document interne indexé pour le RAG
requirements.txt      # Dépendances
dockerfile            # Docker (minuscule) pour build
ci-cd.yaml            # GitHub Actions (build/push/deploy)
```

---

## ⚙️ Installation rapide
```bash
python -m venv .venv && source .venv/bin/activate  # Win: .venv\Scripts\activate
pip install --upgrade pip && pip install -r requirements.txt
# Clés (LLM) : export OPENAI_API_KEY="sk-..."   # ou via un .env non commité
```

---

## 🧹 Préparation des données (EDA)
```bash
python eda.py
# -> génère datafinal.csv (imputations, clipping, debt_ratio, etc.)
```

---

## 🤖 Entraînement du modèle
```bash
python train.py
# -> models/best_model.joblib  (+ meta.json pour l'ordre des features)
```
- 3 modèles comparés (ex. LogisticRegression, DecisionTree, RandomForest).  
- Sélection par **ROC AUC**. Ajuste le seuil métier dans `train.py` si besoin.

---

## 🧩 Agent LangChain (4 outils)
```bash
python langchainagent.py
```
1) **RAG** sur `ragdoc.pdf`  
2) **MLPredict** via `best_model.joblib` (ou MLflow si configuré)  
3) **Calculator** (arithmétique sécurisée)  
4) **WebSearch** (DuckDuckGo)

---

## 💬 Application Streamlit
```bash
streamlit run app.py
```
- Chat unifié (RAG/ML/Calc/Web).  
- Ingestion du document `ragdoc.pdf` pour les réponses contextualisées.

---

## 🐳 Docker
```bash
docker build -f dockerfile -t loan-assistant .
docker run -p 8501:8501 loan-assistant
```

---

## 🚀 CI/CD (GitHub Actions)
- Le workflow **`ci-cd.yaml`** build/push l’image (Docker Hub / ECR).  
- Secrets à définir dans GitHub : `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` (ou variables AWS si ECR).

---

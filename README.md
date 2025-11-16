# 🧠 MLOps × GenAI — Loan Default Assistant (Sorbonnio)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)]()
[![LangChain](https://img.shields.io/badge/LangChain-Agents-green)]()
[![LangGraph](https://img.shields.io/badge/LangGraph-ReAct-brightgreen)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)]()
[![Docker](https://img.shields.io/badge/Docker-Container-blue)]()

## 🔎 Introduction

Ce projet end-to-end combine :

- un **modèle de scoring de défaut de crédit** (scikit-learn, hébergé sur **S3**) ;
- un **RAG** sur le PDF _conditions-tarifaires-particuliers-2025.pdf_ ;
- un **agent LangGraph / LangChain** nommé **Sorbonnio** ;
- une **application Streamlit** avec 3 onglets :  
  **📊 EDA** · **🔮 Prédiction ML** · **💬 Chatbot bancaire**.

Objectif : aider une banque de détail à **analyser le risque de défaut** et **répondre aux questions tarifaires** à partir d’un document interne.

---

## 🗂️ Arborescence

```bash
.github/workflows/
  └── ci-cd-ecs.yml                     # CI/CD GitHub Actions (build & déploiement ECS)

Dockerfile                              # Image Streamlit + agent
Loan_Data.csv                           # Données brutes de crédit
eda.py                                  # Script d’EDA et préparation de datasetfinal.csv
datasetfinal.csv                        # Données nettoyées / feature engineering
train.ipynb                             # Entrainement & MLFLOW
agent.py                                # Agent Sorbonnio (RAG, ML, Calc, Web)
app.py                                  # App Streamlit (EDA, Prédiction ML, Chatbot)
conditions-tarifaires-particuliers-2025.pdf  # Document RAG
README.md                               # Documentation du projet
requirements.txt                        # Dépendances Python
```

> Le modèle ML entraîné (`best_model.pkl`) n’est **pas** dans le repo : il est hébergé sur **S3** et chargé dynamiquement par `agent.py`.

---

## 📦 Données

- **`Loan_Data.csv`** : dataset brut (emprunteurs, variables financières, cible `default`).
- **`datasetfinal.csv`** : version nettoyée / feature-engineered, générée par `eda.py`.

Variables clés :

- `credit_lines_outstanding`, `loan_amt_outstanding`, `total_debt_outstanding`
- `income`, `years_employed`, `fico_score`
- `default` (cible de classification)
- `debt_ratio = total_debt_outstanding / income`

---

## ⚙️ Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Win: .venv\Scriptsctivate
pip install --upgrade pip
pip install -r requirements.txt
```

Variables d’environnement (via `.env` ou export) :

```bash
export OPENAI_API_KEY="sk-..."
export TAVILY_API_KEY="tvly-..."
# optionnels
export OPENAI_MODEL="gpt-4o-mini"
export EMBED_MODEL="text-embedding-3-small"
export CHROMA_DIR="./chroma_store"
```

---

## 🧹 EDA & préparation

```bash
python eda.py
# lit Loan_Data.csv et produit datasetfinal.csv
```

Ce script réalise :

- nettoyage de base,
- création de features (dont `debt_ratio`),
- export d’un dataset prêt pour l’entraînement (le modèle résultant est ensuite uploadé sur S3).

---

## 🤖 Agent Sorbonnio (`agent.py`)

L’agent ReAct (LangGraph + LangChain) expose 4 outils :

1. **`rag_search`**  
   - RAG sur `conditions-tarifaires-particuliers-2025.pdf` indexé dans **ChromaDB**.  
   - Utilisé pour toutes les questions de **tarifs, frais, comptes, cartes, incidents, segments de clientèle**.

2. **`ml_predict`**  
   - Charge un modèle scikit-learn **depuis S3** (`best_model.pkl`).  
   - Accepte un **payload partiel** (par ex. seulement `income`, `fico_score`, `debt_ratio`, `years_employed`).  
   - Complète les features manquantes avec des **valeurs moyennes par défaut**.  
   - Retourne :
     - classe (`Client plutôt sain` / `Défaut probable`),
     - probabilité de défaut,
     - niveau de risque (faible / modéré / élevé),
     - message d’explication indiquant si la prédiction repose sur des données partielles.

3. **`calculator`**  
   - Calculette via `numexpr` pour les calculs de montants et de ratios.

4. **`web_search_tool`**  
   - Recherche web via **Tavily** pour le contexte externe (actualité, macro…).

Le **system prompt** définit la personnalité de Sorbonnio, la priorité d’usage des outils (RAG > ML > Web), et le style de réponse (français, structuré, signé “Sorbonnio, le chatbot bancaire de Kamila Kare”).

---

## 💻 Application Streamlit (`app.py`)

Lancement local :

```bash
streamlit run app.py
```

### 1. Onglet 📊 EDA

- Upload de `Loan_Data.csv` ou d’un autre fichier de crédit.  
- Fonctionnalités :
  - aperçu du dataset,
  - métriques globales (taux de défaut, nombre de clients sains/en défaut),
  - distributions des variables clés par défaut,
  - matrice de corrélation,
  - scatterplots pour visualiser les zones à risque.

### 2. Onglet 🔮 Prédiction ML

- Formulaire métier (une seule colonne) :

  - `credit_lines_outstanding`  
  - `loan_amt_outstanding`  
  - `total_debt_outstanding`  
  - `income`  
  - `years_employed`  
  - `fico_score`

- Calcul automatique de `debt_ratio`.
- Appel de `ml_predict` (modèle S3).
- Affichage :

  - verdict (`Client plutôt sain` / `Défaut probable`),
  - niveau de risque + emoji,
  - probabilité de défaut (%),
  - jauge visuelle (`st.progress`),
  - liste des features utilisées,
  - expander avec la réponse JSON brute.

### 3. Onglet 💬 Chatbot bancaire

- Interface de chat basée sur `st.chat_message`.
- L’agent Sorbonnio peut :

  - répondre aux questions tarifaires via RAG,
  - exécuter `ml_predict` avec un payload partiel,
  - réaliser des calculs,
  - lancer une recherche web.

- Historique de conversation stocké dans `st.session_state`.

---

## 🐳 Docker

Build et run local de l’image :

```bash
docker build -t loan-default-assistant .
docker run -p 8501:8501 loan-default-assistant
```

L’image contient :

- l’app Streamlit,
- l’agent Sorbonnio,
- le chargement dynamique du modèle ML depuis S3,
- l’indexation du PDF de conditions tarifaires.

---

## 🚀 CI/CD (GitHub Actions / ECS)

Le workflow **`.github/workflows/ci-cd-ecs.yml`** automatise :

- le build de l’image Docker,
- le push vers un registre de conteneurs,
- le déploiement sur une infrastructure **AWS ECS** (ou autre cible containerisée selon ta config).

---

## 🎯 Statut & prochaines étapes

- ✅ EDA + dataset final  
- ✅ Modèle entraîné et hébergé sur S3  
- ✅ Agent Sorbonnio (RAG + ML + Calc + Web) opérationnel  
- ✅ App Streamlit multi-onglets  
- ✅ Docker + pipeline CI/CD  
- 🔜 Améliorations possibles :
  - calibration des probabilités de défaut,
  - UI métier encore plus guidée (scénarios types),
  - ajout d’autres documents RAG (brochures produits, FAQ…).

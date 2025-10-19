# -*- coding: utf-8 -*-
"""MLOps End-to-End Application"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
from openai import OpenAI
from tavily import TavilyClient

# Configuration des variables d'environnement
try:
    if hasattr(st, 'secrets') and ('OPENAI_API_KEY' in st.secrets or 'TAVILY_API_KEY' in st.secrets):
        OPENAI_API_KEY = st.secrets.get('OPENAI_API_KEY', '')
        TAVILY_API_KEY = st.secrets.get('TAVILY_API_KEY', '')
        MLFLOW_TRACKING_URI = st.secrets.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    else:
        from dotenv import load_dotenv
        load_dotenv()
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
        TAVILY_API_KEY = os.getenv('TAVILY_API_KEY', '')
        MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
except Exception as e:
    st.warning(f"Echec du chargement des variables d'environnement: {e}")
    OPENAI_API_KEY = ''
    TAVILY_API_KEY = ''
    MLFLOW_TRACKING_URI = 'http://localhost:5000'

# Configuration
st.set_page_config(
    page_title="MLOps End-to-End",
    page_icon="🏦",
    layout="wide"
)

# Session State
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'training_results' not in st.session_state:
    st.session_state.training_results = None

# Fonction RAG
def rag_recherche_web(question: str, max_results: int = 5, region: str = "fr", language: str = "fr"):
    """
    Recherche et synthèse de contenu à partir du web.
    
    Paramètres :
    - question (str) : la question de l'utilisateur.
    - max_results (int) : nombre maximum de résultats à extraire du web.
    - region (str) : région de recherche (par ex. 'fr', 'eu', 'us').
    - language (str) : langue des résultats ('fr' ou 'en').
    
    Retourne :
    - answer (str) : une réponse générée par OpenAI.
    - sources (list[str]) : liste des URLs utilisées pour la synthèse.
    """
    if not question.strip():
        return "Veuillez saisir une question.", []
    if not TAVILY_API_KEY:
        return "❌ Clé Tavily manquante (TAVILY_API_KEY non trouvée).", []
    if not OPENAI_API_KEY:
        return "❌ Clé OpenAI manquante (OPENAI_API_KEY non trouvée).", []

    tavily = TavilyClient(api_key=TAVILY_API_KEY)
    try:
        res = tavily.search(
            query=question,
            max_results=max_results,
            include_answer=False,
            search_depth="advanced"
        )
    except Exception as e:
        return f"Erreur Tavily : {e}", []

    results = res.get("results", []) if isinstance(res, dict) else []
    contexts, sources = [], []
    for r in results:
        content = r.get("content") or r.get("snippet") or ""
        url = r.get("url") or r.get("source") or ""
        if content:
            contexts.append(content)
        if url:
            sources.append(url)

    context_block = "\n\n---\n".join(contexts[:max_results]) if contexts else "Aucun contexte trouvé."

    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""
Tu es un assistant expert en risque de crédit.
En te basant UNIQUEMENT sur le contexte ci-dessous, rédige une réponse claire et concise à la question.
- Langue : français
- Format : 3 à 5 points clés maximum
- Ne génère pas de fausses informations
- Si aucune donnée pertinente n'est trouvée, indique-le simplement.
- Termine par une section "Sources" si disponible.

Question :
{question}

Contexte :
{context_block}
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Erreur OpenAI : {e}", sources

    return answer, sources[:max_results]

# Functions
@st.cache_data
def load_data():
    """Charge les données"""
    df = pd.read_csv('Loan_Data.csv')
    df['debt_ratio'] = df['total_debt_outstanding'] / df['income']
    return df

def train_models(df):
    """Entraîne les 3 modèles"""
    X = df.drop(['default', 'customer_id'], axis=1)
    y = df['default']
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced'
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, max_depth=10, class_weight='balanced'
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=15, 
            class_weight='balanced'
        )
    }
    
    results = {}
    progress = st.progress(0)
    
    for i, (name, model) in enumerate(models.items()):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
        y_proba = model.predict_proba(X_val_scaled)[:, 1]
        
        results[name] = {
            'accuracy': accuracy_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
            'auc': roc_auc_score(y_val, y_proba)
        }
        progress.progress((i + 1) / len(models))
    
    best_name = max(results, key=lambda x: results[x]['f1'])
    best_model = models[best_name]
    best_model.fit(X_train_scaled, y_train)
    
    # Sauvegarde
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(best_model, 'artifacts/best_model.pkl')
    joblib.dump(scaler, 'artifacts/scaler.pkl')
    joblib.dump(X.columns.tolist(), 'artifacts/feature_names.pkl')
    
    return results, best_name

@st.cache_resource
def load_model():
    """Charge le modèle"""
    try:
        model = joblib.load('artifacts/best_model.pkl')
        scaler = joblib.load('artifacts/scaler.pkl')
        features = joblib.load('artifacts/feature_names.pkl')
        return model, scaler, features
    except:
        return None, None, None

# UI
st.title("🏦 MLOps End-to-End: Prédiction Défaut Crédit")
st.markdown("**Pipeline Complet:** EDA → Training → Prediction → Recherche Web → Monitoring")
st.markdown("---")

# Tabs  
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 1. EDA",
    "🤖 2. Training", 
    "🔮 3. Prediction",
    "🔎 4. Recherche Web",
    "📈 5. Monitoring"
])

# TAB 1: EDA
with tab1:
    st.header("Analyse Exploratoire des Données")
    
    if st.button("🔄 Charger les Données", type="primary"):
        st.session_state.df = load_data()
        st.success("✅ Données chargées!")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Métriques
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Lignes", f"{df.shape[0]:,}")
        col2.metric("Colonnes", df.shape[1])
        col3.metric("Défauts", f"{df['default'].sum():,}")
        col4.metric("Taux Défaut", f"{df['default'].mean():.1%}")
        
        # Données
        with st.expander("📋 Voir les données"):
            st.dataframe(df.head(20), use_container_width=True)
        
        # Statistiques
        with st.expander("📊 Statistiques"):
            st.dataframe(df.describe(), use_container_width=True)
        
        # Visualisations
        st.subheader("Visualisations")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                df, names='default',
                title='Distribution des Défauts',
                color_discrete_sequence=['#51cf66', '#ff6b6b']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                df, x='fico_score', color='default',
                title='Distribution FICO Score',
                nbins=30
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Corrélation
        st.subheader("Matrice de Corrélation")
        corr = df.drop('customer_id', axis=1).corr()
        fig = px.imshow(
            corr, text_auto='.2f',
            title="Corrélations",
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👆 Cliquez pour charger les données")

# TAB 2: TRAINING
with tab2:
    st.header("Entraînement des Modèles")
    
    if st.session_state.df is None:
        st.warning("⚠️ Chargez d'abord les données (Tab 1)")
    else:
        st.info("""
        **3 Algorithmes:**
        - Logistic Regression
        - Decision Tree
        - Random Forest
        """)
        
        if st.button("🚀 Lancer l'Entraînement", type="primary"):
            with st.spinner("Entraînement en cours..."):
                results, best_name = train_models(st.session_state.df)
                st.session_state.training_results = {
                    'results': results,
                    'best': best_name
                }
            st.success(f"✅ Meilleur modèle: **{best_name}**")
            st.balloons()
        
        if st.session_state.training_results:
            results = st.session_state.training_results['results']
            best = st.session_state.training_results['best']
            
            # Tableau
            df_results = pd.DataFrame(results).T
            df_results = df_results.round(4)
            
            st.subheader("Résultats")
            st.dataframe(df_results, use_container_width=True)
            
            # Graphique
            fig = go.Figure()
            for metric in ['accuracy', 'f1', 'auc']:
                fig.add_trace(go.Bar(
                    name=metric.upper(),
                    x=list(results.keys()),
                    y=[results[m][metric] for m in results]
                ))
            fig.update_layout(title="Comparaison", barmode='group')
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"🏆 Meilleur: **{best}** (F1={results[best]['f1']:.4f})")

# TAB 3: PREDICTION
with tab3:
    st.header("Prédiction de Défaut")
    
    model, scaler, features = load_model()
    
    if model is None:
        st.warning("⚠️ Entraînez d'abord un modèle (Tab 2)")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Informations Client")
            credit_lines = st.number_input("Lignes de crédit", 0, 50, 5)
            loan_amt = st.number_input("Prêt (€)", 0, 1000000, 15000, 1000)
            total_debt = st.number_input("Dette totale (€)", 0, 1000000, 25000, 1000)
            income = st.number_input("Revenu (€)", 1000, 1000000, 60000, 1000)
        
        with col2:
            st.subheader("Profil Financier")
            years = st.number_input("Années emploi", 0, 50, 10)
            fico = st.number_input("Score FICO", 300, 850, 720)
            debt_ratio = total_debt / income
            st.metric("Debt Ratio", f"{debt_ratio:.2%}")
        
        if st.button("🔮 Prédire", type="primary", use_container_width=True):
            start = time.time()
            
            features_array = np.array([[
                credit_lines, loan_amt, total_debt, 
                income, years, fico, debt_ratio
            ]])
            features_scaled = scaler.transform(features_array)
            
            pred = int(model.predict(features_scaled)[0])
            prob = float(model.predict_proba(features_scaled)[0][1])
            pred_time = time.time() - start
            
            st.session_state.predictions.append({
                'timestamp': datetime.now(),
                'prediction': pred,
                'probability': prob,
                'time': pred_time
            })
            
            if pred == 1:
                st.error(f"### ⚠️ RISQUE DE DÉFAUT")
                st.write(f"**Probabilité:** {prob*100:.1f}%")
            else:
                st.success(f"### ✅ PAS DE RISQUE")
                st.write(f"**Probabilité:** {prob*100:.1f}%")
            
            st.info(f"⏱️ Temps: {pred_time*1000:.2f}ms")

# TAB 4: RECHERCHE WEB (RAG)
with tab4:
    st.header("🔎 Recherche Web (RAG) — Explications & Contexte")
    st.caption("Utilise Tavily (recherche) + OpenAI (synthèse) pour expliquer un résultat ou trouver des infos récentes.")

    if not TAVILY_API_KEY or not OPENAI_API_KEY:
        st.error("API keys manquants. Ajoute `OPENAI_API_KEY` et `TAVILY_API_KEY` dans st.secrets ou .env.")
    else:
        exemples = [
            "Pourquoi le risque de défaut est faible pour un client avec FICO 720 et debt ratio 36% en France ?",
            "Taux de défaut récents sur les prêts conso en Europe (sources fiables) ?",
            "Principaux facteurs qui augmentent la probabilité de défaut selon la littérature.",
        ]
        with st.expander("💡 Exemples de questions", expanded=False):
            for e in exemples:
                st.write(f"- {e}")

        col_a, col_b = st.columns([3, 1])
        with col_a:
            question = st.text_area("Ta question (FR)", value="", height=100, placeholder=exemples[0])
        with col_b:
            max_results = st.slider("Résultats web", 1, 8, 4)
            region = st.selectbox("Région", ["fr", "eu", "us", "global"], index=0)
            language = st.selectbox("Langue", ["fr", "en"], index=0)

        lancer = st.button("🔍 Rechercher & Expliquer", type="primary", use_container_width=True)

        if lancer:
            with st.spinner("Recherche et synthèse en cours..."):
                answer, sources = rag_recherche_web(
                    question=question or exemples[0],
                    max_results=max_results,
                    region=region,
                    language=language
                )
            st.markdown("### 🔎 Réponse")
            st.write(answer)

            if sources:
                st.markdown("### 🔗 Sources")
                for i, url in enumerate(sources, 1):
                    st.markdown(f"{i}. {url}")
            else:
                st.info("Aucune source détectée par la recherche.")

# TAB 5: MONITORING  
with tab5:
    st.header("Monitoring des Prédictions")
    
    if not st.session_state.predictions:
        st.info("Aucune prédiction enregistrée. Utilisez l'onglet Prediction.")
    else:
        df_preds = pd.DataFrame(st.session_state.predictions)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Prédictions", len(df_preds))
        col2.metric("Défauts Prédits", df_preds['prediction'].sum())
        col3.metric("Temps Moyen (ms)", f"{df_preds['time'].mean()*1000:.2f}")
        
        st.subheader("Historique")
        st.dataframe(df_preds, use_container_width=True)
        
        st.subheader("Distribution Probabilités")
        fig = px.histogram(df_preds, x='probability', nbins=20, title="Distribution des probabilités de défaut")
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <strong>Projet MLOps End-to-End</strong><br>
    EDA + Training + Prediction + Monitoring
</div>
""", unsafe_allow_html=True)

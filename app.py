# -*- coding: utf-8 -*-
"""
Application MLOps End-to-End - Prédiction de Défaut de Crédit
Projet Master - Université Paris 1 Panthéon-Sorbonne
"""

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

# Configuration des clés API
try:
    if hasattr(st, 'secrets'):
        OPENAI_API_KEY = st.secrets.get('OPENAI_API_KEY', '')
        TAVILY_API_KEY = st.secrets.get('TAVILY_API_KEY', '')
    else:
        from dotenv import load_dotenv
        load_dotenv()
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
        TAVILY_API_KEY = os.getenv('TAVILY_API_KEY', '')
except:
    OPENAI_API_KEY = ''
    TAVILY_API_KEY = ''

# Import des bibliothèques API (optionnel pour Outil 4)
try:
    from openai import OpenAI
    from tavily import TavilyClient
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# Fonction RAG (Outil 4)
def rag_recherche_web(question, max_results=5):
    """Recherche web + génération de réponse avec RAG"""
    if not RAG_AVAILABLE:
        return "Bibliothèques OpenAI/Tavily non installées.", []
    if not question.strip():
        return "Veuillez poser une question.", []
    if not TAVILY_API_KEY or not OPENAI_API_KEY:
        return "Clés API manquantes (configurez .env ou secrets.toml).", []
    
    try:
        # Recherche web avec Tavily
        tavily = TavilyClient(api_key=TAVILY_API_KEY)
        res = tavily.search(query=question, max_results=max_results, search_depth="advanced")
        results = res.get("results", [])
        
        # Extraction du contenu
        contexts = [r.get("content", "") for r in results if r.get("content")]
        sources = [r.get("url", "") for r in results if r.get("url")]
        context_block = "\n\n".join(contexts[:max_results])
        
        # Génération avec OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = f"""Tu es un expert en risque de crédit. Réponds en français (3-5 points clés).

Question: {question}

Contexte:
{context_block}
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return resp.choices[0].message.content.strip(), sources[:max_results]
    except Exception as e:
        return f"Erreur: {e}", []

# Configuration Streamlit
st.set_page_config(page_title="MLOps Projet", page_icon="🏦", layout="wide")

# Initialisation des états de session
if 'df' not in st.session_state:
    st.session_state.df = None
if 'training_results' not in st.session_state:
    st.session_state.training_results = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# Fonction de chargement des données
@st.cache_data
def load_data():
    """Charge les données CSV et calcule debt_ratio"""
    df = pd.read_csv('Loan_Data.csv')
    df['debt_ratio'] = df['total_debt_outstanding'] / df['income']
    return df

# Fonction d'entraînement
def train_models(df):
    """Entraîne 3 modèles et retourne le meilleur"""
    X = df.drop(['default', 'customer_id'], axis=1)
    y = df['default']
    
    # Split: 60% train, 20% val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Modèles
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15, class_weight='balanced')
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
    
    # Sélection du meilleur
    best_name = max(results, key=lambda x: results[x]['f1'])
    best_model = models[best_name]
    best_model.fit(X_train_scaled, y_train)
    
    # Sauvegarde
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(best_model, 'artifacts/best_model.pkl')
    joblib.dump(scaler, 'artifacts/scaler.pkl')
    joblib.dump(X.columns.tolist(), 'artifacts/feature_names.pkl')
    
    return results, best_name

# Fonction de chargement du modèle
@st.cache_resource
def load_model():
    """Charge le modèle entraîné"""
    try:
        model = joblib.load('artifacts/best_model.pkl')
        scaler = joblib.load('artifacts/scaler.pkl')
        features = joblib.load('artifacts/feature_names.pkl')
        return model, scaler, features
    except:
        return None, None, None

# Interface principale
st.title("🏦 MLOps End-to-End: Prédiction de Défaut de Crédit")
st.markdown("**Pipeline Complet:** EDA → Training → Prediction → Recherche Web")
st.markdown("---")

# Création des onglets
tab1, tab2, tab3, tab4 = st.tabs(["📊 Outil 1: EDA", "🤖 Outil 2: Training", "🔮 Outil 3: Prediction", "🔎 Outil 4: RAG"])

# OUTIL 1: EDA
with tab1:
    st.header("Analyse Exploratoire des Données")
    
    if st.button("🔄 Charger les Données", type="primary"):
        try:
            st.session_state.df = load_data()
            st.success("✅ Données chargées avec succès!")
        except FileNotFoundError:
            st.error("❌ Fichier Loan_Data.csv introuvable!")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Métriques
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Nombre de lignes", f"{df.shape[0]:,}")
        c2.metric("Nombre de colonnes", df.shape[1])
        c3.metric("Nombre de défauts", f"{df['default'].sum():,}")
        c4.metric("Taux de défaut", f"{df['default'].mean():.1%}")
        
        # Aperçu des données
        with st.expander("📋 Aperçu des données (20 premières lignes)"):
            st.dataframe(df.head(20), use_container_width=True)
        
        # Statistiques
        with st.expander("📊 Statistiques descriptives"):
            st.dataframe(df.describe(), use_container_width=True)
        
        # Visualisations
        st.subheader("Visualisations")
        c1, c2 = st.columns(2)
        
        with c1:
            fig = px.pie(df, names='default', title='Répartition des Défauts', 
                        color_discrete_sequence=['#2ecc71', '#e74c3c'])
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            fig = px.histogram(df, x='fico_score', color='default', 
                             title='Distribution du Score FICO', nbins=30)
            st.plotly_chart(fig, use_container_width=True)
        
        # Matrice de corrélation
        st.subheader("Matrice de Corrélation")
        corr = df.drop('customer_id', axis=1).corr()
        fig = px.imshow(corr, text_auto='.2f', title="Corrélations entre variables",
                       color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)

# OUTIL 2: TRAINING
with tab2:
    st.header("Entraînement des Modèles")
    
    if st.session_state.df is None:
        st.warning("⚠️ Veuillez d'abord charger les données dans l'Outil 1")
    else:
        st.info("""
        **3 Algorithmes testés:**
        - Logistic Regression (Régression Logistique)
        - Decision Tree (Arbre de Décision)
        - Random Forest (Forêt Aléatoire)
        
        Le meilleur modèle est sélectionné selon le score F1.
        """)
        
        if st.button("🚀 Lancer l'Entraînement", type="primary"):
            with st.spinner("Entraînement en cours..."):
                results, best_name = train_models(st.session_state.df)
                st.session_state.training_results = {'results': results, 'best': best_name}
            st.success(f"✅ Entraînement terminé! Meilleur modèle: **{best_name}**")
            st.balloons()
        
        if st.session_state.training_results:
            results = st.session_state.training_results['results']
            best = st.session_state.training_results['best']
            
            # Tableau des résultats
            st.subheader("Résultats des Modèles")
            df_results = pd.DataFrame(results).T.round(4)
            st.dataframe(df_results, use_container_width=True)
            
            # Graphique comparatif
            fig = go.Figure()
            for metric in ['accuracy', 'f1', 'auc']:
                fig.add_trace(go.Bar(
                    name=metric.upper(),
                    x=list(results.keys()),
                    y=[results[m][metric] for m in results]
                ))
            fig.update_layout(title="Comparaison des Performances", barmode='group')
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"🏆 Meilleur modèle: **{best}** (F1 Score: {results[best]['f1']:.4f})")

# OUTIL 3: PREDICTION
with tab3:
    st.header("Prédiction de Risque de Défaut")
    
    model, scaler, features = load_model()
    
    if model is None:
        st.warning("⚠️ Veuillez d'abord entraîner un modèle dans l'Outil 2")
    else:
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("📋 Informations Client")
            credit_lines = st.number_input("Nombre de lignes de crédit", 0, 50, 5)
            loan_amt = st.number_input("Montant du prêt (€)", 0, 1000000, 15000, 1000)
            total_debt = st.number_input("Dette totale (€)", 0, 1000000, 25000, 1000)
            income = st.number_input("Revenu annuel (€)", 1000, 1000000, 60000, 1000)
        
        with c2:
            st.subheader("💼 Profil Financier")
            years = st.number_input("Années d'emploi", 0, 50, 10)
            fico = st.number_input("Score FICO", 300, 850, 720)
            debt_ratio = total_debt / income
            st.metric("Ratio Dette/Revenu", f"{debt_ratio:.2%}")
        
        if st.button("🔮 Effectuer la Prédiction", type="primary", use_container_width=True):
            start = time.time()
            
            # Préparation des features
            features_array = np.array([[credit_lines, loan_amt, total_debt, income, years, fico, debt_ratio]])
            features_scaled = scaler.transform(features_array)
            
            # Prédiction
            pred = int(model.predict(features_scaled)[0])
            prob = float(model.predict_proba(features_scaled)[0][1])
            pred_time = time.time() - start
            
            # Sauvegarde dans l'historique
            st.session_state.predictions.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'prediction': "Défaut" if pred == 1 else "Pas de défaut",
                'probabilite': f"{prob*100:.1f}%",
                'temps_ms': f"{pred_time*1000:.2f}"
            })
            
            # Affichage du résultat
            if pred == 1:
                st.error("### ⚠️ RISQUE DE DÉFAUT DÉTECTÉ")
                st.write(f"**Probabilité de défaut:** {prob*100:.1f}%")
            else:
                st.success("### ✅ PAS DE RISQUE DE DÉFAUT")
                st.write(f"**Probabilité de défaut:** {prob*100:.1f}%")
            
            st.info(f"⏱️ Temps de prédiction: {pred_time*1000:.2f} ms")
            
            # Historique
            if len(st.session_state.predictions) > 1:
                with st.expander("📊 Voir l'historique des prédictions"):
                    st.dataframe(pd.DataFrame(st.session_state.predictions), use_container_width=True)

# OUTIL 4: RAG (Recherche Web)
with tab4:
    st.header("🔎 Recherche Web avec RAG")
    st.caption("Utilise Tavily (recherche) + OpenAI (synthèse) pour répondre à vos questions sur le risque de crédit")
    
    if not RAG_AVAILABLE:
        st.error("❌ Bibliothèques non installées. Exécutez: `pip install openai tavily-python`")
    elif not TAVILY_API_KEY or not OPENAI_API_KEY:
        st.warning("⚠️ Clés API manquantes. Configurez `.env` ou `secrets.toml` avec OPENAI_API_KEY et TAVILY_API_KEY")
    else:
        # Exemples de questions
        exemples = [
            "Quels sont les principaux facteurs de risque de défaut de crédit selon la littérature ?",
            "Comment le score FICO impacte-t-il la probabilité de défaut ?",
            "Taux de défaut moyens sur les prêts personnels en France en 2024 ?"
        ]
        
        with st.expander("💡 Exemples de questions"):
            for ex in exemples:
                st.write(f"• {ex}")
        
        # Saisie de la question
        question = st.text_area("Posez votre question en français", height=100, 
                               placeholder="Ex: Pourquoi un FICO de 720 indique un faible risque ?")
        
        col_a, col_b = st.columns([2, 1])
        with col_a:
            rechercher = st.button("🔍 Rechercher et Expliquer", type="primary", use_container_width=True)
        with col_b:
            max_results = st.slider("Nb de sources", 3, 8, 5)
        
        if rechercher and question:
            with st.spinner("🌐 Recherche en cours..."):
                answer, sources = rag_recherche_web(question, max_results)
            
            st.markdown("### 📝 Réponse Générée")
            st.write(answer)
            
            if sources:
                st.markdown("### 🔗 Sources Utilisées")
                for i, url in enumerate(sources, 1):
                    st.markdown(f"{i}. [{url}]({url})")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <strong>Projet MLOps End-to-End</strong><br>
    Master - Université Paris 1 Panthéon-Sorbonne
</div>
""", unsafe_allow_html=True)

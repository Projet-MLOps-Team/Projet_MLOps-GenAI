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
st.markdown("**Pipeline Complet:** EDA → Training → Prediction → Monitoring")
st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. EDA",
    "🤖 2. Training", 
    "🔮 3. Prediction",
    "📈 4. Monitoring"
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

# TAB 4: MONITORING
with tab4:
    st.header("Monitoring des Prédictions")
    
    if len(st.session_state.predictions) > 0:
        df_pred = pd.DataFrame(st.session_state.predictions)
        
        # Métriques
        col1, col2, col3, col4 = st.columns(4)
        total = len(df_pred)
        defaults = (df_pred['prediction'] == 1).sum()
        
        col1.metric("Total", total)
        col2.metric("Défauts", defaults)
        col3.metric("Taux", f"{defaults/total*100:.1f}%")
        col4.metric("Temps Moy.", f"{df_pred['time'].mean()*1000:.2f}ms")
        
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(
                df_pred.reset_index(),
                x='index', y='probability',
                title='Évolution des Probabilités'
            )
            fig.add_hline(y=0.5, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                df_pred, x='probability',
                nbins=20,
                title='Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Tableau
        st.subheader("Historique")
        recent = df_pred.tail(10).copy()
        recent['prediction'] = recent['prediction'].map({0: '✅', 1: '⚠️'})
        recent['probability'] = recent['probability'].apply(lambda x: f"{x*100:.1f}%")
        recent['time'] = recent['time'].apply(lambda x: f"{x*1000:.2f}ms")
        recent['timestamp'] = recent['timestamp'].dt.strftime('%H:%M:%S')
        st.dataframe(recent, use_container_width=True, hide_index=True)
    
    else:
        st.info("📭 Aucune prédiction. Allez dans Tab 3!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <strong>Projet MLOps End-to-End</strong><br>
    EDA + Training + Prediction + Monitoring
</div>
""", unsafe_allow_html=True)

# Configuration des variables d'environnement
try:
    # Utilisation de st.secrets pour Streamlit Cloud
    if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
        OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
        MLFLOW_TRACKING_URI = st.secrets.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    else:
        # Utilisation du fichier .env pour l'environnement local
        from dotenv import load_dotenv
        load_dotenv()
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
        MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
except Exception as e:
    st.warning(f"Echec du chargement des variables d'environnement: {e}")
    OPENAI_API_KEY = ''
    MLFLOW_TRACKING_URI = 'http://localhost:5000'
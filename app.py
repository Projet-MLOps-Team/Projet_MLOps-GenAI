# -*- coding: utf-8 -*-
"""
Outil 4: Application Streamlit avec Monitoring
Auteur: Jiwon
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import os

# Configuration
st.set_page_config(
    page_title="Prédiction Défaut Crédit - MLOps",
    page_icon="🏦",
    layout="wide"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Chargement du modèle
@st.cache_resource
def load_model():
    """Charge le modèle et fichiers"""
    try:
        model = joblib.load('artifacts/best_model.pkl')
        scaler = joblib.load('artifacts/scaler.pkl')
        feature_names = joblib.load('artifacts/feature_names.pkl')
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"❌ Erreur: {e}")
        st.info("💡 Exécutez d'abord: `python train_models.py`")
        st.stop()

@st.cache_data
def load_model_info():
    """Charge info modèle"""
    try:
        with open('artifacts/model_info.txt', 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return "Informations non disponibles"

# Session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
    st.session_state.start_time = datetime.now()

model, scaler, feature_names = load_model()

# En-tête
st.markdown('<div class="main-title"> Système de Prédiction de Défaut de Crédit</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Projet MLOps - Outil 4: Déploiement et Monitoring</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header(" Informations")
    
    # Info modèle
    st.subheader(" Modèle")
    model_info = load_model_info()
    st.text(model_info)
    
    st.markdown("---")
    
    # Statistiques
    st.subheader("📈 Statistiques")
    total = len(st.session_state.predictions)
    
    if total > 0:
        defaults = sum(1 for p in st.session_state.predictions if p['prediction'] == 1)
        no_defaults = total - defaults
        default_rate = defaults / total * 100
        
        st.metric("Total Prédictions", total)
        st.metric("Défauts", defaults)
        st.metric("Pas de Défaut", no_defaults)
        st.metric("Taux de Défaut", f"{default_rate:.1f}%")
        
        # Mini graphique
        fig = go.Figure(data=[go.Pie(
            labels=['Défaut', 'Pas de Défaut'],
            values=[defaults, no_defaults],
            marker_colors=['#ff6b6b', '#51cf66'],
            hole=0.3
        )])
        fig.update_layout(
            height=200,
            margin=dict(t=0, b=0, l=0, r=0),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aucune prédiction")
    
    st.markdown("---")
    st.subheader("🔗 MLflow UI")
    st.code("mlflow ui --port 5000", language="bash")

# Tabs
tab1, tab2, tab3 = st.tabs([" Prédiction", "📈 Monitoring", "📖 Documentation"])

# TAB 1: PRÉDICTION
with tab1:
    st.header("Faire une Prédiction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Informations Client")
        
        credit_lines = st.number_input(
            "Nombre de lignes de crédit",
            min_value=0, max_value=50, value=5,
            help="Nombre total de comptes de crédit actifs"
        )
        
        loan_amt = st.number_input(
            "Montant du prêt en cours (€)",
            min_value=0, max_value=1000000, value=15000, step=1000
        )
        
        total_debt = st.number_input(
            "Dette totale en cours (€)",
            min_value=0, max_value=1000000, value=25000, step=1000
        )
        
        income = st.number_input(
            "Revenu annuel (€)",
            min_value=1000, max_value=1000000, value=60000, step=1000
        )
    
    with col2:
        st.subheader(" Profil Financier")
        
        years_employed = st.number_input(
            "Années d'emploi",
            min_value=0, max_value=50, value=10
        )
        
        fico_score = st.number_input(
            "Score FICO",
            min_value=300, max_value=850, value=720,
            help="Score de crédit entre 300 et 850"
        )
        
        # Calcul debt_ratio
        debt_ratio = total_debt / income if income > 0 else 0
        st.metric(" Ratio d'endettement", f"{debt_ratio:.2%}")
        
        if debt_ratio > 0.4:
            st.warning(" Ratio d'endettement élevé (> 40%)")
        elif debt_ratio > 0.3:
            st.info(" Ratio d'endettement modéré")
        else:
            st.success(" Ratio d'endettement sain")
    
    st.markdown("---")
    
    if st.button(" Prédire le Risque", type="primary", use_container_width=True):
        start_time = time.time()
        
        # Préparation
        features = np.array([[
            credit_lines, loan_amt, total_debt,
            income, years_employed, fico_score, debt_ratio
        ]])
        
        features_scaled = scaler.transform(features)
        
        # Prédiction
        prediction = int(model.predict(features_scaled)[0])
        probability = float(model.predict_proba(features_scaled)[0][1])
        pred_time = time.time() - start_time
        
        # Sauvegarde
        st.session_state.predictions.append({
            'timestamp': datetime.now(),
            'prediction': prediction,
            'probability': probability,
            'time': pred_time,
            'features': {
                'fico': fico_score,
                'debt_ratio': debt_ratio,
                'income': income
            }
        })
        
        # Affichage
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if prediction == 1:
                st.error("###  RISQUE DE DÉFAUT DÉTECTÉ")
                st.markdown(f"**Probabilité de défaut:** `{probability*100:.1f}%`")
                st.markdown(f"**Temps de calcul:** `{pred_time*1000:.2f}ms`")
            else:
                st.success("###  PAS DE RISQUE DE DÉFAUT")
                st.markdown(f"**Probabilité de défaut:** `{probability*100:.1f}%`")
                st.markdown(f"**Temps de calcul:** `{pred_time*1000:.2f}ms`")
        
        with col2:
            # Jauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Risque (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred" if prediction == 1 else "darkgreen"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(t=20, b=0, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

# TAB 2: MONITORING
with tab2:
    st.header("Dashboard de Monitoring")
    
    if len(st.session_state.predictions) > 0:
        df = pd.DataFrame(st.session_state.predictions)
        
        # Métriques
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(" Total", len(df))
        
        with col2:
            avg_prob = df['probability'].mean()
            st.metric(" Prob. Moyenne", f"{avg_prob*100:.1f}%")
        
        with col3:
            avg_time = df['time'].mean() * 1000
            st.metric(" Temps Moyen", f"{avg_time:.2f}ms")
        
        with col4:
            uptime = (datetime.now() - st.session_state.start_time).total_seconds() / 60
            st.metric("⏱️ Uptime", f"{uptime:.0f}min")
        
        st.markdown("---")
        
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📉 Évolution Probabilités")
            fig = px.line(
                df.reset_index(),
                x='index',
                y='probability',
                title='Probabilité de Défaut par Prédiction'
            )
            fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                         annotation_text="Seuil 50%")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Distribution")
            fig = px.histogram(
                df,
                x='probability',
                nbins=20,
                title='Distribution des Probabilités'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Tableau
        st.subheader(" Prédictions Récentes (10 dernières)")
        recent = df.tail(10)[['timestamp', 'prediction', 'probability', 'time']].copy()
        recent['prediction'] = recent['prediction'].map({0: '✅ OK', 1: '⚠️ Défaut'})
        recent['probability'] = recent['probability'].apply(lambda x: f"{x*100:.1f}%")
        recent['time'] = recent['time'].apply(lambda x: f"{x*1000:.2f}ms")
        recent['timestamp'] = recent['timestamp'].dt.strftime('%H:%M:%S')
        st.dataframe(recent, use_container_width=True, hide_index=True)
        
    else:
        st.info(" Aucune prédiction. Allez dans l'onglet 'Prédiction' pour commencer!")

# TAB 3: DOCUMENTATION
with tab3:
    st.header(" Documentation")
    
    st.markdown("""
    ###  Objectif
    Application de prédiction de défaut de crédit utilisant MLOps best practices.
    
    ###  Technologies
    - **ML Framework**: scikit-learn
    - **Experiment Tracking**: MLflow  
    - **Interface**: Streamlit
    - **Deployment**: Docker + GitHub Actions
    
    ###  Modèles Testés
    1. **Logistic Regression** - Baseline simple
    2. **Decision Tree** - Modèle explicable
    3. **Random Forest** - Performance optimale
    
    ###  Features Utilisées
    """)
    
    features_desc = {
        'credit_lines_outstanding': 'Nombre de lignes de crédit actives',
        'loan_amt_outstanding': 'Montant du prêt en cours',
        'total_debt_outstanding': 'Dette totale',
        'income': 'Revenu annuel',
        'years_employed': 'Années d\'emploi',
        'fico_score': 'Score FICO (300-850)',
        'debt_ratio': 'Ratio dette/revenu (calculé)'
    }
    
    for feat, desc in features_desc.items():
        st.markdown(f"- **{feat}**: {desc}")
    
    st.markdown("""
    ###  Utilisation
    1. Entrez les informations du client
    2. Cliquez sur "Prédire"
    3. Consultez le résultat et le monitoring
    
    ###  Métriques MLflow
    Pour voir les expériences:
    ```bash
    mlflow ui --port 5000
    ```
    
    ###  Déploiement Docker
    ```bash
    docker build -t loan-default .
    docker run -p 8501:8501 loan-default
    ```
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p><strong>Projet MLOps</strong> - Outil 3 & 4<br>
    Développé par <strong>Jiwon</strong> | Prédiction de Défaut de Crédit</p>
</div>
""", unsafe_allow_html=True)

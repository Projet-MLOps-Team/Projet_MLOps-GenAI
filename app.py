# -*- coding: utf-8 -*-
"""
Application MLOps End-to-End Complète
Outil 1, 2, 3, 4 - Projet Défaut de Crédit
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(
    page_title="MLOps End-to-End",
    page_icon="🏦",
    layout="wide"
)

# CSS
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

# Session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'training_results' not in st.session_state:
    st.session_state.training_results = None

# Fonctions
@st.cache_data
def load_data():
    """Charge et prépare les données"""
    url = "https://raw.githubusercontent.com/jiwon-yi/Projet_MLOps/main/Loan_Data.csv"
    df = pd.read_csv(url)
    df['debt_ratio'] = df['total_debt_outstanding'] / df['income']
    return df

def train_all_models(df):
    """Entraîne les 3 modèles"""
    with st.spinner("🔥 Entraînement des 3 modèles en cours..."):
        # Préparation
        X = df.drop(['default', 'customer_id'], axis=1)
        y = df['default']
        
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Modèles
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10, class_weight='balanced'),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15, class_weight='balanced')
        }
        
        results = {}
        trained_models = {}
        
        progress_bar = st.progress(0)
        for i, (name, model) in enumerate(models.items()):
            model.fit(X_train_scaled, y_train)
            
            y_pred_val = model.predict(X_val_scaled)
            y_proba_val = model.predict_proba(X_val_scaled)[:, 1]
            
            results[name] = {
                'accuracy': accuracy_score(y_val, y_pred_val),
                'f1': f1_score(y_val, y_pred_val),
                'auc': roc_auc_score(y_val, y_proba_val),
                'precision': precision_score(y_val, y_pred_val),
                'recall': recall_score(y_val, y_pred_val)
            }
            trained_models[name] = model
            progress_bar.progress((i + 1) / len(models))
        
        # Meilleur modèle
        best_name = max(results, key=lambda x: results[x]['f1'])
        best_model = trained_models[best_name]
        
        # Test
        y_pred_test = best_model.predict(X_test_scaled)
        y_proba_test = best_model.predict_proba(X_test_scaled)[:, 1]
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_pred_test),
            'f1': f1_score(y_test, y_pred_test),
            'auc': roc_auc_score(y_test, y_proba_test),
            'precision': precision_score(y_test, y_pred_test),
            'recall': recall_score(y_test, y_pred_test)
        }
        
        # Sauvegarde
        os.makedirs('artifacts', exist_ok=True)
        joblib.dump(best_model, 'artifacts/best_model.pkl')
        joblib.dump(scaler, 'artifacts/scaler.pkl')
        joblib.dump(X.columns.tolist(), 'artifacts/feature_names.pkl')
        
        return results, best_name, test_metrics, scaler, best_model, (len(X_train), len(X_val), len(X_test))

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

# En-tête
st.markdown('<div class="main-title">🏦 Système MLOps End-to-End</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Pipeline Complet: EDA → Preprocessing → Training → Deployment → Monitoring</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Pipeline MLOps")
    st.markdown("""
    ### Étapes:
    - **Outil 1 & 2**: EDA + Preprocessing
    - **Outil 3**: Model Training
    - **Outil 4**: Deployment + Monitoring
    """)
    
    st.markdown("---")
    
    if len(st.session_state.predictions) > 0:
        st.subheader("📈 Stats")
        total = len(st.session_state.predictions)
        defaults = sum(1 for p in st.session_state.predictions if p['prediction'] == 1)
        st.metric("Prédictions", total)
        st.metric("Défauts", defaults)
        st.metric("Taux", f"{defaults/total*100:.1f}%")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Outil 1-2: EDA & Preprocessing",
    "🤖 Outil 3: Training",
    "🔮 Outil 4: Prediction",
    "📈 Monitoring"
])

# ==================== TAB 1: EDA ====================
with tab1:
    st.header("📊 Analyse Exploratoire et Prétraitement")
    
    if st.button("🔄 Charger les Données", type="primary"):
        st.session_state.df = load_data()
        st.success("✅ Données chargées!")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Section 1: Aperçu
        st.subheader("1️⃣ Aperçu des Données")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Lignes", f"{df.shape[0]:,}")
        col2.metric("Colonnes", df.shape[1])
        col3.metric("Défauts", f"{df['default'].sum():,}")
        col4.metric("Taux", f"{df['default'].mean():.1%}")
        
        with st.expander("📋 Voir les données"):
            st.dataframe(df.head(20), use_container_width=True)
        
        # Section 2: Statistiques
        st.subheader("2️⃣ Statistiques Descriptives")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Section 3: Qualité des données
        st.subheader("3️⃣ Qualité des Données")
        col1, col2 = st.columns(2)
        
        with col1:
            nan_count = df.isnull().sum().sum()
            if nan_count == 0:
                st.success(f"✅ Aucune valeur manquante")
            else:
                st.warning(f"⚠️ {nan_count} valeurs manquantes")
        
        with col2:
            dup_count = df.duplicated().sum()
            if dup_count == 0:
                st.success(f"✅ Aucun doublon")
            else:
                st.warning(f"⚠️ {dup_count} doublons")
        
        # Section 4: Visualisations
        st.subheader("4️⃣ Visualisations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig = px.pie(
                df,
                names='default',
                title='Distribution des Défauts',
                color='default',
                color_discrete_map={0: '#51cf66', 1: '#ff6b6b'},
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Histogram FICO
            fig = px.histogram(
                df,
                x='fico_score',
                color='default',
                title='Distribution du Score FICO',
                nbins=30,
                color_discrete_map={0: '#51cf66', 1: '#ff6b6b'},
                barmode='overlay',
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Matrix
        st.subheader("5️⃣ Corrélations")
        corr = df.drop('customer_id', axis=1).corr()
        fig = px.imshow(
            corr,
            text_auto='.2f',
            aspect="auto",
            title="Matrice de Corrélation",
            color_continuous_scale='RdBu',
            zmin=-1, zmax=1
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature Engineering
        st.subheader("6️⃣ Feature Engineering: Debt Ratio")
        st.code("df['debt_ratio'] = df['total_debt_outstanding'] / df['income']")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(
                df,
                x='default',
                y='debt_ratio',
                color='default',
                title='Debt Ratio par Statut',
                color_discrete_map={0: '#51cf66', 1: '#ff6b6b'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                df.sample(1000),
                x='fico_score',
                y='debt_ratio',
                color='default',
                title='FICO vs Debt Ratio',
                color_discrete_map={0: '#51cf66', 1: '#ff6b6b'},
                opacity=0.5
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Preprocessing Summary
        st.subheader("7️⃣ Prétraitement Appliqué")
        st.info("""
        **Étapes de prétraitement:**
        1. ✅ Chargement des données depuis GitHub
        2. ✅ Vérification des valeurs manquantes (0 trouvées)
        3. ✅ Vérification des doublons (0 trouvés)
        4. ✅ Feature engineering: création du debt_ratio
        5. ✅ Normalisation avec StandardScaler
        6. ✅ Division: 60% Train / 20% Val / 20% Test
        """)
    
    else:
        st.info("👆 Cliquez sur 'Charger les Données' pour commencer l'analyse")

# ==================== TAB 2: TRAINING ====================
with tab2:
    st.header("🤖 Entraînement des Modèles")
    
    if st.session_state.df is None:
        st.warning("⚠️ Chargez d'abord les données dans l'onglet EDA")
    else:
        st.subheader("Configuration de l'Entraînement")
        
        st.info("""
        **3 Algorithmes testés:**
        - 🔵 Logistic Regression (baseline)
        - 🟢 Decision Tree (interprétable)
        - 🟣 Random Forest (performance)
        """)
        
        if st.button("🚀 Lancer l'Entraînement", type="primary"):
            results, best_name, test_metrics, scaler, model, split_info = train_all_models(st.session_state.df)
            st.session_state.training_results = {
                'results': results,
                'best_name': best_name,
                'test_metrics': test_metrics,
                'split_info': split_info
            }
            st.success(f"✅ Entraînement terminé! Meilleur modèle: **{best_name}**")
            st.balloons()
        
        if st.session_state.training_results:
            results = st.session_state.training_results['results']
            best_name = st.session_state.training_results['best_name']
            test_metrics = st.session_state.training_results['test_metrics']
            split_info = st.session_state.training_results['split_info']
            
            # Comparaison
            st.subheader("📊 Comparaison des Modèles")
            
            comparison_df = pd.DataFrame(results).T
            comparison_df = comparison_df.round(4)
            comparison_df['Model'] = comparison_df.index
            comparison_df = comparison_df[['Model', 'accuracy', 'precision', 'recall', 'f1', 'auc']]
            
            # Highlight best
            def highlight_best(s):
                is_best = s['Model'] == best_name
                return ['background-color: #d4edda' if is_best else '' for _ in s]
            
            st.dataframe(
                comparison_df.style.apply(highlight_best, axis=1),
                use_container_width=True,
                hide_index=True
            )
            
            # Graphique
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                metrics = ['accuracy', 'f1', 'auc']
                for metric in metrics:
                    fig.add_trace(go.Bar(
                        name=metric.upper(),
                        x=list(results.keys()),
                        y=[results[m][metric] for m in results.keys()]
                    ))
                fig.update_layout(
                    title="Métriques de Validation",
                    barmode='group',
                    yaxis_range=[0, 1]
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Test metrics
                st.markdown(f"### 🏆 {best_name}")
                st.markdown("**Performance sur Test Set:**")
                col_a, col_b = st.columns(2)
                col_a.metric("Accuracy", f"{test_metrics['accuracy']:.2%}")
                col_b.metric("F1-Score", f"{test_metrics['f1']:.2%}")
                col_a.metric("Precision", f"{test_metrics['precision']:.2%}")
                col_b.metric("Recall", f"{test_metrics['recall']:.2%}")
                st.metric("ROC-AUC", f"{test_metrics['auc']:.2%}")
            
            # Split info
            st.subheader("📦 Division des Données")
            col1, col2, col3 = st.columns(3)
            col1.metric("Train", f"{split_info[0]:,}", f"{split_info[0]/(sum(split_info)):.0%}")
            col2.metric("Validation", f"{split_info[1]:,}", f"{split_info[1]/(sum(split_info)):.0%}")
            col3.metric("Test", f"{split_info[2]:,}", f"{split_info[2]/(sum(split_info)):.0%}")

# ==================== TAB 3: PREDICTION ====================
with tab3:
    st.header("🔮 Prédiction de Défaut")
    
    model, scaler, features = load_model()
    
    if model is None:
        st.warning("⚠️ Entraînez d'abord un modèle dans l'onglet Training")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Informations Client")
            credit_lines = st.number_input("Lignes de crédit", 0, 50, 5)
            loan_amt = st.number_input("Montant prêt (€)", 0, 1000000, 15000, 1000)
            total_debt = st.number_input("Dette totale (€)", 0, 1000000, 25000, 1000)
            income = st.number_input("Revenu annuel (€)", 1000, 1000000, 60000, 1000)
        
        with col2:
            st.subheader("💼 Profil Financier")
            years_employed = st.number_input("Années emploi", 0, 50, 10)
            fico_score = st.number_input("Score FICO", 300, 850, 720)
            debt_ratio = total_debt / income
            st.metric("Ratio d'endettement", f"{debt_ratio:.2%}")
        
        if st.button("🔮 Prédire", type="primary", use_container_width=True):
            start = time.time()
            
            features_array = np.array([[credit_lines, loan_amt, total_debt, income, years_employed, fico_score, debt_ratio]])
            features_scaled = scaler.transform(features_array)
            
            prediction = int(model.predict(features_scaled)[0])
            probability = float(model.predict_proba(features_scaled)[0][1])
            pred_time = time.time() - start
            
            st.session_state.predictions.append({
                'timestamp': datetime.now(),
                'prediction': prediction,
                'probability': probability,
                'time': pred_time
            })
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if prediction == 1:
                    st.error("### ⚠️ RISQUE DE DÉFAUT")
                else:
                    st.success("### ✅ PAS DE RISQUE")
                st.info(f"**Probabilité:** {probability*100:.1f}% | **Temps:** {pred_time*1000:.2f}ms")
            
            with col2:
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
                        ]
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 4: MONITORING ====================
with tab4:
    st.header("📈 Monitoring des Prédictions")
    
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
                x='index',
                y='probability',
                title='Évolution des Probabilités'
            )
            fig.add_hline(y=0.5, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                df_pred,
                x='probability',
                nbins=20,
                title='Distribution des Probabilités'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Tableau
        st.subheader("📋 Historique")
        recent = df_pred.tail(10).copy()
        recent['prediction'] = recent['prediction'].map({0: '✅', 1: '⚠️'})
        recent['probability'] = recent['probability'].apply(lambda x: f"{x*100:.1f}%")
        recent['time'] = recent['time'].apply(lambda x: f"{x*1000:.2f}ms")
        recent['timestamp'] = recent['timestamp'].dt.strftime('%H:%M:%S')
        st.dataframe(recent[['timestamp', 'prediction', 'probability', 'time']], use_container_width=True, hide_index=True)
        
    else:
        st.info("📭 Aucune prédiction. Allez dans l'onglet Prediction!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>Projet MLOps End-to-End</strong><br>
    EDA & Preprocessing + Training + Deployment + Monitoring</p>
</div>
""", unsafe_allow_html=True)

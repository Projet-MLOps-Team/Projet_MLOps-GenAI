# -*- coding: utf-8 -*-
"""
Application MLOps - Prédiction de Défaut de Crédit avec Google Gemini AI
Version Corrigée Sans Erreurs

Équipe: 
- Alexandre: EDA + Application
- Patricia & Waï: MLflow + Training
- Jiwon: Application + Déploiement CI/CD

DU Data Analytics - Université Paris 1 Panthéon-Sorbonne
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
from pathlib import Path
import warnings
import os
import requests  # <-- Importé pour Tavily
warnings.filterwarnings('ignore')

# Import Google Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False

warnings.filterwarnings('ignore')

# ========================================
# CONFIGURATION
# ========================================
st.set_page_config(
    page_title="MLOps - Prédiction Défaut Crédit",
    page_icon="🏦",
    layout="wide"
)

# Configuration Tavily
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_AVAILABLE = TAVILY_API_KEY is not None

# Configuration Google Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Import Gemini si disponible
try:
    import google.generativeai as genai
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        # !!! CORRECTION 1: Modèle changé pour plus de stabilité
        # 'gemini-1.5-flash-latest' causait une erreur 404 (v1beta)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash') 
    else:
        gemini_model = None
except:
    gemini_model = None

# ========================================
# SESSION STATE
# ========================================
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

# ========================================
# FONCTIONS PRINCIPALES
# ========================================

# !!! CORRECTION 2: AJOUT DE LA FONCTION TAVILY MANQUANTE
@st.cache_data 
def search_with_tavily(query):
    """Effectue une recherche web avec Tavily"""
    global TAVILY_API_KEY
    
    if not TAVILY_AVAILABLE:
        st.error("❌ Clé API Tavily (TAVILY_API_KEY) non configurée.")
        return None
    
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "basic", # 'basic' pour la rapidité
                "include_answer": True,  # Demande un résumé
                "max_results": 5
            }
        )
        response.raise_for_status() # Lève une exception si erreur HTTP (4xx, 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la recherche Tavily: {e}")
        return None
    except Exception as e:
        st.error(f"Une erreur inattendue est survenue avec Tavily: {e}")
        return None

# ========================================

@st.cache_resource
def load_model():
    """Charge le modèle ML depuis artifacts/"""
    try:
        model_path = Path('artifacts/best_model.joblib')
        if model_path.exists():
            pipeline = joblib.load(model_path)
            return pipeline
        else:
            st.error("❌ Modèle introuvable dans artifacts/best_model.joblib")
            return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None


@st.cache_data
def load_sample_data():
    """Charge les données d'exemple"""
    try:
        if Path('data/Loan_Data.csv').exists():
            df = pd.read_csv('data/Loan_Data.csv')
        else:
            url = "https://raw.githubusercontent.com/jiwon-yi/Projet_MLOps/main/Loan_Data.csv"
            df = pd.read_csv(url)
        
        if 'debt_ratio' not in df.columns:
            # S'assure que la colonne existe si elle n'est pas dans le CSV
            if 'total_debt_outstanding' in df.columns and 'income' in df.columns:
                # Évite la division par zéro
                df['debt_ratio'] = df.apply(lambda row: row['total_debt_outstanding'] / row['income'] if row['income'] > 0 else 0, axis=1)
            else:
                # Crée une colonne fictive si les données sources manquent
                df['debt_ratio'] = 0.0 
                st.warning("Données 'total_debt_outstanding' ou 'income' manquantes pour calculer 'debt_ratio'.")

        return df
    except Exception as e:
        st.warning(f"Données d'exemple non disponibles: {e}")
        return None


def predict_default(pipeline, input_data):
    """Effectue une prédiction de défaut"""
    try:
        pred = pipeline.predict(input_data)[0]
        prob = pipeline.predict_proba(input_data)[0, 1]
        return int(pred), float(prob)
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {e}")
        return None, None


def interpret_fico(score):
    """Interprète un score FICO"""
    if score >= 800:
        return "Exceptionnel", "🟢", "Risque très faible"
    elif score >= 740:
        return "Très bon", "🟢", "Risque faible"
    elif score >= 670:
        return "Bon", "🟡", "Risque modéré"
    elif score >= 580:
        return "Moyen", "🟠", "Risque élevé"
    else:
        return "Faible", "🔴", "Risque très élevé"


def chat_with_gemini(user_message, context=""):
    """Envoie un message à Google Gemini"""
    if not gemini_model:
        return "❌ Chatbot IA non disponible. Veuillez configurer GOOGLE_API_KEY."
    
    try:
        system_prompt = f"""Tu es un assistant expert en analyse de risque crédit pour une banque.
        
Contexte du projet:
- Application MLOps de prédiction de défaut de crédit
- Utilise un modèle Random Forest entraîné sur environ 10,000 dossiers clients
- Variables clés: Score FICO, ratio d'endettement, revenus, ancienneté emploi
- Objectif: Aider les analystes à prendre des décisions d'octroi de prêt

{context}

Réponds de manière professionnelle, claire et en français."""

        full_message = f"{system_prompt}\n\nQuestion utilisateur: {user_message}"
        response = gemini_model.generate_content(full_message)
        
        return response.text
        
    except Exception as e:
        # Renvoie l'erreur spécifique de Gemini à l'utilisateur
        st.error(f"Erreur Gemini: {str(e)}")
        return f"❌ Erreur: {str(e)}"


# ========================================
# HEADER PRINCIPAL
# ========================================
st.title("🏦 Système de Prédiction de Défaut de Crédit")
st.markdown("""
**Plateforme MLOps d'Aide à la Décision pour l'Octroi de Prêts Personnels** *Analyse intelligente du risque de défaut basée sur Machine Learning et IA Générative*
""")
st.markdown("---")

# ========================================
# SIDEBAR
# ========================================
with st.sidebar:
    st.header("📊 Navigation")
    st.markdown("""
    **Sections disponibles:**
    - 📈 Analyse Exploratoire (EDA)
    - 🔮 Prédiction en Temps Réel
    - 💬 Chatbot IA Expert
    - 📊 Historique
    - ℹ️ Documentation
    """)
    
    st.divider()
    
    st.header("👥 Équipe")
    st.markdown("""
    **DU Data Analytics** *Université Paris 1*
    
    - **Alexandre**: EDA + Application
    - **Patricia & Waï**: MLflow + Training
    - **Jiwon**: Application + Déploiement CI/CD
    """)
    
    st.divider()
    
    if st.session_state.predictions:
        st.header("📈 Statistiques")
        st.metric("🔢 Analyses", len(st.session_state.predictions))
        st.metric("⚠️ Défauts", sum(p['prediction'] for p in st.session_state.predictions))
        taux = sum(p['prediction'] for p in st.session_state.predictions) / len(st.session_state.predictions)
        st.metric("📊 Taux Défaut", f"{taux:.1%}")

# ========================================
# ONGLETS PRINCIPAUX
# ========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Analyse EDA",
    "🔮 Prédiction", 
    "💬 Chatbot IA",
    "📊 Historique",
    "ℹ️ Documentation"
])

# ==================== ONGLET 1: ANALYSE EDA ====================
with tab1:
    st.header("📈 Analyse Exploratoire des Données")
    
    df = load_sample_data()
    
    if df is not None:
        st.subheader("📋 Vue d'Ensemble du Portfolio")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📝 Clients Total", f"{df.shape[0]:,}")
        col2.metric("📊 Variables", df.shape[1])
        
        if 'default' in df.columns:
            col3.metric("⚠️ Défauts", f"{df['default'].sum():,}")
            default_rate = df['default'].mean()
            col4.metric("📉 Taux Défaut", f"{default_rate:.1%}")
        else:
            st.warning("Colonne 'default' non trouvée.")

        st.markdown("---")
        
        st.subheader("🎯 Score FICO - Indicateur Principal de Solvabilité")
        
        st.info("""
        **💡 Qu'est-ce que le Score FICO ?**
        
        Le score FICO (Fair Isaac Corporation) est un **score de crédit** évaluant la probabilité qu'une personne 
        rembourse ses dettes. Il varie de **300 à 850 points**.
        
        **Impact direct:** Plus le score est élevé, plus le risque de défaut est faible.
        """)
        
        if 'fico_score' in df.columns and 'default' in df.columns:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = go.Figure()
                
                for status, color, name in [(0, '#2ECC71', '✅ Remboursement régulier'), 
                                            (1, '#E74C3C', '⚠️ Défaut de paiement')]:
                    df_sub = df[df['default'] == status]
                    fig.add_trace(go.Histogram(
                        x=df_sub['fico_score'],
                        name=name,
                        opacity=0.75,
                        marker_color=color,
                        nbinsx=30
                    ))
                
                fig.add_vrect(x0=300, x1=579, fillcolor="red", opacity=0.15, 
                            annotation_text="❌ FAIBLE<br>(Refus probable)", 
                            annotation_position="top left",
                            annotation=dict(font_size=10))
                fig.add_vrect(x0=580, x1=669, fillcolor="orange", opacity=0.15,
                            annotation_text="⚠️ MOYEN<br>(Conditions strictes)", 
                            annotation_position="top left",
                            annotation=dict(font_size=10))
                fig.add_vrect(x0=670, x1=739, fillcolor="yellow", opacity=0.15,
                            annotation_text="✅ BON<br>(Acceptable)", 
                            annotation_position="top left",
                            annotation=dict(font_size=10))
                fig.add_vrect(x0=740, x1=799, fillcolor="lightgreen", opacity=0.15,
                            annotation_text="⭐ TRÈS BON<br>(Favorable)", 
                            annotation_position="top left",
                            annotation=dict(font_size=10))
                fig.add_vrect(x0=800, x1=850, fillcolor="green", opacity=0.15,
                            annotation_text="⭐⭐ EXCEPTIONNEL<br>(Taux préférentiel)", 
                            annotation_position="top left",
                            annotation=dict(font_size=10))
                
                fig.update_layout(
                    title="Distribution des Scores FICO selon Statut de Remboursement",
                    xaxis_title="Score FICO",
                    yaxis_title="Nombre de clients",
                    barmode='overlay',
                    height=450
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**📊 Statistiques Comparatives**")
                
                fico_no_default = df[df['default']==0]['fico_score']
                fico_default = df[df['default']==1]['fico_score']
                
                st.markdown("**✅ Clients sains:**")
                st.metric("Score moyen", f"{fico_no_default.mean():.0f} pts")
                st.metric("Score médian", f"{fico_no_default.median():.0f} pts")
                st.caption(f"Écart-type: {fico_no_default.std():.0f}")
                
                st.markdown("**⚠️ Clients en défaut:**")
                st.metric("Score moyen", f"{fico_default.mean():.0f} pts")
                st.metric("Score médian", f"{fico_default.median():.0f} pts")
                st.caption(f"Écart-type: {fico_default.std():.0f}")
                
                diff = fico_no_default.mean() - fico_default.mean()
                
                st.success(f"""
                **🎯 Insight Clé:**
                
                Écart de **{diff:.0f} points** entre 
                profils sains et à risque.
                
                Le score FICO est **LE prédicteur #1** dans notre modèle ML.
                """)
            
            st.markdown("---")
            
            st.subheader("📉 Taux de Défaut par Tranche de Score FICO")
            
            df['fico_tranche'] = pd.cut(
                df['fico_score'],
                bins=[300, 580, 670, 740, 800, 850],
                labels=['300-579\n(Faible)', '580-669\n(Moyen)', '670-739\n(Bon)', 
                        '740-799\n(Très bon)', '800-850\n(Exceptionnel)']
            )
            
            taux_par_tranche = df.groupby('fico_tranche', observed=True)['default'].agg([
                ('Taux_Defaut', 'mean'),
                ('Nb_Clients', 'count'),
                ('Nb_Defauts', 'sum')
            ]).reset_index()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=taux_par_tranche['fico_tranche'],
                    y=taux_par_tranche['Taux_Defaut'] * 100,
                    marker=dict(
                        color=taux_par_tranche['Taux_Defaut'] * 100,
                        colorscale='RdYlGn_r',
                        showscale=True,
                        colorbar=dict(title="Taux %")
                    ),
                    text=[f"{v:.1f}%" for v in taux_par_tranche['Taux_Defaut'] * 100],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Taux de Défaut (%) par Tranche de Score FICO",
                    xaxis_title="Tranche de Score FICO",
                    yaxis_title="Taux de Défaut (%)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**📈 Analyse par Tranche**")
                st.dataframe(
                    taux_par_tranche.assign(
                        Taux_Defaut=lambda x: x['Taux_Defaut'].apply(lambda v: f"{v:.1%}")
                    )[['fico_tranche', 'Taux_Defaut', 'Nb_Clients', 'Nb_Defauts']],
                    hide_index=True,
                    column_config={
                        "fico_tranche": "Tranche FICO",
                        "Taux_Defaut": "Taux Défaut",
                        "Nb_Clients": "Total Clients",
                        "Nb_Defauts": "Nb Défauts"
                    }
                )
                
                st.warning("""
                **⚠️ Recommandation Bancaire:**
                
                - **< 580:** Refus systématique
                - **580-669:** Conditions strictes
                - **670+:** Conditions standards
                - **740+:** Conditions favorables
                """)
        else:
             st.warning("Données 'fico_score' ou 'default' manquantes pour l'analyse FICO.")
             
        st.markdown("---")
        
        st.subheader("💳 Ratio d'Endettement (Dette / Revenu)")
        
        if 'debt_ratio' in df.columns and 'default' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.box(
                    df, 
                    x='default', 
                    y='debt_ratio',
                    color='default',
                    title="Distribution du Ratio d'Endettement",
                    labels={'default': 'Défaut', 'debt_ratio': 'Ratio (Dette/Revenu)'},
                    color_discrete_map={0: '#2ECC71', 1: '#E74C3C'}
                )
                
                fig.add_hline(y=0.3, line_dash="dash", line_color="orange",
                            annotation_text="⚠️ Seuil 30%")
                fig.add_hline(y=0.4, line_dash="dash", line_color="red",
                            annotation_text="❌ Seuil critique 40%")
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**📊 Statistiques Ratio Endettement**")
                
                debt_ok = df[df['default']==0]['debt_ratio']
                debt_ko = df[df['default']==1]['debt_ratio']
                
                col_a, col_b = st.columns(2)
                col_a.metric("✅ Sans défaut", f"{debt_ok.mean():.1%}")
                col_b.metric("⚠️ Avec défaut", f"{debt_ko.mean():.1%}")
                
                st.success("""
                **💡 Règles d'Or Bancaires:**
                
                - **< 30%:** Situation saine ✅
                - **30-40%:** Vigilance ⚠️
                - **> 40%:** Risque élevé ❌
                - **> 50%:** Surendettement 🚨
                """)
        else:
            st.warning("Données 'debt_ratio' ou 'default' manquantes pour l'analyse d'endettement.")

        st.markdown("---")
        
        st.subheader("🔗 Variables les Plus Corrélées au Défaut")
        
        corr_vars = ['fico_score', 'debt_ratio', 'income', 'total_debt_outstanding', 
                     'years_employed', 'credit_lines_outstanding', 'default']
        
        # Filtre les colonnes pour ne garder que celles qui existent dans le df
        vars_existantes = [v for v in corr_vars if v in df.columns]
        
        if 'default' in vars_existantes:
            corr_with_default = df[vars_existantes].corr()['default'].drop('default').sort_values()
            
            fig = go.Figure(go.Bar(
                x=corr_with_default.values,
                y=corr_with_default.index,
                orientation='h',
                marker=dict(
                    color=corr_with_default.values,
                    colorscale='RdYlGn',
                    cmin=-0.5, cmax=0.5,
                    showscale=True
                ),
                text=[f"{v:.3f}" for v in corr_with_default.values],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Corrélation des Variables avec le Défaut (Pearson)",
                xaxis_title="Coefficient de Corrélation",
                yaxis_title="",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **📌 Interprétation:**
            - **Corrélation négative (rouge):** Plus la variable augmente, moins le risque de défaut est élevé
            - **Corrélation positive (vert):** Plus la variable augmente, plus le risque de défaut est élevé
            - Le **score FICO** a la plus forte corrélation négative → Variable la plus importante
            """)
        else:
            st.warning("Données insuffisantes pour l'analyse de corrélation.")
            
    else:
        st.info("⏳ Chargement des données...")

# ==================== ONGLET 2: PRÉDICTION ====================
with tab2:
    st.header("🔮 Outil de Prédiction de Risque")
    
    pipeline = load_model()
    
    if pipeline is None:
        st.error("❌ Modèle non disponible. Fichier `artifacts/best_model.joblib` requis.")
    else:
        st.success("✅ Modèle ML opérationnel")
        
        st.markdown("---")
        
        with st.expander("📖 Guide d'Utilisation Rapide"):
            st.markdown("""
            ### 🎯 Mode d'Emploi
            
            1. **Remplir** toutes les informations client ci-dessous
            2. **Vérifier** les indicateurs calculés automatiquement  
            3. **Cliquer** sur "Analyser le Risque"
            4. **Interpréter** le résultat et suivre les recommandations
            
            💡 **Cliquez sur les expanders pour obtenir des explications détaillées**
            """)
        
        st.subheader("📝 Informations Client à Saisir")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💳 Informations Financières")
            
            with st.expander("ℹ️ Qu'est-ce qu'une **ligne de crédit** ?"):
                st.markdown("""
                **Définition:** Une ligne de crédit est un compte qui vous permet d'emprunter de l'argent.
                
                **Exemples:**
                - 💳 Cartes bancaires (Visa, Mastercard...)
                - 🏦 Crédits renouvelables (réserve d'argent)
                - 🚗 Prêts automobile
                - 🏠 Prêts immobiliers
                - 🛍️ Crédits magasin (Fnac, Darty...)
                
                **Nombre optimal:** Entre 3 et 7 lignes de crédit
                """)
            
            credit_lines = st.number_input(
                "Nombre de lignes de crédit ouvertes",
                min_value=0, max_value=50, value=5,
                help="Comptez toutes vos cartes et crédits actifs"
            )
            
            with st.expander("ℹ️ **Montant du prêt demandé** - Comment le choisir ?"):
                st.markdown("""
                **C'est quoi ?** La somme d'argent que vous souhaitez emprunter à la banque.
                
                **Exemples d'usage:**
                - 🚗 Achat d'une voiture: 10 000€ - 30 000€
                - 🏠 Travaux immobiliers: 15 000€ - 50 000€
                - 💍 Mariage / Événement: 5 000€ - 20 000€
                """)
            
            loan_amt = st.number_input(
                "Montant du prêt demandé (€)",
                min_value=0, max_value=1000000, value=15000, step=1000,
                help="Somme que vous souhaitez emprunter"
            )
            
            with st.expander("ℹ️ **Dette totale actuelle** - Comment la calculer ?"):
                st.markdown("""
                **C'est quoi ?** La somme de TOUTES vos dettes existantes.
                
                **À inclure dans le calcul:**
                - 🏠 Crédit immobilier restant à payer
                - 🚗 Crédit automobile en cours
                - 💳 Soldes négatifs des cartes bancaires
                - 🛍️ Crédits à la consommation
                """)
            
            total_debt = st.number_input(
                "Dette totale actuelle (€)",
                min_value=0, max_value=1000000, value=25000, step=1000,
                help="Somme de toutes vos dettes actuelles"
            )
        
        with col2:
            st.markdown("#### 👤 Situation Personnelle")
            
            with st.expander("ℹ️ **Revenu annuel** - Que déclarer ?"):
                st.markdown("""
                **C'est quoi ?** Vos revenus AVANT impôts sur une année complète.
                
                **À inclure:**
                - 💼 Salaire brut annuel (x12 mois si mensuel)
                - 💰 Primes et bonus
                - 🏢 Revenus professionnels (si indépendant)
                
                **Exemple:** Salaire mensuel de 3 000€ → 3 000 x 12 = **36 000€/an**
                """)
            
            income = st.number_input(
                "Revenu annuel brut (€)",
                min_value=1000, max_value=1000000, value=60000, step=1000,
                help="Total de vos revenus annuels avant impôts"
            )
            
            with st.expander("ℹ️ **Ancienneté professionnelle** - Pourquoi c'est important ?"):
                st.markdown("""
                **C'est quoi ?** Le nombre d'années dans votre emploi ACTUEL.
                
                **Importance pour la banque:**
                - ⭐ **> 5 ans:** Excellente stabilité financière
                - ✅ **2-5 ans:** Bonne stabilité
                - ⚠️ **1-2 ans:** Acceptable mais surveillé
                - ❌ **< 1 an:** Risque élevé
                """)
            
            years = st.number_input(
                "Ancienneté dans l'emploi actuel (années)",
                min_value=0, max_value=50, value=10,
                help="Nombre d'années dans votre poste actuel"
            )
            
            with st.expander("ℹ️ **Score FICO** - L'indicateur le plus important !"):
                st.markdown("""
                **C'est quoi ?** Votre "note de crédit" qui résume votre fiabilité financière (échelle 300-850).
                
                **Barème d'interprétation:**
                - 800-850: ⭐⭐⭐⭐⭐ Exceptionnel
                - 740-799: ⭐⭐⭐⭐ Très bon
                - 670-739: ⭐⭐⭐ Bon
                - 580-669: ⭐⭐ Moyen
                - 300-579: ⭐ Faible
                """)
            
            fico = st.number_input(
                "Score FICO du client",
                min_value=300, max_value=850, value=720,
                help="Variable LA PLUS IMPORTANTE pour la décision"
            )
        
        debt_ratio = total_debt / income if income > 0 else 0
        
        st.markdown("---")
        st.subheader("📊 Indicateurs Calculés Automatiquement")
        
        col1, col2, col3, col4 = st.columns(4)
        
        debt_color = "🟢" if debt_ratio < 0.3 else "🟡" if debt_ratio < 0.4 else "🔴"
        debt_status = 'Sain' if debt_ratio < 0.3 else 'Acceptable' if debt_ratio < 0.4 else 'Risqué'
        col1.metric("💰 Ratio d'Endettement", f"{debt_ratio:.1%}")
        col1.markdown(f"{debt_color} **{debt_status}**")
        
        fico_cat, fico_emoji, fico_risk = interpret_fico(fico)
        col2.metric("🎯 Catégorie FICO", f"{fico}")
        col2.markdown(f"{fico_emoji} **{fico_cat}**")
        col2.caption(fico_risk)
        
        borrowing_capacity = income * 0.33
        col3.metric("💵 Capacité Emprunt", f"{borrowing_capacity:,.0f} €")
        col3.caption("33% du revenu annuel")
        
        monthly_income = income / 12
        col4.metric("📅 Revenu Mensuel", f"{monthly_income:,.0f} €")
        
        st.markdown("---")
        
        if st.button("🔮 **ANALYSER LE RISQUE DE DÉFAUT**", type="primary", use_container_width=True):
            
            start_time = time.time()
            
            input_df = pd.DataFrame({
                'customer_id': [0], # ID Fictif
                'credit_lines_outstanding': [credit_lines],
                'loan_amt_outstanding': [loan_amt],
                'total_debt_outstanding': [total_debt],
                'income': [income],
                'years_employed': [years],
                'fico_score': [fico],
                'debt_ratio': [debt_ratio]
            })
            
            # S'assure que l'ordre des colonnes correspond au modèle
            # (Cette liste doit correspondre EXACTEMENT à l'entraînement)
            try:
                # Essayez d'obtenir les noms de features depuis le pipeline
                # C'est la méthode la plus robuste si votre pipeline les stocke
                feature_names = pipeline.feature_names_in_
            except AttributeError:
                # Sinon, utilisez une liste codée en dur (moins robuste)
                st.warning("Impossible de récupérer les 'feature_names_in_'. Utilisation d'une liste par défaut.")
                feature_names = ['credit_lines_outstanding', 'loan_amt_outstanding', 
                                 'total_debt_outstanding', 'income', 'years_employed', 
                                 'fico_score', 'debt_ratio']
                # Supprime customer_id qui n'est pas une feature
                input_df = input_df.drop(columns=['customer_id'])


            # Réorganise les colonnes du DataFrame pour correspondre
            try:
                input_data_ordered = input_df[feature_names]
            except KeyError:
                st.error(f"Erreur de feature. Le modèle attend {feature_names} mais a reçu {list(input_df.columns)}")
                input_data_ordered = None

            
            if input_data_ordered is not None:
                prediction, probability = predict_default(pipeline, input_data_ordered)
                pred_time = time.time() - start_time
                
                if prediction is not None:
                    st.session_state.predictions.append({
                        'timestamp': datetime.now(),
                        'prediction': prediction,
                        'probability': probability,
                        'time': pred_time,
                        'fico_score': fico,
                        'debt_ratio': debt_ratio,
                        'income': income,
                        'loan_amt': loan_amt
                    })
                    
                    st.markdown("---")
                    
                    if prediction == 1:
                        st.error("### ⚠️ ALERTE - RISQUE DE DÉFAUT ÉLEVÉ")
                        
                        st.markdown(f"""
                        **🔴 Probabilité de défaut:** {probability*100:.1f}%
                        
                        **🚫 Recommandation:** Refus du prêt ou conditions strictes
                        
                        **📋 Actions Correctives:**
                        - ✅ Demander garanties complémentaires
                        - ✅ Réduire le montant du prêt
                        - ✅ Exiger un co-emprunteur
                        - ✅ Vérification approfondie antécédents
                        - ✅ Majoration du taux d'intérêt
                        """)
                        
                        st.markdown("**⚠️ Facteurs de Risque:**")
                        if fico < 670:
                            st.markdown(f"- 🔴 Score FICO insuffisant ({fico} < 670)")
                        if debt_ratio > 0.4:
                            st.markdown(f"- 🔴 Endettement critique ({debt_ratio:.1%} > 40%)")
                        if years < 2:
                            st.markdown(f"- 🔴 Ancienneté faible ({years} ans < 2 ans)")
                        if loan_amt > borrowing_capacity:
                            st.markdown(f"- 🔴 Montant trop élevé ({loan_amt:,.0f}€ > {borrowing_capacity:,.0f}€)")
                    
                    else:
                        st.success("### ✅ CLIENT ÉLIGIBLE")
                        
                        st.markdown(f"""
                        **🟢 Probabilité de défaut:** {probability*100:.1f}%
                        
                        **✅ Recommandation:** Prêt accordable
                        
                        **📋 Conditions:**
                        - 💰 Taux: Standard/Préférentiel
                        - 💵 Montant max: {borrowing_capacity:,.0f} €
                        - ⏰ Durée: 12-84 mois
                        - 📄 Sans garanties supplémentaires
                        """)
                        
                        st.markdown("**✅ Points Forts:**")
                        if fico >= 740:
                            st.markdown(f"- 🟢 Excellent FICO ({fico} ≥ 740)")
                        if debt_ratio < 0.3:
                            st.markdown(f"- 🟢 Endettement sain ({debt_ratio:.1%} < 30%)")
                        if years >= 5:
                            st.markdown(f"- 🟢 Stabilité professionnelle ({years} ans)")
                        if loan_amt <= borrowing_capacity * 0.8:
                            st.markdown(f"- 🟢 Montant raisonnable")
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=probability * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Probabilité de Défaut (%)", 'font': {'size': 24}},
                        delta={'reference': 50, 'increasing': {'color': "Red"}, 'decreasing': {'color': "Green"}},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkred" if prediction == 1 else "darkgreen"},
                            'steps': [
                                {'range': [0, 20], 'color': "lightgreen"},
                                {'range': [20, 40], 'color': "lightyellow"},
                                {'range': [40, 60], 'color': "orange"},
                                {'range': [60, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {'line': {'color': "red", 'width': 4}, 'value': 50}
                        }
                    ))
                    
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info(f"⏱️ Analyse en {pred_time*1000:.0f} ms")

# ==================== ONGLET 3: CHATBOT IA ====================
with tab3:
    st.header("💬 Assistant IA & Recherche Web")
    
    # Sub-tabs pour les deux fonctionnalités
    subtab1, subtab2 = st.tabs(["🤖 Chatbot IA", "🔍 Recherche Web"])
    
    # SUB-TAB 1: Gemini Chatbot
    with subtab1:
        st.markdown("**Powered by Google Gemini 1.5 Flash** 🤖")
        
        if not gemini_model:
            st.warning("""
            ⚠️ **Chatbot IA non disponible**
            
            Pour activer le chatbot, ajoutez votre clé API Google Gemini:
            1. Obtenez une clé sur https://aistudio.google.com/app/apikey
            2. Ajoutez-la dans Hugging Face Settings → Secrets → `GOOGLE_API_KEY`
            """)
        else:
            st.success("✅ Chatbot opérationnel")
            
            # Affichage de l'historique
            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Zone de saisie
            if prompt := st.chat_input("💬 Posez votre question sur le risque crédit..."):
                st.session_state.chat_messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                context = ""
                if st.session_state.predictions:
                    last = st.session_state.predictions[-1]
                    context = f"""
Contexte de la dernière prédiction:
- Décision: {'DÉFAUT' if last['prediction'] == 1 else 'OK'}
- Probabilité: {last['probability']*100:.1f}%
- FICO: {last['fico_score']}
"""
                
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Réflexion..."):
                        response = chat_with_gemini(prompt, context)
                        if response:
                            st.markdown(response)
                            st.session_state.chat_messages.append({"role": "assistant", "content": response})
                        else:
                            # L'erreur est déjà gérée dans chat_with_gemini
                            pass
            
            if st.button("🔄 Nouvelle Conversation"):
                st.session_state.chat_messages = []
                st.rerun()
    
    # SUB-TAB 2: Tavily Web Search
    with subtab2:
        st.markdown("**Powered by Tavily Search API** 🔍")
        
        if not TAVILY_AVAILABLE:
            st.warning("""
            ⚠️ **Recherche Web non disponible**
            
            Pour activer la recherche Tavily:
            1. Obtenez une clé API sur https://tavily.com
            2. Ajoutez-la dans Hugging Face Settings → Secrets → `TAVILY_API_KEY`
            """)
        else:
            st.success("✅ Recherche web opérationnelle")
            
            st.markdown("""
            **💡 Suggestions de recherche:**
            - Actualités bancaires et crédit immobilier
            - Taux d'intérêt actuels en France
            - Nouvelles réglementations bancaires
            - Tendances du marché du crédit
            """)
            
            search_query = st.text_input(
                "🔎 Rechercher des informations financières:",
                placeholder="Ex: taux crédit immobilier 2024"
            )
            
            if st.button("🔍 Rechercher", type="primary"):
                if search_query:
                    with st.spinner("🔎 Recherche en cours..."):
                        # C'est ici que 'search_with_tavily' est appelée
                        results = search_with_tavily(search_query) 
                        
                        if results and 'results' in results:
                            st.markdown("### 📰 Résultats de recherche")
                            
                            # Résumé si disponible
                            if 'answer' in results and results['answer']:
                                st.info(f"**💡 Résumé:** {results['answer']}")
                                st.markdown("---")
                            
                            # Affichage des résultats
                            for idx, result in enumerate(results['results'][:5], 1):
                                with st.expander(f"📄 {idx}. {result.get('title', 'Sans titre')}"):
                                    st.markdown(f"**🔗 Source:** [{result.get('url', '')}]({result.get('url', '')})")
                                    st.markdown(f"**📝 Extrait:**")
                                    st.markdown(result.get('content', 'Pas de contenu disponible'))
                                    
                                    if 'score' in result:
                                        st.caption(f"Pertinence: {result['score']:.2f}")
                        else:
                            st.error("❌ Aucun résultat trouvé ou erreur de recherche")
                else:
                    st.warning("⚠️ Veuillez saisir une requête de recherche")

        
# ==================== ONGLET 4: HISTORIQUE ====================
with tab4:
    st.header("📊 Historique des Prédictions")
    
    if not st.session_state.predictions:
        st.info("💡 Aucune prédiction. Utilisez l'onglet **🔮 Prédiction** pour commencer.")
    else:
        df_preds = pd.DataFrame(st.session_state.predictions)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🔢 Total", len(df_preds))
        col2.metric("⚠️ Défauts", df_preds['prediction'].sum())
        col3.metric("📊 Taux", f"{df_preds['prediction'].mean():.1%}")
        col4.metric("⏱️ Temps Moy", f"{df_preds['time'].mean()*1000:.0f} ms")
        
        st.markdown("---")
        
        display_df = df_preds.copy()
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['probability'] = (display_df['probability'] * 100).round(1).astype(str) + '%'
        display_df['prediction'] = display_df['prediction'].map({0: '✅ OK', 1: '⚠️ Défaut'})
        
        st.dataframe(display_df[['timestamp', 'prediction', 'probability', 'fico_score']], 
                     use_container_width=True, hide_index=True,
                     column_config={
                         "timestamp": "Date & Heure",
                         "prediction": "Résultat",
                         "probability": "Prob. Défaut",
                         "fico_score": "Score FICO"
                     })
        
        if st.button("💾 Exporter CSV"):
            csv = df_preds.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Télécharger", csv, "historique.csv", "text/csv")

# ==================== ONGLET 5: DOCUMENTATION ====================
with tab5:
    st.header("ℹ️ Documentation Technique")
    
    st.markdown("""
    ## 🎓 Projet MLOps - Prédiction Défaut Crédit
    
    **Formation:** DU Data Analytics  
    **Institution:** Université Paris 1 Panthéon-Sorbonne
    
    ### 🎯 Objectif Business
    
    Prédire la probabilité de défaut de paiement d'un client demandeur de prêt personnel.
    
    ### 📊 Données & Variables
    
    **Dataset:** Environ 10 000 dossiers clients historiques
    
    **Variables prédictives:**
    1. Score FICO (300-850) - **VARIABLE CLÉ**
    2. Ratio d'endettement (Dette/Revenu)
    3. Revenu annuel brut
    4. Ancienneté professionnelle
    5. Nombre de lignes de crédit
    6. Montant du prêt demandé
    7. Dette totale actuelle
    
    ### 🤖 Modèle ML
    
    **Algorithme:** Random Forest Classifier (via `best_model.joblib`)
    **Performance (Exemple):** ~85% accuracy, AUC-ROC ~91%
    **Temps prédiction:** < 100ms
    
    ### 🛠️ Stack Technique
    
    - **ML:** scikit-learn, pandas, numpy, joblib
    - **Tracking:** MLflow (utilisé pour générer `best_model.joblib`)
    - **Interface:** Streamlit + Plotly
    - **IA:** Google Gemini (via `google-generativeai`)
    - **Recherche:** Tavily AI (via `requests`)
    - **DevOps:** GitHub, Hugging Face Spaces (CI/CD)
    
    ### 👥 Équipe
    
    - **Alexandre** - Data Scientist
      - Analyse Exploratoire des Données (EDA)
      - Développement de l'Interface Streamlit
      - Visualisations Plotly et Statistiques
    
    - **Patricia & Waï** - ML Engineers
      - Entraînement & Tuning des Modèles
      - Tracking MLflow et Expérimentations
      - Pipeline de Preprocessing et Feature Engineering
    
    - **Jiwon** - Full Stack & DevOps Engineer
      - Intégration de l'Application Streamlit
      - Déploiement Cloud (Hugging Face)
      - CI/CD (GitHub -> Hugging Face)
      - Intégration API (Gemini, Tavily)
    
    """)

# ========================================
# FOOTER
# ========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>
        🎓 <strong>Projet MLOps - Prédiction Défaut Crédit</strong><br>
        DU Data Analytics - Université Paris 1 Panthéon-Sorbonne<br>
        <br>
        <strong>Équipe:</strong><br>
        Alexandre (EDA + Application) • Patricia & Waï (MLflow + Training) • Jiwon (Application + Déploiement CI/CD)
    </p>
</div>
""", unsafe_allow_html=True)
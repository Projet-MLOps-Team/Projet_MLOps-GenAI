# -*- coding: utf-8 -*-
"""
Outil 3: Entraînement des Modèles avec MLflow
Auteur: Jiwon
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# Configuration MLflow
mlflow.set_experiment("loan_default_prediction")
mlflow.set_tracking_uri("file:./mlruns")

print("\n" + "="*70)
print("OUTIL 3: ENTRAÎNEMENT DES MODÈLES AVEC MLFLOW")
print("="*70)

# Chargement des données
print("\n Chargement des données...")
df = pd.read_csv('Loan_Data.csv')

# Création de debt_ratio (comme dans EDA)
df['debt_ratio'] = df['total_debt_outstanding'] / df['income']

print(f"✓ Données chargées: {df.shape}")
print(f"✓ Taux de défaut: {df['default'].mean():.2%}")

# Séparation features et cible
X = df.drop(['default', 'customer_id'], axis=1)
y = df['default']

print(f"\n📋 Features utilisées:")
for i, col in enumerate(X.columns, 1):
    print(f"  {i}. {col}")

# Division Train/Val/Test (60/20/20)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print(f"\n Division des données:")
print(f"  Train: {len(X_train):>5} échantillons ({len(X_train)/len(X):>5.1%})")
print(f"  Val:   {len(X_val):>5} échantillons ({len(X_val)/len(X):>5.1%})")
print(f"  Test:  {len(X_test):>5} échantillons ({len(X_test)/len(X):>5.1%})")

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Création du dossier artifacts
os.makedirs('artifacts', exist_ok=True)

# Sauvegarde scaler et features
joblib.dump(scaler, 'artifacts/scaler.pkl')
joblib.dump(X.columns.tolist(), 'artifacts/feature_names.pkl')

print(f"✓ Normalisation terminée")

def evaluate_model(y_true, y_pred, y_proba):
    """Calcule les métriques"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba)
    }

def train_model(name, model, params):
    """Entraîne et enregistre avec MLflow"""
    
    with mlflow.start_run(run_name=name):
        print(f"\n{'='*70}")
        print(f" Entraînement: {name}")
        print(f"{'='*70}")
        
        # Log params
        mlflow.log_params(params)
        
        # Entraînement
        model.fit(X_train_scaled, y_train)
        
        # Prédictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_val = model.predict(X_val_scaled)
        y_proba_train = model.predict_proba(X_train_scaled)[:, 1]
        y_proba_val = model.predict_proba(X_val_scaled)[:, 1]
        
        # Métriques
        train_metrics = evaluate_model(y_train, y_pred_train, y_proba_train)
        val_metrics = evaluate_model(y_val, y_pred_val, y_proba_val)
        
        # Log métriques
        for metric_name, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", value)
        
        for metric_name, value in val_metrics.items():
            mlflow.log_metric(f"val_{metric_name}", value)
        
        # Log modèle
        mlflow.sklearn.log_model(model, "model")
        
        # Affichage
        print(f"📈 Train - Acc: {train_metrics['accuracy']:.4f} | "
              f"F1: {train_metrics['f1']:.4f} | "
              f"AUC: {train_metrics['roc_auc']:.4f}")
        print(f"📊 Val   - Acc: {val_metrics['accuracy']:.4f} | "
              f"F1: {val_metrics['f1']:.4f} | "
              f"AUC: {val_metrics['roc_auc']:.4f}")
        
        return model, val_metrics

# Configuration des 3 modèles
print("\n" + "="*70)
print(" ENTRAÎNEMENT DE 3 ALGORITHMES DE CLASSIFICATION")
print("="*70)

models_config = {
    "Logistic_Regression": {
        "model": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ),
        "params": {"model_type": "Logistic Regression", "max_iter": 1000}
    },
    "Decision_Tree": {
        "model": DecisionTreeClassifier(
            random_state=42,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced'
        ),
        "params": {"model_type": "Decision Tree", "max_depth": 10}
    },
    "Random_Forest": {
        "model": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            n_jobs=-1
        ),
        "params": {"model_type": "Random Forest", "n_estimators": 100}
    }
}

# Entraînement
results = {}
trained_models = {}

for name, config in models_config.items():
    trained_model, metrics = train_model(name, config["model"], config["params"])
    results[name] = metrics
    trained_models[name] = trained_model

# Meilleur modèle
best_name = max(results, key=lambda x: results[x]['f1'])
best_model = trained_models[best_name]

print(f"\n{'='*70}")
print(f" MEILLEUR MODÈLE: {best_name}")
print(f"{'='*70}")
print(f"F1-Score (validation): {results[best_name]['f1']:.4f}")
print(f"ROC-AUC (validation):  {results[best_name]['roc_auc']:.4f}")

# Évaluation Test Set
print(f"\n{'='*70}")
print(f" ÉVALUATION SUR L'ENSEMBLE DE TEST")
print(f"{'='*70}")

y_pred_test = best_model.predict(X_test_scaled)
y_proba_test = best_model.predict_proba(X_test_scaled)[:, 1]
test_metrics = evaluate_model(y_test, y_pred_test, y_proba_test)

for metric, value in test_metrics.items():
    print(f"{metric.upper():12s}: {value:.4f}")

print(f"\n Rapport de classification:")
print(classification_report(y_test, y_pred_test, 
                           target_names=['Pas de défaut', 'Défaut']))

# Sauvegarde
joblib.dump(best_model, 'artifacts/best_model.pkl')

with open('artifacts/model_info.txt', 'w', encoding='utf-8') as f:
    f.write(f"Meilleur modèle: {best_name}\n")
    f.write(f"F1-Score (validation): {results[best_name]['f1']:.4f}\n")
    f.write(f"F1-Score (test): {test_metrics['f1']:.4f}\n")
    f.write(f"ROC-AUC (test): {test_metrics['roc_auc']:.4f}\n")

print(f"\n{'='*70}")
print(f" FICHIERS SAUVEGARDÉS DANS artifacts/:")
print(f"   ✓ best_model.pkl")
print(f"   ✓ scaler.pkl")
print(f"   ✓ feature_names.pkl")
print(f"   ✓ model_info.txt")
print(f"\n POUR VOIR LES EXPÉRIENCES MLFLOW:")
print(f"   Commande: mlflow ui --port 5000")
print(f"   URL: http://localhost:5000")
print(f"{'='*70}\n")

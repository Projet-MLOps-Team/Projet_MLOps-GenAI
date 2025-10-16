# src/metrics.py

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, 
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)


import mlflow
import numpy as np
import matplotlib.pyplot as plt
import os


# ──────────────────────────────────────────────────────────────
#  MÉTRIQUES & FIGURES
# ──────────────────────────────────────────────────────────────


def evaluate_model(model, X, y):
    """
    Calcule les métriques de classification clés.
    Log_loss (si proba disponible)
    Roc_auc_ovr (AUC binaire ou multi-classes en one-vs-rest si proba dispo)
    """
    # Vérifie si c'est un modèle scikit-learn (avec .predict et .predict_proba)
    if hasattr(model, 'predict'):
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1_score": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_proba)
        }
    
    else:
        raise TypeError(f"Type de modèle non reconnu: {type(model)}")
    
    return metrics 

def log_metrics(metrics_dict, prefix="val"):
    """
    Enregistre les métriques dans MLflow avec un préfixe donné (ex: 'val_', 'test_').
    """
    for metric_name, value in metrics_dict.items():
        mlflow.log_metric(f"{prefix}_{metric_name}", value)


def log_roc_curve_artifact(model, X, y, model_name): 
    """Génère la courbe ROC sur l'ensemble de Validation et l'enregistre dans MLflow."""
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=180)
    RocCurveDisplay.from_estimator(
        model, X, y, 
        name=f"ROC ({model_name})", 
        ax=ax
    )
    
    # Petites finitions de lisibilité
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("Taux de faux positifs (FPR)")
    ax.set_ylabel("Taux de vrais positifs (TPR)")
    plt.tight_layout()

    #AFTER
    #plt.plot([0, 1], [0, 1], "k--", label="Aléatoire")
    plt.title(f"Courbe ROC - Validation ({model_name})")
    
    filename = f"{model_name}_roc_curve.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    
    mlflow.log_artifact(filename, artifact_path="evaluation_graphs")
        
    os.remove(filename)

def log_confusion_matrix_artifact(model, X, y, model_name):
    cm = confusion_matrix(y, model.predict(X))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                  display_labels=[0, 1])
    
    fig, ax = plt.subplots(figsize=(7, 5), dpi=180)
    disp.plot(ax=ax, values_format="d", colorbar=False, cmap="Blues")
    ax.set_title(f"Matrice de Confusion - Validation ({model_name})")
    plt.tight_layout()
    
    filename = f"{model_name}_confusion_matrix.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)

    mlflow.log_artifact(filename, artifact_path="evaluation_graphs")
        
    os.remove(filename)

def log_precision_recall_curve_artifact(model, X, y, model_name):
    """Génère la courbe Précision-Rappel et l'enregistre dans MLflow."""
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Génération du graphique de Précision-Rappel
    PrecisionRecallDisplay.from_estimator(
        model, X, y,
        name=f"PR ({model_name})",
        ax=ax
    )
    plt.title(f"Courbe Précision-Rappel - Validation ({model_name})")
    
    filename = f"{model_name}_precision_recall_curve.png"
    plt.savefig(filename)
    plt.close(fig)

    mlflow.log_artifact(filename, artifact_path="evaluation_graphs")
    os.remove(filename)


# --- NOUVELLE FONCTION 2 : IMPORTANCE DES CARACTÉRISTIQUES ---

def log_feature_importance_artifact(model, X, model_name):
    """
    Génère le graphique d'importance des caractéristiques et l'enregistre.
    Note : Cette fonction est spécifique aux modèles basés sur les arbres (RF, DT).
    """
    
    # 1. Vérification de l'attribut feature_importances_ (spécifique aux modèles d'arbre)
    if not hasattr(model, 'feature_importances_'):
        print(f"ATTENTION: {model_name} n'a pas l'attribut 'feature_importances_'. Skip.")
        return

    importances = model.feature_importances_
    feature_names = X.columns
    
    # 2. Tri et sélection des 10 premières caractéristiques
    sorted_idx = importances.argsort()[::-1][:10]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_names[sorted_idx][::-1], importances[sorted_idx][::-1], align='center')
    
    plt.xlabel("Importance des Caractéristiques")
    plt.ylabel("Caractéristiques")
    plt.title(f"Top 10 Caractéristiques Importantes ({model_name})")
    plt.tight_layout()
    
    # 3. Sauvegarde et Enregistrement
    filename = f"{model_name}_feature_importance.png"
    plt.savefig(filename)
    plt.close(fig)

    mlflow.log_artifact(filename, artifact_path="evaluation_graphs")
        
    os.remove(filename)
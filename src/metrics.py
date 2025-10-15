# src/metrics.py

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import mlflow

def evaluate_model(model, X, y):
    """
    Calcule les métriques de classification clés.
    """
    # Vérifie si c'est un modèle scikit-learn (avec .predict et .predict_proba)
    if hasattr(model, 'predict'):
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
    
    else:
        raise TypeError(f"Type de modèle non reconnu: {type(model)}")

    # Calcul des métriques
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1_score": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_proba)
    }

def log_metrics(metrics_dict, prefix="val"):
    """
    Enregistre les métriques dans MLflow avec un préfixe donné (ex: 'val_', 'test_').
    """
    for metric_name, value in metrics_dict.items():
        mlflow.log_metric(f"{prefix}_{metric_name}", value)
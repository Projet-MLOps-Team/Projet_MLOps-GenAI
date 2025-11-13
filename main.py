# main.py
import argparse
from src.train_experiment import run_full_experiment

"""
Classification (Régression Logistique, Arbre de Décision, RandomForest)
avec suivi MLflow + gestion du déséquilibre + tuning hyperparamètres
+ AUPRC (PR-AUC) + recherche d'un SEUIL OPTIMAL (max F1 classe minoritaire).

Objectif :
  - Entraîner 3 modèles sur un CSV (features + colonne cible),
  - Gérer le déséquilibre (class_weight + sample_weight),
  - (Optionnel) Chercher de bons hyperparamètres via RandomizedSearchCV,
  - Comparer AUC/F1/Accuracy/log_loss + AUPRC,
  - Chercher un seuil de décision optimisé (sur validation interne) pour la classe positive,
  - Sauvegarder le MEILLEUR pipeline en 'artifacts/best_model.joblib',
  - Logger figures (Confusion Matrix, ROC, Precision-Recall), métriques & tableau comparatif dans MLflow.

  Commande pour lancer : mlflow ui --backend-store-uri mlruns --default-artifact-root mlruns --host 127.0.0.1 --port 5000 
"""

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Lancement du pipeline MLflow.")
    
    # Exécutez l'orchestrateur principal
    run_full_experiment()
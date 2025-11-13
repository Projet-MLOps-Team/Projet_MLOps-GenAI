# --- NOUVELLE FONCTION 3 : APERÇU DES DONNÉES D'ENTRAÎNEMENT ---
import mlflow
import pandas as pd
import os


def log_data_sample_artifact(X_train, y_train):
    """
    Enregistre un échantillon des données d'entraînement (features et target)
    pour la reproductibilité.
    """
    # Combine X et y
    data_sample = X_train.head(50).copy() # Ne garder que les 50 premières lignes
    data_sample['target'] = y_train.head(50)
    
    filename = "training_data_sample.csv"
    data_sample.to_csv(filename, index=False)
    

    mlflow.log_artifact(filename, artifact_path="data_artifacts")
        
    os.remove(filename)

# ---: RAPPORT D'ANALYSE EXPLORATOIRE (EDA) ---

def log_eda_report_artifact(X):
    
    filename = "eda_report.txt"
    with open(filename, "w") as f:
        f.write("--- Rapport EDA Simplifié ---\n")
        f.write(f"Nombre total de lignes: {len(X)}\n")
        f.write(f"Nombre de colonnes: {X.shape[1]}\n")
        f.write("\nStatistiques Clés (Moyenne):\n")
        f.write(X.mean().to_string())
        

    mlflow.log_artifact(filename, artifact_path="data_artifacts")
        
    os.remove(filename)
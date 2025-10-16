import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data():
    data_path = "E:\Formation_Data_Analystic\projet_MLOps_GAI\Projet_MLOps\data\datasetfinal.csv"
    target_col = "default"

    # Charger et préparer les données
    df = pd.read_csv(data_path)
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    return X, y

def split_data(X, y, test_size=0.2, val_size_ratio=0.25, random_state=42):
    """
    Divise les données en ensembles d'entraînement, de validation et de test.
    - X_train_full : Utilisé pour le fit du GridSearch sur LR/RF.
    - X_val : Utilisé pour la sélection des hyperparamètres et le suivi des courbes.
    - X_test : Utilisé UNIQUEMENT pour l'évaluation finale.
    """
    
    # 1. Séparation Train (Full) + Validation vs Test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 2. Séparation Train (Full) en Train et Validation
    # La taille de validation est de val_size_ratio par rapport à X_train_full
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size_ratio, random_state=random_state, stratify=y_train_full
    ) 
    
    
    return X_train, X_val, X_test, y_train, y_val, y_test

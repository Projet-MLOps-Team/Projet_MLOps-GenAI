# src/models.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

def get_models_config():
    """
    Définit les modèles de classification et leurs espaces d'hyperparamètres pour la recherche.
    """
    return {
        "LogisticRegression": {
            "model": LogisticRegression(random_state=42, solver='liblinear'),
            # Hyperparamètres à tester pour la Régression Logistique
            "params": {
                'C': [0.1, 1.0, 10],  # Inverse de la force de régularisation
                'penalty': ['l1', 'l2']
            }
        },
        "RandomForestClassifier": {
            "model": RandomForestClassifier(random_state=42),
            # Hyperparamètres à tester pour le Random Forest
            "params": {
                'n_estimators': [50, 100, 200],  # Nombre d'arbres
                'max_depth': [5, 10, None]       # Profondeur maximale de l'arbre
            }
        },

        "DecisionTreeClassifier": {
            "model": DecisionTreeClassifier(random_state=42),
            # Hyperparamètres à tester pour le Random Forest
            "params": {
                'criterion': ['gini', 'entropy'], 
                'max_depth': [5, 7, None]       # Profondeur maximale de l'arbre
            }
        },
       
    }

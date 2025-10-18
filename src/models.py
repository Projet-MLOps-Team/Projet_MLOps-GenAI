# src/models.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_auc_score, make_scorer

def get_models_config():
    """
    Définit les modèles de classification et leurs espaces d'hyperparamètres pour la recherche.
    """

# --- 2. DÉFINITION DU MODÈLE ET DE LA GRILLE D'HYPERPARAMÈTRES ---

# Métrique de scoring : Utilisation du PR-AUC car plus pertinent en cas de déséquilibre
    pr_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True)
    return {
        "LogisticRegression": {
            "model": LogisticRegression(random_state=42, solver='liblinear'),
            "name":"Logistic_Regression",
            # Hyperparamètres à tester pour la Régression Logistique
            "params": {
                'C': [0.1, 1.0, 10],  
                'penalty': ['l1', 'l2'],
                'class_weight': [None, 'balanced']
            },
            "scoring": pr_auc_scorer # Utilise le PR-AUC pour choisir le meilleur modèle
            
        },
        "RandomForestClassifier": {
            "model": RandomForestClassifier(random_state=42),
            "name":"Random_Forest",
            # Hyperparamètres à tester pour le Random Forest
            "params": {
                'n_estimators': [50, 100, 200],  # Nombre d'arbres
                'max_depth': [5, 10, None],       # Profondeur maximale de l'arbre
                'class_weight': [None, 'balanced']
            }
        },

        "DecisionTreeClassifier": {
            "model": DecisionTreeClassifier(random_state=42),
            "name":"Decision_Tree",
            # Hyperparamètres à tester pour le Random Forest
            "params": {
                'criterion': ['gini', 'entropy'], 
                'max_depth': [5, 7, None],       # Profondeur maximale de l'arbre
                'class_weight': [None, 'balanced'] 
            }
        },
       
    }

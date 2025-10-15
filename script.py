import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# --- 1. Importation du jeu de données ---
data_path = "E:\Formation_Data_Analystic\projet_MLOps_GAI\Projet_MLOps\data\datasetfinal.csv"
target_col = "default"

# Charger et préparer les données
df = pd.read_csv(data_path)
X = df.drop(target_col, axis=1)
y = df[target_col]

# Séparation des ensembles Train (pour l'entraînement et l'optimisation) et Test (pour l'évaluation finale)
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Séparation de l'ensemble d'Entraînement en Train (pour l'ajustement) et Validation (pour l'évaluation du GridSearch)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full) # 0.25 de 0.8 donne 0.2 du total

print(f"Tailles des jeux de données: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")


print("--- Démarrage du Processus MLflow ---")
# --- 2. Configuration MLflow ---

EXPERIMENT_NAME = "Credit_Default_Prediction_Project"
EXPERIMENT_DESCRIPTION = (
    "Comparaison de trois modèles de classification (LogReg, RF, XGBoost) "
    "pour la prédiction de défaut de crédit. Optimisation basée sur l'AUC-ROC de validation. "
    "L'objectif est d'identifier le modèle le plus performant pour la production."
)

try:
    # Crée une nouvelle expérience et utilise le tag 'mlflow.note.content' pour la description
    experiment_id = mlflow.create_experiment(
        name=EXPERIMENT_NAME,
        tags={"mlflow.note.content": EXPERIMENT_DESCRIPTION}
    )
    print(f"Nouvelle expérience créée avec ID: {experiment_id}")
    mlflow.set_experiment(EXPERIMENT_NAME)
except Exception:
    # Si l'expérience existe déjà, on la sélectionne
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"Expérience existante sélectionnée: {EXPERIMENT_NAME}")


# Définition des modèles et des grilles d'hyperparamètres à tester
MODELS_TO_RUN = {
    "LogisticRegression": {
        "model": LogisticRegression(random_state=42, solver='liblinear'),
        "params": {
            'C': [0.1, 1.0, 10],
            'penalty': ['l1', 'l2']
        }
    },
    "RandomForestClassifier": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            'n_estimators': [50, 100],
            'max_depth': [5, 10]
        }
    },
    "XGBoostClassifier": {
        "model": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "params": {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 6]
        }
    }
}

def evaluate_model(model, X, y):
    """Calcule et retourne les métriques de performance."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1_score": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_proba)
    }

# --- 3. Boucle d'Expérimentation ---
best_roc_auc = -1
best_run_id = None
best_model_name = None

for model_name, config in MODELS_TO_RUN.items():
    print(f"\n--- Entraînement et Tuning pour {model_name} ---")

    # Initialisation du GridSearch pour tester les combinaisons d'hyperparamètres
    grid_search = GridSearchCV(
        estimator=config["model"],
        param_grid=config["params"],
        scoring='roc_auc',  # Métrique d'optimisation
        cv=3,  # 3-fold cross-validation sur l'ensemble Train
        verbose=1,
        n_jobs=-1
    )
    
    # Entraînement : le GridSearch utilise X_train pour l'ajustement du modèle et la CV interne
    grid_search.fit(X_train, y_train)

    # Itération sur les résultats de GridSearch pour un suivi détaillé dans MLflow
    for i, params in enumerate(grid_search.cv_results_['params']):
        with mlflow.start_run(run_name=f"{model_name}_Run_{i+1}") as run:
            run_id = run.info.run_id
            
            # --- 3.1. Enregistrer les Hyperparamètres ---
            mlflow.log_params(params)
            
            # --- 3.2. Charger et Évaluer le Modèle Ajusté ---
            # GridSearch a ajusté un modèle pour cette combinaison
            current_model = grid_search.estimator.set_params(**params).fit(X_train, y_train)
            
            # Évaluation sur l'ensemble de Validation
            val_metrics = evaluate_model(current_model, X_val, y_val)
            
            # --- 3.3. Enregistrer les Métriques de Validation ---
            for metric_name, value in val_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", value)
            
            # --- 3.4. Enregistrer le Modèle ---
            # Enregistrement de l'artefact du modèle
            mlflow.sklearn.log_model(
                sk_model=current_model,
                artifact_path="model",
                registered_model_name=f"{model_name}_Credit_Default"
            )

            # Mise à jour du meilleur modèle
            if val_metrics["roc_auc"] > best_roc_auc:
                best_roc_auc = val_metrics["roc_auc"]
                best_run_id = run_id
                best_model_name = model_name
                

print("\n--- 4. Analyse et Évaluation Finale ---")
print(f"Meilleur Modèle (basé sur ROC-AUC de validation): {best_model_name} (Run ID: {best_run_id}) avec ROC-AUC: {best_roc_auc:.4f}")

if best_run_id:
    # Récupérer l'artefact du meilleur modèle
    logged_model_uri = f'runs:/{best_run_id}/model'
    loaded_model = mlflow.sklearn.load_model(logged_model_uri)
    
    # Évaluation finale sur l'ensemble de TEST
    test_metrics = evaluate_model(loaded_model, X_test, y_test)
    
    print("\nMétriques finales sur l'ensemble de TEST:")
    for name, value in test_metrics.items():
        print(f"  {name.upper()}: {value:.4f}")
    
    # Enregistrer les métriques finales dans un nouveau run dédié (ou en tant que tags/métriques finales dans le meilleur run)
    with mlflow.start_run(run_id=best_run_id):
        mlflow.set_tag("Final_Evaluation", "Completed")
        for metric_name, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

    # Marquer le meilleur modèle comme 'Production' dans le Model Registry
    try:
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=f"{best_model_name}_Credit_Default",
            version=1, # Assumons la version 1 pour ce premier enregistrement
            stage="Production"
        )
        print("\nMeilleur modèle transféré vers le stage 'Production' dans le Model Registry.")
    except Exception as e:
        print(f"\nErreur lors du transfert en 'Production' : {e}. Le modèle est déjà enregistré.")

print("\nProcessus terminé. Lancez 'mlflow ui' dans votre terminal pour voir les résultats.")
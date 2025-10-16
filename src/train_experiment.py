# src/train_experiment.py
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import GridSearchCV


# Importation des modules locaux
from src.data_processing import load_data, split_data
from src.models import get_models_config
from src.metrics import (evaluate_model, log_metrics, log_roc_curve_artifact,
                         log_confusion_matrix_artifact, log_precision_recall_curve_artifact, log_feature_importance_artifact)

from src.data_artefacts import log_data_sample_artifact, log_eda_report_artifact

# Définir l'emplacement où MLflow stocke les données
# 'file:./mlruns' indique à MLflow d'utiliser le répertoire local 'mlruns'
mlflow.set_tracking_uri("file:./mlruns")

# Alternativement, utilisez la variable d'environnement (si 'set_tracking_uri' posait un problème d'accès):
# os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"


def run_full_experiment():
    
    # --- 1. Préparation de l'environnement et des données ---

    #Créer ce dossier dans votre workspace 
    #artifacts_dir = Path("./mlflow_artifacts")

    EXPERIMENT_NAME = "Credit_Default_Prediction_Project"
    EXPERIMENT_DESCRIPTION = "Comparaison de LR, RF, et DT avec hyperparamètres variés."
    
    try:
        # Tente de créer l'expérience avec description, sinon la sélectionne
        mlflow.create_experiment(
            name=EXPERIMENT_NAME,
            tags={"mlflow.note.content": EXPERIMENT_DESCRIPTION},
            artifact_location=Path.cwd().joinpath("mlruns").as_uri()
        )
    except Exception:
        pass # L'expérience existe déjà

            
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Chargement et division des données
    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    models_config = get_models_config()

    # SCORE POUR LE MEILLEUR MODELE
    best_roc_auc = -1
    best_run_id = None
    best_model_name = None
    best_model_uri = None

    print(f"Démarrage de l'Expérience MLflow: {EXPERIMENT_NAME}")
    
    # --- 2. Boucle d'Expérimentation ---

    data_artifacts_logged = False

    for model_name, config in models_config.items():

        print(f"\n--- Entraînement et Tuning pour {model_name} ---")
        
        # Utilisation de GridSearchCV pour trouver la meilleure combinaison
        grid_search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            scoring='roc_auc',
            cv=3,
            verbose=0,
            n_jobs=-1
        )
        # Entraîne sur (Train + Validation) pour trouver les meilleurs paramètres
        grid_search.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))


        # Log des meilleurs résultats
        with mlflow.start_run(run_name=f"Best_{model_name}_Run") as run:
            best_model = grid_search.best_estimator_

            mlflow.log_params(grid_search.best_params_)
            
            # LOGS DES METRIQUES NUMERIQUES
            val_metrics = evaluate_model(best_model, X_val, y_val)
            log_metrics(val_metrics, prefix="val")

            # LOGS ARTEFACTS DES DONNEES
            if not data_artifacts_logged:
                print("Enregistrement des artefacts liés aux données (échantillon, rapport EDA)...")
                log_data_sample_artifact(X_train, y_train)
                log_eda_report_artifact(X)
                data_artifacts_logged = True
            
            #LOGS DES ARTEFACTS GRAPHIQUES 
            log_roc_curve_artifact(best_model, X_val, y_val, model_name)
            log_confusion_matrix_artifact(best_model, X_val, y_val, model_name)
            log_precision_recall_curve_artifact(best_model, X_val, y_val, model_name)
            log_feature_importance_artifact(best_model, X_val, model_name)  

            # ENREGISTREMENT DU MODELE
            logged_model = mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                registered_model_name=f"{model_name}_Credit_Default"
            )

            if val_metrics["roc_auc"] > best_roc_auc:
                best_roc_auc = val_metrics["roc_auc"]
                best_run_id = run.info.run_id
                best_model_name = model_name
                best_model_uri = logged_model.model_uri

    # --- 3. Évaluation Finale et Enregistrement du Meilleur Modèle ---
    
    print("\n--- 3. Évaluation Finale ---")
    print(f"Meilleur Modèle (basé sur ROC-AUC de validation): {best_model_name} (Run ID: {best_run_id})")
    

    if best_run_id and best_model_uri:
        # Chargement du meilleur modèle pour l'évaluation finale
        loaded_model = mlflow.sklearn.load_model(best_model_uri)
        
        # EVALUATION DU MODELE AVEC DES VALEURS NON VUS LORS DE SON APPRENTISSAGE
        test_metrics = evaluate_model(loaded_model, X_test, y_test)
        
        print("\nMétriques finales sur l'ensemble de TEST:")
        for name, value in test_metrics.items():
            print(f"  {name.upper()}: {value:.4f}")
        
        # Enregistrer les métriques finales dans le meilleur run
        with mlflow.start_run(run_id=best_run_id):
            mlflow.set_tag("Evaluation_finale", "Completed")
            log_metrics(test_metrics, prefix="test")




        # Marquer le meilleur modèle comme 'Production'
        # try:
        #     client = mlflow.tracking.MlflowClient()
        #     model_name_for_registry = f"{best_model_name}_Credit_Default"
            
        #     # Nous assumons la dernière version du modèle enregistré
        #     versions = client.get_latest_versions(model_name_for_registry) 
        #     if versions:
        #         latest_version = versions[0].version

        #         client.transition_model_version_stage(
        #             name=model_name_for_registry,
        #             version=latest_version,
        #             stage="Production"
        #         )
        #         print(f"\nMeilleur modèle ({model_name_for_registry} V{latest_version}) transféré vers 'Production'.")
        #     else:
        #         print(f"ATTENTION: Aucune version trouvée pour le modèle {model_name_for_registry}. Impossible de le marquer 'Production'.")
        
        # except Exception as e:
        #     print(f"Erreur lors du transfert en 'Production': {e}")
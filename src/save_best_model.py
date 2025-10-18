import mlflow
from mlflow.tracking import MlflowClient

def save_model(best_model_name):
    try:
        client = MlflowClient()
        model_name_for_registry = f"{best_model_name}_Credit_Default"

        # Récupération de la dernière version enregistrée
        versions = client.get_latest_versions(model_name_for_registry)
        if versions:
            latest_version = versions[0].version

            client.set_registered_model_alias(
                name=model_name_for_registry,
                alias="champion",  # alias pour la version en production
                version=latest_version
            )

            print(f"\n✅ Meilleur modèle ({model_name_for_registry} v{latest_version}) assigné à l'alias 'champion' (Production).")

            # ✅ (Optionnel) Chargement du modèle via alias
            model_uri = f"models:/{model_name_for_registry}@champion"
            model = mlflow.pyfunc.load_model(model_uri)
            print("✅ Modèle chargé via alias 'champion'.")

        else:
            print(f"⚠️ Aucune version trouvée pour le modèle '{model_name_for_registry}'. Impossible de le marquer comme 'Production'.")
        
    except mlflow.exceptions.MlflowException as e:
        print(f"❌ Erreur MLflow : {e}")
    except Exception as e:
        print(f"❌ Erreur inattendue : {e}")

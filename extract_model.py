# -*- coding: utf-8 -*-
"""
Script d'Extraction du Meilleur Modèle depuis MLflow
====================================================

Objectif: Extraire le modèle entraîné par Waï depuis MLflow et le sauvegarder
         dans le dossier artifacts/ pour utilisation par l'application Streamlit.

Équipe: Alexandre, Patricia, Waï, Jiwon
Université Paris 1 Panthéon-Sorbonne
"""

import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Run ID fourni par Waï (modèle en production sur MLflow)
RUN_ID = "1da74295cd0b42b28feb9289314db4f8"

# Chemin du tracking MLflow (adapter selon votre configuration)
MLFLOW_TRACKING_URI = "file:///C:/Users/infor/OneDrive/문서/GitHub/Projet_MLOps-GenAI/mlruns"

# Dossier de sortie pour le modèle
OUTPUT_DIR = "artifacts"
OUTPUT_FILE = "best_model.joblib"

# =============================================================================
# EXTRACTION DU MODÈLE
# =============================================================================

def extraire_modele():
    """
    Extrait le modèle depuis MLflow et le sauvegarde localement.
    
    Returns:
        bool: True si succès, False si échec
    """
    
    print("=" * 60)
    print("🔍 EXTRACTION DU MODÈLE DEPUIS MLFLOW")
    print("=" * 60)
    
    try:
        # Configuration de l'URI de tracking MLflow
        print(f"\n📍 Configuration MLflow Tracking URI...")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        print(f"   URI: {MLFLOW_TRACKING_URI}")
        
        # Chargement du modèle depuis MLflow
        print(f"\n🔄 Chargement du modèle (Run ID: {RUN_ID})...")
        model_uri = f"runs:/{RUN_ID}/model"
        model = mlflow.sklearn.load_model(model_uri)
        
        print(f"✅ Modèle chargé avec succès!")
        print(f"   Type: {type(model).__name__}")
        
        # Création du dossier artifacts s'il n'existe pas
        print(f"\n📁 Création du dossier de sortie...")
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(exist_ok=True)
        print(f"   Dossier: {output_path.absolute()}")
        
        # Sauvegarde du modèle au format joblib
        print(f"\n💾 Sauvegarde du modèle...")
        full_output_path = output_path / OUTPUT_FILE
        joblib.dump(model, full_output_path)
        
        print(f"✅ Modèle sauvegardé avec succès!")
        print(f"   Fichier: {full_output_path.absolute()}")
        
        # Vérification de la taille du fichier
        file_size = full_output_path.stat().st_size / (1024 * 1024)  # Convertir en MB
        print(f"   Taille: {file_size:.2f} MB")
        
        print("\n" + "=" * 60)
        print("🎉 EXTRACTION RÉUSSIE!")
        print("=" * 60)
        print("\n📋 Prochaines étapes:")
        print("   1. Vérifier le fichier: dir artifacts\\best_model.joblib")
        print("   2. Lancer l'application: streamlit run app_streamlit_mlops_gemini.py")
        print("   3. Tester la prédiction dans l'onglet 🔮 Prédiction")
        print("   4. Commit sur GitHub: git add artifacts/best_model.joblib")
        
        return True
        
    except mlflow.exceptions.MlflowException as e:
        print(f"\n❌ ERREUR MLFLOW: {e}")
        print("\n🔧 Solutions possibles:")
        print("   1. Vérifier que MLflow UI est accessible: mlflow ui")
        print("   2. Vérifier le Run ID fourni par Waï")
        print("   3. Vérifier que le dossier mlruns/ existe")
        return False
        
    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE: {e}")
        print(f"   Type d'erreur: {type(e).__name__}")
        return False


def afficher_info_modele():
    """
    Affiche les informations détaillées du modèle depuis MLflow.
    """
    
    try:
        print("\n" + "=" * 60)
        print("📊 INFORMATIONS DU MODÈLE")
        print("=" * 60)
        
        # Configuration MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Récupération des informations du run
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(RUN_ID)
        
        # Affichage des informations générales
        print(f"\n🆔 Run ID: {RUN_ID}")
        print(f"📅 Date: {run.info.start_time}")
        print(f"⏱️  Durée: {(run.info.end_time - run.info.start_time) / 1000:.2f}s")
        print(f"✅ Statut: {run.info.status}")
        
        # Affichage des métriques
        if run.data.metrics:
            print(f"\n📈 Métriques de Performance:")
            for key, value in sorted(run.data.metrics.items()):
                print(f"   • {key}: {value:.4f}")
        
        # Affichage des paramètres
        if run.data.params:
            print(f"\n⚙️  Hyperparamètres:")
            for key, value in sorted(run.data.params.items()):
                print(f"   • {key}: {value}")
        
    except Exception as e:
        print(f"\n⚠️  Impossible d'afficher les informations: {e}")


# =============================================================================
# EXÉCUTION PRINCIPALE
# =============================================================================

if __name__ == "__main__":
    """
    Point d'entrée principal du script.
    """
    
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "EXTRACTION MODÈLE MLFLOW" + " " * 24 + "║")
    print("║" + " " * 10 + "Projet MLOps - Paris 1" + " " * 26 + "║")
    print("╚" + "═" * 58 + "╝")
    
    # Affichage des informations du modèle
    afficher_info_modele()
    
    print("\n")
    
    # Extraction du modèle
    success = extraire_modele()
    
    # Code de sortie
    exit(0 if success else 1)

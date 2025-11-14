# -*- coding: utf-8 -*-
"""
Script d'Extraction du Meilleur Modèle depuis MLflow
Équipe: Alexandre, Patricia, Waï, Jiwon
Université Paris 1 Panthéon-Sorbonne
"""

import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path

# Configuration
RUN_ID = "1da74295cd0b42b28feb9289314db4f8"
MLFLOW_TRACKING_URI = "file:///C:/Users/infor/OneDrive/문서/GitHub/Projet_MLOps-GenAI/mlruns"
OUTPUT_DIR = "artifacts"
OUTPUT_FILE = "best_model.joblib"

print("=" * 60)
print("🔍 EXTRACTION DU MODÈLE DEPUIS MLFLOW")
print("=" * 60)

try:
    # Configuration MLflow
    print(f"\n📍 Configuration MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Chargement du modèle
    print(f"\n🔄 Chargement du modèle (Run ID: {RUN_ID})...")
    model_uri = f"runs:/{RUN_ID}/model"
    model = mlflow.sklearn.load_model(model_uri)
    
    print(f"✅ Modèle chargé avec succès!")
    print(f"   Type: {type(model).__name__}")
    
    # Création du dossier
    print(f"\n📁 Création du dossier artifacts...")
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    # Sauvegarde
    print(f"\n💾 Sauvegarde du modèle...")
    full_output_path = output_path / OUTPUT_FILE
    joblib.dump(model, full_output_path)
    
    print(f"✅ Modèle sauvegardé avec succès!")
    print(f"   Fichier: {full_output_path.absolute()}")
    
    file_size = full_output_path.stat().st_size / (1024 * 1024)
    print(f"   Taille: {file_size:.2f} MB")
    
    print("\n" + "=" * 60)
    print("🎉 EXTRACTION RÉUSSIE!")
    print("=" * 60)
    print("\n📋 Prochaines étapes:")
    print("   1. Vérifier: dir artifacts\\best_model.joblib")
    print("   2. Lancer l'app: streamlit run app_streamlit_mlops_gemini.py")
    
except Exception as e:
    print(f"\n❌ ERREUR: {e}")
    print("\n🔧 Vérifier:")
    print("   1. MLflow UI accessible: mlflow ui")
    print("   2. Run ID correct")
    print("   3. Dossier mlruns/ existe")

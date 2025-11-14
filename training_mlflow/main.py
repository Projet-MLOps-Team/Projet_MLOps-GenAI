# main.py
import argparse
from src.train_experiment import run_full_experiment

"""
  Commande pour lancer : mlflow ui --backend-store-uri mlruns --default-artifact-root mlruns --host 127.0.0.1 --port 5000 
"""

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Lancement du pipeline MLflow.")
    # Exécutez l'orchestrateur principal
    run_full_experiment()
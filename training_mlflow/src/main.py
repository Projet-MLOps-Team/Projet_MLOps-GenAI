# main.py
import argparse
from train_experiment import run_full_experiment

"""
EXECUTER 1 : 
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000

EXECUTER 2 : 
python .\main.py
"""

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Lancement du pipeline MLflow.")
    
    # Exécutez l'orchestrateur principal
    run_full_experiment()
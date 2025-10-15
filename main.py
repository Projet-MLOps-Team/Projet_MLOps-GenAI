# main.py
import argparse
from src.train_experiment import run_full_experiment

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Lancement du pipeline MLflow.")
    # Ajoutez ici des arguments si nécessaire (ex: chemin des données)
    
    # Exécutez l'orchestrateur principal
    run_full_experiment()
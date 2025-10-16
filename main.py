# main.py
import argparse
from src.train_experiment import run_full_experiment

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Lancement du pipeline MLflow.")
    
    # Exécutez l'orchestrateur principal
    run_full_experiment()
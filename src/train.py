#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entraîner 3 modèles de classification (Régression Logistique, Arbre de Décision,
RandomForest) avec suivi MLflow, comparer les métriques et sauvegarder le meilleur
pipeline en 'artifacts/best_model.joblib'.

USAGE (exemples) :
  python src/train.py --csv data/datasetfinal.csv --target NOM_CIBLE
  python src/train.py --csv data/datasetfinal.csv                # cible = dernière colonne

Prérequis (requirements.txt) :
  pandas
  numpy
  scikit-learn>=1.2
  mlflow
  joblib
  matplotlib
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

# Préprocesseurs & modèles
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Métriques & courbes
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, log_loss,
    confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay
)

# Les 3 modèles demandés
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
# 1) UTILITAIRES
# ──────────────────────────────────────────────────────────────
def split_features(df: pd.DataFrame, target: str):
    """
    Sépare X (features) et y (cible) + liste les colonnes numériques et catégorielles.
    - X : toutes les colonnes sauf la cible
    - y : colonne cible
    - num_cols : colonnes numériques
    - cat_cols : colonnes non numériques (objets/strings, bool, catégories)
    """
    X = df.drop(columns=[target])
    y = df[target]
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    return X, y, num_cols, cat_cols


def build_preprocess(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    Crée un pré-traitement :
      - Numérique : imputation médiane + standardisation
      - Catégoriel : imputation (modalité la plus fréquente) + One-Hot Encoding
    On retourne un ColumnTransformer qui sera intégré dans le Pipeline sklearn.
    """
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # sparse_output=False pour renvoyer un numpy array dense (plus simple pour le modèle)
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preproc = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols)
        ],
        remainder="drop"  # on ne garde que num_cols + cat_cols
    )
    return preproc


# ──────────────────────────────────────────────────────────────
# 2) MÉTRIQUES & FIGURES
# ──────────────────────────────────────────────────────────────
def metrics_classification(y_true, y_pred, y_prob=None) -> Dict[str, float]:
    """
    Calcule un set de métriques utiles pour comparer les modèles :
      - accuracy
      - f1_weighted (gère le déséquilibre entre classes)
      - log_loss (si proba disponible)
      - roc_auc_ovr (AUC binaire ou multi-classes en one-vs-rest si proba dispo)
    """
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted"))
    }
    # log_loss & AUC demandent des probabilités (ou un score de décision)
    if y_prob is not None:
        # log_loss (plus petit = meilleur)
        try:
            m["log_loss"] = float(log_loss(y_true, y_prob, labels=np.unique(y_true)))
        except Exception:
            pass

        # ROC AUC : binaire -> AUC standard ; multi-classes -> AUC OvR
        try:
            if hasattr(y_prob, "shape") and len(y_prob.shape) == 2 and y_prob.shape[1] > 2:
                # multi-classes
                m["roc_auc_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
            else:
                # binaire : prendre la probabilité de la classe positive (colonne 1)
                pos = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
                m["roc_auc_ovr"] = float(roc_auc_score(y_true, pos))
        except Exception:
            pass
    return m


def selection_score(m: Dict[str, float]) -> float:
    """
    Score unique pour sélectionner le "meilleur" modèle.
    Stratégie :
      1) on privilégie AUC (si disponible),
      2) sinon F1_weighted,
      3) sinon Accuracy.
    (Plus grand = meilleur)
    """
    if "roc_auc_ovr" in m and not np.isnan(m["roc_auc_ovr"]):
        return m["roc_auc_ovr"]
    if "f1_weighted" in m and not np.isnan(m["f1_weighted"]):
        return m["f1_weighted"]
    return m.get("accuracy", -1.0)

from pathlib import Path

def save_confusion_matrix(y_true, y_pred, class_names, out_path: Path):
    """
    Génère une matrice de confusion en **PNG** lisible (grande taille + haute résolution).
    - y_true / y_pred : vraies/pseudo étiquettes du jeu de test
    - class_names : liste des étiquettes (pour afficher les labels)
    - out_path : chemin de sortie (on force .png au cas où)
    """
    # On s’assure d’avoir l’extension .png
    out_path = out_path.with_suffix(".png")

    # Utiliser un backend non-GUI pour éviter les erreurs Tkinter ("main loop")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Calcul de la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)

    # figsize + dpi => plus net et plus grand (visible dans MLflow / Explorateur Windows)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=180)

    # Affichage avec labels + valeurs entières
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, values_format="d", colorbar=False, cmap="Blues")
    ax.set_title("Matrice de confusion")
    plt.tight_layout()

    # Sauvegarde en PNG (bbox_inches='tight' évite les grands marges blanches)
    fig.savefig(out_path, bbox_inches="tight")

    # Libère la figure (important pour les scripts batch)
    plt.close(fig)


def save_roc_curves(y_true, y_prob, classes, out_path: Path):
    """
    Enregistre la/les courbes ROC en **PNG**.
    - Binaire : une seule courbe ROC.
    - Multi-classe : One-vs-Rest (une courbe par classe) si y_prob a la bonne forme.

    y_prob peut venir de predict_proba (shape = [n, n_classes]) ou decision_function.
    """
    # Sans probas, on ne peut pas tracer la ROC
    if y_prob is None:
        return

    out_path = out_path.with_suffix(".png")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay
    from sklearn.preprocessing import label_binarize
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 6), dpi=180)

    # Cas **binaire** : on trace une seule courbe
    if len(classes) == 2:
        # Si y_prob est 2D (n,2), on prend la colonne 1 = proba de la classe positive
        pos = y_prob if getattr(y_prob, "ndim", 1) == 1 else y_prob[:, 1]
        RocCurveDisplay.from_predictions(y_true, pos, ax=ax, name="ROC")
        ax.set_title("Courbe ROC (binaire)")

    else:
        # Cas **multi-classe** : on binarise y_true (One-vs-Rest)
        y_bin = label_binarize(y_true, classes=classes)

        # Vérifie la compatibilité des dimensions (ex : y_prob.shape[1] == nb classes)
        if hasattr(y_prob, "shape") and y_prob.shape[1] == y_bin.shape[1]:
            for i, cls in enumerate(classes):
                RocCurveDisplay.from_predictions(y_bin[:, i], y_prob[:, i], ax=ax, name=str(cls))
            ax.set_title("Courbes ROC (multiclasse, OvR)")
        else:
            # Si dimensions incompatibles, on ne trace rien (cas rare)
            plt.close(fig)
            return

    # Petites finitions de lisibilité
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("Taux de faux positifs (FPR)")
    ax.set_ylabel("Taux de vrais positifs (TPR)")
    plt.tight_layout()

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

# ──────────────────────────────────────────────────────────────
# 3) ENTRAÎNER UN MODÈLE (1 RUN MLflow NESTED)
# ──────────────────────────────────────────────────────────────
def train_one(
    name: str,
    estimator,
    preproc: ColumnTransformer,
    X_train, X_test, y_train, y_test,
    artifacts_dir: Path,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Entraîne un pipeline (prétraitement + modèle), calcule les métriques,
    enregistre figures & artefacts, et loggue tout dans un run MLflow imbriqué.
    """
    with mlflow.start_run(run_name=name, nested=True):
        # 1) Pipeline complet = Preprocess (num+cat) -> Modèle
        pipe = Pipeline(steps=[
            ("preprocess", preproc),
            ("model", estimator)
        ])
        pipe.fit(X_train, y_train)

        # 2) Prédictions & probabilités
        y_pred = pipe.predict(X_test)
        y_prob = None
        try:
            y_prob = pipe.predict_proba(X_test)  # marche pour LogReg / RF / DT si 'predict_proba' dispo
        except Exception:
            try:
                # Certains modèles exposent decision_function (pas besoin ici mais on tente)
                y_prob = pipe.decision_function(X_test)
            except Exception:
                pass

        # 3) Métriques
        mets = metrics_classification(y_test, y_pred, y_prob)

        # 4) Log params / metrics dans MLflow
        mlflow.log_params({"model_name": name})
        for k, v in mets.items():
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                mlflow.log_metric(k, float(v))

        # 5) Figures (matrice de confusion + ROC)
        model_dir = artifacts_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        cm_path = model_dir / "confusion_matrix.png"
        save_confusion_matrix(y_test, y_pred, class_names, cm_path)
        if cm_path.exists():
            mlflow.log_artifact(str(cm_path))

        classes = list(np.unique(y_test))
        roc_path = model_dir / ("roc.png" if len(classes) == 2 else "roc_multiclass.png")
        save_roc_curves(y_test, y_prob, classes, roc_path)
        if roc_path.exists():
            mlflow.log_artifact(str(roc_path))

        # 6) Log du pipeline entraîné (utile pour rechargement via MLflow)
        mlflow.sklearn.log_model(pipe, artifact_path=f"model_{name}")

        return {"name": name, "pipeline": pipe, "metrics": mets}


# ──────────────────────────────────────────────────────────────
# 4) MAIN : CHARGER DONNÉES, LANCER LES 3 MODÈLES, COMPARER, SAUVER LE MEILLEUR
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Classification : LogReg, DecisionTree, RandomForest + MLflow.")
    parser.add_argument("--csv", type=str, required=True, help="Chemin du CSV (ex: data/datasetfinal.csv)")
    parser.add_argument("--target", type=str, default=None, help="Nom de la colonne cible (sinon dernière colonne)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction du test (ex: 0.2 = 20%)")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire pour reproductibilité")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Dossier des sorties/artefacts")
    parser.add_argument("--experiment", type=str, default="mlops-training", help="Nom d'expérience MLflow")
    parser.add_argument("--mlflow-uri", type=str, default="mlruns", help="URI de tracking MLflow (dossier local ou serveur)")
    args = parser.parse_args()

    # 1) Initialiser MLflow (emplacement des runs + expérience)
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    # 2) Charger le CSV et définir la cible
    df = pd.read_csv(args.csv)
    target = args.target or df.columns[-1]  # si non fourni, on prend la dernière colonne
    if target not in df.columns:
        raise ValueError(f"Colonne cible '{target}' introuvable dans {args.csv}.")

    # 3) Séparer X/y et typer les colonnes
    X, y, num_cols, cat_cols = split_features(df, target)
    class_names = list(np.unique(y))

    # 4) Split train/test (stratifié = indispensable pour classification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # 5) Prétraitement (imputation + encodage cat + standardisation)
    preproc = build_preprocess(num_cols, cat_cols)

    # 6) Définir les 3 modèles demandés (avec quelques params raisonnables)
    models = {
        # Régression Logistique (multiclasse = 'auto' -> OvR ou multinomial selon solver/classes)
        "logreg": LogisticRegression(
            max_iter=500,
            # 'lbfgs' gère multinomial ; C=1.0 par défaut ; class_weight=None (ajuste selon besoin)
            solver="lbfgs"
        ),
        # Arbre de décision (limites de profondeur pour éviter l'overfit)
        "decision_tree": DecisionTreeClassifier(
            random_state=args.seed,
            max_depth=None,              # tu peux fixer p.ex. 10 si overfit
            min_samples_leaf=1
        ),
        # Forêt aléatoire (plus robuste que l'arbre simple)
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=args.seed,
            n_jobs=-1
        )
    }

    # 7) Créer le dossier des artefacts
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 8) Lancer un run MLflow global puis 3 runs imbriqués (un par modèle)
    results = []
    with mlflow.start_run(run_name="train_classification"):
        # On évite l'autolog des modèles pour garder un logging "propre" et contrôlé
        mlflow.sklearn.autolog(log_models=False)

        for name, est in models.items():
            res = train_one(
                name=name,
                estimator=est,
                preproc=preproc,
                X_train=X_train, X_test=X_test,
                y_train=y_train, y_test=y_test,
                artifacts_dir=artifacts_dir,
                class_names=class_names
            )
            print(f"[{name}] -> {res['metrics']}")
            results.append(res)

        # 9) Comparaison et sélection du meilleur
        rows = []
        best = None
        best_score = -np.inf
        for r in results:
            m = r["metrics"]
            score = selection_score(m)
            rows.append({"model": r["name"], **m, "selection_score": score})
            if score > best_score:
                best_score = score
                best = r

        # Sauvegarder un tableau comparatif pour traçabilité
        comp_df = pd.DataFrame(rows).sort_values("selection_score", ascending=False)
        comp_csv = artifacts_dir / "models_comparison.csv"
        comp_df.to_csv(comp_csv, index=False)
        mlflow.log_artifact(str(comp_csv))

        # 10) Sauvegarder le MEILLEUR pipeline (préprocess + modèle) en local + MLflow
        best_path = artifacts_dir / "best_model.joblib"
        joblib.dump(best["pipeline"], best_path)
        mlflow.log_artifact(str(best_path))

        # 11) Log d'infos utiles
        mlflow.log_params({
            "best_model": best["name"],
            "target": target,
            "num_features": len(num_cols),
            "cat_features": len(cat_cols),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        })

        print("\n=== RÉSUMÉ ===")
        print(f"Cible : {target}")
        print("Comparaison des modèles :\n", comp_df.to_string(index=False))
        print(f"\nMeilleur modèle : {best['name']} -> sauvegardé : {best_path}")


if __name__ == "__main__":
    main()

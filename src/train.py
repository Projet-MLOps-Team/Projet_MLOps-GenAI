#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification (Régression Logistique, Arbre de Décision, RandomForest)
avec suivi MLflow + gestion du déséquilibre + tuning hyperparamètres
+ AUPRC (PR-AUC) + recherche d'un SEUIL OPTIMAL (max F1 classe minoritaire).

Objectif :
  - Entraîner 3 modèles sur un CSV (features + colonne cible),
  - Gérer le déséquilibre (class_weight + sample_weight),
  - (Optionnel) Chercher de bons hyperparamètres via RandomizedSearchCV,
  - Comparer AUC/F1/Accuracy/log_loss + AUPRC,
  - Chercher un seuil de décision optimisé (sur validation interne) pour la classe positive,
  - Sauvegarder le MEILLEUR pipeline en 'artifacts/best_model.joblib',
  - Logger figures (Confusion Matrix, ROC, Precision-Recall), métriques & tableau comparatif dans MLflow.

USAGE (exemples) :
  # entraînement simple (déséquilibre géré automatiquement)
  python src/train.py --csv data/datasetfinal.csv --target default --mlflow-uri mlruns

  # avec tuning d'hyperparamètres (5 folds)
  python src/train.py --csv data/datasetfinal.csv --target default --mlflow-uri mlruns --tune --cv-splits 5

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
from typing import Dict, Any, List, Optional, Tuple
import warnings

# Base scientifique & I/O
import numpy as np
import pandas as pd
import joblib

# MLflow (tracking)
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Prétraitement & modèles sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV
)

# Métriques
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, log_loss,
    average_precision_score, precision_recall_curve
)
from sklearn.utils.class_weight import compute_sample_weight

# Modèles
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")  # pour garder la console "propre"


# ──────────────────────────────────────────────────────────────
# 1) UTILITAIRES (split, prétraitement, métriques, figures)
# ──────────────────────────────────────────────────────────────
def split_features(df: pd.DataFrame, target: str):
    """
    Sépare X (features) et y (cible) + liste les colonnes numériques et catégorielles.
    - X : toutes les colonnes sauf la cible
    - y : colonne cible
    """
    X = df.drop(columns=[target])
    y = df[target]
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    return X, y, num_cols, cat_cols


def build_preprocess(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    Pré-traitement :
      - Numérique : imputation médiane + standardisation
      - Catégoriel : imputation (modalité la plus fréquente) + One-Hot Encoding
    Retour : ColumnTransformer (sera le step 'preprocess' du Pipeline sklearn).
    """
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # sparse_output=False -> retourne un array dense (plus simple pour les modèles)
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


def metrics_classification(y_true, y_pred, y_prob=None) -> Dict[str, float]:
    """
    Calcule un set de métriques utiles pour comparer les modèles :
      - accuracy
      - f1_weighted (pondère selon la fréquence des classes)
      - log_loss (si proba disponible)
      - roc_auc_ovr (AUC binaire ou multi-classes OvR si proba dispo)
      - auprc (Average Precision / PR-AUC) si binaire & proba dispo
    """
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted"))
    }
    if y_prob is not None:
        # log_loss (plus petit = meilleur)
        try:
            m["log_loss"] = float(log_loss(y_true, y_prob, labels=np.unique(y_true)))
        except Exception:
            pass
        # ROC AUC
        try:
            if hasattr(y_prob, "shape") and y_prob.ndim == 2 and y_prob.shape[1] > 2:
                m["roc_auc_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
            else:
                pos = y_prob if getattr(y_prob, "ndim", 1) == 1 else y_prob[:, 1]
                m["roc_auc_ovr"] = float(roc_auc_score(y_true, pos))
        except Exception:
            pass
        # AUPRC : seulement sensé en binaire
        try:
            if hasattr(y_prob, "shape") and y_prob.ndim == 2 and y_prob.shape[1] == 2:
                m["auprc"] = float(average_precision_score(y_true, y_prob[:, 1]))
            elif np.ndim(y_prob) == 1:
                m["auprc"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            pass
    return m


def selection_score(m: Dict[str, float]) -> float:
    """
    Score unique pour sélectionner le "meilleur" modèle (plus grand = meilleur).
    Stratégie :
      1) on privilégie AUC (si disponible),
      2) sinon F1_weighted,
      3) sinon Accuracy.
    """
    if "roc_auc_ovr" in m and not np.isnan(m["roc_auc_ovr"]):
        return m["roc_auc_ovr"]
    if "f1_weighted" in m and not np.isnan(m["f1_weighted"]):
        return m["f1_weighted"]
    return m.get("accuracy", -1.0)


def save_confusion_matrix(y_true, y_pred, class_names, out_path: Path):
    """
    Génère une matrice de confusion en PNG (backend Agg -> pas besoin d'interface graphique).
    """
    out_path = out_path.with_suffix(".png")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=180)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, values_format="d", colorbar=False, cmap="Blues"
    )
    ax.set_title("Matrice de confusion")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_roc_curves(y_true, y_prob, classes, out_path: Path):
    """
    Enregistre la/les courbes ROC en PNG.
      - Binaire : une seule courbe ROC (proba classe positive)
      - Multiclasse : courbes OvR si y_prob a la bonne forme (n x n_classes)
    """
    if y_prob is None:
        return
    out_path = out_path.with_suffix(".png")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay

    fig, ax = plt.subplots(figsize=(6, 6), dpi=180)
    if len(classes) == 2:
        pos = y_prob if getattr(y_prob, "ndim", 1) == 1 else y_prob[:, 1]
        RocCurveDisplay.from_predictions(y_true, pos, ax=ax, name="ROC")
        ax.set_title("Courbe ROC (binaire)")
    else:
        y_bin = label_binarize(y_true, classes=classes)
        if hasattr(y_prob, "shape") and y_prob.shape[1] == y_bin.shape[1]:
            for i, cls in enumerate(classes):
                RocCurveDisplay.from_predictions(y_bin[:, i], y_prob[:, i], ax=ax, name=str(cls))
            ax.set_title("Courbes ROC (multiclasse, OvR)")
        else:
            plt.close(fig); return

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_pr_curves(y_true, y_prob_pos, out_path: Path):
    """
    Enregistre la courbe Precision-Recall (binaire) en PNG à partir des probas de la classe positive.
    """
    if y_prob_pos is None:
        return
    out_path = out_path.with_suffix(".png")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prec, rec, _ = precision_recall_curve(y_true, y_prob_pos)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=180)
    ax.plot(rec, prec)
    ax.set_title("Courbe Precision-Recall (binaire)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def best_threshold_by_f1(y_true: np.ndarray, y_prob_pos: np.ndarray) -> Tuple[float, float, float]:
    """
    Calcule le SEUIL qui maximise le F1 de la classe positive sur (y_true, y_prob_pos).
    Retourne : (best_threshold, best_f1, recall_at_best)
    """
    prec, rec, thr = precision_recall_curve(y_true, y_prob_pos)
    # precision_recall_curve renvoie n_thresholds = len(thr) ; len(prec)=len(rec)=n_thresholds+1
    # On calcule F1 pour chaque point (en ignorant la première valeur sans threshold)
    f1s = []
    for p, r in zip(prec[1:], rec[1:]):
        if (p + r) > 0:
            f1s.append(2 * p * r / (p + r))
        else:
            f1s.append(0.0)
    if len(f1s) == 0:
        return 0.5, 0.0, 0.0
    idx = int(np.argmax(f1s))
    best_thr = float(thr[idx])
    best_f1 = float(f1s[idx])
    best_rec = float(rec[idx + 1])
    return best_thr, best_f1, best_rec


# ──────────────────────────────────────────────────────────────
# 2) LOG MODEL COMPAT (MLflow 3.x / 2.x)
# ──────────────────────────────────────────────────────────────
def log_model_compat(pipe, model_name: str, signature=None, input_example=None) -> str:
    """
    Log un modèle sklearn dans le run courant et renvoie son URI 'runs:/<run_id>/<name>'.
    Compatible MLflow>=3 (name=...) et versions plus anciennes (artifact_path=...).
    """
    run_id = mlflow.active_run().info.run_id
    try:
        mlflow.sklearn.log_model(
            pipe, name=model_name, signature=signature, input_example=input_example
        )
    except TypeError:
        mlflow.sklearn.log_model(
            pipe, artifact_path=model_name, signature=signature, input_example=input_example
        )
    return f"runs:/{run_id}/{model_name}"


def strip_prefix(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Retire le préfixe 'model__' des meilleurs hyperparamètres (lecture plus claire)."""
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in d.items()}


# ──────────────────────────────────────────────────────────────
# 3) ENTRAÎNER 1 MODÈLE (RUN MLflow IMBRIQUÉ)
# ──────────────────────────────────────────────────────────────
def train_one(
    name: str,
    estimator,
    preproc: ColumnTransformer,
    X_train, X_test, y_train, y_test,
    artifacts_dir: Path,
    class_names: Optional[List[str]] = None,
    tune: bool = False,
    cv_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Entraîne un pipeline (prétraitement + modèle) sous un run MLflow imbriqué :
      - Gère le déséquilibre via class_weight + sample_weight,
      - (Optionnel) tuning via RandomizedSearchCV (StratifiedKFold + scoring AUC),
      - Cherche un **seuil optimal** (max F1) sur une **validation interne** de X_train,
      - Ré-entraine sur tout X_train, évalue sur X_test au seuil 0.5 ET au seuil optimal,
      - Loggue métriques, figures (ROC + PR + matrices confusion), et le pipeline (signature).
    """
    # 0) Poids d'échantillons côté TRAIN pour contrer le déséquilibre
    sample_w_full = compute_sample_weight(class_weight="balanced", y=y_train)

    # 1) Scoring pour le tuning : AUC binaire ou OvR si multiclasses
    n_classes = len(np.unique(y_train))
    scoring = "roc_auc" if n_classes == 2 else "roc_auc_ovr"

    # 2) Option tuning : grilles RandomizedSearchCV (sur le step 'model__*')
    grids = {
        "logreg": {
            "model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
            "model__max_iter": [500, 1000],
        },
        "decision_tree": {
            "model__max_depth": [None, 5, 10, 15, 20, 30],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": [None, "sqrt", "log2"],
        },
        "random_forest": {
            "model__n_estimators": [200, 300, 400, 600, 800],
            "model__max_depth": [None, 10, 20, 30, 40],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": [None, "sqrt", 0.5],
        },
    }

    with mlflow.start_run(run_name=name, nested=True):
        # 3) Construire l'estimateur (tuning éventuel)
        est = estimator
        if tune:
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
            est = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=grids.get(name, {}),
                n_iter=25 if name == "random_forest" else 20,  # un peu plus d'itérations pour RF
                scoring=scoring,
                n_jobs=-1,
                cv=cv,
                refit=True,
                random_state=random_state,
                verbose=0,
            )

        # 4) Pipeline complet = Prétraitement -> Modèle (ou SearchCV)
        pipe = Pipeline(steps=[("preprocess", preproc), ("model", est)])

        # 5) **Validation interne** pour choisir un seuil (binaire seulement)
        best_thr = 0.5
        best_f1_val = None
        if n_classes == 2:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
            )
            w_tr = compute_sample_weight(class_weight="balanced", y=y_tr)
            # fit temporaire sur le sous-ensemble d'entraînement
            pipe.fit(X_tr, y_tr, **{"model__sample_weight": w_tr})
            # probas sur validation
            y_val_prob = None
            try:
                y_val_prob = pipe.predict_proba(X_val)[:, 1]
            except Exception:
                pass
            if y_val_prob is not None:
                thr, f1v, recv = best_threshold_by_f1(y_val.values, y_val_prob)
                best_thr, best_f1_val = float(thr), float(f1v)
                mlflow.log_params({
                    "opt_threshold_from_val": best_thr,
                })
                mlflow.log_metrics({
                    "val_best_f1_positive": best_f1_val,
                    "val_recall_at_best_thr": recv
                })

        # 6) Ré-entraîner **sur tout X_train** avec les poids complets
        pipe.fit(X_train, y_train, **{"model__sample_weight": sample_w_full})

        # 7) Si tuning : logger les meilleurs hyperparamètres (lisibles)
        try:
            if tune and hasattr(pipe.named_steps["model"], "best_params_"):
                best_params = strip_prefix(pipe.named_steps["model"].best_params_, "model__")
                mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        except Exception:
            pass

        # 8) Prédictions (classes + probabilités si dispo)
        y_pred = pipe.predict(X_test)
        y_prob = None
        y_prob_pos = None
        try:
            y_prob = pipe.predict_proba(X_test)
            if n_classes == 2:
                y_prob_pos = y_prob[:, 1]
        except Exception:
            try:
                # fallback éventuel
                y_prob = pipe.decision_function(X_test)
            except Exception:
                pass

        # 9) Métriques par défaut (seuil 0.5) + AUPRC
        mets = metrics_classification(y_test, y_pred, y_prob)

        # 10) Métriques au **seuil optimal** (si binaire + proba dispo)
        if n_classes == 2 and y_prob_pos is not None:
            y_pred_opt = (y_prob_pos >= best_thr).astype(int)
            f1_opt = f1_score(y_test, y_pred_opt, average="binary")
            mlflow.log_metrics({
                "f1_positive_at_opt_thr": float(f1_opt),
                "opt_threshold_used_on_test": float(best_thr)
            })

        # 11) Log des métriques "classiques"
        mlflow.log_params({"model_name": name, "tuned": tune})
        for k, v in mets.items():
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                mlflow.log_metric(k, float(v))

        # 12) Figures -> artefacts (CM à seuil 0.5 + CM au seuil optimal, ROC, PR)
        model_dir = artifacts_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        # CM seuil 0.5
        cm_path = model_dir / "confusion_matrix.png"
        save_confusion_matrix(y_test, y_pred, class_names, cm_path)
        if cm_path.exists(): mlflow.log_artifact(str(cm_path))

        classes = list(np.unique(y_test))
        # ROC
        roc_path = model_dir / ("roc.png" if len(classes) == 2 else "roc_multiclass.png")
        save_roc_curves(y_test, y_prob, classes, roc_path)
        if roc_path.exists(): mlflow.log_artifact(str(roc_path))

        # PR + CM au seuil OPT (binaire seulement)
        if n_classes == 2 and y_prob_pos is not None:
            pr_path = model_dir / "precision_recall.png"
            save_pr_curves(y_test, y_prob_pos, pr_path)
            if pr_path.exists(): mlflow.log_artifact(str(pr_path))

            y_pred_opt = (y_prob_pos >= best_thr).astype(int)
            cm_opt_path = model_dir / "confusion_matrix_opt_threshold.png"
            save_confusion_matrix(y_test, y_pred_opt, class_names, cm_opt_path)
            if cm_opt_path.exists(): mlflow.log_artifact(str(cm_opt_path))

        # 13) Log du pipeline entraîné + signature + input_example (supprime les warnings)
        input_example = X_train.head(5)
        signature = None
        try:
            signature = infer_signature(input_example, pipe.predict(input_example))
        except Exception:
            pass

        model_name = f"model_{name}"
        model_uri = log_model_compat(
            pipe, model_name=model_name, signature=signature, input_example=input_example
        )

        return {
            "name": name,
            "pipeline": pipe,
            "metrics": mets,
            "model_uri": model_uri,
            "opt_threshold": float(best_thr) if n_classes == 2 else None
        }


# ──────────────────────────────────────────────────────────────
# 4) MAIN : CHARGER DONNÉES, LANCER LES 3 MODÈLES, COMPARER, SAUVER LE MEILLEUR
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Classification : 3 modèles + MLflow + déséquilibre + tuning + AUPRC + seuil optimal")
    parser.add_argument("--csv", type=str, required=True, help="Chemin du CSV (ex: data/datasetfinal.csv)")
    parser.add_argument("--target", type=str, default=None, help="Nom de la colonne cible (sinon dernière colonne)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction du test (ex: 0.2 = 20%)")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire pour reproductibilité")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Dossier des sorties/artefacts")
    parser.add_argument("--experiment", type=str, default="mlops-training", help="Nom d'expérience MLflow")
    parser.add_argument("--mlflow-uri", type=str, default="mlruns", help="URI de tracking MLflow (dossier local ou serveur)")
    parser.add_argument("--tune", action="store_true", help="Active le tuning (RandomizedSearchCV)")
    parser.add_argument("--cv-splits", type=int, default=5, help="Nombre de folds (StratifiedKFold) pour le tuning")
    parser.add_argument("--register-best", action="store_true", help="Enregistre le meilleur modèle dans le Model Registry")
    parser.add_argument("--registry-name", type=str, default="mlops_best", help="Nom du Registered Model (onglet Models)")
    args = parser.parse_args()

    # 1) Initialiser MLflow (emplacement des runs + expérience)
    #    -> on force un chemin ABSOLU pour éviter les stores 'cassés' (meta.yaml manquants)
    store_path = Path(args.mlflow_uri).resolve()
    store_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(store_path.as_uri())   # ex: file:///C:/.../Projet_MLOps/mlruns
    mlflow.set_experiment(args.experiment)

    # 2) Charger le CSV et définir la cible
    df = pd.read_csv(args.csv)
    target = args.target or df.columns[-1]  # si non fourni, on prend la dernière colonne
    if target not in df.columns:
        raise ValueError(f"Colonne cible '{target}' introuvable dans {args.csv}.")

    # 3) Séparer X/y et identifier les colonnes numériques/catégorielles
    X, y, num_cols, cat_cols = split_features(df, target)
    class_names = list(np.unique(y))

    # 4) Split train/test (stratifié = indispensable en classification déséquilibrée)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # 5) Prétraitement (imputation + OHE + standardisation)
    preproc = build_preprocess(num_cols, cat_cols)

    # 6) Modèles avec class_weight pour les déséquilibres
    models = {
        # Régression Logistique (solver lbfgs gère le multinomial ; class_weight="balanced")
        "logreg": LogisticRegression(max_iter=500, solver="lbfgs", class_weight="balanced"),
        # Arbre de décision (avec class_weight)
        "decision_tree": DecisionTreeClassifier(random_state=args.seed, class_weight="balanced",
                                                max_depth=None, min_samples_leaf=1),
        # Forêt aléatoire (class_weight="balanced_subsample")
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=args.seed,
                                                n_jobs=-1, class_weight="balanced_subsample"),
    }

    # 7) Dossier des artefacts (sorties locales)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 8) Lancer un run MLflow global puis 3 runs imbriqués (un par modèle)
    results = []
    with mlflow.start_run(run_name="train_classification"):
        # On évite l'autolog des modèles (on loggue proprement ce dont on a besoin)
        mlflow.sklearn.autolog(log_models=False)

        for name, est in models.items():
            res = train_one(
                name=name,
                estimator=est,
                preproc=preproc,
                X_train=X_train, X_test=X_test,
                y_train=y_train, y_test=y_test,
                artifacts_dir=artifacts_dir,
                class_names=class_names,
                tune=args.tune,
                cv_splits=args.cv_splits,
                random_state=args.seed,
            )
            print(f"[{name}] -> {res['metrics']}")
            results.append(res)

        # 9) Comparaison et sélection du meilleur (selon selection_score)
        rows = []
        best = None
        best_score = -np.inf
        for r in results:
            m = r["metrics"]
            score = selection_score(m)
            rows.append({
                "model": r["name"], **m, "selection_score": score,
                "opt_threshold": r.get("opt_threshold", None)
            })
            if score > best_score:
                best_score = score
                best = r

        # Sauvegarder un tableau comparatif pour traçabilité
        comp_df = pd.DataFrame(rows).sort_values("selection_score", ascending=False)
        comp_csv = artifacts_dir / "models_comparison.csv"
        comp_df.to_csv(comp_csv, index=False)
        mlflow.log_artifact(str(comp_csv))

        # 10) Sauvegarder le MEILLEUR pipeline (préprocess + modèle) en local + log MLflow
        best_path = artifacts_dir / "best_model.joblib"
        joblib.dump(best["pipeline"], best_path)
        mlflow.log_artifact(str(best_path))

        # 11) Log d'infos utiles (pratique pour filtrer dans l'UI)
        mlflow.log_params({
            "best_model": best["name"],
            "target": target,
            "num_features": len(num_cols),
            "cat_features": len(cat_cols),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "tuning_enabled": args.tune,
            "cv_splits": args.cv_splits if args.tune else 0,
        })

        # 12) (Optionnel) Enregistrer le meilleur dans le Model Registry
        if args.register_best:
            try:
                print(f"\nEnregistrement dans le Model Registry : {args.registry_name}")
                mlflow.register_model(model_uri=best["model_uri"], name=args.registry_name)
                print("→ OK. Consulte l’onglet 'Models' dans l’UI MLflow.")
            except Exception as e:
                print(f"Impossible d'enregistrer le modèle dans le registry : {e}")

        # 13) Récapitulatif console
        print("\n=== RÉSUMÉ ===")
        print(f"Cible : {target}")
        print("Comparaison des modèles :\n", comp_df.to_string(index=False))
        print(f"\nMeilleur modèle : {best['name']} -> sauvegardé : {best_path}")


# Point d'entrée
if __name__ == "__main__":
    main()


# Classification supervisée avec suivi MLflow

## 1. Présentation du projet

Ce projet implémente une comparaison de modèles de classification supervisée à l’aide de **trois algorithmes** :

* Régression Logistique
* Arbre de Décision
* Random Forest

L’objectif est de proposer une approche complète intégrant la **gestion du déséquilibre des classes**, la **recherche d’hyperparamètres**, l’**optimisation du seuil de décision** et le **suivi expérimental avec MLflow**.

---

## 2. Objectifs principaux

* Entraîner **trois modèles** sur un jeu de données CSV (features + colonne cible).
* Gérer le **déséquilibre de classes** via `class_weight` et/ou `sample_weight`.
* (Optionnel) Effectuer une **recherche d’hyperparamètres** avec `RandomizedSearchCV`.
* Comparer les performances à l’aide des métriques suivantes :

  * AUC (ROC)
  * AUPRC (PR-AUC)
  * Accuracy
  * F1-score
  * Precision
  * Recall
  * Log Score

* Trouver le **seuil de décision optimal** maximisant le F1-score de la **classe minoritaire**.
* Sauvegarder le **meilleur pipeline** dans le répertoire :

  ```
  mlflow_artifacts/nom_modele
  best_model_local.pkl

  ```
* Enregistrer l’ensemble des **métriques, figures et résultats** dans **MLflow**, incluant :

  * Matrice de confusion
  * Courbes ROC et Precision-Recall

---

## 3. Fonctionnalités principales

| Fonctionnalité           | Description                                             |
| ------------------------ | ------------------------------------------------------- |
| Modèles évalués          | Régression Logistique, Arbre de Décision, Random Forest |
| Gestion du déséquilibre  | `class_weight` et/ou `sample_weight`                    |
| Tuning d’hyperparamètres | `RandomizedSearchCV` (optionnel)                        |
| Optimisation du seuil    | Recherche du seuil maximisant le F1-score minoritaire   |
| Suivi expérimental       | Logging complet avec MLflow                             |
| Export final             | `artifacts/best_model.joblib`                           |

---

## 4. Métriques et visualisations

Les performances sont évaluées selon plusieurs indicateurs :

* AUC (ROC)
* AUPRC (PR-AUC)
* Accuracy
* F1-score (global et minoritaire)
* Log loss

Les visualisations enregistrées dans MLflow comprennent :

* Matrice de confusion
* Courbe ROC
* Courbe Precision-Recall
* Comparatif global des modèles

---

## 5. Structure du projet (exemple)

```
.
├── data/
│   └── dataset.csv
├── mlflow_artifacts/
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── mlruns/
│   └── ...
├── src/
│   ├── data_artefacts.py
│   ├── data_processing.py
│   ├── metrics.py
│   ├── models.py
│   ├── save_best_model.py
│   ├── train_experiment.py
│   └── ...
├── best_model_local.pkl
|
├── main.py
│   
├── requirements.txt
└── README.md
```

---

## 6. Lancement rapide

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Lancement de l’interface MLflow

```bash
mlflow ui --backend-store-uri mlruns --default-artifact-root mlruns --host 127.0.0.1 --port 5000 
```

### Entraînement des modèles

```bash
python src/train.py --data data/dataset.csv
```

### Consultation des résultats

* Interface MLflow : [http://localhost:5000](http://localhost:5000)
* Modèle final enregistré : `artifacts/best_model.joblib`

---

## 7. Environnement et dépendances

* Python 3.x
* scikit-learn
* MLflow
* pandas
* numpy
* matplotlib
* seaborn
* joblib
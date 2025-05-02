# === imports ===
from src.config import *
from src.data_loader import load_data

import mlflow
import mlflow.sklearn
from itertools import product
from tqdm import tqdm

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize


def train_and_log_model(model_class, param_ranges, feature_set, cv_folds=5):
    """
    Multi-class model training + hyperparameter search + MLflow logging

    Args:
        model_class: model class (e.g., RandomForestClassifier)
        param_ranges (dict): parameter search ranges (keys are parameter names, values are lists)
        feature_set (str): feature set ID, currently supports '6'
        cv_folds (int): number of cross-validation folds
    """
    best_accuracy_mean = -np.inf
    best_model_params = None
    best_model = None

    # === Load data ===
    (y_train, y_test,
     X_train_6, X_test_6) = load_data(transform=True)

    if feature_set == '6':
        X_train = X_train_6
        X_test = X_test_6
    else:
        raise ValueError("Invalid feature set specified.")

    # === Check and adjust labels (for XGBClassifier) ===
    if model_class == XGBClassifier:
        y_min = min(np.min(y_train), np.min(y_test))
        if y_min != 0:
            print(f"[Note] Adjusting labels: min label is {y_min}, shifting to start from 0 for XGBClassifier.")
            y_train = y_train - y_min
            y_test = y_test - y_min
                
    classes = np.unique(y_train)
    y_test_bin = label_binarize(y_test, classes=classes)

    # === Set experiment name ===
    model_name = model_class.__name__
    mlflow.set_experiment(experiment_name=model_name + '_' + feature_set)

    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())

    for param_combination in tqdm(list(product(*param_values)), desc="Hyperparameter combinations"):
        param_dict = dict(zip(param_names, param_combination))

        with mlflow.start_run():
            model = model_class(**param_dict)

            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

            # Multi-class metrics (macro average)
            f1_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_macro')
            accuracy_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
            precision_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='precision_macro')
            recall_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='recall_macro')

            # Log cross-validation means and standard deviations
            f1_mean, f1_std = np.mean(f1_scores), np.std(f1_scores)
            accuracy_mean, accuracy_std = np.mean(accuracy_scores), np.std(accuracy_scores)
            precision_mean, precision_std = np.mean(precision_scores), np.std(precision_scores)
            recall_mean, recall_std = np.mean(recall_scores), np.std(recall_scores)

            mlflow.log_params(param_dict)
            mlflow.log_metric("cv_f1_macro_mean", f1_mean)
            mlflow.log_metric("cv_f1_macro_std", f1_std)
            mlflow.log_metric("cv_accuracy_mean", accuracy_mean)
            mlflow.log_metric("cv_accuracy_std", accuracy_std)
            mlflow.log_metric("cv_precision_macro_mean", precision_mean)
            mlflow.log_metric("cv_precision_macro_std", precision_std)
            mlflow.log_metric("cv_recall_macro_mean", recall_mean)
            mlflow.log_metric("cv_recall_macro_std", recall_std)

            # Save the current best model
            if accuracy_mean > best_accuracy_mean:
                best_accuracy_mean = accuracy_mean
                best_model_params = param_dict
                best_model = model_class(**param_dict)

    # === Train the best model on the full training set and evaluate on the test set ===
    with mlflow.start_run():
        mlflow.log_params(best_model_params)
        mlflow.log_metric("best_accuracy_cv_mean", best_accuracy_mean)

        best_model.fit(X_train, y_train)
        y_pred_test = best_model.predict(X_test)
        y_prob_test = best_model.predict_proba(X_test)

        # Multi-class metrics (macro average)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        precision_test = precision_score(y_test, y_pred_test, average='macro')
        recall_test = recall_score(y_test, y_pred_test, average='macro')
        f1_test = f1_score(y_test, y_pred_test, average='macro')
        auc_test = roc_auc_score(y_test_bin, y_prob_test, average='macro', multi_class='ovr')

        mlflow.log_metric("test_accuracy", accuracy_test)
        mlflow.log_metric("test_precision_macro", precision_test)
        mlflow.log_metric("test_recall_macro", recall_test)
        mlflow.log_metric("test_f1_macro", f1_test)
        mlflow.log_metric("test_auc_macro_ovr", auc_test)

        mlflow.sklearn.log_model(best_model, "best_model")

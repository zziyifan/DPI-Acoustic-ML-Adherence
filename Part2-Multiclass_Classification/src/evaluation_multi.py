from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import *
from src.data_loader import load_data


class model_evaluation_multi:

    def __init__(self, model_class, params, feature_set, cv_fold=5):
        self.model_class = model_class
        self.params = params
        self.feature_set = feature_set
        self.cv_fold = cv_fold

        # === Load data ===
        y_train, y_test, X_train_6, X_test_6 = load_data(transform=True)

        # === Set feature set ===
        if feature_set == '6':
            self.X_train = X_train_6
            self.X_test = X_test_6
        else:
            raise ValueError(f"Invalid feature set '{feature_set}' specified. Please check your input.")

        # === Standardize data format ===
        self.X_train = self.X_train.values if isinstance(self.X_train, pd.DataFrame) else self.X_train
        self.X_test = self.X_test.values if isinstance(self.X_test, pd.DataFrame) else self.X_test

        # Ensure 1D NumPy arrays by squeezing and converting if needed
        self.y_train = y_train.squeeze().values if isinstance(y_train, (pd.Series, pd.DataFrame)) else np.squeeze(y_train)
        self.y_test = y_test.squeeze().values if isinstance(y_test, (pd.Series, pd.DataFrame)) else np.squeeze(y_test)
        
        # === Check and adjust labels (for XGBClassifier) ===
        if model_class == XGBClassifier:
            y_min = min(np.min(self.y_train), np.min(self.y_test))
            if y_min != 0:
                print(f"[Note] Adjusting labels: min label is {y_min}, shifting to start from 0 for XGBClassifier.")
                self.y_train = self.y_train - y_min
                self.y_test = self.y_test - y_min

        self.model = self.model_class(**self.params)
        self.model.fit(self.X_train, self.y_train)

        self.classes = np.unique(self.y_train)

    def get_cv_results(self):
        skf = StratifiedKFold(n_splits=self.cv_fold, shuffle=True, random_state=42)

        print('\n*************** get_cv_results ***************')
        print('params:', self.params)

        metrics = {
            'f1_macro': 'f1_macro',
            'accuracy': 'accuracy',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro'
        }

        for metric_name, scoring in metrics.items():
            scores = cross_val_score(self.model, self.X_train, self.y_train, cv=skf, scoring=scoring)
            print(f"cv_{metric_name}_mean: {np.mean(scores):.4f}")
            print(f"cv_{metric_name}_std: {np.std(scores):.4f}")

        y_true_all = []
        y_prob_all = []

        for train_idx, test_idx in skf.split(self.X_train, self.y_train):
            X_tr = self.X_train[train_idx]
            X_val = self.X_train[test_idx]
            y_tr = self.y_train[train_idx]
            y_val = self.y_train[test_idx]

            model_fold = self.model_class(**self.params)
            model_fold.fit(X_tr, y_tr)

            y_prob = model_fold.predict_proba(X_val)
            y_true_all.append(y_val)
            y_prob_all.append(y_prob)

        y_true_all = np.hstack(y_true_all)
        y_prob_all = np.vstack(y_prob_all)

        # === Multi-class AUC (One-vs-Rest) ===
        y_bin = label_binarize(y_true_all, classes=self.classes)
        auc = roc_auc_score(y_bin, y_prob_all, average='macro', multi_class='ovr')
        print(f"cv_auc_macro_ovr: {auc:.4f}")

    def get_test_results(self):
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)

        y_test_bin = label_binarize(self.y_test, classes=self.classes)

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='macro')
        recall = recall_score(self.y_test, y_pred, average='macro')
        f1 = f1_score(self.y_test, y_pred, average='macro')
        auc = roc_auc_score(y_test_bin, y_prob, average='macro', multi_class='ovr')

        print('\n*************** get_test_results ***************')
        print(f"test_accuracy: {accuracy:.4f}")
        print(f"test_precision_macro: {precision:.4f}")
        print(f"test_recall_macro: {recall:.4f}")
        print(f"test_f1_macro: {f1:.4f}")
        print(f"test_auc_macro_ovr: {auc:.4f}")

        # === Confusion matrix ===
        ConfusionMatrixDisplay.from_predictions(self.y_test, y_pred, cmap=plt.cm.Blues)
        plt.title("Multiclass Confusion Matrix")
        plt.show()

        # === ROC Curves (One-vs-Rest) ===
        for i, class_label in enumerate(self.classes):
            RocCurveDisplay.from_predictions(
                y_test_bin[:, i], y_prob[:, i], name=f"Class {class_label}"
            )
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
        plt.legend()
        plt.title("AUC-ROC Curves (One-vs-Rest)")
        plt.show()

        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))

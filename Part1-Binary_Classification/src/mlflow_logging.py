# imports
from src.config import *
from src.data_loader import load_data
import mlflow
import mlflow.sklearn
from itertools import product

def train_and_log_model(model_class, param_ranges, feature_set, cv_folds=5):
    """
    Train a model with different hyperparameters using 5-fold cross-validation,
    and log the results with MLflow.

    Args:
        model_class: The model class to be instantiated (e.g., RandomForestClassifier).
        param_ranges (dict): A dictionary containing parameter ranges.
            Example: {
                "n_estimators_range": np.arange(100, 200, 2),
                "max_depth_range": np.arange(1, 5, 2),
                "max_features_range": ["sqrt", "log2"]
            }
        feature_set: in range:['1','2','3','4','5','6'].
        cv_folds (int): Number of cross-validation folds. Default is 5.
        
    """
    best_accuracy_mean = -np.inf
    best_model_params = None
    best_model = None
    
    # Load the data
    (y_train, y_test, 
    X_train_1, X_test_1,
    X_train_2, X_test_2,
    X_train_3, X_test_3,
    X_train_4, X_test_4,
    X_train_5, X_test_5,
    X_train_6, X_test_6) = load_data(transform=True)

    if feature_set == '1':
        X_train = X_train_1
        X_test = X_test_1
    elif feature_set == '2':
        X_train = X_train_2
        X_test = X_test_2
    elif feature_set == '3':
        X_train = X_train_3
        X_test = X_test_3
    elif feature_set == '4':
        X_train = X_train_4
        X_test = X_test_4
    elif feature_set == '5':
        X_train = X_train_5
        X_test = X_test_5
    elif feature_set == '6':
        X_train = X_train_6
        X_test = X_test_6
    else:
        raise ValueError("Invalid feature set specified.")
        
    
    # set experiment
    model_name = model_class.__name__
    mlflow.set_experiment(experiment_name = model_name+'_'+feature_set)

    # Extract parameter names and values
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())

    # Iterate over all combinations of parameter values
    for param_combination in tqdm(list(product(*param_values)), desc="Hyperparameter combinations"):
        param_dict = dict(zip(param_names, param_combination))

        with mlflow.start_run():
            # Initialize the model with the current set of parameters
            # model = model_class(**param_dict, n_jobs=-1)
            model = model_class(**param_dict)
            

            # Perform cross-validation
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            y_pred_cv = cross_val_predict(model, X_train, y_train, cv=skf)
            f1_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1')
            accuracy_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
            precision_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='precision')
            recall_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='recall')
            auc_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc')

            # Calculate mean and variance of metrics
            f1_mean, f1_std = np.mean(f1_scores), np.std(f1_scores)
            accuracy_mean, accuracy_std = np.mean(accuracy_scores), np.std(accuracy_scores)
            precision_mean, precision_std = np.mean(precision_scores), np.std(precision_scores)
            recall_mean, recall_std = np.mean(recall_scores), np.std(recall_scores)
            auc_mean, auc_std = np.mean(auc_scores), np.std(auc_scores)

            # Log parameters
            mlflow.log_params(param_dict)
            # Log mean and variance of metrics
            mlflow.log_metric("cv_f1_mean", f1_mean)
            mlflow.log_metric("cv_f1_std", f1_std)
            mlflow.log_metric("cv_accuracy_mean", accuracy_mean)
            mlflow.log_metric("cv_accuracy_std", accuracy_std)
            mlflow.log_metric("cv_precision_mean", precision_mean)
            mlflow.log_metric("cv_precision_std", precision_std)
            mlflow.log_metric("cv_recall_mean", recall_mean)
            mlflow.log_metric("cv_recall_std", recall_std)
            mlflow.log_metric("cv_auc_mean", auc_mean)
            mlflow.log_metric("cv_auc_std", auc_std)

            # Check if this model is the best based on accuracy mean
            if accuracy_mean > best_accuracy_mean:
                best_accuracy_mean = accuracy_mean
                best_model_params = param_dict
                best_model = model

    # Log the best model parameters
    with mlflow.start_run():
        mlflow.log_params(best_model_params)
        mlflow.log_metric("best_accuracy_mean", best_accuracy_mean)

        # Train the best model on the full training set
        best_model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred_test = best_model.predict(X_test)

        # Calculate performance metrics on the test set
        accuracy_test = accuracy_score(y_test, y_pred_test)
        precision_test = precision_score(y_test, y_pred_test)
        recall_test = recall_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test)
        auc_test = roc_auc_score(y_test, y_pred_test)

        # Log metrics for the test set
        mlflow.log_metric("test_accuracy", accuracy_test)
        mlflow.log_metric("test_precision", precision_test)
        mlflow.log_metric("test_recall", recall_test)
        mlflow.log_metric("test_f1", f1_test)
        mlflow.log_metric("test_auc", auc_test)

        # Log the best model
        mlflow.sklearn.log_model(best_model, "best_model")

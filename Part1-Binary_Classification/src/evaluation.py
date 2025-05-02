# imports
from src.config import *
from src.data_loader import load_data

class model_evaluation:

    def __init__(self, model_class, params, feature_set, cv_fold=5):
        """
        Train a model with different hyperparameters using 5-fold cross-validation.
    
        Args:
            model_class: The model class to be instantiated (e.g., RandomForestClassifier).
            params (dict): A dictionary containing optimal parameters.
            feature_set: in range:['1','2','3','4','5','6'].
            cv_folds (int): Number of cross-validation folds. Default is 5.
            
        """
        self.model_class = model_class
        self.params = params
        self.feature_set = feature_set
        self.cv_fold = cv_fold

        
        # Load the data
        (y_train, y_test, 
        X_train_1, X_test_1,
        X_train_2, X_test_2,
        X_train_3, X_test_3,
        X_train_4, X_test_4,
        X_train_5, X_test_5,
        X_train_6, X_test_6) = load_data(transform=True)

        self.y_train = y_train
        self.y_test = y_test
        
        if feature_set == '1':
            self.X_train = X_train_1
            self.X_test = X_test_1
        elif feature_set == '2':
            self.X_train = X_train_2
            self.X_test = X_test_2
        elif feature_set == '3':
            self.X_train = X_train_3
            self.X_test = X_test_3
        elif feature_set == '4':
            self.X_train = X_train_4
            self.X_test = X_test_4
        elif feature_set == '5':
            self.X_train = X_train_5
            self.X_test = X_test_5
        elif feature_set == '6':
            self.X_train = X_train_6
            self.X_test = X_test_6
        else:
            raise ValueError("Invalid feature set specified.")
            
        # Train the model on the training set
        self.model = self.model_class(**self.params)
        self.model.fit(self.X_train, self.y_train)

    
    ### Cross-validation ###
    def get_cv_results(self):
        # Perform cross-validation
        skf = StratifiedKFold(n_splits=self.cv_fold, shuffle=True, random_state=42)
        y_pred_cv = cross_val_predict(self.model, self.X_train, self.y_train, cv=skf)
        f1_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=skf, scoring='f1')
        accuracy_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=skf, scoring='accuracy')
        precision_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=skf, scoring='precision')
        recall_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=skf, scoring='recall')
        auc_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=skf, scoring='roc_auc')
    
        # Calculate mean and variance of metrics
        f1_mean, f1_std = np.mean(f1_scores), np.std(f1_scores)
        accuracy_mean, accuracy_std = np.mean(accuracy_scores), np.std(accuracy_scores)
        precision_mean, precision_std = np.mean(precision_scores), np.std(precision_scores)
        recall_mean, recall_std = np.mean(recall_scores), np.std(recall_scores)
        auc_mean, auc_std = np.mean(auc_scores), np.std(auc_scores)
    
        # Print parameters
        print('params', self.params)
        print('*******************get_cv_results********************')
        # Print mean and variance of metrics
        print(f"cv_f1_mean: {f1_mean:.4}")
        print(f"cv_f1_std: {f1_std:.4}")
        print(f"cv_accuracy_mean: {accuracy_mean:.4}")
        print(f"cv_accuracy_std: {accuracy_std:.4}")
        print(f"cv_precision_mean: {precision_mean:.4}")
        print(f"cv_precision_std: {precision_std:.4}")
        print(f"cv_recall_mean: {recall_mean:.4}")
        print(f"cv_recall_std: {recall_std:.4}")
        print(f"cv_auc_mean: {auc_mean:.4}")
        print(f"cv_auc_std: {auc_std:.4}")

    ### Evaluation of the test set ###
    def get_test_results(self):
        # Make predictions on the test set
        y_pred_test = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:,1]
        
        # Calculate performance metrics on the test set
        accuracy_test = accuracy_score(self.y_test, y_pred_test)
        precision_test = precision_score(self.y_test, y_pred_test)
        recall_test = recall_score(self.y_test, y_pred_test)
        f1_test = f1_score(self.y_test, y_pred_test)
        auc_test = roc_auc_score(self.y_test, y_pred_test)
    
        # Print metrics for the test set
        print('*******************get_test_results********************')
        print(f"test_accuracy: {accuracy_test:.4}")
        print(f"test_precision: {precision_test:.4}")
        print(f"test_recall: {recall_test:.4}")
        print(f"test_f1: {f1_test:.4}")
        print(f"test_auc: {auc_test:.4}")

        # confusion matrix    
        ConfusionMatrixDisplay.from_predictions(self.y_test, y_pred_test, cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

        # roc    
        RocCurveDisplay.from_predictions(self.y_test, y_prob)
        plt.plot([0,1],[0,1], linestyle="--", label='Random 50:50')
        plt.legend()
        plt.title("AUC-ROC Curve \n")
        plt.show()

        # classification report     
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred_test))

def Accuracy_bar_plot(data):
    df = pd.DataFrame(data)

    # Set font size globally
    plt.rcParams.update({'font.size': 20})
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    for i, (y_col, title) in enumerate(zip(['best_cv_accuracy_mean', 'test_accuracy'], 
                                           ['Best CV Accuracy Mean', 'Test Accuracy'])):
        sns.barplot(x='Algorithms', y=y_col, data=df, ax=axes[i], palette='muted')
        axes[i].set_title(title)
        axes[i].set_ylabel('Values')
        axes[i].set_ylim([0, 1])
        
        # Rotate x-ticks and align them to center
        axes[i].tick_params(axis='x', labelrotation=45)
        axes[i].set_xticklabels(axes[i].get_xticklabels(),  ha='right') 
        
        # Add values on top of bars
        for p in axes[i].patches:
            axes[i].annotate(f'{p.get_height():.4f}',
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center',
                             xytext=(0, 10),
                             textcoords='offset points')

    plt.tight_layout()
    plt.show()
    
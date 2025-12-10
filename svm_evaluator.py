from itertools import product
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC, SVC
import time


class SVMEvaluator:
    """
    Unified evaluator for both Linear and RBF SVM models.
    Uses LinearSVC for linear kernel, SVC(kernel='rbf') for RBF kernel.
    """

    def __init__(self, hyperparameter_grid, kernel):
        self.hyperparameter_grid = hyperparameter_grid
        self.kernel = kernel
        self.results = []
        self.__best_model = None

    def evaluate_all_models(self, x_train, x_test, y_train, y_test):
        """
        Evaluate all SVM parameter combinations.
        Returns DataFrame with all results and stores best model in self.__best_model.
        """

        # Create all parameter combinations
        keys = list(self.hyperparameter_grid.keys())
        combinations = list(product(*self.hyperparameter_grid.values()))

        # Get total number of models
        total_models = len(combinations)

        print(f"[{self.kernel.upper()}] Evaluating {total_models} models...\n")
        
        best_test_accuracy = -1
        
        for index, combo in enumerate(combinations):

            params = dict(zip(keys, combo))
            
            print(f"[{self.kernel.upper()} MODEL {index + 1}/{total_models}] Parameters: {params}\n")
            
            # Create model based on kernel type
            if self.kernel == "linear":
                # Use LinearSVC for linear kernel
                model = LinearSVC(**{k: v for k, v in params.items()})
            else:  # rbf
                # Use SVC for RBF kernel
                model = SVC(kernel="rbf", **params)
            
            # Training
            print(f"[TRAINING] Start")
            start_train = time.time()
            model.fit(x_train, y_train)
            train_time_minutes = (time.time() - start_train) / 60
            print(f"[TRAINING] Finished in {train_time_minutes:.2f} minutes\n")
            
            # Testing
            print(f"[TESTING] Start")
            start_test = time.time()
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            test_time_minutes = (time.time() - start_test) / 60
            print(f"[TESTING] Finished in {test_time_minutes:.2f} minutes\n")
            
            # Accuracy
            print("[ACCURACIES] Calculate")
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
            print(f"[ACCURACIES] Calculated => Train Acc: {train_accuracy:.2f} | Test Acc: {test_accuracy:.2f}")
            print(f"[ACCURACIES] Calculated => Precision: {precision:.2f}")
            print(f"[ACCURACIES] Calculated => Recall: {recall:.2f}")
            print(f"[ACCURACIES] Calculated => F1 Score: {f1:.2f}\n")

            # Divider
            print(f"{150 * '-'}\n")

            # Set model results
            model_results = {
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "train_time": train_time_minutes,
                "test_time": test_time_minutes,
                "params": params,
                "y_test_pred": y_test_pred
            }
            
            # Add data in results list
            self.results.append(model_results)
            
            # Keep track of best model
            if test_accuracy > best_test_accuracy:

                best_test_accuracy = test_accuracy

                # Store the best model along with extra info
                self.__best_model = {
                    "model": model,
                    **model_results
                }

        return self.results
    
    def get_best_model(self):
        """Return the best model information."""
        return self.__best_model

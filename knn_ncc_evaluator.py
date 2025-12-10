from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
import time


class KNNNCCEvaluator:
    """
    Responsible for training and evaluating:
      - KNN (k-Nearest Neighbors, here with k=10)
      - NCC (Nearest Class Centroid)
    """

    def evaluate_models(self, x_train, x_test, y_train, y_test):
        """
        Train and evaluate both KNN and NCC on the given data.

        Returns a dict with metrics and predictions for each model.
        """
        results = {}

        # ===========================
        # ======== KNN (10-NN) ======
        # ===========================
        knn_params = {"n_neighbors": 10}
        knn = KNeighborsClassifier(**knn_params)
        results["KNN"] = self._evaluate_single_model(
            model=knn,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            model_name="KNN",
            params_dict=knn_params
        )

        # ===========================
        # ============ NCC ==========
        # ===========================
        ncc_params = {}
        ncc = NearestCentroid()
        results["NCC"] = self._evaluate_single_model(
            model=ncc,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            model_name="NCC",
            params_dict=ncc_params
        )

        return results

    def _evaluate_single_model(self, model, x_train, x_test, y_train, y_test, model_name, params_dict):
        """
        Common method to train, test, and evaluate a single model.
        Returns a dictionary with all metrics and timing information.
        """
        params_str = str(params_dict).replace("'", "")
        print(f"[{model_name}] Parameters: {params_str}\n")

        # Training
        train_time_minutes = self._train_model(model, x_train, y_train)

        # Testing
        y_train_pred, y_test_pred, test_time_minutes = self._test_model(model, x_train, x_test)

        # Calculate metrics
        metrics = self._calculate_metrics(y_train, y_train_pred, y_test, y_test_pred)
        self._print_metrics(metrics)

        # Divider
        print(f"{120 * '-'}\n")

        # Create results dictionary
        return self._create_results_dict(metrics, train_time_minutes, test_time_minutes, params_dict, y_test_pred)

    def _train_model(self, model, x_train, y_train):
        """Train the model and return training time in minutes."""
        print("[TRAINING] Start")
        start_train = time.time()
        model.fit(x_train, y_train)
        train_time_minutes = (time.time() - start_train) / 60
        print(f"[TRAINING] Finished in {train_time_minutes:.4f} minutes\n")
        return train_time_minutes

    def _test_model(self, model, x_train, x_test):
        """Test the model and return predictions and testing time in minutes."""
        print("[TESTING] Start")
        start_test = time.time()
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        test_time_minutes = (time.time() - start_test) / 60
        print(f"[TESTING] Finished in {test_time_minutes:.4f} minutes\n")
        return y_train_pred, y_test_pred, test_time_minutes

    def _calculate_metrics(self, y_train, y_train_pred, y_test, y_test_pred):
        """Calculate all metrics and return as a dictionary."""
        print("[ACCURACIES] Calculate")
        return {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred, average="macro", zero_division=0),
            "recall": recall_score(y_test, y_test_pred, average="macro", zero_division=0),
            "f1_score": f1_score(y_test, y_test_pred, average="macro", zero_division=0)
        }

    def _print_metrics(self, metrics):
        """Print all calculated metrics."""
        print(f"[ACCURACIES] Calculated => Train Acc: {metrics['train_accuracy']:.2f} | Test Acc: {metrics['test_accuracy']:.2f}")
        print(f"[ACCURACIES] Calculated => Precision: {metrics['precision']:.2f}")
        print(f"[ACCURACIES] Calculated => Recall: {metrics['recall']:.2f}")
        print(f"[ACCURACIES] Calculated => F1 Score: {metrics['f1_score']:.2f}\n")

    def _create_results_dict(self, metrics, train_time, test_time, params, y_test_pred):
        """Create the results dictionary with all metrics and timing information."""
        return {
            "train_accuracy": metrics["train_accuracy"],
            "test_accuracy": metrics["test_accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "train_time": train_time,
            "test_time": test_time,
            "params": params,
            "y_test_pred": y_test_pred
        }

from datetime import datetime
import pandas as pd
from svm_evaluator import SVMEvaluator
from knn_ncc_evaluator import KNNNCCEvaluator
from data_helper import DataHelper
from visualization_helper import VisualizationHelper


class MainHelper:

    @staticmethod
    def main_execution(main_method):

        # Start datetime
        start_datetime = datetime.now()
        print(f"\n[TIME] Start datetime: {start_datetime.strftime('%d/%m/%Y %H:%M:%S')}\n")

        # Execute main
        main_method()

        # End datetime
        end_datetime = datetime.now()
        print(f"[TIME] End datetime: {end_datetime.strftime('%d/%m/%Y %H:%M:%S')}\n")

        # Execution time
        execution_time_minutes = (end_datetime - start_datetime).total_seconds() / 60
        print(f"[TIME] Execution time: {execution_time_minutes:.2f} minutes\n")

    @staticmethod
    def evaluate_all_models(x_train, x_test, y_train, y_test, linear_parameters, rbf_parameters):
        """
        Evaluate all models (Linear SVM, RBF SVM, KNN, NCC) on given data.
        
        Returns:
        --------
        dict with keys: linear_svm_results, best_linear_svm, rbf_svm_results, 
                        best_rbf_svm, knn_ncc_results
        """
        # Evaluate Linear SVM
        linear_svm_evaluator = SVMEvaluator(
            hyperparameter_grid=linear_parameters,
            kernel="linear"
        )
        linear_svm_results = linear_svm_evaluator.evaluate_all_models(
            x_train, x_test, y_train, y_test
        )
        best_linear_svm = linear_svm_evaluator.get_best_model()

        # Evaluate RBF SVM
        rbf_svm_evaluator = SVMEvaluator(
            hyperparameter_grid=rbf_parameters,
            kernel="rbf"
        )
        rbf_svm_results = rbf_svm_evaluator.evaluate_all_models(
            x_train, x_test, y_train, y_test
        )
        best_rbf_svm = rbf_svm_evaluator.get_best_model()

        # Evaluate KNN and NCC
        knn_ncc_evaluator = KNNNCCEvaluator()
        knn_ncc_results = knn_ncc_evaluator.evaluate_models(
            x_train, x_test, y_train, y_test
        )

        return {
            "linear_svm_results": linear_svm_results,
            "best_linear_svm": best_linear_svm,
            "rbf_svm_results": rbf_svm_results,
            "best_rbf_svm": best_rbf_svm,
            "knn_ncc_results": knn_ncc_results
        }

    @staticmethod
    def add_result_to_list(results_list, model_data, model_name, config_info=None):
        """
        Add model results to results list.
        
        Parameters:
        -----------
        results_list : list
            List to append results to
        model_data : dict
            Model results dictionary
        model_name : str
            Name of the model
        config_info : dict, optional
            Configuration info (e.g., KPCA kernel, n_components) for labeling
        """
        params_str = ", ".join([f"{k}={v}" for k, v in model_data.get('params', {}).items()])
        
        display_name = model_name
        if config_info:
            display_name = f"{model_name} (KPCA: {config_info['kpca_kernel']}, n={config_info['kpca_n_components']})"

        results_list.append({
            "Model": display_name,
            "Train Accuracy": f"{model_data['train_accuracy']:.4f}",
            "Test Accuracy": f"{model_data['test_accuracy']:.4f}",
            "Precision": f"{model_data['precision']:.4f}",
            "Recall": f"{model_data['recall']:.4f}",
            "F1 Score": f"{model_data['f1_score']:.4f}",
            "Train Time": f"{model_data['train_time']:.4f}",
            "Test Time": f"{model_data['test_time']:.4f}",
            "Parameters": params_str
        })

    @staticmethod
    def print_sorted_results(results, title="RESULTS"):
        """
        Print results table sorted by test accuracy.
        
        Parameters:
        -----------
        results : list
            List of result dictionaries
        title : str
            Title to print before results
        """
        combined_df = pd.DataFrame(results)
        combined_df['Test Accuracy Float'] = combined_df['Test Accuracy'].astype(float)
        combined_df = combined_df.sort_values('Test Accuracy Float', ascending=False)
        combined_df = combined_df.drop('Test Accuracy Float', axis=1)
        
        print(f"\n### {title} ###\n")
        print(combined_df.to_string(index=False))
        print("\n")

    @staticmethod
    def aggregate_results(model_results, config_info=None):
        """
        Aggregate model results into all_results and best_results lists.
        
        Parameters:
        -----------
        model_results : dict
            Dictionary with keys: linear_svm_results, best_linear_svm, 
                                 rbf_svm_results, best_rbf_svm, knn_ncc_results
        config_info : dict, optional
            Configuration info for labeling (e.g., KPCA kernel, n_components)
        
        Returns:
        --------
        all_results : list
            All model results
        best_results : list
            Best model results only
        """
        all_results = []
        best_results = []

        # Add KNN results
        MainHelper.add_result_to_list(all_results, model_results["knn_ncc_results"]["KNN"], "KNN", config_info)
        MainHelper.add_result_to_list(best_results, model_results["knn_ncc_results"]["KNN"], "KNN", config_info)

        # Add NCC results
        MainHelper.add_result_to_list(all_results, model_results["knn_ncc_results"]["NCC"], "NCC", config_info)
        MainHelper.add_result_to_list(best_results, model_results["knn_ncc_results"]["NCC"], "NCC", config_info)

        # Add all Linear SVM results
        for result in model_results["linear_svm_results"]:
            MainHelper.add_result_to_list(all_results, result, "Linear SVM", config_info)

        # Add only best Linear SVM to best_results
        MainHelper.add_result_to_list(best_results, model_results["best_linear_svm"], "Linear SVM", config_info)

        # Add all RBF SVM results
        for result in model_results["rbf_svm_results"]:
            MainHelper.add_result_to_list(all_results, result, "RBF SVM", config_info)

        # Add only best RBF SVM to best_results
        MainHelper.add_result_to_list(best_results, model_results["best_rbf_svm"], "RBF SVM", config_info)

        return all_results, best_results

    @staticmethod
    def create_visualizations(model_results, original_x_test, y_test, dataset_type, config_info=None):
        """
        Create visualizations for model results.
        
        Parameters:
        -----------
        model_results : dict
            Dictionary with model results
        original_x_test : array
            Original test images (before preprocessing)
        y_test : array
            Test labels
        dataset_type : str
            Dataset type for visualization
        config_info : dict, optional
            Configuration info for labeling
        """
        predictions_size = 3

        if config_info:
            kernel = config_info['kpca_kernel']
            n_components = config_info['kpca_n_components']
            visual_data = {
                f"KNN (KPCA: {kernel}, n={n_components})": model_results["knn_ncc_results"]["KNN"],
                f"NCC (KPCA: {kernel}, n={n_components})": model_results["knn_ncc_results"]["NCC"],
                f"Linear SVM (KPCA: {kernel}, n={n_components})": model_results["best_linear_svm"],
                f"RBF SVM (KPCA: {kernel}, n={n_components})": model_results["best_rbf_svm"]
            }
        else:
            visual_data = {
                "KNN": model_results["knn_ncc_results"]["KNN"],
                "NCC": model_results["knn_ncc_results"]["NCC"],
                "Linear SVM": model_results["best_linear_svm"],
                "RBF SVM": model_results["best_rbf_svm"]
            }

        for key, model_info in visual_data.items():
            print(f"\n[{key} VISUALIZATION] Showing {predictions_size} correct and incorrect predictions...\n")
            VisualizationHelper.random_visualization(
                size=predictions_size,
                model_info=model_info,
                model_name=key,
                desired_y_test=y_test,
                x_test=original_x_test,
                dataset_type=dataset_type
            )

    @staticmethod
    def plot_metrics(best_results, dataset_type):
        """
        Plot performance and time metrics.
        
        Parameters:
        -----------
        best_results : list
            List of best model results
        dataset_type : str
            Dataset type for plot titles
        """
        print("\n[BEST RESULTS VISUALIZATION] Plotting performance metrics...\n")
        VisualizationHelper.plot_performance_metrics(best_results, dataset_type)
        
        print("\n[BEST RESULTS VISUALIZATION] Plotting time metrics...\n")
        VisualizationHelper.plot_time_metrics(best_results, dataset_type)
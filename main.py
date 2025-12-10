from datetime import datetime
import pandas as pd
from data_helper import DataHelper
from dataset_helper import DatasetHelper
from svm_evaluator import SVMEvaluator
from knn_ncc_evaluator import KNNNCCEvaluator
from visualization_helper import VisualizationHelper


# Set program basic information
program_basic_info: dict = {
    "desired_data": "cifar10",
    "linear_parameters": {
        "C": [0.1, 1, 5, 10],
    },
    "rbf_parameters": {
        "C": [0.1, 1, 3],
        "gamma": ["scale", "auto"],
    },
    "minimize_samples": False
}


def main():

    # Load desired data
    # desired_data possible values -> "cifar10", "svhn"
    (x_train, y_train), (x_test, y_test) = DatasetHelper.load_desired_data(
        desired_data=program_basic_info["desired_data"]
    )
    desired_x_train, desired_x_test, desired_y_train, desired_y_test = x_train, x_test, y_train, y_test

    # -------------------------------
    # -------- PREPROCESSING --------
    # -------------------------------

    if program_basic_info["minimize_samples"]:

        # Minimize samples
        desired_x_train, desired_x_test, desired_y_train, desired_y_test = DataHelper.minimize_samples(
            x_train, x_test, y_train, y_test,
            max_train_samples=2000,
            max_test_samples=500
        )

    # Store original test images for visualization (after minimize_samples, before preprocessing)
    original_x_test = desired_x_test.copy()

    # Full preprocessing pipeline:
    # flatten -> normalize -> standardize -> PCA -> ravel labels
    desired_x_train, desired_x_test, desired_y_train, desired_y_test = DataHelper.full_preprocessing_pipeline(
        desired_x_train, desired_x_test, desired_y_train, desired_y_test
    )

    # -------------------------------
    # ------ MODELS EVALUATION ------
    # -------------------------------

    # Evaluate Linear SVM
    linear_svm_evaluator = SVMEvaluator(
        hyperparameter_grid=program_basic_info["linear_parameters"],
        kernel="linear"
    )
    linear_svm_results = linear_svm_evaluator.evaluate_all_models(
        desired_x_train, desired_x_test,
        desired_y_train, desired_y_test
    )
    best_linear_svm = linear_svm_evaluator.get_best_model()

    # Evaluate RBF SVM
    rbf_svm_evaluator = SVMEvaluator(
        hyperparameter_grid=program_basic_info["rbf_parameters"],
        kernel="rbf"
    )
    rbf_svm_results = rbf_svm_evaluator.evaluate_all_models(
        desired_x_train, desired_x_test,
        desired_y_train, desired_y_test
    )
    best_rbf_svm = rbf_svm_evaluator.get_best_model()

    # Evaluate KNN and NCC
    knn_ncc_evaluator = KNNNCCEvaluator()
    knn_ncc_results = knn_ncc_evaluator.evaluate_models(
        desired_x_train, desired_x_test,
        desired_y_train, desired_y_test
    )

    # -------------------------------
    # ----------- RESULTS -----------
    # -------------------------------

    # Add desired values in given all results list
    def add_values_in_all_results(given_all_results, given_data, model_value):

        params_str = ", ".join([f"{k}={v}" for k, v in given_data.get('params', {}).items()])

        given_all_results.append({
            "Model": model_value,
            "Train Accuracy": f"{given_data['train_accuracy']:.4f}",
            "Test Accuracy": f"{given_data['test_accuracy']:.4f}",
            "Precision": f"{given_data['precision']:.4f}",
            "Recall": f"{given_data['recall']:.4f}",
            "F1 Score": f"{given_data['f1_score']:.4f}",
            "Train Time": f"{given_data['train_time']:.4f}",
            "Test Time": f"{given_data['test_time']:.4f}",
            "Parameters": params_str
        })

    # Display combined results
    def print_combined_and_sorted_results(given_results):

        combined_df = pd.DataFrame(given_results)
        # Sort by test accuracy (convert to float for sorting, then format back)
        combined_df['Test Accuracy Float'] = combined_df['Test Accuracy'].astype(float)
        combined_df = combined_df.sort_values('Test Accuracy Float', ascending=False)
        combined_df = combined_df.drop('Test Accuracy Float', axis=1)
        print(combined_df.to_string(index=False))
        print("\n")


    # Print info messages
    print("\n### RESULTS ###\n")
    print("\n[INFO] Sorted by test accuracy - Time in minutes\n")

    # Create combined results array
    all_results = []
    best_results = []

    # Add KNN results
    add_values_in_all_results(all_results, knn_ncc_results["KNN"], "KNN")
    add_values_in_all_results(best_results, knn_ncc_results["KNN"], "KNN")

    # Add NCC results
    add_values_in_all_results(all_results, knn_ncc_results["NCC"], "NCC")
    add_values_in_all_results(best_results, knn_ncc_results["NCC"], "NCC")

    # Add all Linear SVM results
    for result in linear_svm_results:
        add_values_in_all_results(all_results, result, "Linear SVM")

    # Add only best Linear SVM to best_results
    add_values_in_all_results(best_results, best_linear_svm, "Linear SVM")

    # Add all RBF SVM results
    for result in rbf_svm_results:
        add_values_in_all_results(all_results, result, "RBF SVM")

    # Add only best RBF SVM to best_results
    add_values_in_all_results(best_results, best_rbf_svm, "RBF SVM")

    print("\n### ALL RESULTS ###\n")
    print_combined_and_sorted_results(given_results=all_results)

    print("\n### BEST RESULTS ###\n")
    print_combined_and_sorted_results(given_results=best_results)

    # Print best model info
    DataHelper.print_best_model_info(best_model=best_linear_svm, best_title="LINEAR SVM")

    # Print best model info
    DataHelper.print_best_model_info(best_model=best_rbf_svm, best_title="RBF SVM")

    # -------------------------------
    # -------- VISUALIZATIONS -------
    # -------------------------------

    # Visualize predictions for all models
    print("\n### VISUALIZATIONS ###\n")

    visual_data = {
        "KNN": knn_ncc_results["KNN"],
        "NCC": knn_ncc_results["NCC"],
        "Linear SVM": best_linear_svm,
        "RBF SVM": best_rbf_svm
    }

    # Get dataset type for visualization
    dataset_type = program_basic_info["desired_data"]

    for key in visual_data.keys():

        # Visualization
        predictions_size=3
        print(f"\n[{key} VISUALIZATION] Showing {predictions_size} correct and incorrect predictions...\n")
        VisualizationHelper.random_visualization(
            size=predictions_size,
            model_info=visual_data[key],
            model_name=key,
            desired_y_test=desired_y_test,
            x_test=original_x_test,
            dataset_type=dataset_type
        )

    # -------------------------------
    # ------ BEST RESULTS GRAPH -----
    # -------------------------------

    print("\n[BEST RESULTS VISUALIZATION] Plotting performance metrics...\n")
    VisualizationHelper.plot_performance_metrics(best_results, dataset_type)
    
    print("\n[BEST RESULTS VISUALIZATION] Plotting time metrics...\n")
    VisualizationHelper.plot_time_metrics(best_results, dataset_type)

# Program execution
if __name__ == '__main__':

    # Start datetime
    start_datetime = datetime.now()
    print(f"\n[TIME] Start datetime: {start_datetime.strftime('%d/%m/%Y %H:%M:%S')}\n")

    # Execute main
    main()

    # End datetime
    end_datetime = datetime.now()
    print(f"[TIME] End datetime: {end_datetime.strftime('%d/%m/%Y %H:%M:%S')}\n")

    # Execution time
    execution_time_minutes = (end_datetime - start_datetime).total_seconds()/60
    print(f"[TIME] Execution time: {execution_time_minutes:.2f} minutes\n")
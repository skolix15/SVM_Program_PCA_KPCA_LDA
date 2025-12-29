from data_helper import DataHelper
from dataset_helper import DatasetHelper
from main_helper import MainHelper


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

    model_results = MainHelper.evaluate_all_models(
        desired_x_train, desired_x_test,
        desired_y_train, desired_y_test,
        program_basic_info["linear_parameters"],
        program_basic_info["rbf_parameters"]
    )

    # -------------------------------
    # ----------- RESULTS -----------
    # -------------------------------

    print("\n### RESULTS ###\n")
    print("\n[INFO] Sorted by test accuracy - Time in minutes\n")

    all_results, best_results = MainHelper.aggregate_results(model_results)

    MainHelper.print_sorted_results(all_results, "ALL RESULTS")
    MainHelper.print_sorted_results(best_results, "BEST RESULTS")

    # Print best model info
    DataHelper.print_best_model_info(best_model=model_results["best_linear_svm"], best_title="LINEAR SVM")
    DataHelper.print_best_model_info(best_model=model_results["best_rbf_svm"], best_title="RBF SVM")

    # -------------------------------
    # -------- VISUALIZATIONS -------
    # -------------------------------

    print("\n### VISUALIZATIONS ###\n")
    dataset_type = program_basic_info["desired_data"]
    
    MainHelper.create_visualizations(model_results, original_x_test, desired_y_test, dataset_type)
    MainHelper.plot_metrics(best_results, dataset_type)

# Program execution
if __name__ == '__main__':

    MainHelper.main_execution(main_method=main)
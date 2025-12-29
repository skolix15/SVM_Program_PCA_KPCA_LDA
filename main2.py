from data_helper import DataHelper
from dataset_helper import DatasetHelper
from main_helper import MainHelper


# Set program basic information
program_basic_info: dict = {
    "desired_data": "faces",
    "linear_parameters": {
        "C": [0.1, 1, 5, 10],
    },
    "rbf_parameters": {
        "C": [0.1, 1, 3],
        "gamma": ["scale", "auto"],
    },
    "minimize_samples": False,
    "kpca_parameters": {
        "kernels": ["rbf", "poly"],
        "n_components": 100,
    }
}


def main():

    # Load desired data
    # desired_data possible values -> "cifar10", "svhn", "faces"
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
            max_train_samples=20000,
            max_test_samples=7000
        )

    # Store original test images for visualization (after minimize_samples, before preprocessing)
    original_x_test = desired_x_test.copy()

    # -------------------------------
    # ------ MODELS EVALUATION ------
    # -------------------------------

    # Dictionary to store all results for each KPCA configuration
    all_kernel_results = {}

    # Get standard n_components value
    kpca_n_components = program_basic_info["kpca_parameters"]["n_components"]
    
    # Loop through each KPCA kernel
    for kpca_kernel in program_basic_info["kpca_parameters"]["kernels"]:

        config_key = f"{kpca_kernel}_{kpca_n_components}"
        
        print(f"\n{'='*60}")
        print(f"### PROCESSING WITH KPCA KERNEL: {kpca_kernel.upper()}, N_COMPONENTS: {kpca_n_components} ###")
        print(f"{'='*60}\n")

        # Full preprocessing pipeline with KPCA and LDA:
        # flatten -> normalize -> standardize -> KPCA -> LDA -> ravel labels
        desired_x_train, desired_x_test, desired_y_train, desired_y_test = DataHelper.full_preprocessing_pipeline_with_kpca_and_lda(
            desired_x_train, desired_x_test, desired_y_train, desired_y_test,
            kpca_kernel=kpca_kernel,
            kpca_n_components=kpca_n_components
        )

        # Evaluate all models
        model_results = MainHelper.evaluate_all_models(
            desired_x_train, desired_x_test,
            desired_y_train, desired_y_test,
            program_basic_info["linear_parameters"],
            program_basic_info["rbf_parameters"]
        )

        # Store results for this configuration
        all_kernel_results[config_key] = {
            "kpca_kernel": kpca_kernel,
            "kpca_n_components": kpca_n_components,
            **model_results,
            "desired_x_test": desired_x_test,
            "desired_y_test": desired_y_test
        }

    # -------------------------------
    # ----------- RESULTS -----------
    # -------------------------------

    print("\n### RESULTS WITH KPCA + LDA ###\n")
    print("\n[INFO] Sorted by test accuracy - Time in minutes\n")

    all_results = []
    best_results = []

    # Process results for each configuration
    for config_key, config_data in all_kernel_results.items():

        config_info = {
            "kpca_kernel": config_data["kpca_kernel"],
            "kpca_n_components": config_data["kpca_n_components"]
        }
        
        config_all, config_best = MainHelper.aggregate_results(config_data, config_info)

        all_results.extend(config_all)
        best_results.extend(config_best)

    MainHelper.print_sorted_results(all_results, "ALL RESULTS")
    MainHelper.print_sorted_results(best_results, "BEST RESULTS")

    # Print best model info for each configuration
    for config_key, config_data in all_kernel_results.items():

        kernel = config_data["kpca_kernel"]
        n_components = config_data["kpca_n_components"]

        print(f"\n### BEST RESULTS FOR KPCA KERNEL: {kernel.upper()}, N_COMPONENTS: {n_components} ###\n")

        # Print best model info
        DataHelper.print_best_model_info(best_model=config_data["best_linear_svm"], best_title=f"LINEAR SVM (KPCA: {kernel.upper()}, n={n_components})")
        DataHelper.print_best_model_info(best_model=config_data["best_rbf_svm"], best_title=f"RBF SVM (KPCA: {kernel.upper()}, n={n_components})")

    # -------------------------------
    # -------- VISUALIZATIONS -------
    # -------------------------------

    print("\n### VISUALIZATIONS ###\n")
    dataset_type = program_basic_info["desired_data"]

    # Visualize best model from each configuration
    for config_key, config_data in all_kernel_results.items():
        config_info = {
            "kpca_kernel": config_data["kpca_kernel"],
            "kpca_n_components": config_data["kpca_n_components"]
        }
        MainHelper.create_visualizations(
            config_data, original_x_test, config_data["desired_y_test"], 
            dataset_type, config_info
        )

    MainHelper.plot_metrics(best_results, dataset_type)

# Program execution
if __name__ == '__main__':

    MainHelper.main_execution(main_method=main)

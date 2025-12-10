import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer


class DataHelper:
    """
    Responsible for preprocessing / editing samples
    (e.g., minimizing, flattening, standardizing, raveling labels, applying PCA)
    """

    @staticmethod
    def minimize_samples(x_train, x_test, y_train, y_test, max_train_samples=4000, max_test_samples=500):
        """
        Randomly subsample the train and test sets to reduce dataset size.
        """

        print("[MINIMIZE SAMPLES] Start...")
        np.random.seed(0)
        train_idx = np.random.choice(len(x_train), max_train_samples, replace=False)
        test_idx = np.random.choice(len(x_test), max_test_samples, replace=False)
        print("[MINIMIZE SAMPLES] Finished\n")

        return x_train[train_idx], x_test[test_idx], y_train[train_idx], y_test[test_idx]

    # Flatten images to vectors
    @staticmethod
    def flatten_samples(given_x_train, given_x_test):
        """
        Reshape image tensors into 2D matrices of flattened feature vectors.
        """
        x_train = given_x_train.reshape(given_x_train.shape[0], -1)
        x_test = given_x_test.reshape(given_x_test.shape[0], -1)
        return x_train, x_test

    @staticmethod
    def normalize_samples(given_x_train, given_x_test):
        """
        Normalize feature vectors (e.g., L2 norm = 1).
        """

        print("[NORMALIZATION] Start...")
        normalizer = Normalizer()
        x_train = normalizer.fit_transform(given_x_train)
        x_test = normalizer.transform(given_x_test)
        print("[NORMALIZATION] Finished\n")

        return x_train, x_test

    @staticmethod
    def standardize_samples(given_x_train, given_x_test):
        """
        Standardize features to zero mean and unit variance.
        This z-score standardization is typically done before PCA.
        """
        print("[STANDARDIZATION] Start...")
        scaler = StandardScaler()
        x_train = scaler.fit_transform(given_x_train)
        x_test = scaler.transform(given_x_test)
        print("[STANDARDIZATION] Finished\n")

        return x_train, x_test

    # Remove extra label dimension  (shape becomes (N,) instead of (N,1))
    @staticmethod
    def ravel_samples(given_y_train, given_y_test):
        """
        Remove extra label dimensions so that labels become 1D arrays.
        """
        y_train = given_y_train.ravel()
        y_test = given_y_test.ravel()
        return y_train, y_test

    @staticmethod
    def pca_samples(given_x_train, given_x_test):
        """
        Apply PCA retaining 90% of variance and transform train and test features.
        Assumes that inputs are already flattened and standardized.
        """

        print("[PCA] Apply...")
        pca = PCA(n_components=0.90)
        x_train = pca.fit_transform(given_x_train)
        x_test = pca.transform(given_x_test)
        print("[PCA] Applied\n")

        print(f"Training set after PCA: {x_train.shape}")
        print(f"Testing set after PCA: {x_test.shape}\n")

        return x_train, x_test

    @staticmethod
    def full_preprocessing_pipeline(given_x_train, given_x_test, given_y_train, given_y_test):
        """
        Apply the full preprocessing pipeline:
        1. Flatten images
        2. Normalize feature vectors
        3. Standardize features
        4. Apply PCA
        5. Ravel labels
        Returns: (x_train, x_test, y_train, y_test)
        """

        # Flatten data
        x_train, x_test = DataHelper.flatten_samples(given_x_train, given_x_test)

        # Normalize data
        x_train, x_test = DataHelper.normalize_samples(x_train, x_test)

        # Standardize data
        x_train, x_test = DataHelper.standardize_samples(x_train, x_test)

        # Apply PCA
        x_train, x_test = DataHelper.pca_samples(x_train, x_test)

        # Ravel labels
        y_train, y_test = DataHelper.ravel_samples(given_y_train, given_y_test)

        return x_train, x_test, y_train, y_test

    @staticmethod
    def print_best_model_info(best_model, best_title):
        best_test_acc = best_model["test_accuracy"]
        print(f"[BEST {best_title}] Test Accuracy: {best_test_acc:.4f}")
        print(f"[BEST {best_title}] Parameters: {best_model.get('params', {})}\n")
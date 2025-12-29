import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.kernel_approximation import Nystroem
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
    def kpca_samples(given_x_train, given_x_test, kernel, n_components):
        """
        Apply KPCA with the given kernel, n_components and gamma and transform train and test features.
        Assumes that inputs are already flattened and standardized.
        """

        # Apply Kernel PCA
        # Set gamma for kernels that need it (rbf and poly)
        if kernel in ["rbf", "poly"]:
            kpca_gamma = 1.0 / given_x_train.shape[1]  # safe default gamma (1/n_features)
        else:
            kpca_gamma = None

        print("[KPCA] Apply...")

        # For CIFAR10 dataset
        kpca = KernelPCA(n_components, kernel=kernel, gamma=kpca_gamma)

        # For SVHN dataset
        # Due to the quadratic memory complexity of Kernel PCA, used the Nystroem approximation to
        # obtain a scalable kernel-based feature mapping
        # Approximate KPCA
        # kpca = Nystroem(kernel=kernel, n_components=n_components, gamma=kpca_gamma)

        x_train = kpca.fit_transform(given_x_train)
        x_test = kpca.transform(given_x_test)
        print("[KPCA] Applied\n")
        
        print(f"Training set after KPCA: {x_train.shape}")
        print(f"Testing set after KPCA: {x_test.shape}\n")
        
        return x_train, x_test

    @staticmethod
    def lda_samples(given_x_train, given_x_test, given_y_train):
        """
        Apply Linear Discriminant Analysis (LDA) to transform train and test features.
        Assumes that inputs are already flattened and standardized.
        LDA is a supervised method, so it requires training labels.
        """

        # LDA max components = min(n_classes - 1, n_features)
        n_classes = len(np.unique(given_y_train))
        n_components = min(n_classes - 1, given_x_train.shape[1])

        print(f"[LDA] Apply (n_components = {n_components})...")
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        x_train = lda.fit_transform(given_x_train, given_y_train)
        x_test = lda.transform(given_x_test)
        print("[LDA] Applied\n")
        
        print(f"Training set after LDA: {x_train.shape}")
        print(f"Testing set after LDA: {x_test.shape}\n")
        
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
    def full_preprocessing_pipeline_with_kpca_and_lda(given_x_train, given_x_test, given_y_train, given_y_test, kpca_kernel, kpca_n_components):
        """
        Apply the extra full preprocessing pipeline:
        1. Flatten images
        2. Normalize feature vectors
        3. Standardize features
        4. Apply Kernel PCA (KPCA)
        5. Apply Linear Discriminant Analysis (LDA)
        6. Ravel labels
        Returns: (x_train, x_test, y_train, y_test)
        """

        # Flatten data
        x_train, x_test = DataHelper.flatten_samples(given_x_train, given_x_test)

        # Normalize data
        x_train, x_test = DataHelper.normalize_samples(x_train, x_test)

        # Standardize data
        x_train, x_test = DataHelper.standardize_samples(x_train, x_test)

        # Apply Kernel PCA
        x_train, x_test = DataHelper.kpca_samples(x_train, x_test, kpca_kernel, kpca_n_components)

        # Apply LDA
        x_train, x_test = DataHelper.lda_samples(x_train, x_test, given_y_train)

        # Ravel labels
        y_train, y_test = DataHelper.ravel_samples(given_y_train, given_y_test)

        return x_train, x_test, y_train, y_test

    @staticmethod
    def print_best_model_info(best_model, best_title):
        best_test_acc = best_model["test_accuracy"]
        print(f"[BEST {best_title}] Test Accuracy: {best_test_acc:.4f}")
        print(f"[BEST {best_title}] Parameters: {best_model.get('params', {})}\n")
from tensorflow.keras.datasets import cifar10
import numpy as np
from scipy.io import loadmat
import os
import requests


class DatasetHelper:
    """
    Responsible only for loading / reading datasets from disk.
    All preprocessing / editing of samples remains in DataHelper.
    """

    @staticmethod
    def load_desired_data(desired_data):
        """
        Dispatch loading based on dataset name.
        Returns raw pixel values in [0, 255].
        """
        if desired_data == "cifar10":
            return DatasetHelper.load_cifar10_data()
        elif desired_data == "svhn":
            return DatasetHelper.load_svhn_data()
        else:
            raise Exception("[ERROR] Wrong desired data was given!")

    @staticmethod
    def load_cifar10_data():
        """
        Load CIFAR-10 dataset using Keras helper (raw pixels in [0, 255]).
        """
        print("[LOAD CIFAR10] Start...")
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        print("[LOAD CIFAR10] Finished\n")

        print(f"Training set: {x_train.shape}")
        print(f"Testing set: {x_test.shape}\n")

        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def custom_load_mat(path):

        """
        Helper to load SVHN .mat files.
        """
        data = loadmat(path)
        x = data["X"]  # shape: (32, 32, 3, N)
        y = data["y"].flatten()  # shape: (N,) # flatten

        # Move images to (N, 32, 32, 3)
        x = np.transpose(x, (3, 0, 1, 2))

        # SVHN uses label "10" to represent "0"
        y[y == 10] = 0
        return x, y

    @staticmethod
    def load_svhn_data():
        """
        Load SVHN dataset (raw pixels in [0, 255]).

        First tries to read local .mat files from ./svhn_data.
        If they are not found, downloads them from the official URL
        using HTTP requests and then loads them.
        """
        print("[LOAD SVHN] Start...")

        data_dir = "./svhn_data"
        os.makedirs(data_dir, exist_ok=True)

        train_path = os.path.join(data_dir, "train_32x32.mat")
        test_path = os.path.join(data_dir, "test_32x32.mat")

        # URLs from the official SVHN website
        train_url = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
        test_url = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"

        def download_if_needed(url, path):
            if not os.path.exists(path):
                print(f"[LOAD SVHN] Downloading {url} ...")
                resp = requests.get(url)
                resp.raise_for_status()
                with open(path, "wb") as f:
                    f.write(resp.content)
                print(f"[LOAD SVHN] Saved to {path}")

        # Ensure files exist (download if missing)
        download_if_needed(train_url, train_path)
        download_if_needed(test_url, test_path)

        # Load all sets (no normalization here)
        x_train, y_train = DatasetHelper.custom_load_mat(train_path)
        x_test, y_test = DatasetHelper.custom_load_mat(test_path)

        print("[LOAD SVHN] Finished\n")

        print(f"Training set: {x_train.shape}")
        print(f"Testing set: {x_test.shape}\n")

        return (x_train, y_train), (x_test, y_test)

from tensorflow.keras.datasets import cifar10
import numpy as np
from scipy.io import loadmat
import os
import requests
from PIL import Image
import glob


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
        elif desired_data == "faces":
            return DatasetHelper.load_faces_data()
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

    @staticmethod
    def load_faces_data():
        """
        Load faces/core images dataset from local folder containing PNG images.
        
        Reads PNG images from subfolders in faces_data directory.
        Each subfolder represents an emotion (label), and the folder name is used as the label.
        Data is split into 60% training and 40% testing, maintaining class balance.
        
        Returns:
        --------
        (x_train, y_train), (x_test, y_test) : tuple
            Training and testing sets with images and labels.
            Images are in shape (N, H, W, 3) for RGB images.
            Labels are numeric (0, 1, 2, ...) based on folder names.
        """
        # Default values
        faces_data_dir = "./faces_data"
        train_split = 0.6  # 60% train, 40% test
        
        print("[LOAD FACES] Start...")
        
        if not os.path.exists(faces_data_dir):
            raise Exception(f"[ERROR] Faces data directory not found: {faces_data_dir}")
        
        images = []
        labels = []
        label_to_id = {}
        current_label_id = 0
        standard_size = None  # Will be set from first image
        
        # Get all subdirectories (emotion folders)
        subdirs = [d for d in os.listdir(faces_data_dir) 
                   if os.path.isdir(os.path.join(faces_data_dir, d)) and not d.startswith('.')]
        
        if len(subdirs) == 0:
            raise Exception(f"[ERROR] No subdirectories found in {faces_data_dir}. Expected emotion folders.")
        
        print(f"[LOAD FACES] Found {len(subdirs)} emotion folders: {sorted(subdirs)}")
        
        # Read images from each subfolder (emotion)
        for emotion_folder in sorted(subdirs):
            
            emotion_path = os.path.join(faces_data_dir, emotion_folder)
            
            # Assign label ID to this emotion
            if emotion_folder not in label_to_id:
                label_to_id[emotion_folder] = current_label_id
                current_label_id += 1
            
            label_id = label_to_id[emotion_folder]
            
            # Find all PNG images in this emotion folder
            png_files = glob.glob(os.path.join(emotion_path, "*.png")) + \
                       glob.glob(os.path.join(emotion_path, "*.PNG"))
            
            print(f"[LOAD FACES] Loading {len(png_files)} images from '{emotion_folder}' (label: {label_id})")
            
            for img_path in sorted(png_files):
                try:
                    img = Image.open(img_path)
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        if img.mode == 'L':
                            img = img.convert('RGB')
                        elif img.mode == 'RGBA':
                            img = img.convert('RGB')
                        else:
                            img = img.convert('RGB')
                    
                    # Resize image to a standard size (use first image size as reference)
                    if standard_size is None:
                        # Use first image's size as the standard size
                        standard_size = img.size  # (width, height)
                        print(f"[LOAD FACES] Standard image size set to: {standard_size}")
                    else:
                        # Resize to match standard size
                        if img.size != standard_size:
                            img = img.resize(standard_size, Image.Resampling.LANCZOS)
                    
                    img_array = np.array(img, dtype=np.uint8)
                    images.append(img_array)
                    labels.append(label_id)
                except Exception as e:
                    print(f"[WARNING] Could not load image {img_path}: {e}")
        
        if len(images) == 0:
            raise Exception(f"[ERROR] No PNG images found in {faces_data_dir}")
        
        images = np.array(images)  # shape: (N, H, W, 3)
        labels = np.array(labels, dtype=np.int32)
        
        print(f"[LOAD FACES] Loaded {len(images)} images with {len(np.unique(labels))} unique labels")
        print(f"[LOAD FACES] Label mapping: {label_to_id}")
        
        # Split into train and test sets (60% train, 40% test) maintaining class balance
        np.random.seed(0)
        x_train_list = []
        y_train_list = []
        x_test_list = []
        y_test_list = []
        
        # Split each class separately to maintain balance
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            n_label_samples = len(label_indices)
            n_train_label = int(n_label_samples * train_split)
            
            # Shuffle indices for this class
            np.random.shuffle(label_indices)
            
            train_indices = label_indices[:n_train_label]
            test_indices = label_indices[n_train_label:]
            
            x_train_list.append(images[train_indices])
            y_train_list.append(labels[train_indices])
            x_test_list.append(images[test_indices])
            y_test_list.append(labels[test_indices])
        
        # Concatenate all classes
        x_train = np.concatenate(x_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        x_test = np.concatenate(x_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        
        # Shuffle train and test sets
        train_shuffle_idx = np.random.permutation(len(x_train))
        test_shuffle_idx = np.random.permutation(len(x_test))
        
        x_train = x_train[train_shuffle_idx]
        y_train = y_train[train_shuffle_idx]
        x_test = x_test[test_shuffle_idx]
        y_test = y_test[test_shuffle_idx]
        
        n_train = len(x_train)
        n_test = len(x_test)
        print(f"[LOAD FACES] Splitting data: {n_train} images ({train_split*100:.0f}%) for training, {n_test} images ({(1-train_split)*100:.0f}%) for testing")
        
        # Reshape y to match other datasets format (N, 1)
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        
        print("[LOAD FACES] Finished\n")
        
        print(f"Training set: {x_train.shape}")
        print(f"Testing set: {x_test.shape}\n")
        
        return (x_train, y_train), (x_test, y_test)

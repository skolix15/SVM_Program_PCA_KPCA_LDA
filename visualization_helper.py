from matplotlib import pyplot as plt
import numpy as np
import os
import re
import glob


class VisualizationHelper:

    @staticmethod
    def _sanitize_filename(filename):
        """Remove invalid characters from filename."""
        filename = re.sub(r'[<>:"/\\|?*\[\]]', '_', filename)
        filename = filename.replace(' ', '_')
        return filename

    @staticmethod
    def save_grid_examples(indices, images, y_true, y_pred, title, dataset_type):

        # Get class names based on dataset type
        if dataset_type == "cifar10":
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        elif dataset_type == "svhn":
            class_names = [str(i) for i in range(10)]  # digits 0–9
        elif dataset_type == "faces":
            # Get emotion names from faces_data subfolders
            faces_data_dir = "./faces_data"
            if os.path.exists(faces_data_dir):
                subdirs = [d for d in os.listdir(faces_data_dir) 
                          if os.path.isdir(os.path.join(faces_data_dir, d)) and not d.startswith('.')]
                # Sort to match the label order (labels are assigned based on sorted folder names)
                class_names = sorted(subdirs)
            else:
                # Fallback: use generic names
                max_label = max(int(y_true.max()), int(y_pred.max())) if len(y_true) > 0 else 0
                class_names = [f"Emotion_{i}" for i in range(max_label + 1)]
        else:
            raise Exception(f"[ERROR] Wrong dataset type was given: {dataset_type}")

        plt.figure(figsize=(12, 4))

        for idx, sample_idx in enumerate(indices):
            img = images[sample_idx]
            
            # Reshape if needed (handle flattened images)
            if len(img.shape) == 1:
                side = int(np.sqrt(img.shape[0] / 3))
                img = img.reshape(side, side, 3) if side * side * 3 == img.shape[0] else img.reshape(32, 32, 3)

            # Get labels
            true_label_idx = int(y_true[sample_idx] if y_true.ndim == 1 else y_true[sample_idx][0])
            pred_label_idx = int(y_pred[sample_idx] if y_pred.ndim == 1 else y_pred[sample_idx][0])
            
            true_label = class_names[true_label_idx] if true_label_idx < len(class_names) else f"Class_{true_label_idx}"
            pred_label = class_names[pred_label_idx] if pred_label_idx < len(class_names) else f"Class_{pred_label_idx}"

            plt.subplot(1, len(indices), idx + 1)

            # Detect if normalized
            if images[sample_idx].max() <= 1:
                plt.imshow(img)  # float32 normalized image
            else:
                plt.imshow(img.astype(np.uint8))  # uint8 image

            plt.axis("off")
            plt.title(f"T: {true_label}\nP: {pred_label}")

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        filename = VisualizationHelper._sanitize_filename(f"{title}.png")
        save_path = os.path.join('visualizations', filename)
        os.makedirs('visualizations', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] {save_path}")
        plt.close()

    @staticmethod
    def random_visualization(size, model_info, model_name, desired_y_test, x_test, dataset_type):

        class_y_test_pred = model_info["y_test_pred"]

        # Find correct and incorrect indices
        correct_idx = np.where(class_y_test_pred == desired_y_test)[0]
        incorrect_idx = np.where(class_y_test_pred != desired_y_test)[0]

        # Random correct classifications
        correct_title = f"[{model_name}] Correct Classifications - [{dataset_type.upper()}]"
        random_indexes = np.random.choice(correct_idx, size=size, replace=False)
        VisualizationHelper.save_grid_examples(
            random_indexes,
            x_test, desired_y_test, class_y_test_pred,
            correct_title,
            dataset_type
        )

        # Random incorrect classifications
        incorrect_title = f"[{model_name}] Incorrect Classifications - [{dataset_type.upper()}]"
        random_indexes = np.random.choice(incorrect_idx, size=size, replace=False)
        VisualizationHelper.save_grid_examples(
            random_indexes,
            x_test, desired_y_test, class_y_test_pred,
            incorrect_title,
            dataset_type
        )

    @staticmethod
    def plot_performance_metrics(best_results_data, dataset_type):
        """Plot performance metrics comparison for all best models."""
        model_names = [r['Model'] for r in best_results_data]
        metrics = [
            ([float(r['Train Accuracy']) for r in best_results_data], 'Train Accuracy', '#3498db'),
            ([float(r['Test Accuracy']) for r in best_results_data], 'Test Accuracy', '#2ecc71'),
            ([float(r['Precision']) for r in best_results_data], 'Precision', '#e74c3c'),
            ([float(r['Recall']) for r in best_results_data], 'Recall', '#f39c12'),
            ([float(r['F1 Score']) for r in best_results_data], 'F1 Score', '#9b59b6')
        ]
        
        x, width = np.arange(len(model_names)), 0.15
        fig, ax = plt.subplots(figsize=(12, 6))
        
        max_val = max(max(v[0]) for v in metrics)
        for idx, (values, label, color) in enumerate(metrics):
            offset = (idx - 2) * width
            ax.bar(x + offset, values, width, label=label, color=color, alpha=0.8, edgecolor='black')
            for i, val in enumerate(values):
                ax.text(float(x[i]) + offset, val + max_val * 0.02, f'{val:.3f}', 
                       ha='center', va='bottom', fontsize=7, fontweight='bold')
        
        ax.set(xlabel='Models', ylabel='Score', title=f'{dataset_type.upper()} - Performance Metrics', ylim=[0, max_val * 1.2])
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        filename = VisualizationHelper._sanitize_filename(f'Performance_Metrics_{dataset_type.upper()}.png')
        save_path = os.path.join('visualizations', filename)
        os.makedirs('visualizations', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] {save_path}")
        plt.close()

    @staticmethod
    def plot_time_metrics(best_results_data, dataset_type):
        """Plot time metrics comparison for all best models."""
        model_names = [r['Model'] for r in best_results_data]
        train_time = [float(r['Train Time']) for r in best_results_data]
        test_time = [float(r['Test Time']) for r in best_results_data]
        
        x, width = np.arange(len(model_names)), 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.bar(x - width/2, train_time, width, label='Train Time (min)', color='#16a085', alpha=0.8, edgecolor='black')
        ax.bar(x + width/2, test_time, width, label='Test Time (min)', color='#27ae60', alpha=0.8, edgecolor='black')
        
        max_time = max(max(train_time), max(test_time)) or 1
        for i, (train_val, test_val) in enumerate(zip(train_time, test_time)):
            ax.text(float(x[i]) - width/2, train_val + max_time * 0.02, f'{train_val:.4f}', 
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
            ax.text(float(x[i]) + width/2, test_val + max_time * 0.02, f'{test_val:.4f}', 
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set(xlabel='Models', ylabel='Time (minutes)', title=f'{dataset_type.upper()} - Time Metrics', ylim=[0, max_time * 1.2])
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        filename = VisualizationHelper._sanitize_filename(f'Time_Metrics_{dataset_type.upper()}.png')
        save_path = os.path.join('visualizations', filename)
        os.makedirs('visualizations', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] {save_path}")
        plt.close()

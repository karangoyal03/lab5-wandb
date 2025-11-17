"""Data loading and preprocessing utilities."""

import os
import numpy as np
import xgboost as xgb
import wandb
import pandas as pd
from urllib.request import urlretrieve
from config import DATASET_URL, DATASET_FILE, TRAIN_SPLIT


def download_dataset(url: str, filepath: str) -> None:
    """Download dataset from URL if it doesn't exist."""
    if not os.path.exists(filepath):
        print(f"Downloading dataset from {url}...")
        urlretrieve(url, filepath)
        print("Download complete.")
    else:
        print(f"Dataset already exists at {filepath}")


def log_dataset_statistics(train_X: np.ndarray, train_Y: np.ndarray, 
                          test_X: np.ndarray, test_Y: np.ndarray) -> None:
    """
    Log dataset statistics and visualizations to Wandb.
    
    Args:
        train_X: Training features
        train_Y: Training labels
        test_X: Test features
        test_Y: Test labels
    """
    # Log dataset sizes
    wandb.log({
        "dataset/train_size": len(train_Y),
        "dataset/test_size": len(test_Y),
        "dataset/total_size": len(train_Y) + len(test_Y),
        "dataset/train_split": TRAIN_SPLIT,
        "dataset/num_features": train_X.shape[1],
        "dataset/num_classes": len(np.unique(train_Y))
    })
    
    # Log class distribution
    train_class_counts = np.bincount(train_Y.astype(int))
    test_class_counts = np.bincount(test_Y.astype(int))
    
    class_distribution = {
        f"dataset/train_class_{i}_count": int(count) 
        for i, count in enumerate(train_class_counts)
    }
    class_distribution.update({
        f"dataset/test_class_{i}_count": int(count) 
        for i, count in enumerate(test_class_counts)
    })
    wandb.log(class_distribution)
    
    # Create and log class distribution visualization
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Train distribution
        axes[0].bar(range(len(train_class_counts)), train_class_counts)
        axes[0].set_title('Train Set Class Distribution')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Count')
        axes[0].set_xticks(range(len(train_class_counts)))
        
        # Test distribution
        axes[1].bar(range(len(test_class_counts)), test_class_counts)
        axes[1].set_title('Test Set Class Distribution')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Count')
        axes[1].set_xticks(range(len(test_class_counts)))
        
        plt.tight_layout()
        wandb.log({"dataset/class_distribution": wandb.Image(fig)})
        plt.close(fig)
    except ImportError:
        pass  # matplotlib not available
    
    # Log sample data as table
    try:
        sample_size = min(100, len(train_X))
        sample_indices = np.random.choice(len(train_X), sample_size, replace=False)
        sample_data = train_X[sample_indices]
        sample_labels = train_Y[sample_indices]
        
        # Create DataFrame for table
        df = pd.DataFrame(sample_data, columns=[f"feature_{i}" for i in range(train_X.shape[1])])
        df['label'] = sample_labels
        
        wandb.log({"dataset/train_samples": wandb.Table(dataframe=df.head(50))})
    except Exception as e:
        print(f"Could not log data table: {e}")


def load_data(filepath: str, log_stats: bool = True) -> tuple:
    """
    Load and preprocess the dermatology dataset.
    
    Args:
        filepath: Path to the dataset file
        log_stats: Whether to log dataset statistics to Wandb
        
    Returns:
        Tuple of (train_X, train_Y, test_X, test_Y, xg_train, xg_test)
    """
    # Load data with converters for missing values and label adjustment
    data = np.loadtxt(
        filepath,
        delimiter=',',
        converters={
            33: lambda x: int(x == '?'),  # Convert missing values
            34: lambda x: int(x) - 1  # Convert labels to 0-based indexing
        }
    )
    
    sz = data.shape
    
    # Split into train and test
    split_idx = int(sz[0] * TRAIN_SPLIT)
    train = data[:split_idx, :]
    test = data[split_idx:, :]
    
    # Separate features and labels
    train_X = train[:, :33]
    train_Y = train[:, 34]
    test_X = test[:, :33]
    test_Y = test[:, 34]
    
    # Log dataset statistics to Wandb
    if log_stats:
        log_dataset_statistics(train_X, train_Y, test_X, test_Y)
    
    # Create XGBoost DMatrix objects
    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_test = xgb.DMatrix(test_X, label=test_Y)
    
    return train_X, train_Y, test_X, test_Y, xg_train, xg_test


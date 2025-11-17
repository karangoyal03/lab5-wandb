"""Main entry point for XGBoost model training with Wandb."""

import wandb
import platform
import psutil
from config import WANDB_PROJECT, WANDB_RUN_NAME, DATASET_URL, DATASET_FILE
from data_loader import download_dataset, load_data
from model_trainer import train_model, evaluate_model


def log_system_info() -> None:
    """Log system information to Wandb."""
    try:
        wandb.log({
            "system/platform": platform.platform(),
            "system/python_version": platform.python_version(),
            "system/cpu_count": psutil.cpu_count(),
            "system/memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "system/memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        })
    except Exception as e:
        print(f"Could not log system info: {e}")


def main():
    """Main function to orchestrate the training pipeline."""
    # Login to Wandb
    wandb.login()
    
    # Initialize Wandb run with additional metadata
    run = wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        tags=["xgboost", "dermatology", "classification"],
        notes="XGBoost model training with comprehensive Wandb logging"
    )
    
    try:
        # Log system information
        log_system_info()
        
        # Download dataset if needed
        download_dataset(DATASET_URL, DATASET_FILE)
        
        # Log dataset as artifact
        try:
            dataset_artifact = wandb.Artifact("dermatology_dataset", type="dataset")
            dataset_artifact.add_file(DATASET_FILE)
            wandb.log_artifact(dataset_artifact)
        except Exception as e:
            print(f"Could not log dataset artifact: {e}")
        
        # Load and preprocess data
        train_X, train_Y, test_X, test_Y, xg_train, xg_test = load_data(DATASET_FILE, log_stats=True)
        
        # Train model
        bst, xg_train = train_model(xg_train, xg_test, run)
        
        # Evaluate model
        error_rate = evaluate_model(bst, xg_test, test_Y, train_X, run)
        
        print(f"Training completed. Error rate: {error_rate}")
        
        # Log final summary
        wandb.alert(
            title="Training Complete",
            text=f"Model training finished with error rate: {error_rate:.4f}"
        )
        
    finally:
        # Finish Wandb run
        run.finish()


if __name__ == "__main__":
    main()


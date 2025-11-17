"""Model training and evaluation utilities."""

import os
import numpy as np
import wandb
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from config import MODEL_PARAMS, NUM_ROUNDS


def train_model(xg_train: xgb.DMatrix, xg_test: xgb.DMatrix, run) -> tuple:
    """
    Train XGBoost model with Wandb logging.
    
    Args:
        xg_train: Training data as XGBoost DMatrix
        xg_test: Test data as XGBoost DMatrix
        run: Wandb run object
        
    Returns:
        Tuple of (Trained XGBoost booster model, training data DMatrix)
    """
    # Update wandb config with model parameters
    wandb.config.update(MODEL_PARAMS)
    
    # Create watchlist for monitoring
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    
    # Train model with Wandb callback
    bst = xgb.train(
        MODEL_PARAMS,
        xg_train,
        NUM_ROUNDS,
        watchlist,
        callbacks=[wandb.xgboost.WandbCallback()]
    )
    
    return bst, xg_train


def log_feature_importance(bst: xgb.Booster, num_features: int = 20) -> None:
    """
    Log feature importance to Wandb.
    
    Args:
        bst: Trained XGBoost booster model
        num_features: Number of top features to log
    """
    try:
        # Get feature importance
        importance = bst.get_score(importance_type='gain')
        
        # Sort by importance
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        # Log top features
        top_features = sorted_importance[:num_features]
        feature_names = [f"feature_{int(k[1:])}" for k, v in top_features]
        feature_values = [v for k, v in top_features]
        
        # Log as bar chart
        wandb.log({
            "feature_importance": wandb.plot.bar(
                wandb.Table(
                    data=[[name, val] for name, val in zip(feature_names, feature_values)],
                    columns=["Feature", "Importance"]
                ),
                "Feature", "Importance",
                title="Top Feature Importance"
            )
        })
        
        # Log individual feature importances
        for i, (name, val) in enumerate(zip(feature_names, feature_values)):
            wandb.log({f"feature_importance/{name}": val})
            
    except Exception as e:
        print(f"Could not log feature importance: {e}")


def log_model_artifact(bst: xgb.Booster, artifact_name: str = "xgboost_model") -> None:
    """
    Save and log model as Wandb artifact.
    
    Args:
        bst: Trained XGBoost booster model
        artifact_name: Name for the artifact
    """
    try:
        model_path = f"{artifact_name}.json"
        bst.save_model(model_path)
        
        artifact = wandb.Artifact(artifact_name, type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        
        # Clean up local file
        if os.path.exists(model_path):
            os.remove(model_path)
    except Exception as e:
        print(f"Could not log model artifact: {e}")


def evaluate_model(bst: xgb.Booster, xg_test: xgb.DMatrix, test_Y: np.ndarray, 
                  train_X: np.ndarray, run) -> float:
    """
    Evaluate model and log comprehensive results to Wandb.
    
    Args:
        bst: Trained XGBoost booster model
        xg_test: Test data as XGBoost DMatrix
        test_Y: True test labels
        train_X: Training features (for feature importance)
        run: Wandb run object
        
    Returns:
        Error rate as float
    """
    # Get predictions
    pred = bst.predict(xg_test)
    
    # Calculate metrics
    error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
    accuracy = accuracy_score(test_Y, pred)
    precision = precision_score(test_Y, pred, average='weighted', zero_division=0)
    recall = recall_score(test_Y, pred, average='weighted', zero_division=0)
    f1 = f1_score(test_Y, pred, average='weighted', zero_division=0)
    
    print(f'Test error using softmax = {error_rate}')
    print(f'Test accuracy = {accuracy}')
    print(f'Test precision = {precision}')
    print(f'Test recall = {recall}')
    print(f'Test F1 score = {f1}')
    
    # Log metrics to wandb summary
    run.summary.update({
        'Error Rate': error_rate,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })
    
    # Log metrics as wandb metrics
    wandb.log({
        'metrics/error_rate': error_rate,
        'metrics/accuracy': accuracy,
        'metrics/precision': precision,
        'metrics/recall': recall,
        'metrics/f1_score': f1
    })
    
    # Plot confusion matrix
    class_labels = [0., 1., 2., 3., 4., 5.]
    wandb.sklearn.plot_confusion_matrix(test_Y, pred, class_labels)
    
    # Log classification report
    try:
        report = classification_report(test_Y, pred, output_dict=True, zero_division=0)
        wandb.log({"classification_report": wandb.Table(
            data=[[k, v.get('precision', 0), v.get('recall', 0), v.get('f1-score', 0), v.get('support', 0)] 
                  for k, v in report.items() if isinstance(v, dict)],
            columns=["Class", "Precision", "Recall", "F1-Score", "Support"]
        )})
    except Exception as e:
        print(f"Could not log classification report: {e}")
    
    # Log feature importance
    log_feature_importance(bst, num_features=20)
    
    # Log model as artifact
    log_model_artifact(bst)
    
    return error_rate


import lightgbm as lgb
import joblib
from pathlib import Path

def get_robust_classifier():
    """
    Returns an uninitialized LightGBM classifier optimized for our feature vectors.
    Uses class_weight='balanced' for addressing class imbalance explicitly without destroying gradients.
    """
    # Define hyperparams - trying GPU if possible. If python lightgbm isn't 
    # compiled with GPU, this might fallback to CPU or raise an error.
    # Note: If it errors out locally with "GPU Tree Learner was not enabled",
    # the user can simply remove `device='gpu'` or `device_type='gpu'` here.
    return lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,             # Allow deeper trees to capture complex decision boundaries
        num_leaves=64,           # Increased leaves to allow splits
        min_data_in_leaf=20,     # Default to allow finer splits on the minority class
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,          
        objective='binary',
        metric='binary_logloss',
        device_type='gpu',
        random_state=42,
        class_weight='balanced'  # Allows LightGBM to natively inflate the REAL gradient weight
    )

def train_model(X_train, y_train, X_val=None, y_val=None):
    """
    Trains the LightGBM classifier properly resolving extreme imbalances.
    """
    model = get_robust_classifier()
    
    # Setup eval sets for early stopping if validation data is provided.
    eval_set = [(X_train, y_train)]
    if X_val is not None and y_val is not None:
        eval_set.append((X_val, y_val))
    
    # We use early_stopping in the fit method using callbacks
    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=True),
        lgb.log_evaluation(period=50)
    ]
    
    model.fit(
        X_train, 
        y_train,
        eval_set=eval_set,
        eval_metric='binary_logloss',
        callbacks=callbacks
    )
    
    return model

def save_model_package(package_dict, output_path):
    """ Safely save the bundled dictionary format (model + scaler + selector + threshold). """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package_dict, str(output_path))
    print(f"Model saved successfully to {output_path}")

def load_model_package(model_path):
    """ Loads the bundled dictionary for inference. """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}.")
    return joblib.load(str(model_path))

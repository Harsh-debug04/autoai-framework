import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

def train_model(df: pd.DataFrame, target_column: str, task: str, model_name: str = 'xgboost'):
    """
    Trains and evaluates a machine learning model.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target variable column.
        task (str): The type of task ('classification' or 'regression').
        model_name (str): The name of the model to train.

    Returns:
        A tuple containing the trained model and its evaluation score.
    """
    print(f"\n--- Starting Model Training ---")
    print(f"Model: {model_name}, Task: {task}, Target: {target_column}")

    try:
        # 1. Separate features (X) and target (y)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # 2. One-hot encode categorical features
        X = pd.get_dummies(X, drop_first=True)

        # Align columns after getting dummies - crucial for prediction later
        # For now, this is handled implicitly by splitting after encoding.

        # 3. Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"Data split into training ({X_train.shape[0]} rows) and testing ({X_test.shape[0]} rows).")

        # 4. Define model mappings
        models = {
            'classification': {
                'xgboost': XGBClassifier(random_state=42, eval_metric='logloss'),
                'lightgbm': LGBMClassifier(random_state=42),
                'catboost': CatBoostClassifier(random_state=42, verbose=0)
            },
            'regression': {
                'xgboost': XGBRegressor(random_state=42),
                'lightgbm': LGBMRegressor(random_state=42),
                'catboost': CatBoostRegressor(random_state=42, verbose=0)
            }
        }

        if task not in models or model_name not in models[task]:
            raise ValueError(f"Unsupported task '{task}' or model '{model_name}'.")

        model = models[task][model_name]

        # 5. Train the model
        print(f"Training {model_name} model...")
        model.fit(X_train, y_train)
        print("Training complete.")

        # 6. Make predictions
        y_pred = model.predict(X_test)

        # 7. Evaluate the model
        if task == 'classification':
            score = accuracy_score(y_test, y_pred)
            print(f"Model Accuracy: {score:.4f}")
        else: # regression
            score = r2_score(y_test, y_pred)
            print(f"Model R-squared: {score:.4f}")

        print("--- Model Training Finished ---")
        return model, score

    except Exception as e:
        print(f"An error occurred during model training: {e}")
        return None, None

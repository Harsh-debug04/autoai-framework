import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from autoai.explainability.explainer import generate_explanation_report

def train_model(df: pd.DataFrame, target_column: str, task: str, model_name: str = 'xgboost', tune_hyperparameters: bool = False, explain: bool = False):
    """
    Trains and evaluates a machine learning model, with an option for hyperparameter tuning.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target variable column.
        task (str): The type of task ('classification' or 'regression').
        model_name (str): The name of the model to train.
        tune_hyperparameters (bool): Whether to perform hyperparameter tuning.
        explain (bool): Whether to generate an explanation report.

    Returns:
        A tuple containing the trained model and its evaluation score on the test set.
    """
    print(f"\n--- Starting Model Training ---")
    print(f"Model: {model_name}, Task: {task}, Target: {target_column}, Tuning: {tune_hyperparameters}")

    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X = pd.get_dummies(X, drop_first=True, dtype=int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            'classification': {'xgboost': XGBClassifier, 'lightgbm': LGBMClassifier, 'catboost': CatBoostClassifier},
            'regression': {'xgboost': XGBRegressor, 'lightgbm': LGBMRegressor, 'catboost': CatBoostRegressor}
        }

        model_class = models[task][model_name]
        best_params = {}

        if tune_hyperparameters:
            print("\n--- Starting Hyperparameter Tuning with Optuna ---")

            def _objective(trial):
                if model_name == 'xgboost':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    }
                elif model_name == 'lightgbm':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    }
                elif model_name == 'catboost':
                    params = {
                        'iterations': trial.suggest_int('iterations', 100, 1000),
                        'depth': trial.suggest_int('depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    }

                model = model_class(**params, random_state=42)
                if model_name == 'catboost':
                    model.set_params(verbose=0)

                X_train_part, X_val, y_train_part, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
                model.fit(X_train_part, y_train_part)
                preds = model.predict(X_val)

                if task == 'classification':
                    return accuracy_score(y_val, preds)
                else:
                    return r2_score(y_val, preds)

            study = optuna.create_study(direction='maximize')
            study.optimize(_objective, n_trials=20)

            best_params = study.best_params
            print(f"--- Tuning Finished ---")
            print(f"Best trial score: {study.best_value:.4f}")
            print(f"Best parameters found: {best_params}")

        final_model = model_class(**best_params, random_state=42)
        if model_name == 'catboost':
             final_model.set_params(verbose=0)

        print("\nTraining final model...")
        final_model.fit(X_train, y_train)
        print("Training complete.")

        y_pred = final_model.predict(X_test)

        if task == 'classification':
            score = accuracy_score(y_test, y_pred)
            print(f"\nFinal Model Accuracy on Test Set: {score:.4f}")
        else:
            score = r2_score(y_test, y_pred)
            print(f"\nFinal Model R-squared on Test Set: {score:.4f}")

        if explain:
            generate_explanation_report(final_model, X_test, y_test, task)

        print("--- Model Training Finished ---")
        return final_model, score

    except Exception as e:
        print(f"An error occurred during model training: {e}")
        return None, None

import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error

def generate_explanation_report(model, X_test: pd.DataFrame, y_test: pd.Series, task: str):
    """
    Generates and prints a comprehensive evaluation and explainability report.

    Args:
        model: The trained machine learning model.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The test target variable.
        task (str): The type of task ('classification' or 'regression').
    """
    print("\n--- Generating Model Evaluation & Explainability Report ---")

    y_pred = model.predict(X_test)

    # --- Enhanced Evaluation Metrics ---
    print("\n--- Performance Metrics ---")
    if task == 'classification':
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    else: # regression
        print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
        print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.4f}")

    # --- SHAP Explainability ---
    print("\n--- SHAP Feature Importance ---")
    try:
        # Use shap.Explainer for compatibility with different model types
        explainer = shap.Explainer(model, X_test)
        shap_values = explainer(X_test)

        # Generate SHAP summary plot
        print("Generating SHAP summary plot...")
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)

        # Save the plot to a file
        plt.savefig("shap_summary.png", bbox_inches='tight')
        plt.close() # Close the plot to free up memory

        print("SHAP summary plot saved as shap_summary.png")

    except Exception as e:
        print(f"Could not generate SHAP plot. Error: {e}")
        print("SHAP plot generation is often only supported for tree-based models like XGBoost, LightGBM, and CatBoost.")

    print("\n--- Report Generation Finished ---")

import argparse
from autoai.core.data_loader import load_data
from autoai.eda.report import generate_eda_report
from autoai.core.cleaner import impute_missing_values
from autoai.models.trainer import train_model

def main():
    """Main function to run the AutoAI pipeline."""
    parser = argparse.ArgumentParser(description="AutoAI: End-to-End Automated Machine Learning")

    # --- Data and Task Specification ---
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the input data file (CSV, text, or image folder)."
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Name of the target column for model training. If not provided, only data loading and EDA will be performed."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="auto",
        choices=["auto", "classification", "regression"],
        help="Type of machine learning task. 'auto' will try to infer the task."
    )

    # --- Feature Engineering & Preprocessing ---
    parser.add_argument(
        '--clean',
        action='store_true',
        help='If set, the data will be automatically cleaned (e.g., missing value imputation).'
    )

    # --- Model Training ---
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "lightgbm", "catboost"],
        help="The machine learning model to train."
    )

    # --- Reporting ---
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to save the generated EDA HTML report. e.g., --report eda_report.html"
    )

    args = parser.parse_args()

    print(f"--- AutoAI Pipeline Started ---")
    print(f"Data path: {args.data}")

    # Load the data
    df = load_data(args.data)

    if df is not None:
        print(f"\nSuccessfully loaded data: {df.shape[0]} rows, {df.shape[1]} columns.")

        # Clean the data if requested
        if args.clean:
            df = impute_missing_values(df)

        # Generate EDA report if requested
        if args.report:
            if not args.report.endswith(".html"):
                print("\nWarning: The report filename should end with .html for best results.")
            generate_eda_report(df, args.report)

        # --- Model Training ---
        if args.target:
            if args.target not in df.columns:
                print(f"Error: Target column '{args.target}' not found in the dataset.")
                return

            # Simple task auto-detection
            if args.task == 'auto':
                if pd.api.types.is_numeric_dtype(df[args.target]) and df[args.target].nunique() > 15:
                    task = 'regression'
                else:
                    task = 'classification'
                print(f"Task auto-detected as: {task}")
            else:
                task = args.task

            train_model(
                df=df,
                target_column=args.target,
                task=task,
                model_name=args.model
            )
        else:
            print("\nNo target column specified. Skipping model training.")

    print(f"\n--- AutoAI Pipeline Finished ---")


if __name__ == "__main__":
    main()

import argparse
from autoai.core.data_loader import load_data

def main():
    """Main function to run the AutoAI pipeline."""
    parser = argparse.ArgumentParser(description="AutoAI: End-to-End Automated Machine Learning")

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the input data file (CSV, text, or image folder)."
    )

    parser.add_argument(
        "--task",
        type=str,
        default="auto",
        choices=["auto", "classification", "regression"],
        help="Type of machine learning task. 'auto' will try to infer the task."
    )

    args = parser.parse_args()

    print(f"Data path: {args.data}")
    print(f"Task: {args.task}")

    # Load the data
    df = load_data(args.data)

    if df is not None:
        print("\nData loaded successfully. First 5 rows:")
        print(df.head())

if __name__ == "__main__":
    main()

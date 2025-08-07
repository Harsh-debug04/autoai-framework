import argparse
from autoai.core.data_loader import load_data
from autoai.eda.report import generate_eda_report

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

    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to save the generated EDA HTML report. e.g., --report eda_report.html"
    )

    args = parser.parse_args()

    print(f"Data path: {args.data}")
    print(f"Task: {args.task}")

    # Load the data
    df = load_data(args.data)

    if df is not None:
        print("\nData loaded successfully. First 5 rows:")
        print(df.head())

        # Generate EDA report if requested
        if args.report:
            if not args.report.endswith(".html"):
                print("\nWarning: The report filename should end with .html for best results.")
            generate_eda_report(df, args.report)

if __name__ == "__main__":
    main()

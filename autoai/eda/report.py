import pandas as pd
from ydata_profiling import ProfileReport

def generate_eda_report(df: pd.DataFrame, output_path: str):
    """
    Generates an EDA report using ydata-profiling and saves it to an HTML file.

    Args:
        df (pd.DataFrame): The input DataFrame to be profiled.
        output_path (str): The path to save the HTML report.
    """
    print(f"Generating EDA report for the dataset...")

    try:
        # Create a ProfileReport object
        profile = ProfileReport(
            df,
            title="Automated EDA Report",
            explorative=True
        )

        # Save the report to an HTML file
        profile.to_file(output_path)

        print(f"EDA report successfully generated and saved to {output_path}")

    except Exception as e:
        print(f"An error occurred during EDA report generation: {e}")

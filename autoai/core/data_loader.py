import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a given file path.
    Currently supports CSV files.

    Args:
        file_path (str): The path to the data file.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    # For now, we'll just handle CSVs.
    # We can add support for other file types later.
    if file_path.endswith('.csv'):
        try:
            df = pd.read_csv(file_path)
            print(f"Successfully loaded data from {file_path}")
            return df
        except FileNotFoundError:
            print(f"Error: The file at {file_path} was not found.")
            return None
        except Exception as e:
            print(f"An error occurred while reading the CSV file: {e}")
            return None
    else:
        print(f"Error: Unsupported file type. Currently, only .csv files are supported.")
        return None

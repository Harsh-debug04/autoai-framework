import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values in a DataFrame.

    It uses the mean for numerical columns and the most frequent value
    for categorical columns.

    Args:
        df (pd.DataFrame): The input DataFrame with potential missing values.

    Returns:
        pd.DataFrame: The DataFrame with missing values imputed.
    """
    print("Starting missing value imputation...")

    # Make a copy to avoid modifying the original DataFrame
    df_cleaned = df.copy()

    # Identify numerical and categorical columns
    numerical_cols = df_cleaned.select_dtypes(include=np.number).columns
    categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns

    # Impute numerical columns
    if not df_cleaned[numerical_cols].isnull().sum().sum() == 0:
        print(f"Imputing {len(numerical_cols)} numerical columns with mean strategy.")
        num_imputer = SimpleImputer(strategy='mean')
        df_cleaned[numerical_cols] = num_imputer.fit_transform(df_cleaned[numerical_cols])
    else:
        print("No missing values found in numerical columns.")

    # Impute categorical columns
    if not df_cleaned[categorical_cols].isnull().sum().sum() == 0:
        print(f"Imputing {len(categorical_cols)} categorical columns with most_frequent strategy.")
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_cleaned[categorical_cols] = cat_imputer.fit_transform(df_cleaned[categorical_cols])
    else:
        print("No missing values found in categorical columns.")

    print("Missing value imputation complete.")
    return df_cleaned

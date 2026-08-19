import pandas as pd
from IPython.display import display


def explore_df(df: pd.DataFrame) -> None:
    """
    Prints an overview of a DataFrame: shape, info, summary statistics, and a sample of rows.

    Args:
        df (DataFrame): The DataFrame to explore.
    """
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns\n")
    
    print("Info:")
    df.info()
    
    print("\nDescribe:")
    display(df.describe(include="all"))
    
    print("\nSample:")
    display(df.head())


def show_df(df: pd.DataFrame, n: int = 10) -> None:
    """
    Displays the first n rows of a DataFrame.

    Args:
        df (DataFrame): The DataFrame to display.
        n (int): Number of rows to display. Defaults to 10.
    """
    display(df.head(n))


def show_first_value(df: pd.DataFrame, column: str) -> None:
    """
    Prints the value of the first row for a column in a DataFrame.

    Args:
        df (DataFrame): The DataFrame to inspect.
        column (str): Name of the column to inspect.
    """
    if column not in df.columns:
        print(f"Column '{column}' not found. Available columns: {list(df.columns)}")
        return

    print(f"{column}: {df[column].iloc[0]}")


def check_nulls(df: pd.DataFrame, column: str) -> None:
    """
    Prints the count and percentage of null values for a column in a DataFrame.

    Args:
        df (DataFrame): The DataFrame to check.
        column (str): Name of the column to check for null values.
    """
    if column not in df.columns:
        print(f"Column '{column}' not found. Available columns: {list(df.columns)}")
        return

    null_count = df[column].isna().sum()
    total = df.shape[0]
    pct = (null_count / total * 100) if total else 0.0

    print(f"Column: {column}")
    print(f"Nulls: {null_count:,} of {total:,} rows ({pct:.2f}%)")
    print(f"Non-null: {total - null_count:,}")

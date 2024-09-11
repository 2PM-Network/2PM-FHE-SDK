from io import StringIO

import pandas as pd


def convert_file_to_df(file_path: str) -> pd.DataFrame:
    """Converts a file to a DataFrame.

    Args:
        file_name (str): The name of the file.

    Returns:
        pd.DataFrame: The DataFrame representation of the file.
    """
    with open(file_path, "r") as f:
        content = StringIO(f.read())
    if file_path.endswith(".csv"):
        return pd.read_csv(content)
    elif file_path.endswith(".xlsx"):
        return pd.read_excel(content)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

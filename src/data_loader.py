import pandas as pd


def load_diabetes_data(path: str) -> pd.DataFrame:
    """
    Load and validate the diabetes dataset.

    Parameters
    ----------
    path : str
        Path to CSV file

    Returns
    -------
    pd.DataFrame
        Validated diabetes dataset
    """

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Loaded dataset is empty")

    if df.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains missing values")

    if "Outcome" not in df.columns:
        raise ValueError("Target column 'Outcome' not found")

    return df

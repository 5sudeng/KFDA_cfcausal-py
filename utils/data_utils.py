from sklearn.model_selection import train_test_split
import pandas as pd

def load_dataset(path):
    """
    Load a dataset from a given path.
    
    Parameters:
    path (str): The path to the dataset file.
    
    Returns:
    pd.DataFrame: The loaded dataset as a pandas DataFrame. or Tuple (X, Y, T)
    """
    
    return pd.read_csv(path)

def split_train_calibration(data, train_ratio=0.8, seed=42):
    """
    Split the dataset into training and calibration sets. (Applied stratified split)
    
    Parameters:
    data (pd.DataFrame): The dataset to split.
    train_ratio (float): The proportion of the dataset to include in the training set.
    seed (int): Random seed for reproducibility.
    
    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: The training and calibration datasets.
    """
    
    train_data, calibration_data = train_test_split(
        data,
        train_size=train_ratio,
        stratify=data['treatment'],
        random_state=seed
    )
    return train_data, calibration_data

def split_by_treatment(dataset, treatment_col='treatment'):
    """
    Split the dataset into treated and control groups based on the treatment column.
    
    Parameters:
    dataset (pd.DataFrame): The dataset to split.
    treatment_col (str): The name of the treatment column.
    
    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: The treated and control datasets.
    """
    
    treated_df = data[data[treatment_col] == 1]
    control_df = data[data[treatment_col] == 0]
    
    return treated_df, control_df
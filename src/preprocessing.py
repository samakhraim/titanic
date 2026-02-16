import pandas as pd

def clean_titanic_data(df: pd.DataFrame, age_median=None, embarked_mode=None, fare_mean=None) -> pd.DataFrame:
    """
    Clean Titanic dataset by handling missing values, encoding, and one-hot encoding.
    
    Args:
        df: Raw dataframe
        age_median: Median age for filling missing values (computed from train if None)
        embarked_mode: Most common embarked port (computed from train if None)
        fare_mean: Mean fare for filling missing values (computed from train if None)
    
    Returns:
        Cleaned dataframe
    """
    df = df.copy()
    
    # Drop unnecessary columns
    drop_cols = ["PassengerId", "Ticket", "Name", "Cabin"]
    df = df.drop(columns=drop_cols, errors="ignore")
    
    # Compute stats from train data if not provided
    if age_median is None:
        age_median = df["Age"].median()
    if embarked_mode is None:
        embarked_mode = df["Embarked"].mode()[0]
    if fare_mean is None:
        fare_mean = df["Fare"].mean()
    
    # Fill missing values
    df["Age"] = df["Age"].fillna(age_median)
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)
    df["Fare"] = df["Fare"].fillna(fare_mean)
    
    # Encode Sex
    df["Sex"] = df["Sex"].astype(str).str.strip().str.lower().map({"male": 0, "female": 1})
    
    # One-hot encode Embarked
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=False)
    
    return df, age_median, embarked_mode, fare_mean

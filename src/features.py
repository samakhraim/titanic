import pandas as pd


class TitanicFeatures:
    """Handle feature engineering for Titanic dataset."""
    
    def __init__(self, X):
        """
        Initialize with feature data.
        
        Args:
            X: Feature dataframe
        """
        self.X = X
        self.feature_columns = list(X.columns)
        self.n_features = X.shape[1]
    
    def get_feature_info(self):
        """
        Get information about features.
        
        Returns:
            Dictionary with feature information
        """
        return {
            'n_features': self.n_features,
            'feature_columns': self.feature_columns,
            'dtypes': self.X.dtypes.to_dict()
        }
    
    def get_missing_values(self):
        """
        Get missing values count for each feature.
        
        Returns:
            Series with missing value counts
        """
        return self.X.isnull().sum()
    
    def get_statistics(self):
        """
        Get basic statistics for features.
        
        Returns:
            DataFrame with feature statistics
        """
        return self.X.describe()

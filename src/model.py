from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TitanicModel:
    """Logistic Regression model for Titanic survival prediction."""
    
    def __init__(self, random_state=42):
        """
        Initialize the model.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.model = LogisticRegression(random_state=random_state, max_iter=1000)
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """
        Train the model on training data.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("✓ Model trained successfully")
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def evaluate(self, X_val, y_val):
        """
        Evaluate model performance on validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X_val)
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred)
        }
        
        return metrics

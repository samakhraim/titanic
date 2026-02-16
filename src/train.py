import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import clean_titanic_data
from model import TitanicModel
from features import TitanicFeatures
import os

# Get the directory of this script and construct absolute path to data
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "data")

# Load raw data
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))

# Clean data
train_df, age_median, embarked_mode, fare_median = clean_titanic_data(train_df)

# Separate features and target
X = train_df.drop(columns=["Survived"])
y = train_df["Survived"]

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

7  # Get feature information
features = TitanicFeatures(X_train)
print("\n" + "=" * 50)
print("FEATURE INFORMATION")
print("=" * 50)
print(f"Total Features: {features.n_features}")
print(f"Features: {features.feature_columns}")
print("=" * 50 + "\n")

# Train model
model = TitanicModel(random_state=42)
model.train(X_train, y_train)

# Evaluate model
metrics = model.evaluate(X_val, y_val)

print("=" * 50)
print("MODEL EVALUATION")
print("=" * 50)
print(f"Accuracy:  {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"F1 Score:  {metrics['f1']:.4f}")
print("=" * 50)

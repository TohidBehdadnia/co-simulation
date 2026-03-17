# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from pathlib import Path


np.random.seed(42)

# Load data csv
dataFolder = Path("./data").absolute()
data = pd.read_csv(dataFolder / "final_data.csv")

def train_test_split(X, y, test_size=0.2, random_state=42, splitColumn=None):
    if splitColumn is not None and splitColumn in X.columns:
        # If a split column is provided, use it to split the data (each unique element of that column must appear only in either train or test set)
        unique_values = X[splitColumn].unique()
        np.random.shuffle(unique_values)
        split_index = int(len(unique_values) * (1 - test_size))
        train_values = unique_values[:split_index]
        test_values = unique_values[split_index:]
        
        X_train = X[X[splitColumn].isin(train_values)].drop(columns=[splitColumn])
        y_train = y[X[splitColumn].isin(train_values)]
        X_test = X[X[splitColumn].isin(test_values)].drop(columns=[splitColumn])
        y_test = y[X[splitColumn].isin(test_values)]
    else:
        # If no split column is provided, use random splitting
        from sklearn.model_selection import train_test_split as sk_train_test_split
        X_train, X_test, y_train, y_test = sk_train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test


classifiers = [
    RandomForestClassifier(random_state=42)
]


time_window_widths = [(0,1), (1,2), (2,4), (4,6), (6,8), (8,10)]

for w in time_window_widths:
    start = w[0]
    end = w[1]
    # Consider the samples belonging to the training window
    data_window = data[(data["t-from-trip"] >= start) & (data["t-from-trip"] < end)]
    print(f"Training classifiers on data from the first {w} seconds of the simulation...")
    # Split the data into features and target variable
    X = data_window.drop("class", axis=1)  # Features
    y = data_window["class"]  # Target variable
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, splitColumn="run-id")
    for clf in classifiers:
        print(f"Training and evaluating {clf.__class__.__name__}...")
        # Train the model
        clf.fit(X_train, y_train)
        # Make predictions
        y_pred = clf.predict(X_test)
        # Evaluate the model
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))



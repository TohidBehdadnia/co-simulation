from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import time
from learner_classifier import classifiers, data, resultsFolder

np.random.seed(42)

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

if __name__ == "__main__":
    # Testing classification response speed on the whole dataset
    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=["class"]), data["class"], test_size=0.2, random_state=42, splitColumn="run-id") 
    times_rows = []
    for clfName, clf in classifiers.items():
        print(f"Training {clfName} on the whole dataset...")
        clf.fit(X_train, y_train)
        # Perform the classification one by one to test the response speed of the classifier on the whole dataset (instead of using batch prediction) and print the time taken for each prediction.
        times = []
        for i in range(len(X_test)):
            start_time = time.time()
            pred = clf.predict(X_test.iloc[i:i+1])
            end_time = time.time()
            times.append(end_time - start_time)
        print(f"Average prediction time for {clfName}: {np.mean(times)*1000:.4f} ms")
        times_rows.append({
            "Classifier": clfName,
            "Average Prediction Time (ms)": np.mean(times)*1000,
            "Standard Deviation (ms)": np.std(times)*1000
        })
    timesResults = pd.DataFrame(times_rows)
    timesResults.to_csv(resultsFolder / "classifier_prediction_times.csv", index=False)
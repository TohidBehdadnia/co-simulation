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
from tqdm import tqdm
import time
from learner_classifier import classifiers, data, resultsFolder, plotsFolder, _apply_latex_rcparams
import argparse

np.random.seed(42)
bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the prediction time of classifiers on the test set.")
    parser.add_argument("--only-plot", action=argparse.BooleanOptionalAction, help="Only plot the stored results without running the evaluation.")
    parser.add_argument("--latex", action=argparse.BooleanOptionalAction, help="Use LaTeX formatting for the plot.")
    return parser.parse_args()

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

def compute_time_granularity_distribution(data: pd.DataFrame) -> np.ndarray:
    # Compute the distribution of time granularity in the dataset by looking at the differences between consecutive timestamps in the "t-from-trip" run-id by run-id
    inter_times = []
    for run_id, group in data.groupby("run-id"):
        t_values = group["t-from-trip"].values
        inter_times.extend(np.diff(t_values))
    return np.array(inter_times)

def main(args: argparse.Namespace):
    if not args.only_plot:
        # Testing classification response speed on the whole dataset
        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=["class"]), data["class"], test_size=0.2, random_state=42, splitColumn="run-id") 
        times_rows = []
        for clfName, clf in tqdm(classifiers.items(), desc=f"Training {clfName} on the training set", total=len(classifiers), bar_format=bar_format):
            clf.fit(X_train, y_train)
            # Perform the classification one by one to test the response speed of the classifier on the whole dataset (instead of using batch prediction) and print the time taken for each prediction.
            times = []
            for i in tqdm(range(len(X_test)), desc=f"Testing {clfName} on the test set", leave=False, total=len(X_test), bar_format=bar_format):
                start_time = time.time()
                pred = clf.predict(X_test.iloc[i:i+1])
                end_time = time.time()
                times.append(end_time - start_time)
            times_rows.append({
                "Classifier": clfName,
                "Average Prediction Time (ms)": np.mean(times)*1000,
                "Standard Deviation (ms)": np.std(times)*1000
            })
        timesResults = pd.DataFrame(times_rows)
        timesResults.to_csv(resultsFolder / "classifier_prediction_times.csv", index=False)
    else:
        inter_times = compute_time_granularity_distribution(data)
        min_time = np.min(inter_times)
        max_time = np.max(inter_times)
        median_time = np.median(inter_times)
        avg_time = np.mean(inter_times)
        print(f"Time granularity distribution of the full dataset: min={min_time:.6f}s, max={max_time:.6f}s, median={median_time:.6f}s, average={avg_time:.6f}s")

        sns.set_theme()
        if args.latex:
            _apply_latex_rcparams()
        timesResults = pd.read_csv(resultsFolder / "classifier_prediction_times.csv")
        y_column = "Average Prediction Time (ms)"
        fig, ax = plt.subplots(figsize=(8, 6))
        ax = sns.barplot(data=timesResults, x="Classifier", y=y_column, hue="Classifier", dodge=False, edgecolor=".3", capsize=0.1)
        if ax.legend_ is not None:
            ax.legend_.remove()
        plt.errorbar(x=timesResults["Classifier"], y=timesResults[y_column], yerr=timesResults["Standard Deviation (ms)"], fmt='none', c='black', capsize=5, elinewidth=0.9, ecolor="0.2")

        median_granularity_ms = median_time * 1000
        median_line = ax.axhline(
            median_granularity_ms,
            color="red",
            linestyle="--",
            linewidth=1.6,
            label=f"Dataset median granularity ({median_granularity_ms:.3f} ms)",
        )


        # Add average/std labels slightly above the error bars.
        annotation_offset_ms = 0.02 * (
            timesResults[y_column] + timesResults["Standard Deviation (ms)"]
        ).max()
        for patch, (_, row) in zip(ax.patches, timesResults.iterrows()):
            # Compute also how many inter-times are below the average value (convert iter_times to milliseconds)
            percLost = (len([t for t in (inter_times * 1000) if t < row[y_column]]) / len(inter_times)) * 100

            x = patch.get_x() + patch.get_width() / 2
            y = row[y_column] + row["Standard Deviation (ms)"] + annotation_offset_ms
            ax.text(
                x,
                y,
                f"avg: {row[y_column]:.3f} ms\nstd: {row['Standard Deviation (ms)']:.3f} ms" if not args.latex else f"${row[y_column]:.3f}$ $\pm {row['Standard Deviation (ms)']:.3f}$ ms",
                ha="center",
                va="bottom",
                rotation=0,
                fontsize=8,
            )

            y2 = row[y_column] + row["Standard Deviation (ms)"] + (annotation_offset_ms * 6)
            ax.text(
                x,
                y2,
                f"{percLost:.2f}%\nof queries lost" if not args.latex else f"\\textbf{{$\\sim$ {percLost:.2f}\\%\\\\of queries lost}}",
                ha="center",
                va="bottom",
                rotation=0,
                color="red",
                fontweight="bold",
                fontsize=8,
            )

        #plt.title("Average Prediction Time of Classifiers on the Test Set")
        plt.xlabel("Classifier")
        plt.ylabel("Average Prediction Time (ms)") if not args.latex else plt.ylabel("Average Prediction Time $\pm std$ (ms)")
        line_y_frac = median_granularity_ms / ax.get_ylim()[1]
        legend_y = min(max(line_y_frac - 0.04, 0.08), 0.92)
        ax.legend(handles=[median_line], loc="upper left", bbox_to_anchor=(0.01, legend_y), frameon=True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig.set_size_inches(6, 4, forward=True)
        plt.savefig(plotsFolder / "classifier_prediction_times.pdf", bbox_inches="tight")

if __name__ == "__main__":
    args = _parse_args()
    main(args)
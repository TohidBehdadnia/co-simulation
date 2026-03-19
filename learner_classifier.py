# %%
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

np.random.seed(42)

dataFolder = Path("./data").absolute()
resultsFolder = Path("./results").absolute()
plotsFolder = Path("./plots").absolute()
data = pd.read_csv(dataFolder / "final_data.csv")
latex = True  # Set to True to use LaTeX rendering for text in plots

def k_fold_cv(X, y, k=5, random_state=42, splitColumn=None):
    if splitColumn is not None and splitColumn in X.columns:
        unique_values = X[splitColumn].unique()
        np.random.shuffle(unique_values)
        fold_size = len(unique_values) // k
        folds = [unique_values[i*fold_size:(i+1)*fold_size] for i in range(k)]
        
        for i in range(k):
            test_values = folds[i]
            train_values = np.concatenate([folds[j] for j in range(k) if j != i])
            
            X_train = X[X[splitColumn].isin(train_values)].drop(columns=[splitColumn])
            y_train = y[X[splitColumn].isin(train_values)]
            X_test = X[X[splitColumn].isin(test_values)].drop(columns=[splitColumn])
            y_test = y[X[splitColumn].isin(test_values)]
            
            yield X_train, y_train, X_test, y_test
    else:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            yield X_train, y_train, X_test, y_test

def _apply_latex_rcparams() -> None:
    # NOTE: This does not validate that a LaTeX distribution is installed.
    # If missing, matplotlib will raise at render/save time.
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            # Use LaTeX fonts instead of matplotlib rcfonts.
            "pgf.rcfonts": False,
            "pgf.texsystem": "pdflatex",
        }
    )

classifiers = {
    "MNB": MultinomialNB(),
    "MLP": MLPClassifier(max_iter=300, random_state=42, n_iter_no_change=10),
    "KNN": KNeighborsClassifier(n_neighbors=20, n_jobs=-1),
    "DT": DecisionTreeClassifier(max_depth=10, random_state=42),
    "RFC": RandomForestClassifier(max_depth=10, random_state=42, n_jobs=-1)
}

if __name__ == "__main__":
    # Time windows are defined as (start, end) in seconds relative to the trip event
    time_window_widths = [(0,1), (1,2), (2,3), (3,4)]

    maitain_it_features = [False, True]

    scores = pd.DataFrame(columns=["Classifier", "Time Window", "Fold", "Precision", "Recall", "F1-Score", "Maintain IT Features"])

    for maintain_it in maitain_it_features:
        if not maintain_it:
            print("Training classifiers without IT features...")
            data_working = data.drop(columns=["setpoint-count", "average-delay"])
        else:
            print("Training classifiers with IT features...")
            data_working = data.copy()
        for w in time_window_widths:
            start = w[0]
            end = w[1]
            # Just consider the samples belonging to the training window
            data_window = data_working[(data_working["t-from-trip"] >= start) & (data_working["t-from-trip"] < end)]
            print(f"Training classifiers on data between {start+1} and {end+1} seconds...")
            X = data_window.drop("class", axis=1)
            y = data_window["class"] 

            for clfName, clf in classifiers.items():
                print(f"Training and evaluating {clfName} on data between {start+1} and {end+1} seconds...")
                for fold, (X_train, y_train, X_test, y_test) in enumerate(k_fold_cv(X, y, k=5, random_state=42, splitColumn="run-id")):
                    print(f"Fold {fold + 1}")
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    
                    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                    # The time window is reported in absolute time and not relative to the trip event (for consistency with the previous plots)
                    scores = pd.concat([scores, pd.DataFrame({
                        "Classifier": clfName,
                        "Time Window": f"{start+1}-{end+1} seconds",
                        "Fold": fold + 1,
                        "Precision": report["weighted avg"]["precision"],
                        "Recall": report["weighted avg"]["recall"],
                        "F1-Score": report["weighted avg"]["f1-score"],
                        "Maintain IT Features": maintain_it
                    }, index=[0])], ignore_index=True)
    scores.to_csv(resultsFolder / "classifier_scores.csv", index=False)

    sns.set_theme()

    if latex:
        try:
            plt.use("pgf")
        except Exception:
            pass
        try:
            _apply_latex_rcparams()
        except Exception:
            pass

    scores["Condition"] = scores["Maintain IT Features"].apply(
        lambda x: "With IT Features" if x else "Without IT Features"
    )

    # Keep time windows in chronological order (e.g., 1-2, 2-3, ...)
    time_windows = sorted(
        scores["Time Window"].unique(),
        key=lambda s: int(str(s).split("-")[0])
    )

    desired_order =  list(classifiers.keys())
    classifiers_present = set(scores["Classifier"].unique())
    classifiers_list = [clf for clf in desired_order if clf in classifiers_present]

    conditions = ["Without IT Features", "With IT Features"]

    # Visual mapping: same color per classifier, different hatches per condition
    colors_map = {
        clf: color
        for clf, color in zip(classifiers_list, sns.color_palette(n_colors=len(classifiers_list)))
    }
    hatches = {"Without IT Features": "", "With IT Features": "//"}

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=False)

    x_centers = np.arange(len(time_windows))
    n_classifiers = len(classifiers_list)
    n_conditions = len(conditions)

    # Keep bars intentionally small within each time-window group
    group_width = 0.80
    bar_width = group_width / (n_classifiers * n_conditions)

    for c_idx, clf in enumerate(classifiers_list):
        clf_data = scores[scores["Classifier"] == clf]
        summary = (
            clf_data.groupby(["Time Window", "Condition"])["F1-Score"]
            .agg(["mean", "std"])
            .reset_index()
        )

        for k_idx, condition in enumerate(conditions):
            cond_summary = summary[summary["Condition"] == condition].set_index("Time Window")
            cond_summary = cond_summary.reindex(time_windows)

            # Position each bar in a compact cluster around the time-window center
            local_index = c_idx * n_conditions + k_idx
            offset = -group_width / 2 + (local_index + 0.5) * bar_width

            ax.bar(
                x_centers + offset,
                cond_summary["mean"],
                width=bar_width,
                color=colors_map[clf],
                hatch=hatches[condition],
                edgecolor=".2",
                linewidth=0.8,
                yerr=cond_summary["std"],
                capsize=2,
                error_kw={"ecolor": ".2", "elinewidth": 0.9},
            )

    #ax.set_title("Classifier Performance With vs Without IT Features", pad=14)
    ax.set_xlabel("Time Window")
    if not latex: ax.set_ylabel("Macro F1-Score (± std)") 
    else: ax.set_ylabel("Macro F1-Score $(\pm std$)")  
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x_centers)
    ax.set_xticklabels(time_windows, rotation=15)

    # Single legend box with section headers for classifiers and dataset conditions
    legend_handles = [
        Patch(facecolor="none", edgecolor="none", label="Classifier"),
    ]
    legend_handles += [
        Patch(facecolor=colors_map[clf], edgecolor=".2", label=clf)
        for clf in classifiers_list
    ]
    legend_handles += [
        Patch(facecolor="none", edgecolor="none", label="Dataset"),
    ]
    legend_handles += [
        Patch(facecolor="white", edgecolor=".2", hatch=hatches[cond], label=cond)
        for cond in conditions
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.00),
        frameon=True,
        ncol=1,
    )

    # Reserve room on the right for legend
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(plotsFolder / "classifier_comparison.pdf", bbox_inches="tight")

    # Store the average F1 score and std for each classifier and time window in a CSV file for later analysis
    summary_data = []
    for clf in classifiers_list:
        clf_data = scores[scores["Classifier"] == clf]
        summary = (
            clf_data.groupby(["Time Window", "Condition"])["F1-Score"]
            .agg(["mean", "std"])
            .reset_index()
        )
        for _, row in summary.iterrows():
            summary_data.append({
                "Classifier": clf,
                "Time Window": row["Time Window"],
                "Condition": row["Condition"],
                "Mean F1-Score": row["mean"],
                "Std F1-Score": row["std"]
            })
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(resultsFolder / "classifier_summary.csv", index=False)
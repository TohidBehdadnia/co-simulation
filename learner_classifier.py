# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import time
import argparse
import sys
import ast
from tqdm import tqdm

np.random.seed(42)

dataFolder = Path("./data").absolute()
resultsFolder = Path("./results").absolute()
plotsFolder = Path("./plots").absolute()
data = pd.read_csv(dataFolder / "final_data.csv")

classifiers = {
    "MNB": MultinomialNB(),
    "MLP": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(max_iter=300, random_state=42, n_iter_no_change=10)),
        ]
    ),
    "KNN": KNeighborsClassifier(n_neighbors=20, n_jobs=-1),
    "DT": DecisionTreeClassifier(max_depth=10, random_state=42),
    "RFC": RandomForestClassifier(max_depth=10, random_state=42, n_jobs=-1)
}

bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"

def delete_last_line():
    "Deletes the last line in the STDOUT"
    # cursor up one line
    sys.stdout.write('\x1b[1A')
    # delete last line
    sys.stdout.write('\x1b[2K')

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
            # Keep LaTeX in control of typeface selection.
            "pgf.rcfonts": False,
            "pgf.texsystem": "pdflatex",
        }
    )


def _scale_font_rcparams(scale: float) -> None:
    """Scale matplotlib text sizes while preserving LaTeX-managed font families."""
    scale = float(scale)

    keys = [
        "font.size",
        "axes.titlesize",
        "axes.labelsize",
        "xtick.labelsize",
        "ytick.labelsize",
        "legend.fontsize",
        "figure.titlesize",
    ]
    for key in keys:
        value = plt.rcParams.get(key)
        if isinstance(value, (int, float)):
            plt.rcParams[key] = float(value) * scale

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Script for training and evaluating ML models on the curated data")
    parser.add_argument(
        "--only-plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Chose to generate the plot based on the stored results")
    
    parser.add_argument(
        "--latex",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable LaTeX text rendering in plots (PGF backend when possible). "
            "Requires a working LaTeX installation (e.g. pdflatex)."
        )
    )

    parser.add_argument(
        "--poster",
        action="store_true",
        help="Set seaborn context to poster"
    )
    
    return parser.parse_args()

def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)

def _parse_confusion_matrix(value) -> np.ndarray:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return np.asarray(value, dtype=int)

def _parse_label_order(value) -> list[str]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return [str(v) for v in value]

def main(args: argparse.Namespace):
    # Time windows are defined as (start, end) in seconds relative to the trip event
    time_window_widths = [(0,1), (1,2), (2,3)]

    physical_features = ["t-from-trip", "freq", "freq-lag-50ms", "freq-lag-100ms"]
    cyber_features = ["t-from-trip", "setpoint-count", "average-delay"]
    combined_features = list(dict.fromkeys(physical_features + cyber_features))
    feature_sets = [
        {"name": "Physical Only", "features": physical_features},
        {"name": "Cyber Only", "features": cyber_features},
        {"name": "Physical + Cyber", "features": combined_features},
    ]

    if not bool(args.only_plot):
        score_rows = []

        for feature_set in tqdm(feature_sets, desc="Dataset conditions", total=len(feature_sets), bar_format=bar_format):
            feature_cols = feature_set["features"]
            for w in tqdm(time_window_widths, desc="Time windows", total=len(time_window_widths), bar_format=bar_format, leave=False):
                start = w[0]
                end = w[1]
                # Just consider the samples belonging to the training window
                data_window = data[(data["t-from-trip"] >= start) & (data["t-from-trip"] < end)]
                X = data_window[feature_cols + ["run-id"]]
                y = data_window["class"] 

                for clfName, clf in tqdm(classifiers.items(), desc="Classifiers", total=len(classifiers), bar_format=bar_format, leave=False):
                    k = 5
                    for fold, (X_train, y_train, X_test, y_test) in tqdm(enumerate(k_fold_cv(X, y, k=k, random_state=42, splitColumn="run-id")), total=k, desc="Fold", bar_format=bar_format, leave=False):
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        
                        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                        labels = y.unique()
                        # The time window is reported in absolute time and not relative to the trip event (for consistency with the previous plots)
                        score_rows.append({
                            "Classifier": clfName,
                            "Time Window": f"{start+1}-{end+1} seconds",
                            "Fold": fold + 1,
                            "Precision": report["weighted avg"]["precision"],
                            "Recall": report["weighted avg"]["recall"],
                            "F1-Score": report["weighted avg"]["f1-score"],
                            "Confusion Matrix Order": labels.tolist(),
                            "Confusion Matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
                            "Feature Set": feature_set["name"]
                        })
        scores = pd.DataFrame(score_rows)
        scores.to_csv(resultsFolder / "classifier_scores.csv", index=False)
    else:
        scores = pd.read_csv(resultsFolder / "classifier_scores.csv")

    if "Feature Set" not in scores.columns and "Maintain Cyber-Layer Features" in scores.columns:
        scores["Maintain Cyber-Layer Features"] = scores["Maintain Cyber-Layer Features"].apply(_to_bool)
        scores["Feature Set"] = scores["Maintain Cyber-Layer Features"].apply(
            lambda x: "Physical + Cyber" if x else "Physical Only"
        )
    scores["Confusion Matrix"] = scores["Confusion Matrix"].apply(_parse_confusion_matrix)
    scores["Confusion Matrix Order"] = scores["Confusion Matrix Order"].apply(_parse_label_order)

    sns.set_theme()

    latex = bool(args.latex)
    if latex:
        try:
            plt.use("pgf")
        except Exception:
            pass
        try:
            _apply_latex_rcparams()
        except Exception:
            pass
            
    if args.poster:
        _scale_font_rcparams(1.30)

    scores["Condition"] = scores["Feature Set"].astype(str)

    # Keep time windows in chronological order (e.g., 1-2, 2-3, ...)
    time_windows = sorted(
        scores["Time Window"].unique(),
        key=lambda s: int(str(s).split("-")[0])
    )

    desired_order =  list(classifiers.keys())
    classifiers_present = set(scores["Classifier"].unique())
    classifiers_list = [clf for clf in desired_order if clf in classifiers_present]

    conditions = ["Physical Only", "Cyber Only", "Physical + Cyber"]

    # Visual mapping: same color per classifier, different hatches per condition
    colors_map = {
        clf: color
        for clf, color in zip(classifiers_list, sns.color_palette(n_colors=len(classifiers_list)))
    }
    hatches = {
        "Physical Only": "",
        "Cyber Only": "//",
        "Physical + Cyber": "xx",
    }

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

    if not latex:
        ax.set_xlabel("Time Window")
    else:
        ax.set_xlabel("$\\mathcal{{W}}_{{q}} \\rightarrow [\\tau_q^{{\\min}}, \\tau_q^{{\\max}}) $ (s)")
    if not latex: ax.set_ylabel("Macro F1-score ± SD") 
    else: ax.set_ylabel("Macro F1-score $\pm$ SD")  
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x_centers)
    if latex:
        latex_labels = []
        for window in time_windows:
            window_text = str(window).replace(" seconds", "")
            parts = window_text.split("-")
            start = parts[0].strip() if len(parts) > 0 else ""
            end = parts[1].strip() if len(parts) > 1 else ""
            latex_labels.append(f"$[{start},{end})$")
        ax.set_xticklabels(latex_labels, rotation=0)
    else:
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
        Patch(
            facecolor="white",
            edgecolor=".2",
            hatch=hatches[cond],
            label=(
                "$z^{{(phys)}}$"
                if cond == "Physical Only"
                else "$z^{{(cyb)}}$"
                if cond == "Cyber Only"
                else "$z^{{(cp)}}$"
            ),
        )
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
    if args.poster:
        plt.savefig(plotsFolder / "classifier_comparison.png", bbox_inches="tight", dpi=700)
    else:
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

    if args.poster:
        sns.set_theme(font_scale=1, font="serif")

    # Build one confusion-matrix grid for the 1-2 seconds window:
    # columns are classifiers and rows are conditions.
    preferred_order = ["normal", "delay", "drop", "mod"]
    available_labels = data["class"].dropna().astype(str).unique().tolist()
    class_labels = [label for label in preferred_order if label in available_labels]
    class_labels += [label for label in available_labels if label not in class_labels]
    n_labels = len(class_labels)

    target_time_window = "1-2 seconds"
    if target_time_window not in time_windows and len(time_windows) > 0:
        target_time_window = time_windows[0]

    aggregated_cm = {}
    for clf in classifiers_list:
        clf_data = scores[
            (scores["Classifier"] == clf)
            & (scores["Time Window"] == target_time_window)
        ]
        cm_aggregates = {condition: np.zeros((n_labels, n_labels), dtype=int) for condition in conditions}

        for _, row in clf_data.iterrows():
            row_labels = row["Confusion Matrix Order"]
            cm = np.asarray(row["Confusion Matrix"], dtype=int)

            cm_df = pd.DataFrame(cm, index=row_labels, columns=row_labels)
            cm_aligned = cm_df.reindex(index=class_labels, columns=class_labels, fill_value=0).to_numpy(dtype=int)

            condition = row["Condition"]
            if condition in cm_aggregates:
                cm_aggregates[condition] += cm_aligned

        aggregated_cm[clf] = cm_aggregates

    n_rows = len(conditions)
    n_cols = len(classifiers_list)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(3.2 * n_cols, 2.9 * n_rows),
        squeeze=False,
        constrained_layout=False,
    )
    cbar_ax = fig.add_axes([0.51, 0.11, 0.010, 0.4])

    condition_layout = conditions
    normalization_mode = "row"
    for row_idx, condition in enumerate(condition_layout):
        for col_idx, clf in enumerate(classifiers_list):
            ax = axes[row_idx][col_idx]
            cm = aggregated_cm[clf][condition].astype(float)

            if normalization_mode == "row":
                norm_den = cm.sum(axis=1, keepdims=True)
            else:
                norm_den = cm.sum(axis=0, keepdims=True)
            norm_den = np.where(norm_den == 0, 1.0, norm_den)
            cm_normalized = cm / norm_den
            cm_labels = np.vectorize(lambda value: f"{value:.2f}".lstrip("0"))(cm_normalized)

            show_x_ticklabels = row_idx == n_rows - 1
            show_y_ticklabels = col_idx == 0
            sns.heatmap(
                cm_normalized,
                annot=cm_labels,
                fmt="",
                cmap="Blues",
                cbar=(row_idx == 0 and col_idx == n_cols - 1),
                cbar_ax=cbar_ax,
                square=True,
                linewidths=0.3,
                linecolor="white",
                vmin=0.0,
                vmax=1.0,
                xticklabels=class_labels if show_x_ticklabels else False,
                yticklabels=class_labels if show_y_ticklabels else False,
                ax=ax,
            )

            if row_idx == 0:
                ax.set_title(clf)

            if col_idx == 0:
                if latex:
                    if condition == "Physical Only":
                        condition_short = "Trained on $z^{{(phys)}}$\n"
                    elif condition == "Cyber Only":
                        condition_short = "Trained on $z^{{(cyb)}}$\n"
                    else:
                        condition_short = "Trained on $z^{{(cp)}}$\n"
                    ax.set_ylabel(f"{condition_short}\nTrue Label", fontsize=10)
                else:
                    if condition == "Physical Only":
                        condition_short = "Physical\nFeatures\n"
                    elif condition == "Cyber Only":
                        condition_short = "Cyber\nFeatures\n"
                    else:
                        condition_short = "Physical + Cyber\nFeatures\n"
                    ax.set_ylabel(f"{condition_short}\nTrue Label", fontsize=10)
            else:
                ax.set_ylabel("")

            if row_idx == n_rows - 1:
                ax.set_xlabel("Predicted Label", fontsize=10)
            else:
                ax.set_xlabel("")

            if show_x_ticklabels:
                ax.set_xticklabels(class_labels, rotation=35, ha="right", fontsize=9)
            if show_y_ticklabels:
                ax.set_yticklabels(class_labels, rotation=35, fontsize=9)

    #fig.suptitle(f"Aggregated Confusion Matrices - {target_time_window}", y=0.995)
    fig.subplots_adjust(left=0.06, right=0.50, bottom=0.10, top=0.52, wspace=0.25, hspace=0.22)
    if args.poster:
        plt.savefig(plotsFolder / "confusion_matrices_by_model_1_2s.png", bbox_inches="tight", dpi=700)
    else:
        plt.savefig(plotsFolder / "confusion_matrices_by_model_1_2s.pdf", bbox_inches="tight")

if __name__ == "__main__":
    args = _parse_args()
    main(args)
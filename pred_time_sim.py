import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

from learner_classifier import (
    classifiers,
    data,
    resultsFolder,
    plotsFolder,
    _apply_latex_rcparams,
    _scale_font_rcparams,
)

np.random.seed(42)
bar_format = "{l_bar}{bar:10}{r_bar}{bar:-10b}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate real-time query arrivals per run-id using measured prediction times. "
            "If a query arrives while the model is busy, it is dropped (queue length 0)."
        )
    )
    parser.add_argument(
        "--only-plot",
        action=argparse.BooleanOptionalAction,
        help="Only plot the stored results without running the evaluation.",
    )
    parser.add_argument(
        "--latex",
        action=argparse.BooleanOptionalAction,
        help="Use LaTeX formatting for the plot.",
    )
    parser.add_argument(
        "--poster",
        action="store_true",
        help="Set seaborn context to poster",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of run-ids reserved for testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/test run-id split.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=resultsFolder / "classifier_prediction_sim.csv",
        help="Output CSV file for the simulation summary.",
    )
    return parser.parse_args()


def train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    splitColumn: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if splitColumn is not None and splitColumn in X.columns:
        # If a split column is provided, use it to split the data
        unique_values = X[splitColumn].unique()
        rng = np.random.default_rng(random_state)
        rng.shuffle(unique_values)
        split_index = int(len(unique_values) * (1 - test_size))
        train_values = unique_values[:split_index]
        test_values = unique_values[split_index:]

        X_train = X[X[splitColumn].isin(train_values)].copy()
        y_train = y[X[splitColumn].isin(train_values)]
        X_test = X[X[splitColumn].isin(test_values)].copy()
        y_test = y[X[splitColumn].isin(test_values)]
    else:
        from sklearn.model_selection import train_test_split as sk_train_test_split

        X_train, X_test, y_train, y_test = sk_train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    return X_train, X_test, y_train, y_test


def compute_time_granularity_distribution(data: pd.DataFrame) -> np.ndarray:
    # Compute the distribution of time granularity in the dataset by looking at the differences
    # between consecutive timestamps in "t-from-trip", run-id by run-id.
    inter_times = []
    for _, group in data.groupby("run-id"):
        t_values = group["t-from-trip"].values
        inter_times.extend(np.diff(t_values))
    return np.array(inter_times)


def _plot_time_granularity_distribution(inter_times: np.ndarray, args: argparse.Namespace) -> None:
    min_time = np.min(inter_times)
    max_time = np.max(inter_times)
    median_time = np.median(inter_times)
    avg_time = np.mean(inter_times)
    print(
        "Time granularity distribution of the full dataset: "
        f"min={min_time:.6f}s, max={max_time:.6f}s, median={median_time:.6f}s, average={avg_time:.6f}s"
    )

    sns.set_theme()
    if args.latex:
        _apply_latex_rcparams()
    if args.poster:
        _scale_font_rcparams(1.20)

    ax = plt.gca()
    sns.histplot(inter_times * 1000, bins=50, edgecolor="black", ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("Inter-query time granularity (ms)")
    if args.latex:
        ax.set_xlabel("Inter-arrival time, $\\Delta t^{{arr}}$ (ms)")
    ax.set_ylabel("Observation count (log scale)")

    import matplotlib.patches as patches

    rects_and_texts = [
        {
            "coords": (-0.5, 3, 15, 5000),
            "text": "Bursts of setpoint commands",
            "txt_pos": (0.2, 5500),
            "color": "red",
        },
        {
            "coords": (-1.0, 31.5, 15, 30000),
            "text": "Spontaneous events (e.g. the first setpoint command arriving)",
            "txt_pos": (0.7, 33000),
            "color": "darkorange",
        },
        {
            "coords": (32.0, 34.0, 15, 300000),
            "text": "Periodic events (e.g. physical measurements)",
            "txt_pos": (11.6, 200000),
            "color": "green",
        },
    ]

    for rt in rects_and_texts:
        xmin, xmax, ymin, ymax = rt["coords"]
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle(
            (xmin, ymin),
            width,
            height,
            linewidth=2,
            edgecolor=rt["color"],
            facecolor="none",
            alpha=0.7,
        )
        ax.add_patch(rect)

        txt_x, txt_y = rt["txt_pos"]
        ax.text(
            txt_x,
            txt_y,
            rt["text"],
            color=rt["color"],
            fontsize=10,
            ha="left",
            va="bottom",
        )

    if args.poster:
        plt.savefig(plotsFolder / "time_granularity_distribution_sim.png", bbox_inches="tight", dpi=700)
    else:
        plt.savefig(plotsFolder / "time_granularity_distribution_sim.pdf", bbox_inches="tight")
    plt.close()


def _simulate_single_run(arrivals: np.ndarray, service_times: np.ndarray) -> dict:
    busy_until = -np.inf
    dropped_count = 0
    processed_count = 0
    total_service = 0.0

    for arrival, service in zip(arrivals, service_times):
        if arrival < busy_until:
            dropped_count += 1
            continue
        processed_count += 1
        total_service += float(service)
        busy_until = arrival + service

    total_count = len(arrivals)
    if total_count == 0:
        return {
            "total_count": 0,
            "dropped_count": 0,
            "processed_count": 0,
            "total_service": 0.0,
            "run_duration": 0.0,
        }

    first_arrival = float(arrivals[0])
    last_arrival = float(arrivals[-1])
    run_end = max(busy_until, last_arrival)
    run_duration = max(run_end - first_arrival, 0.0)
    return {
        "total_count": total_count,
        "dropped_count": dropped_count,
        "processed_count": processed_count,
        "total_service": total_service,
        "run_duration": run_duration,
    }


def _simulate_by_run(sim_df: pd.DataFrame) -> dict:
    total_count = 0
    dropped_count = 0
    processed_count = 0
    total_service = 0.0
    total_duration = 0.0

    for _, group in sim_df.groupby("run-id"):
        group_sorted = group.sort_values("t-from-trip")
        arrivals = group_sorted["t-from-trip"].to_numpy(dtype=float)
        service_times = group_sorted["pred-time"].to_numpy(dtype=float)
        run_stats = _simulate_single_run(arrivals, service_times)

        total_count += run_stats["total_count"]
        dropped_count += run_stats["dropped_count"]
        processed_count += run_stats["processed_count"]
        total_service += run_stats["total_service"]
        total_duration += run_stats["run_duration"]

    avg_service = total_service / processed_count if processed_count > 0 else 0.0
    avg_system = avg_service
    dropped_pct = (dropped_count / total_count) * 100 if total_count > 0 else 0.0
    utilization = (total_service / total_duration) * 100 if total_duration > 0 else 0.0

    return {
        "total_count": total_count,
        "dropped_count": dropped_count,
        "processed_count": processed_count,
        "dropped_pct": dropped_pct,
        "avg_service": avg_service,
        "avg_system": avg_system,
        "utilization": utilization,
    }


def _plot_results(results: pd.DataFrame, args: argparse.Namespace) -> None:
    inter_times = compute_time_granularity_distribution(data)
    median_time = np.median(inter_times)

    _plot_time_granularity_distribution(inter_times, args)

    sns.set_theme()
    if args.latex:
        _apply_latex_rcparams()
    if args.poster:
        _scale_font_rcparams(1.30)

    y_column = "Average Prediction Time (ms)"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.barplot(
        data=results,
        x="Classifier",
        y=y_column,
        hue="Classifier",
        dodge=False,
        edgecolor=".3",
        capsize=0.1,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()

    plt.errorbar(
        x=results["Classifier"],
        y=results[y_column],
        yerr=results["Standard Deviation (ms)"],
        fmt="none",
        c="black",
        capsize=5,
        elinewidth=0.9,
        ecolor="0.2",
    )

    median_granularity_ms = median_time * 1000
    median_line = ax.axhline(
        median_granularity_ms,
        color="red",
        linestyle="--",
        linewidth=1.6,
        label=(
            f"Median inter-arrival time, $\\Delta t^{{arr}}$ ({median_granularity_ms:.3f} ms)"
            if not args.poster
            else f"Inter-query\nmedian time granularity ({median_granularity_ms:.3f} ms)"
        ),
    )

    annotation_offset_ms = 0.02 * (
        results[y_column] + results["Standard Deviation (ms)"]
    ).max()
    for patch, (_, row) in zip(ax.patches, results.iterrows()):
        x = patch.get_x() + patch.get_width() / 2
        y = row[y_column] + row["Standard Deviation (ms)"] + annotation_offset_ms
        ax.text(
            x,
            y,
            (
                f"avg: {row[y_column]:.3f} ms\nstd: {row['Standard Deviation (ms)']:.3f} ms"
                if not args.latex
                else f"${row[y_column]:.3f}$ $\\pm {row['Standard Deviation (ms)']:.3f}$ ms"
            ),
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=8 if not args.latex else 9,
        )

        dropped_pct = row["Dropped Queries (%)"]
        y_offset_multiplier = 5 if row["Classifier"] in ["MNB", "MLP", "DT"] else 4.5
        y2 = row[y_column] + row["Standard Deviation (ms)"] + (
            annotation_offset_ms * y_offset_multiplier
        )
        ax.text(
            x,
            y2,
            (
                f"{dropped_pct:.2f}%\nof queries lost"
                if not args.latex
                else f"\\textbf{{$\\rho_{{loss}} =$ {dropped_pct:.2f} \\%}}"
            ),
            ha="center",
            va="bottom",
            rotation=0,
            color="red",
            fontweight="bold",
            fontsize=8 if not args.latex else 10,
        )

    plt.xlabel("Classifier")
    if not args.latex:
        plt.ylabel("Average Prediction Time (ms)")
    else:
        plt.ylabel("Average inference time, $\\overline{{T}} \\pm$ SD (ms)")

    line_y_frac = median_granularity_ms / ax.get_ylim()[1]
    legend_y = min(max(line_y_frac + 1, 0.08), 0.94)
    ax.legend(handles=[median_line], loc="upper left", bbox_to_anchor=(0.01, legend_y), frameon=True)
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.set_size_inches(6, 4, forward=True)
    if args.poster:
        plt.savefig(plotsFolder / "classifier_prediction_sim.png", bbox_inches="tight", dpi=700)
    else:
        plt.savefig(plotsFolder / "classifier_prediction_sim.pdf", bbox_inches="tight")


def main(args: argparse.Namespace) -> None:
    results_path = args.out_csv
    # Suppress the specific configuration propagation warning
    warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed should be used.*")

    if not args.only_plot:
        X = data.drop(columns=["class"])
        y = data["class"]
        X_train, X_test, y_train, _ = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.seed,
            splitColumn="run-id",
        )

        X_train_features = X_train.drop(columns=["run-id"]).reset_index(drop=True)
        X_test_features = X_test.drop(columns=["run-id"]).reset_index(drop=True)
        X_test_meta = X_test[["run-id", "t-from-trip"]].reset_index(drop=True)

        sim_rows = []
        for clfName, clf in tqdm(
            classifiers.items(),
            desc="Training classifiers",
            total=len(classifiers),
            bar_format=bar_format,
        ):
            clf.fit(X_train_features, y_train)

            pred_times = []
            for i in tqdm(
                range(len(X_test_features)),
                desc=f"Timing {clfName} predictions",
                leave=False,
                total=len(X_test_features),
                bar_format=bar_format,
            ):
                start_time = time.perf_counter()
                _ = clf.predict(X_test_features.iloc[i : i + 1])
                end_time = time.perf_counter()
                pred_times.append(end_time - start_time)

            sim_df = X_test_meta.copy()
            sim_df["pred-time"] = np.asarray(pred_times, dtype=float)

            sim_stats = _simulate_by_run(sim_df)
            pred_times_ms = np.asarray(pred_times, dtype=float) * 1000
            sim_rows.append(
                {
                    "Classifier": clfName,
                    "Average Prediction Time (ms)": float(np.mean(pred_times_ms)),
                    "Standard Deviation (ms)": float(np.std(pred_times_ms)),
                    "Total Queries": sim_stats["total_count"],
                    "Processed Queries": sim_stats["processed_count"],
                    "Dropped Queries": sim_stats["dropped_count"],
                    "Dropped Queries (%)": sim_stats["dropped_pct"],
                    "Avg System Time (ms)": sim_stats["avg_system"] * 1000,
                    "Utilization (%)": sim_stats["utilization"],
                }
            )

        results = pd.DataFrame(sim_rows)
        results.to_csv(results_path, index=False)
        print("Saved simulation results to:", results_path)
        print(results)
    else:
        results = pd.read_csv(results_path)

    _plot_results(results, args)


if __name__ == "__main__":
    args = _parse_args()
    main(args)

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

# Event times
t_trip = 1.0

# Colors and markers for up to 10 scenarios. Matplotlib will cycle through these.
markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X"]


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    label: str


def _parse_cli_scenario_spec(spec: str) -> ScenarioSpec:
    """Parse CLI scenario spec.

    This parser is ONLY for the command-line UX (positional args).
    For programmatic usage, prefer passing `ScenarioSpec` or dicts.
    """
    raw = (spec or "").strip()
    if not raw:
        raise ValueError("Empty scenario spec")

    if "=" in raw:
        scenario_id, label = raw.split("=", 1)
        scenario_id = scenario_id.strip()
        label = label.strip() or scenario_id
        if not scenario_id:
            raise ValueError(f"Invalid scenario spec (missing id): {spec!r}")
        return ScenarioSpec(scenario_id=scenario_id, label=label)

    return ScenarioSpec(scenario_id=raw, label=raw)


def _normalize_specs(args: argparse.Namespace) -> list[ScenarioSpec]:
    """Normalize scenario specs from either structured inputs or CLI strings."""
    # Preferred programmatic path
    if hasattr(args, "scenario_specs") and args.scenario_specs:
        specs: list[ScenarioSpec] = []
        for item in args.scenario_specs:
            if isinstance(item, ScenarioSpec):
                specs.append(item)
            elif isinstance(item, dict):
                scenario_id = str(item.get("scenario_id") or item.get("scenario") or "").strip()
                label = str(item.get("label") or scenario_id).strip()
                if not scenario_id:
                    raise ValueError(f"Invalid scenario_specs item (missing scenario_id): {item!r}")
                specs.append(ScenarioSpec(scenario_id=scenario_id, label=label))
            else:
                raise TypeError(f"scenario_specs items must be ScenarioSpec or dict, got: {type(item)}")
        return specs

    # CLI path (positional strings)
    if not getattr(args, "scenarios", None):
        raise ValueError("No scenarios provided")
    return [_parse_cli_scenario_spec(s) for s in args.scenarios]


def _resolve_npz_path(scenario: str) -> Path:
    name = scenario.strip()
    if not name:
        raise ValueError("Empty scenario name")

    filename = name if name.endswith(".npz") else f"{name}.npz"

    # Backward-compatible flat layout: results/<name>.npz
    flat = RESULTS_DIR / filename
    if flat.exists():
        return flat

    # New layout: results/<scenario>/<scenario>_<id>.npz
    scenario_id = name[:-4] if name.endswith(".npz") else name
    candidate_dir = RESULTS_DIR / scenario_id
    if candidate_dir.exists() and candidate_dir.is_dir():
        files = sorted(candidate_dir.glob(f"{scenario_id}_*.npz"))
        if files:
            return files[-1]

    return flat


def _resolve_npz_paths(scenario: str) -> list[Path]:
    """Resolve scenario input to one or more npz files.

    - If scenario refers to a folder in results/, load all runs in that folder.
    - Else fall back to the legacy behavior and return a single .npz path.
    """
    name = (scenario or "").strip()
    if not name:
        raise ValueError("Empty scenario name")

    # Explicit file
    if name.endswith(".npz"):
        return [RESULTS_DIR / name] if not (RESULTS_DIR / name).exists() else [RESULTS_DIR / name]

    # Folder of runs
    folder = RESULTS_DIR / name
    if folder.exists() and folder.is_dir():
        paths = sorted(folder.glob(f"{name}_*.npz"))
        if not paths:
            paths = sorted(folder.glob("*.npz"))
        return paths

    # Legacy single file
    return [_resolve_npz_path(name)]


def _mean_and_band(values: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean and a symmetric band.

    mode
    ----
    - "ci95": 95% CI of the mean (default)
    - "sd": ±1 standard deviation across runs
    """
    values = np.asarray(values, dtype=float)
    mode = str(mode or "ci95").lower()

    mean = np.nanmean(values, axis=0)
    n = int(values.shape[0])
    if n <= 1:
        return mean, mean, mean

    std = np.nanstd(values, axis=0, ddof=1)
    if mode == "sd":
        hw = std
    else:
        from scipy.stats import t 
        tcrit = t.ppf(0.975, df=n - 1)
        sem = std / np.sqrt(float(n))
        hw = float(tcrit) * sem
    return mean, mean - hw, mean + hw


def _load_run(npz_path: Path) -> tuple[np.ndarray, np.ndarray, float | None]:
    data = np.load(npz_path)
    if "t" not in data or "freq" not in data:
        raise KeyError(f"{npz_path.name} must contain arrays 't' and 'freq'")
    t = np.asarray(data["t"], dtype=float)
    freq = np.asarray(data["freq"], dtype=float)
    if t.shape != freq.shape:
        raise ValueError(f"Mismatched shapes in {npz_path.name}: t{t.shape} vs freq{freq.shape}")

    last_cmd_time = None
    if "last_cmd_time" in data:
        try:
            v = float(data["last_cmd_time"])
            if np.isfinite(v):
                last_cmd_time = v
        except Exception:
            last_cmd_time = None

    if last_cmd_time is None and "measured_delay" in data:
        try:
            d = float(data["measured_delay"])
            if np.isfinite(d):
                last_cmd_time = t_trip + d
        except Exception:
            last_cmd_time = None

    return t, freq, last_cmd_time


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Overlay frequency-vs-time curves from multiple scenarios. "
            "Scenarios are loaded from the results/ folder as <scenario>.npz."
        )
    )
    ap.add_argument(
        "scenarios",
        nargs="+",
        help=(
            "Scenarios to plot from results/. Each item can be either 'id' or 'id=Pretty name'. "
            "(CLI convenience only; programmatic use should pass structured ScenarioSpec dicts.)"
        ),
    )
    ap.add_argument(
        "--out-name",
        default=str("freq_comparison.pdf"),
        help="Output PDF name (default: freq_comparison.pdf)",
    )
    ap.add_argument(
        "--title",
        default="Frequency comparison",
        help="Plot title",
    )
    ap.add_argument(
        "--band",
        choices=["ci95", "sd"],
        default="ci95",
        help="Uncertainty band to show for frequency and last-cmd-time: 95% CI of mean (ci95) or ±1 SD (sd).",
    )
    return ap.parse_args()


def plot_comparison(args: argparse.Namespace) -> None:

    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"results folder not found: {RESULTS_DIR}")

    band_mode = str(getattr(args, "band", "ci95")).lower()
    fig, ax = plt.subplots()

    mean_handles: list[plt.Line2D] = []
    last_cmd_stats: list[tuple[str, str, float | None, float]] = []  # (label, color, mean, hw)

    specs = _normalize_specs(args)

    i = 1
    for spec in specs:
        scenario_id = spec.scenario_id
        label = spec.label
        npz_paths = _resolve_npz_paths(scenario_id)
        npz_paths = [p for p in npz_paths if p.exists()]
        if not npz_paths:
            available = sorted(p.name for p in RESULTS_DIR.glob("*.npz"))
            available_dirs = sorted(p.name for p in RESULTS_DIR.iterdir() if p.is_dir())
            raise FileNotFoundError(
                f"Scenario not found: {scenario_id!r}\n"
                f"Available files in results/: {', '.join(available) if available else '(none)'}\n"
                f"Available folders in results/: {', '.join(available_dirs) if available_dirs else '(none)'}"
            )

        runs_t: list[np.ndarray] = []
        runs_freq: list[np.ndarray] = []
        last_times: list[float] = []

        t_ref = None
        for p in npz_paths:
            t, f, last_t = _load_run(p)
            if t_ref is None:
                t_ref = t
            if t_ref.shape != t.shape or np.max(np.abs(t - t_ref)) > 1e-9:
                # Interpolate onto reference grid
                f = np.interp(t_ref, t, f)
            runs_t.append(t_ref)
            runs_freq.append(f)
            if last_t is not None:
                last_times.append(float(last_t))

        t_plot = t_ref
        freq_mat = np.vstack([np.asarray(x, dtype=float) for x in runs_freq])
        mean_f, lo_f, hi_f = _mean_and_band(freq_mat, mode=band_mode)

        (line,) = ax.plot(t_plot, mean_f, marker=markers[i % len(markers)], markevery=15, label=f"{label} (mean)", linewidth=2.0)
        mean_handles.append(line)
        if freq_mat.shape[0] > 1:
            ax.fill_between(
                t_plot,
                lo_f,
                hi_f,
                color=line.get_color(),
                alpha=0.20,
                linewidth=0,
                label="_nolegend_",
            )

        if len(last_times) >= 1:
            last_arr = np.asarray(last_times, dtype=float)
            last_arr = last_arr[np.isfinite(last_arr)]
            if last_arr.size >= 1:
                m = float(np.nanmean(last_arr))
                if last_arr.size > 1:
                    s = float(np.nanstd(last_arr, ddof=1))
                    if band_mode == "sd":
                        hw = float(s)
                    else:
                        from scipy.stats import t
                        tcrit = t.ppf(0.975, df=last_arr.size - 1)
                        hw = float(tcrit) * s / np.sqrt(float(last_arr.size))
                else:
                    hw = 0.0
                last_cmd_stats.append((label, line.get_color(), m, float(hw)))
            else:
                # We had last-cmd samples, but none are finite (treat as no commands).
                last_cmd_stats.append((label, line.get_color(), None, 0.0))
        else:
            # No finite last-command times in this scenario.
            last_cmd_stats.append((label, line.get_color(), None, 0.0))
        i += 1

    # Draw last-command mean vertical lines and uncertainty spans after curves are plotted.
    last_cmd_handles: list[plt.Line2D] = []
    if last_cmd_stats:
        for (label, color, m, hw) in last_cmd_stats:
            if m is None:
                # Legend-only entry when no commands arrived in any run.
                (h,) = ax.plot(
                    [],
                    [],
                    color=color,
                    linestyle="--",
                    linewidth=1.6,
                    label=f"Last cmd: none received",
                )
                last_cmd_handles.append(h)
                continue

            if hw > 0:
                ax.axvspan(m - hw, m + hw, color=color, alpha=0.12, linewidth=0, zorder=1)

            band_txt = "±95% CI" if band_mode == "ci95" else "±1 SD"
            h = ax.axvline(
                m,
                color=color,
                linestyle="--",
                linewidth=1.6,
                alpha=0.95,
                zorder=5,
                label=f"Last cmd mean={m:.3f}s {band_txt}",
            )
            last_cmd_handles.append(h)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(str(args.title))

    # Time ticks as integer seconds: 1, 2, 3, ... (avoid Matplotlib auto-fractions).
    xmin, xmax = ax.get_xlim()
    if np.isfinite(xmin) and np.isfinite(xmax):
        # Guard against extremely long horizons creating thousands of ticks.
        if (xmax - xmin) <= 60.0:
            tick_start = max(1, int(np.ceil(xmin)))
            tick_end = int(np.floor(xmax))
            if tick_end >= tick_start:
                ax.set_xticks(np.arange(tick_start, tick_end + 1, 1))

    # Trip event marker (legend entry should appear last)
    trip_handle = ax.axvline(
        float(t_trip),
        color="k",
        linestyle=":",
        linewidth=1.4,
        alpha=0.85,
        zorder=0,
        label=f"Trip time: t={float(t_trip):.3f} s",
    )

    # Compact legend: place below the axes (avoids covering curves).
    # Legend order (as requested): all frequency means, then trip time, then last-cmd entries.
    handles: list[plt.Line2D] = []
    handles.extend(mean_handles)
    handles.append(trip_handle)
    handles.extend(last_cmd_handles)

    ax.legend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        labelspacing=0.3,
        handlelength=2.0,
        handletextpad=0.6,
        columnspacing=1.2,
    )

    # Always store plots as PDF
    out_name = Path(str(args.out_name)).name
    out_name = str(Path(out_name).with_suffix(".pdf"))
    out_path = PLOTS_DIR / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=0.25)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def main() -> None:
    args = _parse_args()
    plot_comparison(args)


if __name__ == "__main__":
    main()
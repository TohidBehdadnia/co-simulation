import argparse
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


def _parse_scenario_spec(spec: str) -> tuple[str, str]:
    """Parse a scenario argument.

    Supported forms
    ---------------
    - case_iec104_direct_all
    - case_iec104_direct_all=Direct (all-at-once)
    - case_iec104_direct_all.npz=Direct (all-at-once)

    Returns
    -------
    (scenario_id, label)
    """
    raw = (spec or "").strip()
    if not raw:
        raise ValueError("Empty scenario spec")

    if "=" in raw:
        scenario_id, label = raw.split("=", 1)
        scenario_id = scenario_id.strip()
        label = label.strip()
        if not scenario_id:
            raise ValueError(f"Invalid scenario spec (missing id): {spec!r}")
        if not label:
            # Allow "id=" and fall back to id as label.
            label = scenario_id
        return scenario_id, label

    # Default: label is the id itself.
    return raw, raw


def _resolve_npz_path(scenario: str) -> Path:
    name = scenario.strip()
    if not name:
        raise ValueError("Empty scenario name")

    filename = name if name.endswith(".npz") else f"{name}.npz"
    return RESULTS_DIR / filename


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
            "Example: case_iec104_direct_all='Direct (all-at-once)'"
        ),
    )
    ap.add_argument(
        "--out-name",
        default=str("freq_comparison.png"),
        help="Output PNG name (default: freq_comparison.png)",
    )
    ap.add_argument(
        "--title",
        default="Frequency comparison",
        help="Plot title",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"results folder not found: {RESULTS_DIR}")

    fig, ax = plt.subplots()

    i = 0
    for spec in args.scenarios:
        scenario_id, label = _parse_scenario_spec(spec)
        npz_path = _resolve_npz_path(scenario_id)
        if not npz_path.exists():
            available = sorted(p.name for p in RESULTS_DIR.glob("*.npz"))
            raise FileNotFoundError(
                f"Scenario file not found: {npz_path}\n"
                f"Available in results/: {', '.join(available) if available else '(none)'}"
            )

        data = np.load(npz_path)
        if "t" not in data or "freq" not in data:
            raise KeyError(f"{npz_path.name} must contain arrays 't' and 'freq'")

        t = np.asarray(data["t"], dtype=float)
        freq = np.asarray(data["freq"], dtype=float)
        last_cmd_time = t_trip + float(data["measured_delay"]) if "measured_delay" in data else None
        if t.shape != freq.shape:
            raise ValueError(
                f"Mismatched shapes in {npz_path.name}: t{t.shape} vs freq{freq.shape}"
            )

        # Add sparse markers so multiple curves remain readable.
        n = int(t.size)
        mark_every = max(1, n // 30) if n > 0 else 1
        (line,) = ax.plot(
            t,
            freq,
            label=label,
            marker=markers[i % len(markers)],
            markevery=mark_every,
            markersize=3,
            linewidth=1.5,
        )
        if last_cmd_time is not None:
            t_last = last_cmd_time
            ax.axvline(
                t_last,
                color=line.get_color(),
                linestyle="--",
                label=f"Last command: t={t_last:.3f} s",
                linewidth=1.2,
                alpha=0.9,
            )
        i += 1

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(str(args.title))
    ax.legend(loc="best")

    out_path = PLOTS_DIR / args.out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
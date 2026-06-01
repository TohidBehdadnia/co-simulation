import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import sys

import matplotlib

# Enable LaTeX/PGF backend as early as possible (must happen before importing pyplot).
# - CLI usage: `python plotter_comparator.py ... --latex`
# - Programmatic usage: set env `CO_SIM_USE_LATEX=1` before importing this module.
_WANT_LATEX = ("--latex" in sys.argv) or (str(os.environ.get("CO_SIM_USE_LATEX", "0")).strip() in {"1", "true", "yes"})
_WANT_POSTER = ("--poster" in sys.argv)


def _apply_latex_rcparams() -> None:
    # NOTE: This does not validate that a LaTeX distribution is installed.
    # If missing, matplotlib will raise at render/save time.
    matplotlib.rcParams.update(
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
        value = matplotlib.rcParams.get(key)
        if isinstance(value, (int, float)):
            matplotlib.rcParams[key] = float(value) * scale

import seaborn as sns

sns.set_theme()

if _WANT_LATEX:
    try:
        # Prefer a LaTeX-native backend.
        matplotlib.use("pgf")
    except Exception:
        # Backend switch can fail in some environments; keep going with default backend.
        pass
    try:
        _apply_latex_rcparams()
    except Exception:
        # Fall back to default rcParams; we can still try to enable usetex later.
        pass

if _WANT_POSTER:
    _scale_font_rcparams(1.20)

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

# Event times
t_trip = 1.0

# Colors and markers for up to 10 scenarios. Matplotlib will cycle through these.
markers = ["o", "s", "^", "X", "v", "<", ">", "*", "P"]

def _strategy_palette(n_colors: int) -> list[object]:
    """Return the default categorical palette, reordered.

    Keeps the current seaborn theme palette (set by `sns.set_theme()`, typically
    the less-vivid "deep" palette), but reorders the first strategies to:
    blue → green → yellow → red.
    """

    n = max(1, int(n_colors))
    base = sns.color_palette(n_colors=10)

    preferred = [0, 2, 1, 3]
    remaining = [i for i in range(10) if i not in preferred]
    order = preferred + remaining

    pal = [base[i] for i in order]
    if n <= 10:
        return pal[:n]
    # More than 10: fall back to seaborn's extended palette.
    return sns.color_palette(n_colors=n)


def _extract_run_id_from_npz_name(path: Path) -> int | None:
    """Extract trailing numeric run id from a run npz filename.

    Expected form: <scenario_name>_<run_id>.npz
    """
    name = path.name
    if not name.endswith(".npz"):
        return None
    stem = name[:-4]
    if "_" not in stem:
        return None
    last = stem.rsplit("_", 1)[-1]
    try:
        return int(last)
    except Exception:
        return None


def _iter_iec104_type_timestamps(
    pcap_path: Path,
    *,
    type_ids: tuple[int, ...],
    src_ports: tuple[int, ...] | None = None,
    dst_ports: tuple[int, ...] | None = None,
        # Parser kept for backward compatibility; dpkt is always used.
) -> list[float]:
    """Return epoch timestamps of IEC-104 ASDUs matching type_ids in a pcap.

    Uses a lightweight dpkt-based parser that scans TCP payloads for IEC-104 APDUs.

    Filters (optional):
    - src_ports: only packets whose TCP source port is in src_ports
    - dst_ports: only packets whose TCP destination port is in dst_ports
    """

    type_ids = tuple(int(x) for x in (type_ids or ()))
    if not type_ids:
        return []

    src_ports_set = set(int(p) for p in src_ports) if src_ports is not None else None
    dst_ports_set = set(int(p) for p in dst_ports) if dst_ports is not None else None

    # dpkt-based scanner.
    try:
        import dpkt  # type: ignore
    except Exception:
        return []

    timestamps: list[float] = []
    try:
        with open(pcap_path, "rb") as f:
            pcap = dpkt.pcap.Reader(f)
            dl = pcap.datalink()

            for ts, buf in pcap:
                try:
                    if dl == dpkt.pcap.DLT_EN10MB:
                        frame = dpkt.ethernet.Ethernet(buf)
                        payload = frame.data
                    elif dl == getattr(dpkt.pcap, "DLT_LINUX_SLL", -1):
                        frame = dpkt.sll.SLL(buf)
                        payload = frame.data
                    elif dl == getattr(dpkt.pcap, "DLT_LINUX_SLL2", -2):
                        # dpkt may not have sll2 in older versions.
                        sll2 = getattr(dpkt, "sll2", None)
                        if sll2 is None:
                            continue
                        frame = sll2.SLL2(buf)
                        payload = frame.data
                    else:
                        continue

                    ip = payload
                    if isinstance(ip, dpkt.ip.IP):
                        tcp = ip.data
                    elif isinstance(ip, dpkt.ip6.IP6):
                        tcp = ip.data
                    else:
                        continue

                    if not isinstance(tcp, dpkt.tcp.TCP):
                        continue
                    if dst_ports_set is not None and int(tcp.dport) not in dst_ports_set:
                        continue
                    if src_ports_set is not None and int(tcp.sport) not in src_ports_set:
                        continue

                    data = bytes(tcp.data or b"")
                    if not data:
                        continue

                    i = 0
                    n = len(data)
                    while i + 2 <= n:
                        if data[i] != 0x68:
                            i += 1
                            continue
                        apdu_len = int(data[i + 1])
                        total = 2 + apdu_len
                        if i + total > n:
                            break
                        # Control field is 4 bytes after start+len. ASDU begins at offset 6.
                        if apdu_len >= 5:
                            type_id = int(data[i + 6])
                            if type_id in type_ids:
                                timestamps.append(float(ts))
                        i += total
                except Exception:
                    continue
    except Exception:
        return []

    return timestamps


def _iter_iec104_startdt_con_timestamps(
    pcap_path: Path,
    *,
    src_ports: tuple[int, ...] | None = None,
    dst_ports: tuple[int, ...] | None = None,
) -> list[float]:
    """Return epoch timestamps of IEC-104 StartDT con U-frames in a pcap.

    This marks the end of the IEC-104 handshake (StartDT activation confirmed).
    The parser is dpkt-based and scans each TCP payload for IEC-104 APDUs.
    """

    src_ports_set = set(int(p) for p in src_ports) if src_ports is not None else None
    dst_ports_set = set(int(p) for p in dst_ports) if dst_ports is not None else None

    try:
        import dpkt  # type: ignore
    except Exception:
        return []

    timestamps: list[float] = []
    try:
        with open(pcap_path, "rb") as f:
            pcap = dpkt.pcap.Reader(f)
            dl = pcap.datalink()

            for ts, buf in pcap:
                try:
                    if dl == dpkt.pcap.DLT_EN10MB:
                        frame = dpkt.ethernet.Ethernet(buf)
                        payload = frame.data
                    elif dl == getattr(dpkt.pcap, "DLT_LINUX_SLL", -1):
                        frame = dpkt.sll.SLL(buf)
                        payload = frame.data
                    elif dl == getattr(dpkt.pcap, "DLT_LINUX_SLL2", -2):
                        sll2 = getattr(dpkt, "sll2", None)
                        if sll2 is None:
                            continue
                        frame = sll2.SLL2(buf)
                        payload = frame.data
                    else:
                        continue

                    ip = payload
                    if isinstance(ip, dpkt.ip.IP):
                        tcp = ip.data
                    elif isinstance(ip, dpkt.ip6.IP6):
                        tcp = ip.data
                    else:
                        continue

                    if not isinstance(tcp, dpkt.tcp.TCP):
                        continue
                    if dst_ports_set is not None and int(tcp.dport) not in dst_ports_set:
                        continue
                    if src_ports_set is not None and int(tcp.sport) not in src_ports_set:
                        continue

                    data = bytes(tcp.data or b"")
                    if not data:
                        continue

                    i = 0
                    n = len(data)
                    while i + 2 <= n:
                        if data[i] != 0x68:
                            i += 1
                            continue
                        apdu_len = int(data[i + 1])
                        total = 2 + apdu_len
                        if i + total > n:
                            break

                        # U-format StartDT con: 68 04 0B 00 00 00
                        if apdu_len == 4 and i + 6 <= n:
                            c1 = data[i + 2]
                            c2 = data[i + 3]
                            c3 = data[i + 4]
                            c4 = data[i + 5]
                            if c1 == 0x0B and c2 == 0x00 and c3 == 0x00 and c4 == 0x00:
                                timestamps.append(float(ts))

                        i += total
                except Exception:
                    continue
    except Exception:
        return []

    return timestamps


def _iter_iec104_type_cot_timestamps(
    pcap_path: Path,
    *,
    type_ids: tuple[int, ...],
    cot_causes: tuple[int, ...],
    src_ports: tuple[int, ...] | None = None,
    dst_ports: tuple[int, ...] | None = None,
) -> list[float]:
    """Return epoch timestamps of IEC-104 ASDUs matching type_ids and COT causes.

    COT cause is extracted from the first COT byte (low 6 bits) of the ASDU header.
    """

    type_ids = tuple(int(x) for x in (type_ids or ()))
    cot_causes = tuple(int(x) for x in (cot_causes or ()))
    if not type_ids or not cot_causes:
        return []

    src_ports_set = set(int(p) for p in src_ports) if src_ports is not None else None
    dst_ports_set = set(int(p) for p in dst_ports) if dst_ports is not None else None

    try:
        import dpkt  # type: ignore
    except Exception:
        return []

    timestamps: list[float] = []
    try:
        with open(pcap_path, "rb") as f:
            pcap = dpkt.pcap.Reader(f)
            dl = pcap.datalink()

            for ts, buf in pcap:
                try:
                    if dl == dpkt.pcap.DLT_EN10MB:
                        frame = dpkt.ethernet.Ethernet(buf)
                        payload = frame.data
                    elif dl == getattr(dpkt.pcap, "DLT_LINUX_SLL", -1):
                        frame = dpkt.sll.SLL(buf)
                        payload = frame.data
                    elif dl == getattr(dpkt.pcap, "DLT_LINUX_SLL2", -2):
                        sll2 = getattr(dpkt, "sll2", None)
                        if sll2 is None:
                            continue
                        frame = sll2.SLL2(buf)
                        payload = frame.data
                    else:
                        continue

                    ip = payload
                    if isinstance(ip, dpkt.ip.IP):
                        tcp = ip.data
                    elif isinstance(ip, dpkt.ip6.IP6):
                        tcp = ip.data
                    else:
                        continue

                    if not isinstance(tcp, dpkt.tcp.TCP):
                        continue
                    if dst_ports_set is not None and int(tcp.dport) not in dst_ports_set:
                        continue
                    if src_ports_set is not None and int(tcp.sport) not in src_ports_set:
                        continue

                    data = bytes(tcp.data or b"")
                    if not data:
                        continue

                    i = 0
                    n = len(data)
                    while i + 2 <= n:
                        if data[i] != 0x68:
                            i += 1
                            continue
                        apdu_len = int(data[i + 1])
                        total = 2 + apdu_len
                        if i + total > n:
                            break

                        # I-format with ASDU: control field (4 bytes) + ASDU header.
                        # ASDU begins at offset 6: type(1), vsq(1), cot(2), ca(2)...
                        if apdu_len >= 10 and i + 10 <= n:
                            type_id = int(data[i + 6])
                            if type_id in type_ids:
                                cot_cause = int(data[i + 8]) & 0x3F
                                if cot_cause in cot_causes:
                                    timestamps.append(float(ts))

                        i += total
                except Exception:
                    continue
    except Exception:
        return []

    return timestamps


def _iter_iec104_type50_timestamps(
    pcap_path: Path,
    *,
    dst_ports: tuple[int, ...] = (2404, 2405),
    # Parser kept for backward compatibility; dpkt is always used.
) -> list[float]:
    """Back-compat wrapper: timestamps of C_SE_NC_1 (typeid 50) packets."""
    return _iter_iec104_type_timestamps(
        pcap_path,
        type_ids=(50,),
        dst_ports=dst_ports,
        # parser parameter is no longer used.
    )


def _maybe_generate_pcap_plots(
    *,
    out_stem: str,
    scenario_specs: list["ScenarioSpec"],
    scenario_run_ids: dict[str, list[int]],
    max_runs: int,
    pcap_parser: str,
    scenario_order: list[str],
    scenario_colors: dict[str, object],
    poster: bool,
) -> None:
    """Generate extra plots derived from tcpdump pcaps, when present.

    Produces strategy-summary plots (when data is available), considering only
    IEC-104 C_SE_NC_1 (typeid 50) setpoint APDUs going to the outstation/server
    (dst port 2404):

    - delay strategies (+ direct): distribution of inter-arrival times (IATs)
    - drop strategies (+ direct): distribution of per-run setpoint counts

    Note: this function is independent of the removed 3D distribution plot feature.
    """

    logs_dir = BASE_DIR / "logs"
    if not logs_dir.exists():
        return

    label_by_id = {s.scenario_id: s.label for s in scenario_specs}

    scenario_ids = [str(s.scenario_id) for s in scenario_specs]
    has_delay_strategies = any("mitm_delay" in sid for sid in scenario_ids)
    has_drop_strategies = any("mitm_drop" in sid for sid in scenario_ids)

    rel_arrival_delay_group: dict[str, list[float]] = {}
    counts_drop_group: dict[str, list[int]] = {}
    direct_label: str | None = None

    for scenario_id, run_ids in scenario_run_ids.items():
        run_ids_int = [int(x) for x in run_ids]
        if max_runs > 0 and len(run_ids_int) > max_runs:
            run_ids_int = run_ids_int[-max_runs:]

        sid = str(scenario_id)
        is_direct = "direct" in sid
        is_delay = "mitm_delay" in sid
        is_drop = "mitm_drop" in sid
        if not (is_direct or is_delay or is_drop):
            continue

        label = str(label_by_id.get(scenario_id, scenario_id))
        if is_direct and direct_label is None:
            direct_label = label

        for run_id in run_ids_int:
            pcap_path = logs_dir / f"run_{int(run_id)}" / "traffic.pcap"
            if not pcap_path.exists():
                continue

            ts = _iter_iec104_type50_timestamps(pcap_path, dst_ports=(2404,))

            if has_delay_strategies and (is_direct or is_delay) and len(ts) >= 1:
                ts_sorted = sorted(float(x) for x in ts)

                # Reference time: C_SC_NA_1 Activation Confirmation from server.
                # This is immediately before the (delayed) measurement issuance in our scenario,
                # and is a more representative "start" for the introduced delay.
                # Fallback to first measurement snapshot (type 13), then to first setpoint.
                t_refs = _iter_iec104_type_cot_timestamps(
                    pcap_path,
                    type_ids=(45,),
                    cot_causes=(7,),
                    src_ports=(2404,),
                )
                if t_refs:
                    t_ref = float(min(t_refs))
                else:
                    t_meas = _iter_iec104_type_timestamps(pcap_path, type_ids=(13,), src_ports=(2404,))
                    t_ref = float(min(t_meas)) if t_meas else float(ts_sorted[0])

                rel = [float(t - t_ref) for t in ts_sorted if np.isfinite(t_ref) and t >= t_ref]
                if rel:
                    rel_arrival_delay_group.setdefault(label, []).extend(rel)

            if has_drop_strategies and (is_direct or is_drop):
                counts_drop_group.setdefault(label, []).append(int(len(ts)))

    def _palette_for(order: list[str]) -> dict[str, object]:
        # Only include colors for categories we actually plot.
        pal = {lab: scenario_colors.get(lab) for lab in order}
        # Fill any missing entries with a default palette.
        missing = [k for k, v in pal.items() if v is None]
        if missing:
            fallback = sns.color_palette(n_colors=len(missing))
            for k, c in zip(missing, fallback):
                pal[k] = c
        return pal

    def _add_median_labels(
        *,
        ax: plt.Axes,
        df: "pd.DataFrame",
        order: list[str],
        fmt: str,
    ) -> None:
        medians = df.groupby("Scenario", sort=False)["Value"].median()
        values = df["Value"].to_numpy(dtype=float)
        if values.size == 0:
            return
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        span = vmax - vmin
        offset = span * 0.01 if span > 0.0 else max(0.01, abs(vmax) * 0.01)

        for idx, label in enumerate(order):
            if label not in medians:
                continue
            val = float(medians[label])
            ax.text(
                float(idx),
                val + offset,
                fmt.format(val),
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
            )

    def _save_count_boxplot(
        *,
        data: dict[str, list[int]],
        out_path: Path,
        dpi: int,
        title: str,
        order: list[str],
        baseline_label: str | None,
    ) -> None:
        baseline_mean: float | None = None
        if baseline_label and baseline_label in data:
            base_vals = [int(v) for v in data.get(baseline_label, []) if v is not None]
            if base_vals:
                baseline_mean = float(np.mean(base_vals))

        rows: list[tuple[str, int]] = []
        for lab, values in data.items():
            for v in values:
                if baseline_mean is not None and baseline_mean > 0.0:
                    rows.append((str(lab), float(v) * 100.0 / baseline_mean))
                else:
                    rows.append((str(lab), int(v)))
        if not rows:
            return

        labs = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_axisbelow(True)

        import pandas as pd

        df = pd.DataFrame({"Scenario": labs, "Value": vals})
        sns.boxplot(
            data=df,
            x="Scenario",
            y="Value",
            hue="Scenario",
            order=order,
            palette=_palette_for(order),
            dodge=False,
            ax=ax,
            medianprops={"color": "black", "linewidth": 1.5}
        )
        if baseline_mean is not None and baseline_mean > 0.0:
            fmt = "{:.1f}\\%" if _WANT_LATEX else "{:.1f}%"
        else:
            fmt = "{:.0f}"
        _add_median_labels(ax=ax, df=df, order=order, fmt=fmt)
        ax.legend_.remove() if ax.legend_ else None
        ax.set_xlabel("Scenario") 
        if _WANT_LATEX:
            if baseline_mean is not None and baseline_mean > 0.0:
                ax.set_ylabel("Cumulative received commands, $c(\\tau_j)$ (\\%)")
            else:
                ax.set_ylabel("Cumulative received commands, $c(\\tau_j)$")
        else:
            if baseline_mean is not None and baseline_mean > 0.0:
                ax.set_ylabel("Setpoint commands received (% of direct)")
            else:
                ax.set_ylabel("N. Setpoint commands received")
        if title is not None:
            ax.set_title(str(title))
        ax.tick_params(axis="x", rotation=15)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.set_size_inches(6, 4, forward=True)
        fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        print(f"Saved pcap setpoint count boxplot to {out_path}")

    def _save_rel_arrival_boxplot(
        *,
        data: dict[str, list[float]],
        out_path: Path,
        dpi: int,
        title: str,
        order: list[str],
    ) -> None:
        rows: list[tuple[str, float]] = []
        for lab, values in data.items():
            for v in values:
                try:
                    fv = float(v)
                except Exception:
                    continue
                if np.isfinite(fv) and fv >= 0.0:
                    rows.append((str(lab), fv))
        if not rows:
            return

        labs = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_axisbelow(True)

        import pandas as pd

        df = pd.DataFrame({"Scenario": labs, "Value": vals})
        sns.boxplot(
            data=df,
            x="Scenario",
            y="Value",
            hue="Scenario",
            order=order,
            palette=_palette_for(order),
            dodge=False,
            ax=ax,
            medianprops={"color": "black", "linewidth": 1.5}
        )
        _add_median_labels(ax=ax, df=df, order=order, fmt="{:.2f}")
        ax.legend_.remove() if ax.legend_ else None
        ax.set_xlabel("Scenario")
        if _WANT_LATEX:
            ax.set_ylabel("Setpoint commands arrival time, $t_k^{{arr}}$ (s)")
        else:
            ax.set_ylabel("Setpoint commands arrival time (s)")

        from matplotlib.ticker import MultipleLocator

        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        if title is not None:
            ax.set_title(str(title))
        ax.tick_params(axis="x", rotation=15)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.set_size_inches(6, 4, forward=True)
        fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        print(f"Saved pcap setpoint relative-arrival boxplot to {out_path}")

    pcap_ext = "png" if bool(poster) else "pdf"
    pcap_dpi = 700 if bool(poster) else 300

    if has_delay_strategies and rel_arrival_delay_group:
        order = [lab for lab in scenario_order if (lab in rel_arrival_delay_group)]
        _save_rel_arrival_boxplot(
            data=rel_arrival_delay_group,
            out_path=PLOTS_DIR / f"pcap_setpoint_arrival_rel_to_meas_delay.{pcap_ext}",
            dpi=pcap_dpi,
            title=None,#"Setpoint arrival times relative to trip notify (per scenario)",
            order=order,
        )
    if has_drop_strategies and counts_drop_group:
        order = [lab for lab in scenario_order if (lab in counts_drop_group)]
        _save_count_boxplot(
            data=counts_drop_group,
            out_path=PLOTS_DIR / f"pcap_setpoint_count_to_server_drop.{pcap_ext}",
            dpi=pcap_dpi,
            title=None,#"N. Setpoint commands to server (per scenario)",
            order=order,
            baseline_label=direct_label,
        )


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
        help="Uncertainty band to show for frequency and last-cmd-time: 95%% CI of mean (ci95) or ±1 SD (sd).",
    )

    ap.add_argument(
        "--latex",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable LaTeX text rendering (PGF backend when possible). "
            "Requires a working LaTeX installation (e.g. pdflatex)."
        ),
    )

    ap.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help=(
            "Maximum number of runs (.npz files) to load per scenario when the scenario is a results/ folder. "
            "0 means load all runs (default). If > 0, the most recent runs (by filename sort) are used."
        ),
    )

    ap.add_argument(
        "--pcap-parser",
        choices=["none", "dpkt"],
        default="none",
        help=(
            "Parser to use for optional pcap-derived plots. "
            "'none' disables pcap plots (default). 'dpkt' enables pcap plots."
        ),
    )

    ap.add_argument(
        "--poster",
        action="store_true",
        help="Set seaborn context to poster"
    )

    return ap.parse_args()


def plot_comparison(args: argparse.Namespace) -> None:

    # If module was imported without the early backend switch, still enable usetex when requested.
    if bool(getattr(args, "latex", False)):
        try:
            _apply_latex_rcparams()
        except Exception:
            pass
    
    if getattr(args, "poster", False):
        _scale_font_rcparams(1.20)

    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"results folder not found: {RESULTS_DIR}")

    specs = _normalize_specs(args)

    scenario_ids = [str(s.scenario_id) for s in specs]
    has_delay = any("delay" in sid for sid in scenario_ids)
    has_drop = any("drop" in sid for sid in scenario_ids)
    has_mod = any("mod" in sid for sid in scenario_ids)

    max_runs = int(getattr(args, "max_runs", 0) or 0)
    pcap_parser = str(getattr(args, "pcap_parser", "none") or "none").lower().strip()
    want_pcap_plots = pcap_parser != "none" and (has_delay or has_drop or has_mod)

    # When generating pcap-derived plots, skip the 2D frequency comparison plot.
    want_freq_plot = not want_pcap_plots
    need_freq_data = want_freq_plot

    band_mode = str(getattr(args, "band", "ci95")).lower()

    fig = None
    ax = None
    mean_handles: list[plt.Line2D] = []
    last_cmd_stats: list[tuple[str, str, float | None, float, bool]] = []  # (label, color, mean, hw, is_attack)
    # Per-strategy summary stats to export when plotting mean ±95% CI.
    # Columns requested: average, median, minimum, maximum, relative accuracy.
    # Here, these are computed on all frequency samples across runs after the trip time.
    scenario_summary_rows: list[dict[str, str]] = []
    if want_freq_plot:
        fig, ax = plt.subplots()
        ax.set_axisbelow(True)

    # Deterministic palette: blue→green→yellow→red from Direct to last strategy.
    base_palette = _strategy_palette(len(specs))
    out_name_base = Path(str(args.out_name)).name
    out_stem = str(Path(out_name_base).with_suffix(""))

    scenario_run_ids: dict[str, list[int]] = {}
    scenario_order: list[str] = [str(s.label) for s in specs]
    # Assign Direct the first color (blue) even if user-provided specs are not ordered.
    scenario_colors: dict[str, object] = {}
    direct_idx = None
    for idx, s in enumerate(specs):
        if "direct" in str(s.scenario_id).lower():
            direct_idx = idx
            break
    pal_iter = iter(base_palette)
    if direct_idx is not None:
        scenario_colors[str(specs[direct_idx].label)] = next(pal_iter)
        for j, s in enumerate(specs):
            if j == direct_idx:
                continue
            scenario_colors[str(s.label)] = next(pal_iter, base_palette[-1])
    else:
        for s in specs:
            scenario_colors[str(s.label)] = next(pal_iter, base_palette[-1])

    i = 1
    for spec in specs:
        scenario_id = spec.scenario_id
        is_attack = "direct" not in str(scenario_id).lower()
        label = spec.label
        npz_paths = _resolve_npz_paths(scenario_id)
        npz_paths = [p for p in npz_paths if p.exists()]
        if max_runs > 0 and len(npz_paths) > max_runs:
            npz_paths = npz_paths[-max_runs:]
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

        # Track run ids (for optional pcap-derived plots).
        run_ids: list[int] = []
        for p in npz_paths:
            rid = _extract_run_id_from_npz_name(Path(p))
            if rid is not None:
                run_ids.append(int(rid))
        scenario_run_ids[str(scenario_id)] = run_ids

        if need_freq_data:
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

            # Collect summary stats for CSV export (only written for ci95 plots).
            # Use post-trip samples so the pre-trip steady-state does not dominate the stats.
            try:
                t_mask = np.asarray(t_plot, dtype=float) >= float(t_trip)
                if np.any(t_mask):
                    samples = np.asarray(freq_mat[:, t_mask], dtype=float).ravel()
                else:
                    samples = np.asarray(freq_mat, dtype=float).ravel()

                samples = samples[np.isfinite(samples)]
                if samples.size:
                    avg_v = float(np.nanmean(samples))
                    med_v = float(np.nanmedian(samples))
                    min_v = float(np.nanmin(samples))
                    max_v = float(np.nanmax(samples))
                else:
                    avg_v = float("nan")
                    med_v = float("nan")
                    min_v = float("nan")
                    max_v = float("nan")

                # Relative accuracy (as requested): CI semiwidth / average value * 100.
                # It is meaningful only for ci95 band; leave blank otherwise.
                rel_acc_pct: float | None = None
                max_rel_acc_pct: float | None = None
                if band_mode == "ci95":
                    mean_f, lo_f, hi_f = _mean_and_band(freq_mat, mode=band_mode)
                    # CI half-width per time step: (hi - lo) / 2.
                    hw = 0.5 * (np.asarray(hi_f, dtype=float) - np.asarray(lo_f, dtype=float))
                    if np.any(t_mask):
                        mean_sel = np.asarray(mean_f, dtype=float)[t_mask]
                        hw_sel = np.asarray(hw, dtype=float)[t_mask]
                    else:
                        mean_sel = np.asarray(mean_f, dtype=float)
                        hw_sel = np.asarray(hw, dtype=float)

                    mean_sel = np.asarray(mean_sel, dtype=float)
                    hw_sel = np.asarray(hw_sel, dtype=float)
                    finite_mask = np.isfinite(mean_sel) & np.isfinite(hw_sel)
                    denom_sel = np.abs(mean_sel)
                    finite_mask = finite_mask & (denom_sel != 0.0)

                    if np.any(finite_mask):
                        denom = float(np.nanmean(denom_sel[finite_mask]))
                        numer = float(np.nanmean(hw_sel[finite_mask]))
                        if np.isfinite(denom) and np.isfinite(numer):
                            rel_acc_pct = 100.0 * (numer / denom)
                            ratio_series = 100.0 * (hw_sel[finite_mask] / denom_sel[finite_mask])
                            if ratio_series.size and np.any(np.isfinite(ratio_series)):
                                max_rel_acc_pct = float(np.nanmax(ratio_series))
                        elif np.isfinite(denom) and np.isfinite(numer) and numer == 0.0:
                            rel_acc_pct = 0.0
                            max_rel_acc_pct = 0.0

                scenario_summary_rows.append(
                    {
                        "strategy": str(label),
                        "average": f"{avg_v:.6g}" if np.isfinite(avg_v) else "",
                        "median": f"{med_v:.6g}" if np.isfinite(med_v) else "",
                        "minimum": f"{min_v:.6g}" if np.isfinite(min_v) else "",
                        "maximum": f"{max_v:.6g}" if np.isfinite(max_v) else "",
                        "average_relative_accuracy": f"{rel_acc_pct:.6g}" if rel_acc_pct is not None and np.isfinite(rel_acc_pct) else "",
                        "max_relative_accuracy": f"{max_rel_acc_pct:.6g}" if max_rel_acc_pct is not None and np.isfinite(max_rel_acc_pct) else "",
                    }
                )
            except Exception:
                # Never fail the plot if stats export cannot be computed.
                pass

            if want_freq_plot:
                assert ax is not None and fig is not None
                mean_f, lo_f, hi_f = _mean_and_band(freq_mat, mode=band_mode)
                n_lines_before = len(ax.lines)
                band_txt = "±95% CI" if band_mode == "ci95" else "±1 SD"
                label_str = f"{label}"#f"{label} (mean {band_txt})"
                # Enforce the shared strategy palette (blue→green→yellow→red).
                line_color = scenario_colors.get(str(label))
                if line_color is None:
                    line_color = base_palette[(i - 1) % len(base_palette)]
                    scenario_colors[str(label)] = line_color
                sns.lineplot(
                    x=t_plot,
                    y=mean_f,
                    ax=ax,
                    label=label_str,
                    linewidth=2.0,
                    markersize=6,
                    color=line_color,
                )
                if len(ax.lines) <= n_lines_before:
                    raise RuntimeError("Failed to create mean line plot")
                line = ax.lines[-1]
                line.set_marker(markers[(i - 1) % len(markers)])
                line.set_markevery(15)
                mean_handles.append(line)
                # Keep scenario_colors stable (do not overwrite with seaborn defaults).

                if freq_mat.shape[0] > 1:
                    ax.fill_between(
                        t_plot,
                        lo_f,
                        hi_f,
                        color=line_color,
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
                        last_cmd_stats.append((label, line_color, m, float(hw), is_attack))
                    else:
                        last_cmd_stats.append((label, line_color, None, 0.0, is_attack))
                else:
                    last_cmd_stats.append((label, line_color, None, 0.0, is_attack))
        i += 1

    if want_freq_plot:
        assert ax is not None and fig is not None

        # Draw last-command mean vertical lines and uncertainty spans after curves are plotted.
        last_cmd_handles: list[plt.Line2D] = []
        if last_cmd_stats and not (has_drop or has_mod):
            for (label, color, m, hw, is_attack) in last_cmd_stats:
                if m is None:
                    if _WANT_LATEX:
                        label_str = "$\\tilde{m}_K$: none received" if is_attack else "$m_K$: none received"
                    else:
                        label_str = " m_K: none received" if not is_attack else "m_K~: none received"
                    (h,) = ax.plot(
                        [],
                        [],
                        color=color,
                        linestyle="--",
                        linewidth=1.6,
                        label=label_str,
                    )
                    last_cmd_handles.append(h)
                    continue

                if hw > 0:
                    ax.axvspan(m - hw, m + hw, color=color, alpha=0.12, linewidth=0, zorder=1)

                band_txt = "±95% CI" if band_mode == "ci95" else "±1 SD"
                band_txt = f"(mean {band_txt})"
                label_str = f"m_K t={m:.3f}s"
                if _WANT_LATEX:
                    label_str = f"$m_K$: $t={m:.3f}\\mathrm{{s}}$"
                    if is_attack:
                        label_str = f"$\\tilde{{m}}_K$: $t={m:.3f}\\mathrm{{s}}$"
                else:
                    if is_attack:
                        label_str = f"m_K~: t={m:.3f}s"
                h = ax.axvline(
                    m,
                    color=color,
                    linestyle="--",
                    linewidth=1.6,
                    alpha=0.95,
                    zorder=5,
                    label=label_str
                )
                last_cmd_handles.append(h)

        ax.set_xlabel("Time, t (s)")
        if args.latex:
            ax.set_xlabel("Time, $t$ (s)")
            if band_mode == "ci95":
                ax.set_ylabel("Frequency, $\omega(t) \pm 95\%$ CI (Hz)")
            else:
                ax.set_ylabel("Frequency, $\omega(t) \pm \mathrm{{SD}}$ (Hz)")
        else:
            if band_mode == "ci95":
                ax.set_ylabel("Frequency ±95% CI (Hz)")
            else:
                ax.set_ylabel("Frequency ±SD (Hz)")
        #ax.set_title(str(args.title))

        xmin, xmax = ax.get_xlim()
        if np.isfinite(xmin) and np.isfinite(xmax):
            if (xmax - xmin) <= 60.0:
                tick_start = max(0, int(np.ceil(xmin)))
                tick_end = int(np.floor(xmax))
                if tick_end >= tick_start:
                    ax.set_xticks(np.arange(tick_start, tick_end + 1, 1))

        label_str = f"Gen 2 trip: t={float(t_trip):.1f}s"
        if _WANT_LATEX:
            label_str = f"Gen 2 trip: $t={float(t_trip):.1f}\\mathrm{{s}}$"
        
        trip_handle = ax.axvline(
            float(t_trip),
            color="k",
            linestyle=":",
            linewidth=1.8,
            alpha=0.95,
            zorder=10,
            label=label_str,
        )

        handles: list[plt.Line2D] = []
        handles.extend(mean_handles)
        handles.append(trip_handle)
        handles.extend(last_cmd_handles)
        legend_fontsize = 8 if not args.poster else 10
        
        if not (has_drop or has_mod):
            ax.legend(
                handles=handles,
                labels=[h.get_label() for h in handles],
                loc="best",
                ncol=2,
            fontsize=legend_fontsize,
                frameon=True,
                framealpha=0.9,
                handlelength=2.0,
                handletextpad=0.6,
                columnspacing=1.2
            ).set_zorder(20)
        else:
            ax.legend(
                handles=handles,
                labels=[h.get_label() for h in handles],
                loc="best",
                fontsize=legend_fontsize,
                frameon=True,
                framealpha=0.9,
                handlelength=2.0,
                handletextpad=0.6,
                columnspacing=1.2
            ).set_zorder(20)

        if args.poster:
            out_name = str(Path(out_name_base).with_suffix(".png"))
        else:
            out_name = str(Path(out_name_base).with_suffix(".pdf"))
        out_path = PLOTS_DIR / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.subplots_adjust(bottom=0.30)
        fig.set_size_inches(7, 5.5, forward=True) if args.poster else fig.set_size_inches(6, 5, forward=True)
        fig.savefig(out_path, dpi=700, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {out_path}")

        # Export per-strategy stats when plotting with 95% CI.
        if band_mode == "ci95" and scenario_summary_rows:
            csv_path = RESULTS_DIR / f"{out_stem}_strategy_stats_ci95.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "strategy",
                        "average",
                        "median",
                        "minimum",
                        "maximum",
                        "average_relative_accuracy",
                        "max_relative_accuracy",
                    ],
                )
                w.writeheader()
                w.writerows(scenario_summary_rows)
            print(f"Saved strategy stats CSV to {csv_path}")

    if want_pcap_plots:
        _maybe_generate_pcap_plots(
            out_stem=out_stem,
            scenario_specs=specs,
            scenario_run_ids=scenario_run_ids,
            max_runs=max_runs,
            pcap_parser=pcap_parser,
            scenario_order=scenario_order,
            scenario_colors=scenario_colors,
            poster=bool(getattr(args, "poster", False)),
        )


def main() -> None:
    args = _parse_args()
    plot_comparison(args)


if __name__ == "__main__":
    main()
import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

# Event times
t_trip = 1.0

# Colors and markers for up to 10 scenarios. Matplotlib will cycle through these.
markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X"]


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
    dist_tag: str,
    scenario_specs: list["ScenarioSpec"],
    scenario_run_ids: dict[str, list[int]],
    max_runs: int,
    pcap_parser: str,
    scenario_order: list[str],
    scenario_colors: dict[str, object],
) -> None:
    """Generate extra plots derived from tcpdump pcaps, when present.

    Produces strategy-summary plots (when data is available), considering only
    IEC-104 C_SE_NC_1 (typeid 50) setpoint APDUs going to the outstation/server
    (dst port 2404):

    - delay strategies (+ direct): distribution of inter-arrival times (IATs)
    - drop strategies (+ direct): distribution of per-run setpoint counts

    dist_tag is kept for backward compatibility (used by other plot naming), but not used here.
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

    def _save_count_boxplot(
        *,
        data: dict[str, list[int]],
        out_path: Path,
        title: str,
        order: list[str],
    ) -> None:
        rows: list[tuple[str, int]] = []
        for lab, values in data.items():
            for v in values:
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
        )
        ax.legend_.remove() if ax.legend_ else None
        ax.set_xlabel("Scenario")
        ax.set_ylabel("# Setpoint commands received by server")
        ax.set_title(str(title))
        ax.tick_params(axis="x", rotation=15)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved pcap setpoint count boxplot to {out_path}")

    def _save_rel_arrival_boxplot(
        *,
        data: dict[str, list[float]],
        out_path: Path,
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
        )
        ax.legend_.remove() if ax.legend_ else None
        ax.set_xlabel("Scenario")
        ax.set_ylabel("Setpoint commands arrival time [s]")

        from matplotlib.ticker import MultipleLocator

        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.set_title(str(title))
        ax.tick_params(axis="x", rotation=15)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved pcap setpoint relative-arrival boxplot to {out_path}")

    if has_delay_strategies and rel_arrival_delay_group:
        order = [lab for lab in scenario_order if (lab in rel_arrival_delay_group)]
        _save_rel_arrival_boxplot(
            data=rel_arrival_delay_group,
            out_path=PLOTS_DIR / f"pcap_setpoint_arrival_rel_to_meas_delay.pdf",
            title="Setpoint arrival times relative to trip notify (per scenario)",
            order=order,
        )
    if has_drop_strategies and counts_drop_group:
        order = [lab for lab in scenario_order if (lab in counts_drop_group)]
        _save_count_boxplot(
            data=counts_drop_group,
            out_path=PLOTS_DIR / f"pcap_setpoint_count_to_server_drop.pdf",
            title="# Setpoint commands to server (per scenario)",
            order=order,
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
        "--dist3d",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also generate a 3D plot showing the distribution of frequency across runs over time "
            "(surface: time × frequency bins × density)."
        ),
    )
    ap.add_argument(
        "--dist3d-bins",
        type=int,
        default=80,
        help="Number of frequency bins to use for the 3D distribution plot (default: 80).",
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
    return ap.parse_args()


def _plot_frequency_distribution_3d(
    *,
    ax,
    t: np.ndarray,
    freq_mat: np.ndarray,
    title: str,
    edges: np.ndarray,
    color=None,
) -> None:
    """Plot a 3D surface of the frequency distribution over time onto an Axes3D.

    For each time index, we compute a histogram across runs, then plot a surface:
    X=time, Y=frequency bin centers, Z=density.
    """
    t = np.asarray(t, dtype=float)
    freq_mat = np.asarray(freq_mat, dtype=float)
    if t.ndim != 1 or freq_mat.ndim != 2:
        raise ValueError("Expected t as 1D and freq_mat as 2D")
    if freq_mat.shape[1] != t.shape[0]:
        raise ValueError(f"Mismatched shapes: freq_mat{freq_mat.shape} vs t{t.shape}")

    finite = np.isfinite(freq_mat)
    if not np.any(finite):
        raise ValueError("No finite frequency samples to plot")

    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or edges.size < 8:
        raise ValueError("Invalid histogram edges")
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Histogram per time step across runs.
    z = np.empty((t.shape[0], centers.shape[0]), dtype=float)
    for j in range(t.shape[0]):
        col = freq_mat[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            z[j, :] = np.nan
            continue
        h, _ = np.histogram(col, bins=edges, density=True)
        z[j, :] = h

    # Create a mesh for surface: time x freq.
    tt, ff = np.meshgrid(t, centers, indexing="xy")
    zz = z.T  # (freq_bins, time)

    surf_kwargs = dict(rstride=1, cstride=1, linewidth=0, antialiased=True)
    if color is not None:
        surf_kwargs.update(color=color, alpha=0.90)
    ax.plot_surface(tt, ff, zz, **surf_kwargs)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_zlabel("Density")
    ax.set_title(str(title))


def _plot_frequency_distribution_3d_grid(
    *,
    scenarios: list[tuple[str, object | None, np.ndarray, np.ndarray]],
    out_path: Path,
    bins: int,
    title: str,
) -> None:
    """Plot a single PDF with a grid of 3D distribution subplots.

    scenarios: list of (label, color, t, freq_mat)
    """
    if not scenarios:
        return

    bins = int(bins)
    if bins <= 5:
        raise ValueError("--dist3d-bins must be > 5")

    # Use common bin edges across all scenarios for comparability.
    fmin = float(np.nanmin([np.nanmin(m) for _, _, _, m in scenarios]))
    fmax = float(np.nanmax([np.nanmax(m) for _, _, _, m in scenarios]))
    if not np.isfinite(fmin) or not np.isfinite(fmax) or fmax <= fmin:
        raise ValueError(f"Invalid frequency range: fmin={fmin}, fmax={fmax}")
    pad = max(1e-6, 0.01 * (fmax - fmin))
    edges = np.linspace(fmin - pad, fmax + pad, bins + 1, dtype=float)

    # Layout: 2 columns for up to 8 scenarios (common batch sizes), else 3.
    n = len(scenarios)
    cols = 2 if n <= 8 else 3
    rows = int(np.ceil(n / cols))

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(6.0 * cols, 4.5 * rows))
    fig.suptitle(str(title))
    for idx, (label, color, t, mat) in enumerate(scenarios, start=1):
        ax = fig.add_subplot(rows, cols, idx, projection="3d")
        _plot_frequency_distribution_3d(
            ax=ax,
            t=t,
            freq_mat=mat,
            title=str(label),
            edges=edges,
            color=color,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(args: argparse.Namespace) -> None:

    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"results folder not found: {RESULTS_DIR}")

    specs = _normalize_specs(args)

    # Tag used to disambiguate combined 3D distribution filenames.
    # (Common case: direct+delay batch vs direct+drop batch with same --out-name.)
    scenario_ids = [str(s.scenario_id) for s in specs]
    has_delay = any("delay" in sid for sid in scenario_ids)
    has_drop = any("drop" in sid for sid in scenario_ids)
    if has_delay and not has_drop:
        dist_tag = "delay"
    elif has_drop and not has_delay:
        dist_tag = "drop"
    elif has_delay and has_drop:
        dist_tag = "mixed"
    else:
        dist_tag = "scenarios"

    want_dist3d = bool(getattr(args, "dist3d", False))
    dist3d_bins = int(getattr(args, "dist3d_bins", 80))
    max_runs = int(getattr(args, "max_runs", 0) or 0)
    pcap_parser = str(getattr(args, "pcap_parser", "none") or "none").lower().strip()
    want_pcap_plots = pcap_parser != "none"

    # When generating dist/pcap plots, skip the 2D frequency comparison plot.
    want_freq_plot = not (want_dist3d or want_pcap_plots)
    need_freq_data = want_freq_plot or want_dist3d

    band_mode = str(getattr(args, "band", "ci95")).lower()

    fig = None
    ax = None
    mean_handles: list[plt.Line2D] = []
    last_cmd_stats: list[tuple[str, str, float | None, float]] = []  # (label, color, mean, hw)
    if want_freq_plot:
        fig, ax = plt.subplots()
        ax.set_axisbelow(True)

    # Deterministic palette matching seaborn's default categorical palette.
    base_palette = sns.color_palette(n_colors=max(1, len(specs)))
    out_name_base = Path(str(args.out_name)).name
    out_stem = str(Path(out_name_base).with_suffix(""))

    dist3d_scenarios: list[tuple[str, object | None, np.ndarray, np.ndarray]] = []
    scenario_run_ids: dict[str, list[int]] = {}
    scenario_order: list[str] = [str(s.label) for s in specs]
    scenario_colors: dict[str, object] = {str(s.label): base_palette[i] for i, s in enumerate(specs)}

    i = 1
    for spec in specs:
        scenario_id = spec.scenario_id
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

            if want_freq_plot:
                assert ax is not None and fig is not None
                mean_f, lo_f, hi_f = _mean_and_band(freq_mat, mode=band_mode)
                n_lines_before = len(ax.lines)
                band_txt = "±95% CI" if band_mode == "ci95" else "±1 SD"
                label_str = f"{label} (mean {band_txt})"
                sns.lineplot(x=t_plot, y=mean_f, ax=ax, label=label_str, linewidth=2.0)
                if len(ax.lines) <= n_lines_before:
                    raise RuntimeError("Failed to create mean line plot")
                line = ax.lines[-1]
                line.set_marker(markers[i % len(markers)])
                line.set_markevery(15)
                mean_handles.append(line)
                line_color = line.get_color()
                scenario_colors[str(label)] = line_color

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
                        last_cmd_stats.append((label, line_color, m, float(hw)))
                    else:
                        last_cmd_stats.append((label, line_color, None, 0.0))
                else:
                    last_cmd_stats.append((label, line_color, None, 0.0))
            else:
                # No 2D plot: still pick a consistent color for dist3d.
                line_color = scenario_colors.get(str(label), base_palette[(i - 1) % len(base_palette)])
                scenario_colors[str(label)] = line_color

            if want_dist3d:
                dist3d_scenarios.append(
                    (
                        str(label),
                        line_color,
                        np.asarray(t_plot, dtype=float),
                        np.asarray(freq_mat, dtype=float),
                    )
                )
        i += 1

    if want_freq_plot:
        assert ax is not None and fig is not None

        # Draw last-command mean vertical lines and uncertainty spans after curves are plotted.
        last_cmd_handles: list[plt.Line2D] = []
        if last_cmd_stats:
            for (label, color, m, hw) in last_cmd_stats:
                if m is None:
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
                band_txt = f"(mean {band_txt})"
                h = ax.axvline(
                    m,
                    color=color,
                    linestyle="--",
                    linewidth=1.6,
                    alpha=0.95,
                    zorder=5,
                    label=f"Last cmd t={m:.3f}s {band_txt}",
                )
                last_cmd_handles.append(h)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_title(str(args.title))

        xmin, xmax = ax.get_xlim()
        if np.isfinite(xmin) and np.isfinite(xmax):
            if (xmax - xmin) <= 60.0:
                tick_start = max(0, int(np.ceil(xmin)))
                tick_end = int(np.floor(xmax))
                if tick_end >= tick_start:
                    ax.set_xticks(np.arange(tick_start, tick_end + 1, 1))

        trip_handle = ax.axvline(
            float(t_trip),
            color="k",
            linestyle=":",
            linewidth=1.8,
            alpha=0.95,
            zorder=10,
            label=f"Gen 2 trip t={float(t_trip):.1f} s",
        )

        handles: list[plt.Line2D] = []
        handles.extend(mean_handles)
        handles.append(trip_handle)
        handles.extend(last_cmd_handles)

        ax.legend(
            handles=handles,
            labels=[h.get_label() for h in handles],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            fontsize=8,
            frameon=True,
            framealpha=0.9,
            labelspacing=0.3,
            handlelength=2.0,
            handletextpad=0.6,
            columnspacing=1.2,
        )

        out_name = str(Path(out_name_base).with_suffix(".pdf"))
        out_path = PLOTS_DIR / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.subplots_adjust(bottom=0.30)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {out_path}")

    if want_dist3d and dist3d_scenarios:
        dist3d_all_path = PLOTS_DIR / f"{out_stem}_dist3d_all_{dist_tag}.pdf"
        _plot_frequency_distribution_3d_grid(
            scenarios=dist3d_scenarios,
            out_path=dist3d_all_path,
            bins=dist3d_bins,
            title=f"Frequency distribution (3D) - all scenarios",
        )
        print(f"Saved 3D distribution plot to {dist3d_all_path}")

    if want_pcap_plots:
        _maybe_generate_pcap_plots(
            out_stem=out_stem,
            dist_tag=dist_tag,
            scenario_specs=specs,
            scenario_run_ids=scenario_run_ids,
            max_runs=max_runs,
            pcap_parser=pcap_parser,
            scenario_order=scenario_order,
            scenario_colors=scenario_colors,
        )


def main() -> None:
    args = _parse_args()
    plot_comparison(args)


if __name__ == "__main__":
    main()
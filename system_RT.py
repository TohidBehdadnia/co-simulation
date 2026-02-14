# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 17:09:48 2025

@author: tbehdadn
"""

# -*- coding: utf-8 -*-
"""
lfc_comm_example.py

Three cases:

Case A: local controller (direct compute in this script).
Case B: remote controller over TCP, direct to controller_server.py.
Case C: remote controller over TCP, through noisy_proxy_with_wireshark.py.

Each case:
    - measure control delay to set t_shed
    - run TDS in two stages (0 -> t_shed, then t_shed -> t_final)
    - save a frequency plot with fixed y limits
    - save (t, freq) for first GENROU as .npz for replay
"""

import json
import socket
import argparse
from pathlib import Path
import time

import andes
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Folder of this script, used for saving plots and data
BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
RESULTS_DIR = BASE_DIR / "results"

# Case files
ieee14_raw = andes.get_case("ieee14/ieee14.raw")
ieee14_dyr = andes.get_case("ieee14/ieee14.dyr")

# Event times
t_trip = 1.0        # time of generator trip [s]
t_final = 10.0      # final simulation time [s]

# Candidate load shedding buses
shed_buses = [2, 3, 4, 5, 6, 9]

# Controller server addresses
CTRL_HOST = "127.0.0.1"
CTRL_PORT_DIRECT = 5001  # direct connection to controller_server.py
CTRL_PORT_PROXY = 5002   # connection through noisy proxy

# y limits for frequency plots
FREQ_Y_LIMS = (59.6, 60.1)


def compute_load_shedding(lost_p, loads):
    """
    lost_p : float, lost active power in pu
    loads  : list of dicts with keys: id, Ppf, bus

    Simple scheme: shed in proportion to Ppf.
    """
    p_vals = np.array([ld["Ppf"] for ld in loads], dtype=float)
    total_p = p_vals.sum()

    if total_p <= 0:
        per = lost_p / max(len(loads), 1)
        return {ld["id"]: per for ld in loads}

    shed = lost_p * p_vals / total_p

    result = {}
    for ld, s in zip(loads, shed):
        result[ld["id"]] = float(s)

    return result


def build_info_system():
    """
    Run a power flow to get:
    - lost generation from GENROU_2
    - candidate loads (PQ ids and Ppf values)
    """
    ss = andes.load(ieee14_raw, addfile=ieee14_dyr)

    ss.PFlow.run()

    lost_p = float(ss.PV.p0.v[0])

    pq_ids = ss.PQ.find_idx(keys="bus", values=shed_buses)
    pq_p = ss.PQ.get(src="Ppf", idx=pq_ids, attr="v")
    pq_bus = ss.PQ.get(src="bus", idx=pq_ids, attr="v")

    loads = []
    for dev, p, b in zip(pq_ids, pq_p, pq_bus):
        loads.append({"id": dev, "Ppf": float(p), "bus": int(b)})

    return lost_p, loads


def call_remote_controller(lost_p, loads, host, port, label):
    """
    Talk to the controller server over TCP at (host, port).
    Measure round trip time as communication plus computation delay.
    """
    payload = {
        "lost_p": float(lost_p),
        "loads": loads,
    }

    print(f"[{label}] connecting to controller at {host}:{port}")
    with socket.create_connection((host, port), timeout=10.0) as sock:
        f = sock.makefile("rw", encoding="utf-8")

        t0 = time.perf_counter()
        line_out = json.dumps(payload) + "\n"
        print(f"[{label}] sending: {line_out.strip()}")
        f.write(line_out)
        f.flush()

        resp_line = f.readline()
        t1 = time.perf_counter()

    if not resp_line:
        raise RuntimeError(f"[{label}] no response from controller server")

    print(f"[{label}] received: {resp_line.strip()}")
    resp = json.loads(resp_line)
    shed = {k: float(v) for k, v in resp["shed"].items()}
    delay = t1 - t0

    print(f"[{label}] round trip delay: {delay:.6f} s")
    return delay, shed


def build_system(t_shed, shed_dict):
    """
    Build an ANDES system for one case.

    t_shed   : time when load shedding is applied (s)
    shed_dict: dict PQ_id -> amount to shed (pu)
    """
    ss = andes.load(ieee14_raw, addfile=ieee14_dyr, setup=False)

    ss.add(
        "Toggle",
        dict(
            name="Trip_GENROU_2",
            model="SynGen",
            dev="GENROU_2",
            t=t_trip,
        ),
    )

    ss.setup()

    if len(ss.Toggle.u.v) >= 2:
        ss.Toggle.u.v[[0, 1]] = 0

    ss.PQ.config.p2p = 1
    ss.PQ.config.q2q = 1
    ss.PQ.config.p2z = 0
    ss.PQ.config.q2z = 0
    ss.PQ.pq2z = 0

    ss.PFlow.run()

    pq_ids = list(shed_dict.keys())
    pq_p_before = ss.PQ.get(src="Ppf", idx=pq_ids, attr="v")

    pq_p_after = []
    for dev, p0 in zip(pq_ids, pq_p_before):
        p_new = float(p0) - shed_dict[dev]
        pq_p_after.append(p_new)

    pq_p_after = np.array(pq_p_after)

    ss._pq_ids_to_shed = pq_ids
    ss._pq_p_after = pq_p_after
    ss._t_shed = t_shed

    return ss


def run_two_stage_tds(ss):
    """
    Run TDS in two stages:
      1) from 0 to t_shed with no shedding
      2) at t_shed, change Ppf, then continue to t_final
    """
    t_shed = ss._t_shed
    pq_ids = ss._pq_ids_to_shed
    pq_p_after = ss._pq_p_after

    # Stage 1
    ss.TDS.config.criteria = 0
    ss.TDS.config.tf = t_shed
    ss.TDS.run()

    # apply shedding
    ss.PQ.set(src="Ppf", idx=pq_ids, attr="v", value=pq_p_after)

    # Stage 2
    ss.TDS.config.tf = t_final
    ss.TDS.run()


def extract_time_and_freq(ss):
    """
    Extract time vector and frequency (Hz) of first GENROU from dae time series.
    """
    ts = ss.dae.ts.t
    omega_mat = ss.dae.ts.x[:, ss.GENROU.omega.a]   # shape (N, n_gen)
    freq_mat = 60.0 * omega_mat
    freq_first = freq_mat[:, 0]                     # first generator
    return ts, freq_first


def plot_frequency_from_arrays(t, freq, title, filename, y_limits=None, last_cmd_time=None):
    """
    Plot frequency (Hz) from arrays and save to file.
    """
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    sns.lineplot(x=t, y=freq, ax=ax, label="Frequency")
    if last_cmd_time is not None:
        t_last = float(last_cmd_time)
        ax.axvline(
            t_last,
            color="r",
            linestyle="--",
            label=f"Last command arrival: t={t_last:.3f} s",
        )

    # Trip event marker (keep last so it appears last in the legend)
    ax.axvline(
        float(t_trip),
        color="k",
        linestyle=":",
        linewidth=1.8,
        alpha=0.95,
        zorder=10,
        label=f"Gen 2 trip time: {float(t_trip):.1f} s",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(title)

    if y_limits is not None:
        ax.set_ylim(y_limits[0], y_limits[1])

    ax.legend(loc="best")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    # Always store plots as PDF
    out_name = Path(str(filename)).name
    out_name = str(Path(out_name).with_suffix(".pdf"))
    out_path = PLOTS_DIR / out_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {out_path}")

    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single TCP JSON scenario multiple times")
    parser.add_argument(
        "--scenario",
        choices=["local", "direct", "proxy"],
        default="local",
        help="Scenario to run (default: local).",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of repeated runs for the selected scenario (default: 1).",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=1,
        help="Starting numeric id used in output filenames (default: 1).",
    )
    return parser.parse_args()


def _run_single_scenario(scenario: str, lost_p: float, loads: list[dict]):
    if scenario == "local":
        t0_local = time.perf_counter()
        shed = compute_load_shedding(lost_p, loads)
        t1_local = time.perf_counter()
        delay = t1_local - t0_local
        title = "Frequency - local controller (no TCP)"
    elif scenario == "direct":
        delay, shed = call_remote_controller(
            lost_p, loads, CTRL_HOST, CTRL_PORT_DIRECT, "direct"
        )
        title = "Frequency - remote controller (direct TCP)"
    else:
        delay, shed = call_remote_controller(
            lost_p, loads, CTRL_HOST, CTRL_PORT_PROXY, "proxy"
        )
        title = "Frequency - remote controller (TCP via proxy)"

    t_shed = t_trip + delay
    ss = build_system(t_shed=t_shed, shed_dict=shed)
    run_two_stage_tds(ss)
    t_arr, f_arr = extract_time_and_freq(ss)
    return t_arr, f_arr, float(delay), float(t_shed), title


def _save_run_outputs(scenario: str, run_id: int, t_arr, f_arr, measured_delay: float, t_shed: float, title: str) -> None:
    scenario_name = f"case_{scenario}"
    scenario_dir = RESULTS_DIR / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    npz_name = f"{scenario_name}_{int(run_id)}.npz"
    np.savez(scenario_dir / npz_name, t=t_arr, freq=f_arr, measured_delay=float(measured_delay))

    plot_frequency_from_arrays(
        t_arr,
        f_arr,
        title,
        f"freq_{scenario}_{int(run_id)}.pdf",
        y_limits=FREQ_Y_LIMS,
        last_cmd_time=float(t_shed),
    )
    print(f"Saved run {run_id} to {scenario_dir / npz_name}")


def main():
    andes.config_logger(stream_level=20)
    args = _parse_args()
    if args.num_runs <= 0:
        raise ValueError("--num-runs must be > 0")

    lost_p, loads = build_info_system()
    print(f"[main] scenario={args.scenario} num_runs={args.num_runs} start_id={args.start_id}")
    print(f"[main] lost generation from GENROU_2: {lost_p:.3f} pu")

    for run_id in range(int(args.start_id), int(args.start_id) + int(args.num_runs)):
        print(f"[main] starting run_id={run_id}")
        t_arr, f_arr, delay, t_shed, title = _run_single_scenario(args.scenario, lost_p, loads)
        _save_run_outputs(args.scenario, run_id, t_arr, f_arr, delay, t_shed, title)


if __name__ == "__main__":
    main()

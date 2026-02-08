# -*- coding: utf-8 -*-
"""system_RT_iec104.py

Run the same load-shedding delay analysis as system_RT.py, but using IEC 60870-5-104
communication via the `c104` Python library.

Roles:
- This script acts as the IEC-104 outstation/server.
- A separate script (controller_iec104_master.py) acts as the IEC-104 master/client.

Protocol mapping (simple lab mapping):
- Common address (CA): 1
- IOA 9000: lost generation active power `lost_p` (M_ME_NC_1)
- IOA 10000 + bus: load active power `Ppf` for shedding-candidate bus (M_ME_NC_1)
- IOA 20000 + bus: shed command amount (delta P to subtract) for that bus (C_SE_NC_1)

The delay is measured from the time the outstation transmits the measurements to the
moment it has received all shed commands.
"""

import argparse
import os
import threading
import time
from pathlib import Path

import andes
import c104
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"

# Case files
ieee14_raw = andes.get_case("ieee14/ieee14.raw")
ieee14_dyr = andes.get_case("ieee14/ieee14.dyr")

# Event times
t_trip = 1.0
t_final = 10.0

# Candidate load shedding buses
shed_buses = [2, 3, 4, 5, 6, 9]

# IEC-104 server configuration
IEC104_IP = "127.0.0.1"
IEC104_PORT = 2404
IEC104_CA = 1


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _clamp_tick_ms(value: int) -> int:
    # c104 enforces tickRate_ms >= 50
    return max(int(value), 50)

# IOA mapping
IOA_START = 8999
IOA_LOST_P = 9000
IOA_PPF_BASE = 10000
IOA_SHED_BASE = 20000

# y limits for frequency plots
FREQ_Y_LIMS = (59.6, 60.1)


def build_info_system():
    """Run a power flow and extract lost_p and candidate PQ loads.

    Returns
    -------
    lost_p : float
        Lost active power in pu.
    loads : list[dict]
        Each dict has keys: id (PQ device id), Ppf (float), bus (int)
    """
    ss = andes.load(ieee14_raw, addfile=ieee14_dyr)
    ss.PFlow.run()

    lost_p = float(ss.PV.p0.v[0])

    pq_ids = ss.PQ.find_idx(keys="bus", values=shed_buses)
    pq_p = ss.PQ.get(src="Ppf", idx=pq_ids, attr="v")
    pq_bus = ss.PQ.get(src="bus", idx=pq_ids, attr="v")

    loads: list[dict] = []
    for dev, p, b in zip(pq_ids, pq_p, pq_bus):
        loads.append({"id": dev, "Ppf": float(p), "bus": int(b)})

    return lost_p, loads


def build_system(t_shed: float, shed_dict: dict) -> andes.system.System:
    """Build an ANDES system and schedule shedding at t_shed by adjusting PQ.Ppf."""
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
        pq_p_after.append(float(p0) - float(shed_dict[dev]))

    ss._pq_ids_to_shed = pq_ids
    ss._pq_p_after = np.array(pq_p_after, dtype=float)
    ss._t_shed = float(t_shed)

    return ss


def build_base_system() -> andes.system.System:
    """Build an ANDES system (trip scheduled) without any shedding applied yet."""
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
    return ss


def run_multi_stage_tds(ss: andes.system.System, events: list[tuple[float, dict]]) -> None:
    """Run TDS in multiple stages applying PQ.Ppf updates at specified times.

    Parameters
    ----------
    ss : andes.system.System
        Initialized ANDES system.
    events : list of (t_event, updates)
        t_event is the absolute simulation time in seconds.
        updates is a dict {pq_id: new_Ppf_value} to set at t_event.
    """
    ss.TDS.config.criteria = 0

    # Ensure chronological order and clamp event times.
    events_sorted = sorted([(float(t), upd) for t, upd in events], key=lambda x: x[0])
    t_cur = 0.0
    for t_evt, updates in events_sorted:
        # Skip no-op or backwards events.
        if t_evt <= t_cur + 1e-9:
            # Apply immediately at current state.
            if updates:
                pq_ids = list(updates.keys())
                pq_vals = [float(updates[i]) for i in pq_ids]
                ss.PQ.set(src="Ppf", idx=pq_ids, attr="v", value=np.array(pq_vals, dtype=float))
            continue

        ss.TDS.config.tf = float(t_evt)
        ss.TDS.run()
        t_cur = float(t_evt)

        if updates:
            pq_ids = list(updates.keys())
            pq_vals = [float(updates[i]) for i in pq_ids]
            ss.PQ.set(src="Ppf", idx=pq_ids, attr="v", value=np.array(pq_vals, dtype=float))

    ss.TDS.config.tf = t_final
    ss.TDS.run()


def run_two_stage_tds(ss: andes.system.System) -> None:
    t_shed = ss._t_shed
    pq_ids = ss._pq_ids_to_shed
    pq_p_after = ss._pq_p_after

    ss.TDS.config.criteria = 0
    ss.TDS.config.tf = t_shed
    ss.TDS.run()

    ss.PQ.set(src="Ppf", idx=pq_ids, attr="v", value=pq_p_after)

    ss.TDS.config.tf = t_final
    ss.TDS.run()


def extract_time_and_freq(ss: andes.system.System):
    ts = ss.dae.ts.t
    omega_mat = ss.dae.ts.x[:, ss.GENROU.omega.a]
    freq_mat = 60.0 * omega_mat
    return ts, freq_mat[:, 0]


def plot_frequency(t, freq, title: str, filename: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(t, freq)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(title)
    ax.set_ylim(FREQ_Y_LIMS[0], FREQ_Y_LIMS[1])

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def _delay_tag(seconds: float) -> str:
    s = f"{seconds:.3f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IEC-104 outstation + ANDES simulation")
    parser.add_argument(
        "--scenario",
        choices=["direct", "mitm"],
        default=os.environ.get("IEC104_SCENARIO", "direct"),
        help="Label outputs as direct IEC-104 or MITM-proxied IEC-104 (default: direct).",
    )
    parser.add_argument(
        "--induced-delay",
        type=float,
        default=(float(os.environ["IEC104_INDUCED_DELAY"]) if "IEC104_INDUCED_DELAY" in os.environ else None),
        help="Induced delay configured in the MITM (seconds). Required when --scenario mitm.",
    )
    parser.add_argument(
        "--wait-start",
        action=argparse.BooleanOptionalAction,
        default=(os.environ.get("IEC104_WAIT_START", "1") not in {"0", "false", "False", "no", "NO"}),
        help="Wait for a start command from the master before transmitting measurements (default: enabled).",
    )
    parser.add_argument(
        "--wait-start-timeout",
        type=float,
        default=float(os.environ.get("IEC104_WAIT_START_TIMEOUT", "0")),
        help="Max seconds to wait for the start command (0 = wait forever).",
    )
    parser.add_argument(
        "--tick-ms",
        type=int,
        default=_clamp_tick_ms(_env_int("IEC104_TICK_MS", 50)),
        help="IEC-104 server tick rate in milliseconds (min 50ms in c104).",
    )
    parser.add_argument(
        "--shed-mode",
        choices=["all_at_once", "per_command"],
        default=os.environ.get("IEC104_SHED_MODE", "all_at_once"),
        help=(
            "How to apply shedding in the ANDES simulation. "
            "all_at_once: apply all shed deltas at t_trip + delay(last command). "
            "per_command: apply each bus's shed delta at its own command-arrival time."
        ),
    )
    return parser.parse_args()


def main():
    andes.config_logger(stream_level=20)

    args = _parse_args()
    if args.scenario == "mitm" and args.induced_delay is None:
        raise ValueError("For --scenario mitm you must pass --induced-delay (or set IEC104_INDUCED_DELAY).")

    if args.scenario == "mitm":
        print(
            "[iec104-system] NOTE: --scenario mitm only affects output labeling. "
            "The real outstation still listens on 2404; to actually run through the MITM, "
            "start the proxy on 2405 and point the controller to port 2405.",
            flush=True,
        )

    lost_p, loads = build_info_system()
    print(f"[iec104-system] lost_p: {lost_p:.6f} pu")
    print("[iec104-system] candidate loads:")
    for ld in loads:
        print(f"  bus={ld['bus']:>2}  PQ={ld['id']}  Ppf={ld['Ppf']:.6f} pu")

    expected_buses = sorted({int(ld["bus"]) for ld in loads})
    pq_by_bus = {int(ld["bus"]): ld["id"] for ld in loads}

    received_shed_by_bus: dict[int, float] = {}
    received_shed_time_by_bus: dict[int, float] = {}
    all_received = threading.Event()
    start_received = threading.Event()

    # Set once the measurement snapshot is successfully transmitted.
    tx_t0: dict[str, float | None] = {"t0": None}

    def on_shed_command(point: c104.Point, previous_info: c104.Information, message: c104.IncomingMessage) -> c104.ResponseState:
        bus = int(point.io_address - IOA_SHED_BASE)
        t_rx = time.perf_counter()
        try:
            # Prefer the incoming message payload (more reliable than point.value timing).
            received_shed_by_bus[bus] = float(message.info.value)
        except Exception:
            try:
                received_shed_by_bus[bus] = float(point.value)
            except Exception:
                return c104.ResponseState.FAILURE

        received_shed_time_by_bus[bus] = float(t_rx)

        if all(bus in received_shed_by_bus for bus in expected_buses):
            all_received.set()

        return c104.ResponseState.SUCCESS

    def on_start_command(point: c104.Point, previous_info: c104.Information, message: c104.IncomingMessage) -> c104.ResponseState:
        start_received.set()
        return c104.ResponseState.SUCCESS

    # Start outstation
    server = c104.Server(ip=IEC104_IP, port=IEC104_PORT, tick_rate_ms=_clamp_tick_ms(int(args.tick_ms)))
    station = server.add_station(common_address=IEC104_CA)
    if station is None:
        raise RuntimeError("Failed to add IEC-104 station")

    pt_start = station.add_point(io_address=IOA_START, type=c104.Type.C_SC_NA_1)
    if pt_start is None:
        raise RuntimeError("Failed to add start command point")
    pt_start.on_receive(callable=on_start_command)

    pt_lost_p = station.add_point(io_address=IOA_LOST_P, type=c104.Type.M_ME_NC_1)
    if pt_lost_p is None:
        raise RuntimeError("Failed to add lost_p point")

    ppf_points: dict[int, c104.Point] = {}
    cmd_points: dict[int, c104.Point] = {}

    for bus in expected_buses:
        ppf_pt = station.add_point(io_address=IOA_PPF_BASE + bus, type=c104.Type.M_ME_NC_1)
        cmd_pt = station.add_point(io_address=IOA_SHED_BASE + bus, type=c104.Type.C_SE_NC_1)
        if ppf_pt is None or cmd_pt is None:
            raise RuntimeError(f"Failed to add IEC-104 points for bus {bus}")
        cmd_pt.on_receive(callable=on_shed_command)
        ppf_points[bus] = ppf_pt
        cmd_points[bus] = cmd_pt

    server.start()
    print(f"[iec104-system] outstation listening on {IEC104_IP}:{IEC104_PORT} (CA={IEC104_CA})")

    # Optional but helps some masters: signal station initialized.
    try:
        station.signal_initialized(cause=c104.Coi.LOCAL_POWER_ON)
    except Exception:
        pass

    if args.wait_start:
        if args.wait_start_timeout and args.wait_start_timeout > 0:
            print(f"[iec104-system] waiting for start command (timeout={args.wait_start_timeout:.1f}s)...")
            ok = start_received.wait(timeout=float(args.wait_start_timeout))
            if not ok:
                print("[iec104-system] start command timeout; proceeding anyway")
        else:
            print("[iec104-system] waiting for start command...")
            start_received.wait()

    # Publish measurements and start delay timer.
    # Note: depending on link startup timing, has_active_connections can lag behind.
    # We instead retry transmit until it returns True (i.e., delivered to a connected master).
    pt_lost_p.value = float(lost_p)
    for ld in loads:
        bus = int(ld["bus"])
        ppf_points[bus].value = float(ld["Ppf"])

    # Prefer batch transmit to deliver a consistent snapshot.
    batch = c104.Batch(cause=c104.Cot.SPONTANEOUS, points=[pt_lost_p, *[ppf_points[b] for b in expected_buses]])

    t_tx_wait0 = time.perf_counter()
    t0 = None
    while True:
        t_try = time.perf_counter()
        ok = server.transmit_batch(batch)
        if ok:
            t0 = t_try
            tx_t0["t0"] = float(t0)
            break
        if t_try - t_tx_wait0 > 20.0:
            server.stop()
            raise TimeoutError("Failed to transmit measurements (no master?) within 20s")
        time.sleep(0.01)

    print("[iec104-system] measurements transmitted, waiting for shed commands...")

    if not all_received.wait(timeout=20.0):
        # In per_command mode we can still proceed with partial commands.
        if args.shed_mode == "all_at_once":
            server.stop()
            raise TimeoutError("Did not receive all shed commands within 20s")
        print(
            f"[iec104-system] WARNING: not all shed commands received within timeout; proceeding with {len(received_shed_by_bus)}/{len(expected_buses)} commands",
            flush=True,
        )

    if not received_shed_by_bus:
        server.stop()
        raise TimeoutError("Did not receive any shed commands")

    # Stop comms before running the offline simulation.
    server.stop()

    if args.shed_mode == "all_at_once":
        t1 = time.perf_counter()
        delay = t1 - float(t0)
        t_shed = t_trip + delay

        shed_dict: dict = {}
        for bus in expected_buses:
            pq_id = pq_by_bus[bus]
            shed_dict[pq_id] = float(received_shed_by_bus[bus])

        print(f"[iec104-system] received all commands; delay={delay:.6f} s; t_shed={t_shed:.6f} s")

        ss = build_system(t_shed=t_shed, shed_dict=shed_dict)
        run_two_stage_tds(ss)
        t_arr, f_arr = extract_time_and_freq(ss)
        measured_delay = float(delay)
        per_bus_delays = None

    else:
        # Apply each command at its own arrival time (relative to measurement transmit time).
        t0_tx = tx_t0["t0"]
        if t0_tx is None:
            raise RuntimeError("Internal error: transmit t0 not set")

        per_bus_delays_map: dict[int, float] = {}
        events: list[tuple[float, dict]] = []

        # Build a base system and compute baseline Ppf for all candidate PQs.
        ss = build_base_system()

        pq_ids_all = [pq_by_bus[b] for b in expected_buses]
        ppf_base = ss.PQ.get(src="Ppf", idx=pq_ids_all, attr="v")
        ppf_base_by_pq = {pq_id: float(p) for pq_id, p in zip(pq_ids_all, ppf_base)}

        for bus, shed_amt in received_shed_by_bus.items():
            if bus not in pq_by_bus:
                continue
            if bus not in received_shed_time_by_bus:
                continue
            t_rx = float(received_shed_time_by_bus[bus])
            d = max(0.0, t_rx - float(t0_tx))
            per_bus_delays_map[int(bus)] = float(d)
            t_evt = float(t_trip + d)

            pq_id = pq_by_bus[int(bus)]
            new_ppf = float(ppf_base_by_pq[pq_id]) - float(shed_amt)

            # Keep events separate per command to make individual steps visible.
            events.append((float(t_evt), {pq_id: float(new_ppf)}))

        # If two commands are truly simultaneous, merge them to avoid zero-length stages.
        events_sorted = sorted(events, key=lambda x: x[0])
        merged: list[tuple[float, dict]] = []
        eps = 1e-6
        for t_evt, upd in events_sorted:
            if not merged:
                merged.append((float(t_evt), dict(upd)))
                continue
            t_last, upd_last = merged[-1]
            if abs(float(t_evt) - float(t_last)) <= eps:
                upd_last.update(upd)
                merged[-1] = (t_last, upd_last)
            else:
                merged.append((float(t_evt), dict(upd)))
        events = merged

        measured_delay = max(per_bus_delays_map.values()) if per_bus_delays_map else 0.0
        per_bus_delays = np.array([per_bus_delays_map.get(int(b), np.nan) for b in expected_buses], dtype=float)

        print(
            f"[iec104-system] per_command shedding events: n={len(events)} last_delay={measured_delay:.6f} s",
            flush=True,
        )

        # Log the per-command schedule (sorted by event time).
        try:
            schedule_rows = []
            for bus in sorted(per_bus_delays_map.keys()):
                d = float(per_bus_delays_map[bus])
                schedule_rows.append((float(t_trip + d), int(bus), d))
            for t_evt, bus, d in sorted(schedule_rows, key=lambda x: x[0]):
                print(f"[iec104-system]   bus={bus:>2} delay={d:.6f} s -> t_shed={t_evt:.6f} s", flush=True)
        except Exception:
            pass

        run_multi_stage_tds(ss, events)
        t_arr, f_arr = extract_time_and_freq(ss)

    shed_tag = "percmd" if args.shed_mode == "per_command" else "all"

    if args.scenario == "direct":
        title = "Frequency - IEC-104 (direct)"
        png_name = f"freq_iec104_direct_{shed_tag}.png"
        npz_name = f"case_iec104_direct_{shed_tag}.npz"
    else:
        induced = float(args.induced_delay)
        title = f"Frequency - IEC-104 (MITM, induced delay={induced:.3f} s)"
        tag = _delay_tag(induced)
        png_name = f"freq_iec104_mitm_delay{tag}_{shed_tag}.png"
        npz_name = f"case_iec104_mitm_delay{tag}_{shed_tag}.npz"

    if args.shed_mode == "per_command":
        title = title + " [per-command shedding]"

    plot_frequency(t_arr, f_arr, title, png_name)
    if per_bus_delays is None:
        np.savez(BASE_DIR / npz_name, t=t_arr, freq=f_arr, measured_delay=float(measured_delay))
    else:
        np.savez(
            BASE_DIR / npz_name,
            t=t_arr,
            freq=f_arr,
            measured_delay=float(measured_delay),
            buses=np.array(expected_buses, dtype=int),
            per_bus_delay=per_bus_delays,
        )


if __name__ == "__main__":
    main()

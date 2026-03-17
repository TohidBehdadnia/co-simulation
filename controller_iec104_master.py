# -*- coding: utf-8 -*-
"""controller_iec104_master.py

IEC 60870-5-104 master/client controller that mirrors controller_RT.py logic.

It connects to the outstation started by system_RT_iec104.py, receives:
- lost_p (IOA 9000)
- Ppf values for candidate buses (IOA 10000 + bus)

Then it computes proportional shedding and sends shed deltas back as setpoints:
- shed_delta (IOA 20000 + bus)

This is intended for local lab analysis (delay/jitter experiments, MITM on TCP, etc.).
"""

import argparse
import os
import threading
import time
import numpy as np
import c104

DEFAULT_DIRECT_PORT = 2404
DEFAULT_MITM_PORT = 2405

shed_buses = [2, 3, 4, 5, 6, 9]

IOA_LOST_P = 9000
IOA_PPF_BASE = 10000
IOA_SHED_BASE = 20000
IOA_START = 8999


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _clamp_tick_ms(value: int) -> int:
    # c104 enforces tickRate_ms >= 50
    return max(int(value), 50)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="IEC-104 master/controller")
    ap.add_argument(
        "--scenario",
        choices=["direct", "mitm"],
        default=os.environ.get("IEC104_SCENARIO", "direct"),
        help=(
            "Connection scenario. In 'mitm' mode the default port becomes 2405 "
            "(unless --port or IEC104_PORT is provided)."
        ),
    )
    ap.add_argument("--ip", default=os.environ.get("IEC104_IP", "127.0.0.1"))

    # If IEC104_PORT is explicitly set in the environment, honor it.
    if "IEC104_PORT" in os.environ:
        default_port = int(os.environ.get("IEC104_PORT", str(DEFAULT_DIRECT_PORT)))
    else:
        default_port = DEFAULT_MITM_PORT if ap.parse_known_args()[0].scenario == "mitm" else DEFAULT_DIRECT_PORT

    ap.add_argument("--port", type=int, default=default_port)
    ap.add_argument("--ca", type=int, default=_env_int("IEC104_CA", 1))
    ap.add_argument(
        "--tick-ms",
        type=int,
        default=_clamp_tick_ms(_env_int("IEC104_TICK_MS", 50)),
        help="IEC-104 client tick rate in milliseconds (min 50ms in c104).",
    )
    ap.add_argument(
        "--inter-command-sleep-ms",
        type=float,
        default=float(os.environ.get("IEC104_INTER_COMMAND_SLEEP_MS", "0")),
        help="Optional sleep between successive shed commands (ms). Use to make per-command effects visible.",
    )
    ap.add_argument(
        "--debug-timing",
        action="store_true",
        default=bool(int(os.environ.get("IEC104_DEBUG_TIMING", "0"))),
        help="Print controller timing diagnostics (perf_counter-based).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None if os.environ.get("IEC104_SEED") is None else int(os.environ.get("IEC104_SEED")),
        help="Optional random seed for reproducibility (e.g., to get the same bus shuffling across runs). If not set, the shuffling will be different across runs."
    )
    return ap.parse_args()


def compute_shed_by_bus(lost_p: float, ppf_by_bus: dict[int, float]) -> dict[int, float]:
    total = sum(float(v) for v in ppf_by_bus.values())
    if total <= 0.0:
        n = max(len(ppf_by_bus), 1)
        per = float(lost_p) / n
        return {bus: per for bus in ppf_by_bus}

    return {bus: float(lost_p) * float(ppf) / total for bus, ppf in ppf_by_bus.items()}


def run_controller_once(args: argparse.Namespace) -> None:

    # Rarely, the c104 link handshake can get stuck in OPEN_MUTED (TCP connected, but STARTDT
    # not completed). In that case no ASDUs flow and the outstation keeps waiting for the start
    # point. Empirically this can happen intermittently across runs (see logs where only TESTFR
    # act/con appear and STARTDT act/con is missing). To make campaigns robust, we retry the
    # entire client startup a few times if we observe a prolonged OPEN_MUTED with no progress.
    MAX_RETRIES = 3
    OPEN_MUTED_STUCK_S = 5.0

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            ok = _run_controller_once_inner(args, open_muted_stuck_s=float(OPEN_MUTED_STUCK_S))
            if ok:
                return
            last_exc = RuntimeError("controller did not reach OPEN / did not complete handshake")
        except Exception as exc:
            last_exc = exc

        if attempt < MAX_RETRIES:
            print(
                f"[iec104-controller] WARNING: retrying controller startup ({attempt}/{MAX_RETRIES})",
                flush=True,
            )
            time.sleep(0.5)

    # If we get here, all attempts failed.
    raise TimeoutError(f"IEC-104 controller failed after {MAX_RETRIES} attempts: {last_exc}")


def _run_controller_once_inner(args: argparse.Namespace, *, open_muted_stuck_s: float) -> bool:

    expected_buses = list(sorted(set(int(b) for b in shed_buses)))

    got_lost_p = None
    got_ppf_by_bus: dict[int, float] = {}
    got_lost_p_rx = False
    got_ppf_rx: set[int] = set()
    sent = False

    # perf_counter timestamps for the most recent callback per signal.
    lost_p_rx_t: float | None = None
    ppf_rx_t_by_bus: dict[int, float] = {}

    rx_event = threading.Event()
    open_event = threading.Event()

    last_conn_state: c104.ConnectionState | None = None
    open_muted_since: float | None = None

    tick_rate_ms = _clamp_tick_ms(int(args.tick_ms))

    client = c104.Client(tick_rate_ms=tick_rate_ms, command_timeout_ms=10000)
    # Avoid automatic initialization (e.g., implicit interrogation) because when running
    # through the MITM it can return default-zero values before the real outstation has
    # transmitted its snapshot.
    conn = client.add_connection(ip=args.ip, port=int(args.port), init=c104.Init.NONE)
    if conn is None:
        raise RuntimeError("Failed to add IEC-104 connection")

    def _raw_is_asdu(data: bytes, type_id: int) -> bool:
        try:
            s = str(c104.explain_bytes(apdu=data))
            # Observed format includes 'ASDU=13' etc.
            return f"ASDU={int(type_id)}" in s
        except Exception:
            return False

    raw_seen: dict[str, bool | int] = {"rx_meas": False, "tx_cmd": False, "rx_n": 0, "tx_n": 0}
    if bool(getattr(args, "debug_timing", False)):
        raw_max = 20
        try:
            raw_max = int(os.environ.get("IEC104_DEBUG_RAW_MAX", "20"))
        except Exception:
            raw_max = 20

        def _on_rx_raw(connection: c104.Connection, data: bytes) -> None:
            try:
                if int(raw_seen["rx_n"]) < int(raw_max):
                    raw_seen["rx_n"] = int(raw_seen["rx_n"]) + 1
                    try:
                        msg = str(c104.explain_bytes(apdu=data))
                    except Exception:
                        msg = "<unparsed>"
                    print(f"[iec104-controller][timing][raw] rx t={time.perf_counter():.6f} {msg}", flush=True)
            except Exception:
                pass

            if (not bool(raw_seen["rx_meas"])) and _raw_is_asdu(data, 13):
                raw_seen["rx_meas"] = True
                print(f"[iec104-controller][timing][raw] rx_asdu13 t={time.perf_counter():.6f}", flush=True)

        def _on_tx_raw(connection: c104.Connection, data: bytes) -> None:
            try:
                if int(raw_seen["tx_n"]) < int(raw_max):
                    raw_seen["tx_n"] = int(raw_seen["tx_n"]) + 1
                    try:
                        msg = str(c104.explain_bytes(apdu=data))
                    except Exception:
                        msg = "<unparsed>"
                    print(f"[iec104-controller][timing][raw] tx t={time.perf_counter():.6f} {msg}", flush=True)
            except Exception:
                pass

            if (not bool(raw_seen["tx_cmd"])) and _raw_is_asdu(data, 50):
                raw_seen["tx_cmd"] = True
                print(f"[iec104-controller][timing][raw] tx_asdu50 t={time.perf_counter():.6f}", flush=True)

        try:
            conn.on_receive_raw(callable=_on_rx_raw)
        except Exception as exc:
            print(f"[iec104-controller][timing][raw] WARNING: failed to register on_receive_raw: {exc}", flush=True)
        try:
            conn.on_send_raw(callable=_on_tx_raw)
        except Exception as exc:
            print(f"[iec104-controller][timing][raw] WARNING: failed to register on_send_raw: {exc}", flush=True)

    station = conn.add_station(common_address=int(args.ca))
    if station is None:
        raise RuntimeError("Failed to add IEC-104 station")

    # Monitoring points
    pt_lost_p = station.add_point(io_address=IOA_LOST_P, type=c104.Type.M_ME_NC_1)
    if pt_lost_p is None:
        raise RuntimeError("Failed to add lost_p monitoring point")

    # Start handshake point (lets the outstation know we're ready).
    pt_start = station.add_point(io_address=IOA_START, type=c104.Type.C_SC_NA_1)
    if pt_start is None:
        raise RuntimeError("Failed to add start command point")

    ppf_points = {}
    for bus in expected_buses:
        pt = station.add_point(io_address=IOA_PPF_BASE + bus, type=c104.Type.M_ME_NC_1)
        if pt is None:
            raise RuntimeError(f"Failed to add Ppf monitoring point for bus {bus}")
        ppf_points[bus] = pt

    # Command points
    cmd_points = {}
    for bus in expected_buses:
        pt = station.add_point(io_address=IOA_SHED_BASE + bus, type=c104.Type.C_SE_NC_1)
        if pt is None:
            raise RuntimeError(f"Failed to add shed command point for bus {bus}")
        cmd_points[bus] = pt

    accepted_monitor_cots = {
        c104.Cot.SPONTANEOUS,
        c104.Cot.INTERROGATED_BY_STATION,
        c104.Cot.INTERROGATED_BY_GROUP_1,
        c104.Cot.INTERROGATED_BY_GROUP_2,
        c104.Cot.INTERROGATED_BY_GROUP_3,
        c104.Cot.INTERROGATED_BY_GROUP_4,
        c104.Cot.INTERROGATED_BY_GROUP_5,
        c104.Cot.INTERROGATED_BY_GROUP_6,
        c104.Cot.INTERROGATED_BY_GROUP_7,
        c104.Cot.INTERROGATED_BY_GROUP_8,
        c104.Cot.INTERROGATED_BY_GROUP_9,
        c104.Cot.INTERROGATED_BY_GROUP_10,
        c104.Cot.INTERROGATED_BY_GROUP_11,
        c104.Cot.INTERROGATED_BY_GROUP_12,
        c104.Cot.INTERROGATED_BY_GROUP_13,
        c104.Cot.INTERROGATED_BY_GROUP_14,
        c104.Cot.INTERROGATED_BY_GROUP_15,
        c104.Cot.INTERROGATED_BY_GROUP_16,
        c104.Cot.PERIODIC,
        c104.Cot.BACKGROUND_SCAN,
    }

    def send_commands(lost_p: float, ppf_by_bus: dict[int, float]) -> None:
        np.random.seed(args.seed) # Set random seed for reproducibility of bus shuffling.
        nonlocal sent
        shed_by_bus = compute_shed_by_bus(lost_p, ppf_by_bus)

        inter_sleep_s = max(0.0, float(args.inter_command_sleep_ms)) / 1000.0

        print(f"[iec104-controller] lost_p={lost_p:.6f} pu", flush=True)
        for bus in expected_buses:
            print(f"  bus={bus:>2} Ppf={ppf_by_bus[bus]:.6f} -> shed={shed_by_bus[bus]:.6f} pu", flush=True)

        t0 = time.perf_counter()
        all_ok = True
        # Shuffle expected busses to modify the order of commands accross runs
        np.random.shuffle(expected_buses)
        for bus in expected_buses:
            cmd_points[bus].value = float(shed_by_bus[bus])
            ok = cmd_points[bus].transmit(c104.Cot.ACTIVATION)
            all_ok = all_ok and bool(ok)
            if inter_sleep_s > 0:
                time.sleep(inter_sleep_s)
        t1 = time.perf_counter()

        if bool(getattr(args, "debug_timing", False)):
            rx_ts = []
            if lost_p_rx_t is not None:
                rx_ts.append(float(lost_p_rx_t))
            rx_ts.extend(float(v) for v in ppf_rx_t_by_bus.values())
            t_last_meas = max(rx_ts) if rx_ts else None
            if t_last_meas is None:
                print(
                    f"[iec104-controller][timing] cmd_window={(t1 - t0):.6f}s (no measurement rx timestamps available)",
                    flush=True,
                )
            else:
                print(
                    "[iec104-controller][timing] "
                    f"t_last_meas_rx={t_last_meas:.6f} "
                    f"t_cmd_tx_start={t0:.6f} (dt={(t0 - t_last_meas):.6f}s) "
                    f"t_cmd_tx_end={t1:.6f} (dt={(t1 - t_last_meas):.6f}s) "
                    f"cmd_window={(t1 - t0):.6f}s",
                    flush=True,
                )

        sent = True
        print(f"[iec104-controller] commands sent in {(t1 - t0):.6f} s (ok={all_ok})", flush=True)

    def on_lost_p(point: c104.Point, previous_info: c104.Information, message: c104.IncomingMessage) -> c104.ResponseState:
        nonlocal got_lost_p, got_lost_p_rx, lost_p_rx_t
        if message.cot not in accepted_monitor_cots:
            return c104.ResponseState.SUCCESS
        lost_p_rx_t = float(time.perf_counter())
        try:
            got_lost_p = float(message.info.value)
        except Exception:
            try:
                got_lost_p = float(point.value)
            except Exception:
                return c104.ResponseState.FAILURE
        got_lost_p_rx = True
        rx_event.set()
        return c104.ResponseState.SUCCESS

    def make_on_ppf(bus: int):
        def _cb(point: c104.Point, previous_info: c104.Information, message: c104.IncomingMessage) -> c104.ResponseState:
            nonlocal got_ppf_by_bus, got_ppf_rx, ppf_rx_t_by_bus
            if message.cot not in accepted_monitor_cots:
                return c104.ResponseState.SUCCESS
            ppf_rx_t_by_bus[int(bus)] = float(time.perf_counter())
            try:
                got_ppf_by_bus[bus] = float(message.info.value)
            except Exception:
                try:
                    got_ppf_by_bus[bus] = float(point.value)
                except Exception:
                    return c104.ResponseState.FAILURE
            got_ppf_rx.add(bus)
            rx_event.set()
            return c104.ResponseState.SUCCESS

        return _cb

    pt_lost_p.on_receive(callable=on_lost_p)
    for bus, pt in ppf_points.items():
        pt.on_receive(callable=make_on_ppf(bus))

    def on_state_change(connection: c104.Connection, state: c104.ConnectionState) -> None:
        nonlocal last_conn_state
        nonlocal open_muted_since
        if state != last_conn_state:
            last_conn_state = state
            print(f"[iec104-controller] connection state={state}", flush=True)

        if state == c104.ConnectionState.OPEN_MUTED:
            if open_muted_since is None:
                open_muted_since = float(time.perf_counter())
        else:
            open_muted_since = None

        if state == c104.ConnectionState.OPEN:
            open_event.set()
            try:
                # Handshake: tell the outstation to begin the measurement->command timing window.
                pt_start.value = True
                pt_start.transmit(c104.Cot.ACTIVATION)
            except Exception:
                pass
            # With the start handshake enabled on the outstation, we rely on the outstation
            # to transmit a measurement snapshot after start.

    conn.on_state_change(callable=on_state_change)

    client.start()
    if args.scenario == "mitm" and int(args.port) == DEFAULT_DIRECT_PORT and "IEC104_PORT" not in os.environ:
        print(
            "[iec104-controller] WARNING: scenario=mitm but connecting to port 2404. "
            "Did you forget to start the proxy and use port 2405?",
            flush=True,
        )
    print(
        f"[iec104-controller] scenario={args.scenario} client started; target={args.ip}:{int(args.port)} (CA={int(args.ca)})",
        flush=True,
    )
    if bool(getattr(args, "debug_timing", False)):
        print(
            f"[iec104-controller][timing] tick_ms={int(tick_rate_ms)} inter_command_sleep_ms={float(args.inter_command_sleep_ms):.3f}",
            flush=True,
        )

    # Keep process alive long enough to receive and send.
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < 30.0:
        # If the link got stuck in OPEN_MUTED, restart the client so we can re-attempt STARTDT.
        if open_muted_since is not None and (time.perf_counter() - float(open_muted_since)) > float(open_muted_stuck_s):
            print(
                f"[iec104-controller] WARNING: stuck in OPEN_MUTED for >{open_muted_stuck_s:.1f}s; restarting client",
                flush=True,
            )
            break
        if open_event.is_set() and (not sent) and got_lost_p_rx and all(bus in got_ppf_rx for bus in expected_buses):
            if bool(getattr(args, "debug_timing", False)):
                rx_ts = []
                if lost_p_rx_t is not None:
                    rx_ts.append(float(lost_p_rx_t))
                rx_ts.extend(float(v) for v in ppf_rx_t_by_bus.values())
                t_last_meas = max(rx_ts) if rx_ts else None
                t_now = float(time.perf_counter())
                if t_last_meas is not None:
                    print(
                        "[iec104-controller][timing] "
                        f"decision_t={t_now:.6f} dt_since_last_meas={(t_now - t_last_meas):.6f}s",
                        flush=True,
                    )
            send_commands(float(got_lost_p), {b: float(got_ppf_by_bus[b]) for b in expected_buses})
            break
        if (not sent) and got_lost_p_rx and all(bus in got_ppf_rx for bus in expected_buses):
            if bool(getattr(args, "debug_timing", False)):
                rx_ts = []
                if lost_p_rx_t is not None:
                    rx_ts.append(float(lost_p_rx_t))
                rx_ts.extend(float(v) for v in ppf_rx_t_by_bus.values())
                t_last_meas = max(rx_ts) if rx_ts else None
                t_now = float(time.perf_counter())
                if t_last_meas is not None:
                    print(
                        "[iec104-controller][timing] "
                        f"decision_t={t_now:.6f} dt_since_last_meas={(t_now - t_last_meas):.6f}s (open_event not set)",
                        flush=True,
                    )
            send_commands(float(got_lost_p), {b: float(got_ppf_by_bus[b]) for b in expected_buses})
            break
        # Event-driven wait: avoids 50ms polling quantization.
        rx_event.wait(timeout=0.5)
        rx_event.clear()

    time.sleep(0.05)
    client.stop()

    return bool(sent)


def main():
    args = _parse_args()
    run_controller_once(args)


if __name__ == "__main__":
    main()

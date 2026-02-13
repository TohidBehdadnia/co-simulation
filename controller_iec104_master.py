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
    return ap.parse_args()


def compute_shed_by_bus(lost_p: float, ppf_by_bus: dict[int, float]) -> dict[int, float]:
    total = sum(float(v) for v in ppf_by_bus.values())
    if total <= 0.0:
        n = max(len(ppf_by_bus), 1)
        per = float(lost_p) / n
        return {bus: per for bus in ppf_by_bus}

    return {bus: float(lost_p) * float(ppf) / total for bus, ppf in ppf_by_bus.items()}


def main():
    args = _parse_args()

    expected_buses = list(sorted(set(int(b) for b in shed_buses)))

    got_lost_p = None
    got_ppf_by_bus: dict[int, float] = {}
    got_lost_p_rx = False
    got_ppf_rx: set[int] = set()
    sent = False

    rx_event = threading.Event()
    open_event = threading.Event()

    last_conn_state: c104.ConnectionState | None = None

    tick_rate_ms = _clamp_tick_ms(int(args.tick_ms))

    client = c104.Client(tick_rate_ms=tick_rate_ms, command_timeout_ms=10000)
    # Avoid automatic initialization (e.g., implicit interrogation) because when running
    # through the MITM it can return default-zero values before the real outstation has
    # transmitted its snapshot.
    conn = client.add_connection(ip=args.ip, port=int(args.port), init=c104.Init.NONE)
    if conn is None:
        raise RuntimeError("Failed to add IEC-104 connection")

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
        nonlocal sent
        shed_by_bus = compute_shed_by_bus(lost_p, ppf_by_bus)

        inter_sleep_s = max(0.0, float(args.inter_command_sleep_ms)) / 1000.0

        print(f"[iec104-controller] lost_p={lost_p:.6f} pu", flush=True)
        for bus in expected_buses:
            print(f"  bus={bus:>2} Ppf={ppf_by_bus[bus]:.6f} -> shed={shed_by_bus[bus]:.6f} pu", flush=True)

        t0 = time.perf_counter()
        all_ok = True
        for bus in expected_buses:
            cmd_points[bus].value = float(shed_by_bus[bus])
            ok = cmd_points[bus].transmit(c104.Cot.ACTIVATION)
            all_ok = all_ok and bool(ok)
            if inter_sleep_s > 0:
                time.sleep(inter_sleep_s)
        t1 = time.perf_counter()

        sent = True
        print(f"[iec104-controller] commands sent in {(t1 - t0):.6f} s (ok={all_ok})", flush=True)

    def on_lost_p(point: c104.Point, previous_info: c104.Information, message: c104.IncomingMessage) -> c104.ResponseState:
        nonlocal got_lost_p, got_lost_p_rx
        if message.cot not in accepted_monitor_cots:
            return c104.ResponseState.SUCCESS
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
            nonlocal got_ppf_by_bus, got_ppf_rx
            if message.cot not in accepted_monitor_cots:
                return c104.ResponseState.SUCCESS
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
        if state != last_conn_state:
            last_conn_state = state
            print(f"[iec104-controller] connection state={state}", flush=True)

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

    # Keep process alive long enough to receive and send.
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < 30.0:
        if open_event.is_set() and (not sent) and got_lost_p_rx and all(bus in got_ppf_rx for bus in expected_buses):
            send_commands(float(got_lost_p), {b: float(got_ppf_by_bus[b]) for b in expected_buses})
            break
        if (not sent) and got_lost_p_rx and all(bus in got_ppf_rx for bus in expected_buses):
            send_commands(float(got_lost_p), {b: float(got_ppf_by_bus[b]) for b in expected_buses})
            break
        # Event-driven wait: avoids 50ms polling quantization.
        rx_event.wait(timeout=0.5)
        rx_event.clear()

    time.sleep(0.05)
    client.stop()


if __name__ == "__main__":
    main()

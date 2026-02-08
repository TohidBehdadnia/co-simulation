# -*- coding: utf-8 -*-
"""mitm_proxy_iec104_delay.py

IEC 60870-5-104 Man-in-the-Middle proxy implemented using the `c104` library.

Design goal (lab experiments):
- Act as an IEC-104 outstation (server) towards the *real master/controller*.
- Act as an IEC-104 master (client) towards the *real outstation*.
- Forward monitoring traffic in both directions.
- Inject an additional delay ONLY for the shedding setpoint command packets.

This proxy is tailored to the IOA mapping used by system_RT_iec104.py:
- CA 1
- IOA 9000: lost_p measurement (M_ME_NC_1)
- IOA 10000+bus: Ppf measurement (M_ME_NC_1)
- IOA 20000+bus: shed delta command (C_SE_NC_1)

Run example (3 terminals):
- Outstation:  .venv/bin/python system_RT_iec104.py
- MITM:       .venv/bin/python mitm_proxy_iec104_delay.py --listen-port 2405 --upstream-port 2404 --shed-delay 0.20
- Controller: IEC104_PORT=2405 .venv/bin/python controller_iec104_master.py
"""

import argparse
import random
import threading
import time
import sys

import c104


IOA_LOST_P = 9000
IOA_PPF_BASE = 10000
IOA_SHED_BASE = 20000
IOA_START = 8999


def is_shed_command_ioa(ioa: int, shed_buses: list[int]) -> bool:
    if ioa < IOA_SHED_BASE:
        return False
    bus = ioa - IOA_SHED_BASE
    return int(bus) in set(int(b) for b in shed_buses)


def main() -> None:
    ap = argparse.ArgumentParser(description="IEC-104 MITM proxy (c104) delaying only shed commands")
    ap.add_argument("--listen-host", default="127.0.0.1")
    ap.add_argument("--listen-port", type=int, default=2405, help="where the real controller/master connects")
    ap.add_argument("--upstream-host", default="127.0.0.1")
    ap.add_argument("--upstream-port", type=int, default=2404, help="real outstation server port")
    ap.add_argument("--ca", type=int, default=1, help="station common address")
    ap.add_argument("--shed-delay", type=float, default=0.0, help="additional delay (seconds) applied ONLY to shed commands")
    ap.add_argument(
        "--shed-delay-min",
        type=float,
        default=None,
        help="if set together with --shed-delay-max, apply a per-packet random delay U[min,max] to EACH shed command packet",
    )
    ap.add_argument(
        "--shed-delay-max",
        type=float,
        default=None,
        help="upper bound for per-packet random delay (seconds)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for repeatable per-packet delays (only used with --shed-delay-min/--shed-delay-max)",
    )
    ap.add_argument("--tick-ms", type=int, default=50, help="tick rate (ms), min 50ms in c104")
    ap.add_argument("--command-timeout-ms", type=int, default=10000)
    ap.add_argument("--shed-buses", default="2,3,4,5,6,9", help="comma-separated bus list for shed commands")
    ap.add_argument("--log-messages", action="store_true", default=True, help="log received/forwarded messages (default: enabled)")
    ap.add_argument("--no-log-messages", action="store_false", dest="log_messages", help="disable message logging")
    args = ap.parse_args()

    shed_buses = [int(x.strip()) for x in args.shed_buses.split(",") if x.strip()]
    ca = int(args.ca)

    use_random_per_packet = (args.shed_delay_min is not None) and (args.shed_delay_max is not None)
    rng = random.Random(args.seed) if use_random_per_packet else None

    t0 = time.monotonic()

    try:
        # Ensure logs show up immediately when redirected (e.g. via nohup).
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    def log(msg: str) -> None:
        if not args.log_messages:
            return
        dt = time.monotonic() - t0
        print(f"[iec104-mitm +{dt:9.3f}s] {msg}", flush=True)

    print(
        f"[iec104-mitm] listen {args.listen_host}:{args.listen_port} (as outstation) | "
        f"upstream {args.upstream_host}:{args.upstream_port} (as master) | CA={ca} | shed_delay={args.shed_delay}s"
        + (
            f" | per_packet_delay=U[{float(args.shed_delay_min):.3f},{float(args.shed_delay_max):.3f}]s seed={args.seed}"
            if use_random_per_packet
            else ""
        )
        ,
        flush=True,
    )

    # --------------------------
    # Upstream side (to real outstation)
    # --------------------------
    tick_ms = max(int(args.tick_ms), 50)

    upstream_client = c104.Client(tick_rate_ms=tick_ms, command_timeout_ms=args.command_timeout_ms)
    # Disable automatic init/interrogation upstream; the real outstation will transmit
    # the measurement snapshot after it receives the start command.
    upstream_con = upstream_client.add_connection(ip=args.upstream_host, port=args.upstream_port, init=c104.Init.NONE)
    if upstream_con is None:
        raise RuntimeError("Failed to create upstream IEC-104 connection")

    upstream_open = threading.Event()

    def on_upstream_state_change(connection: c104.Connection, state: c104.ConnectionState) -> None:
        if state == c104.ConnectionState.OPEN:
            upstream_open.set()
            log("upstream connection OPEN")
        else:
            upstream_open.clear()
            log(f"upstream connection state={state}")

    upstream_con.on_state_change(callable=on_upstream_state_change)

    upstream_station = upstream_con.add_station(common_address=ca)
    if upstream_station is None:
        raise RuntimeError("Failed to add upstream station")

    # Points on upstream (client-side)
    up_start = upstream_station.add_point(io_address=IOA_START, type=c104.Type.C_SC_NA_1)
    if up_start is None:
        raise RuntimeError("Failed to add upstream start point")

    up_lost_p = upstream_station.add_point(io_address=IOA_LOST_P, type=c104.Type.M_ME_NC_1)
    if up_lost_p is None:
        raise RuntimeError("Failed to add upstream lost_p point")

    up_ppf: dict[int, c104.Point] = {}
    up_shed_cmd: dict[int, c104.Point] = {}
    for bus in shed_buses:
        p = upstream_station.add_point(io_address=IOA_PPF_BASE + bus, type=c104.Type.M_ME_NC_1)
        c = upstream_station.add_point(io_address=IOA_SHED_BASE + bus, type=c104.Type.C_SE_NC_1)
        if p is None or c is None:
            raise RuntimeError(f"Failed to create upstream points for bus {bus}")
        up_ppf[bus] = p
        up_shed_cmd[bus] = c

    # --------------------------
    # Downstream side (to real controller/master)
    # --------------------------
    downstream_server = c104.Server(ip=args.listen_host, port=args.listen_port, tick_rate_ms=tick_ms)
    downstream_station = downstream_server.add_station(common_address=ca)
    if downstream_station is None:
        raise RuntimeError("Failed to add downstream station")

    dn_start = downstream_station.add_point(io_address=IOA_START, type=c104.Type.C_SC_NA_1)
    if dn_start is None:
        raise RuntimeError("Failed to add downstream start point")

    dn_lost_p = downstream_station.add_point(io_address=IOA_LOST_P, type=c104.Type.M_ME_NC_1)
    if dn_lost_p is None:
        raise RuntimeError("Failed to add downstream lost_p point")

    dn_ppf: dict[int, c104.Point] = {}
    dn_shed_cmd: dict[int, c104.Point] = {}
    for bus in shed_buses:
        p = downstream_station.add_point(io_address=IOA_PPF_BASE + bus, type=c104.Type.M_ME_NC_1)
        c = downstream_station.add_point(io_address=IOA_SHED_BASE + bus, type=c104.Type.C_SE_NC_1)
        if p is None or c is None:
            raise RuntimeError(f"Failed to create downstream points for bus {bus}")
        dn_ppf[bus] = p
        dn_shed_cmd[bus] = c

    # --------------------------
    # Monitoring value cache (for proper interrogation responses)
    # --------------------------
    mon_values: dict[int, float] = {}
    mon_ready: dict[int, threading.Event] = {}

    all_mon_ioas = [IOA_LOST_P, *[IOA_PPF_BASE + b for b in shed_buses]]
    for ioa in all_mon_ioas:
        mon_ready[int(ioa)] = threading.Event()

    accepted_monitor_cots = {
        c104.Cot.SPONTANEOUS,
        c104.Cot.PERIODIC,
        c104.Cot.BACKGROUND_SCAN,
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
    }

    def publish_downstream_measurement(ioa: int, value: float) -> None:
        ioa = int(ioa)
        val = float(value)
        mon_values[ioa] = val
        if ioa in mon_ready:
            mon_ready[ioa].set()

        log(f"RX upstream->MITM meas ioa={ioa} value={val}")

        if ioa == IOA_LOST_P:
            dn_lost_p.value = val
            dn_lost_p.transmit(c104.Cot.SPONTANEOUS)
            return

        if ioa >= IOA_PPF_BASE:
            bus = ioa - IOA_PPF_BASE
            pt = dn_ppf.get(int(bus))
            if pt is not None:
                pt.value = val
                pt.transmit(c104.Cot.SPONTANEOUS)

    # --------------------------
    # Delay ONLY shedding commands
    #   - Default: buffer + flush once after fixed --shed-delay (legacy behavior)
    #   - Optional: per-packet random delay U[min,max] for EACH shed command
    # --------------------------
    start_lock = threading.Lock()
    pending_start: bool | None = None
    start_retry_timer: threading.Timer | None = None

    def _try_forward_start() -> None:
        nonlocal pending_start, start_retry_timer
        with start_lock:
            start_retry_timer = None
            val = pending_start

        if val is None:
            return

        if not upstream_open.is_set():
            log("FWD start deferred: upstream not open")
            with start_lock:
                if start_retry_timer is None:
                    start_retry_timer = threading.Timer(0.1, _try_forward_start)
                    start_retry_timer.daemon = True
                    start_retry_timer.start()
            return

        try:
            up_start.value = bool(val)
            ok = up_start.transmit(c104.Cot.ACTIVATION)
        except Exception:
            ok = False

        log(f"FWD MITM->upstream start ioa={IOA_START} value={bool(val)} ok={bool(ok)}")
        if ok:
            with start_lock:
                if pending_start == val:
                    pending_start = None
            return

        with start_lock:
            if start_retry_timer is None:
                start_retry_timer = threading.Timer(0.1, _try_forward_start)
                start_retry_timer.daemon = True
                start_retry_timer.start()

    cmd_lock = threading.Lock()
    pending_shed: dict[int, float] = {}  # ioa -> value
    flush_timer: threading.Timer | None = None

    def _sample_shed_delay() -> float:
        if not use_random_per_packet:
            return max(0.0, float(args.shed_delay))
        lo = float(args.shed_delay_min)
        hi = float(args.shed_delay_max)
        if lo > hi:
            lo, hi = hi, lo
        if hi <= 0.0:
            return 0.0
        lo = max(0.0, lo)
        hi = max(0.0, hi)
        # rng is set when use_random_per_packet=True
        return float(rng.uniform(lo, hi))

    def _forward_one_shed(ioa: int, value: float, sampled_delay_s: float) -> None:
        ioa = int(ioa)
        value = float(value)
        sampled_delay_s = max(0.0, float(sampled_delay_s))

        def _attempt_send() -> None:
            if not upstream_open.is_set():
                log(f"FWD shed deferred (upstream not open) ioa={ioa}")
                t = threading.Timer(0.1, _attempt_send)
                t.daemon = True
                t.start()
                return

            bus = int(ioa - IOA_SHED_BASE)
            pt = up_shed_cmd.get(bus)
            if pt is None:
                return
            try:
                pt.value = float(value)
                ok = pt.transmit(c104.Cot.ACTIVATION)
            except Exception:
                ok = False

            if not ok:
                log(f"FWD shed failed; retrying ioa={ioa}")
                t = threading.Timer(0.1, _attempt_send)
                t.daemon = True
                t.start()
                return

            log(f"FWD MITM->upstream shed ioa={ioa} value={float(value)}")

        log(f"ARM per-packet shed timer ioa={ioa} delay={sampled_delay_s:.3f}s")
        t = threading.Timer(sampled_delay_s, _attempt_send)
        t.daemon = True
        t.start()

    def flush_shed_commands() -> None:
        nonlocal flush_timer
        with cmd_lock:
            to_send = dict(pending_shed)
            pending_shed.clear()
            flush_timer = None

        if not to_send:
            return

        if not upstream_open.is_set():
            log("FLUSH deferred: upstream not open")
            with cmd_lock:
                pending_shed.update(to_send)
                if flush_timer is None:
                    flush_timer = threading.Timer(0.1, flush_shed_commands)
                    flush_timer.daemon = True
                    flush_timer.start()
            return

        log(f"FLUSH MITM->upstream shed n={len(to_send)} (delay={float(args.shed_delay):.3f}s)")

        # Forward to real outstation (upstream) without further delay.
        for ioa, value in to_send.items():
            bus = int(ioa - IOA_SHED_BASE)
            pt = up_shed_cmd.get(bus)
            if pt is None:
                continue
            try:
                pt.value = float(value)
                ok = pt.transmit(c104.Cot.ACTIVATION)
                if not ok:
                    raise RuntimeError("transmit returned False")
            except Exception:
                # Re-queue and retry.
                with cmd_lock:
                    pending_shed[int(ioa)] = float(value)
                    if flush_timer is None:
                        flush_timer = threading.Timer(0.1, flush_shed_commands)
                        flush_timer.daemon = True
                        flush_timer.start()

    # --------------------------
    # Callbacks (do minimal work; enqueue)
    # --------------------------
    def on_upstream_meas(point: c104.Point, previous_info: c104.Information, message: c104.IncomingMessage) -> c104.ResponseState:
        if message.cot not in accepted_monitor_cots:
            return c104.ResponseState.SUCCESS
        try:
            value = float(message.info.value)
        except Exception:
            try:
                value = float(point.value)
            except Exception:
                return c104.ResponseState.SUCCESS

        try:
            publish_downstream_measurement(int(message.io_address), float(value))
        except Exception:
            pass
        return c104.ResponseState.SUCCESS

    up_lost_p.on_receive(callable=on_upstream_meas)
    for bus, pt in up_ppf.items():
        pt.on_receive(callable=on_upstream_meas)

    def on_downstream_command(point: c104.Point, previous_info: c104.Information, message: c104.IncomingMessage) -> c104.ResponseState:
        ioa = int(message.io_address)

        # Log receipt from downstream master.
        if args.log_messages:
            try:
                raw_val = message.info.value
            except Exception:
                raw_val = None
            log(f"RX downstream->MITM cmd ioa={ioa} cot={getattr(message, 'cot', None)} value={raw_val}")

        # Start handshake command: forward immediately.
        if ioa == IOA_START:
            try:
                val = bool(message.info.value)
            except Exception:
                try:
                    val = bool(point.value)
                except Exception:
                    return c104.ResponseState.FAILURE
            with start_lock:
                nonlocal pending_start
                pending_start = bool(val)
            _try_forward_start()
            # Ack downstream immediately; forwarding is retried until it succeeds.
            return c104.ResponseState.SUCCESS

        try:
            value = float(message.info.value)
        except Exception:
            try:
                value = float(point.value)
            except Exception:
                return c104.ResponseState.FAILURE

        if is_shed_command_ioa(ioa, shed_buses):
            if use_random_per_packet:
                # Independent delay for each shed command packet.
                with cmd_lock:
                    delay_s = _sample_shed_delay()
                _forward_one_shed(int(ioa), float(value), float(delay_s))
            else:
                # Legacy: buffer and flush once after fixed --shed-delay.
                nonlocal flush_timer
                with cmd_lock:
                    pending_shed[ioa] = float(value)
                    log(f"BUF shed ioa={ioa} value={float(value)} pending={len(pending_shed)}")
                    if flush_timer is None:
                        delay_s = max(0.0, float(args.shed_delay))
                        log(f"ARM flush timer in {delay_s:.3f}s")
                        if delay_s == 0.0:
                            flush_timer = threading.Timer(0.0, flush_shed_commands)
                        else:
                            flush_timer = threading.Timer(delay_s, flush_shed_commands)
                        flush_timer.daemon = True
                        flush_timer.start()

            # Ack immediately; the forwarding is delayed.
            return c104.ResponseState.SUCCESS

        # Non-shed commands: forward immediately (currently only shed points are modelled).
        bus = ioa - IOA_SHED_BASE
        pt = up_shed_cmd.get(int(bus))
        if pt is None:
            return c104.ResponseState.FAILURE
        try:
            pt.value = float(value)
            ok = pt.transmit(c104.Cot.ACTIVATION)
            return c104.ResponseState.SUCCESS if ok else c104.ResponseState.FAILURE
        except Exception:
            return c104.ResponseState.FAILURE

    dn_start.on_receive(callable=on_downstream_command)
    for bus, pt in dn_shed_cmd.items():
        pt.on_receive(callable=on_downstream_command)

    # Gate interrogation/read responses until we have seen upstream values.
    def make_on_before_read(ioa: int):
        def _cb(point: c104.Point) -> None:
            ev = mon_ready.get(int(ioa))
            if ev is not None:
                ev.wait(timeout=5.0)
            if int(ioa) in mon_values:
                try:
                    point.value = float(mon_values[int(ioa)])
                except Exception:
                    pass

        return _cb

    dn_lost_p.on_before_read(callable=make_on_before_read(IOA_LOST_P))
    for bus, p in dn_ppf.items():
        p.on_before_read(callable=make_on_before_read(IOA_PPF_BASE + int(bus)))

    # --------------------------
    # Start both ends
    # --------------------------
    downstream_server.start()
    try:
        downstream_station.signal_initialized(cause=c104.Coi.LOCAL_POWER_ON)
    except Exception:
        pass

    upstream_client.start()

    # Keep the process alive
    print("[iec104-mitm] running", flush=True)
    log("MITM started")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[iec104-mitm] stopping")
    finally:
        try:
            upstream_client.stop()
        except Exception:
            pass
        try:
            downstream_server.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()

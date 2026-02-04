# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 15:50:53 2025

@author: tbehdadn
"""

# -*- coding: utf-8 -*-
"""
advanced_mitm_proxy_all_modes.py

Proxy between LFC client (lfc_comm_example.py) and controller_server.py.

- Listens on 127.0.0.1:5002
- Connects to controllers on 127.0.0.1:5001 (main) and optional 5003 (alt)
- Expects one JSON request line and one JSON response line per connection

Modes:
    baseline          forward request and response as they are
    delay             add fixed delay on request and response
    slow              send response to client one byte at a time
    rate_limit        send response to client in small chunks
    noise             add random noise to shed values
    bias              apply gain and offset to shed values
    targeted          add noise only on selected PQ ids
    replay            sometimes reuse stored response
    freeze            after N live responses, keep sending stored one
    smart_drop        drop requests with large lost_p, no response
    conditional_slow  slow response only when lost_p is large
    dup               sometimes send duplicate response line
    multi_ctrl        pick main or alt controller at random

Use only in local lab tests.
"""

import socket
import json
import threading
import time
import random
import subprocess

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------

# Where the LFC client connects
LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 5002   # client connects here

# Main controller
CTRL_HOST = "127.0.0.1"
CTRL_PORT_MAIN = 5001     # controller_server.py listens here

# Alt controller for multi_ctrl mode (start a second controller there if needed)
CTRL_PORT_ALT = 5003

# Proxy mode
MODE = "delay"
# Allowed values:
# "baseline", "delay", "slow", "rate_limit",
# "noise", "bias", "targeted",
# "replay", "freeze", "smart_drop",
# "conditional_slow", "dup", "multi_ctrl"

# For "delay" mode
REQ_DELAY_SEC = 0.20
RESP_DELAY_SEC = 0.20

# For "slow" mode (response)
SLOW_BYTE_DELAY_SEC = 0.02  # sleep between bytes

# For "rate_limit" mode (response)
RATE_LIMIT_CHUNK = 64       # bytes per send
RATE_LIMIT_TICK = 0.01      # seconds between chunks

# For "noise" mode
NOISE_STD_FRAC = 0.05       # relative std dev

# For "bias" mode
BIAS_GAIN = 1.10            # 10 percent gain
BIAS_OFFSET = 0.0           # additive offset

# For "targeted" mode - only disturb these PQ ids
CRITICAL_IDS = {"PQ_5", "PQ_6"}   # PQ ids to disturb
TARGET_NOISE_STD_FRAC = 1     # relative std dev for targets

# For "replay" mode
REPLAY_PROB = 0.4           # probability to reuse stored response

# For "freeze" mode
FREEZE_AFTER_N = 0          # after this many live responses, freeze

# For "smart_drop" mode
LOST_P_DROP_THRESHOLD = 0.5

# For "conditional_slow" mode
COND_SLOW_LOST_P = 0.3

# For "dup" mode
DUP_PROB = 0.3

# For "multi_ctrl" mode
MULTI_CTRL_BAD_PROB = 0.5   # chance to use alt controller

# Wireshark path (adjust if needed)
WIRESHARK_PATH = r"C:\Program Files\Wireshark\Wireshark.exe"

# --------------------------------------------------------------------
# Global state for replay / freeze
# --------------------------------------------------------------------

_last_response_line = None
_last_response_lock = threading.Lock()

_freeze_line = None
_freeze_count = 0
_freeze_frozen = False
_freeze_lock = threading.Lock()

# --------------------------------------------------------------------


def log(msg):
    print(f"[proxy-{MODE}] {msg}")


# ---------- Wireshark helpers ------------------------------------------------


def detect_loopback_interface():
    """
    Ask Wireshark for the list of interfaces and return the index
    of the first one whose line contains 'loopback'.
    Fall back to "1" if nothing is found.
    """
    try:
        result = subprocess.run(
            [WIRESHARK_PATH, "-D"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        log("Wireshark not found at WIRESHARK_PATH, capture will not start.")
        return None
    except subprocess.CalledProcessError as e:
        log("Error running 'wireshark -D':")
        print(e.stdout)
        return None

    output = result.stdout
    log("Wireshark interfaces from 'wireshark -D':")
    print(output)

    chosen = None
    for line in output.splitlines():
        line_stripped = line.strip()
        if "loopback" in line_stripped.lower():
            parts = line_stripped.split(".", 1)
            if parts:
                idx = parts[0].strip()
                if idx.isdigit():
                    chosen = idx
                    break

    if chosen:
        log(f"selected loopback interface index: {chosen}")
        return chosen

    log("no interface with 'loopback' found, falling back to index 1")
    return "1"


def start_wireshark_capture():
    """
    Start Wireshark GUI with a capture filter on the proxy and controller ports.
    Returns the Popen object or None on failure.
    """
    iface = detect_loopback_interface()
    if iface is None:
        return None

    capture_filter = f"tcp port {LISTEN_PORT} or tcp port {CTRL_PORT_MAIN}"

    cmd = [
        WIRESHARK_PATH,
        "-i", iface,
        "-k",
        "-f", capture_filter,
    ]

    log("starting Wireshark capture:")
    print("  Executable:", WIRESHARK_PATH)
    print("  Interface index:", iface)
    print("  Filter:", capture_filter)

    try:
        proc = subprocess.Popen(cmd)
        log("Wireshark started for live capture.")
        return proc
    except FileNotFoundError:
        log("Wireshark executable not found at WIRESHARK_PATH.")
        return None
    except Exception as e:
        log(f"could not start Wireshark: {e}")
        return None


# ---------- mode helpers -----------------------------------------------------


def maybe_add_delay_on_request():
    if MODE == "delay" and REQ_DELAY_SEC > 0.0:
        log(f"sleep {REQ_DELAY_SEC:.3f} s before sending request")
        time.sleep(REQ_DELAY_SEC)


def maybe_add_delay_on_response():
    if MODE == "delay" and RESP_DELAY_SEC > 0.0:
        log(f"sleep {RESP_DELAY_SEC:.3f} s before sending response")
        time.sleep(RESP_DELAY_SEC)


def should_use_slow(req_obj):
    """
    Decide if slow sending should be used for this connection.
    """
    if MODE == "slow":
        return True
    if MODE == "conditional_slow":
        lost_p = float(req_obj.get("lost_p", 0.0))
        if lost_p > COND_SLOW_LOST_P:
            log(f"conditional_slow: lost_p={lost_p:.4f} above threshold, will slow")
            return True
    return False


def choose_upstream():
    """
    Pick upstream controller for this connection.
    """
    if MODE == "multi_ctrl":
        if random.random() < MULTI_CTRL_BAD_PROB:
            log(f"multi_ctrl: picking alt controller {CTRL_HOST}:{CTRL_PORT_ALT}")
            return CTRL_HOST, CTRL_PORT_ALT
        else:
            log(f"multi_ctrl: picking main controller {CTRL_HOST}:{CTRL_PORT_MAIN}")
            return CTRL_HOST, CTRL_PORT_MAIN

    return CTRL_HOST, CTRL_PORT_MAIN


def tamper_request(req_obj):
    """
    Request side tampering.
    smart_drop: do not forward if lost_p large.
    For other modes, pass through.
    Returns (should_forward, req_obj_or_none).
    """
    if MODE != "smart_drop":
        return True, req_obj

    lost_p = float(req_obj.get("lost_p", 0.0))
    log(f"smart_drop: lost_p={lost_p:.4f}")
    if lost_p > LOST_P_DROP_THRESHOLD:
        log("smart_drop: lost_p above threshold, dropping request")
        return False, None
    return True, req_obj


def tamper_response_obj(resp_obj):
    """
    Response side tampering on JSON object.

    noise     random noise on all shed values
    bias      gain and offset on all shed values
    targeted  add noise only on selected PQ ids
    other     pass through
    """
    if not isinstance(resp_obj, dict):
        return resp_obj

    shed = resp_obj.get("shed")
    if not isinstance(shed, dict):
        return resp_obj

    # Copy so we do not modify in place by accident
    new_shed = dict(shed)

    if MODE == "noise":
        for k, v in list(new_shed.items()):
            try:
                base = float(v)
            except (TypeError, ValueError):
                continue
            factor = 1.0 + random.gauss(0.0, NOISE_STD_FRAC)
            new_val = base * factor
            new_shed[k] = new_val
            log(f"noise: shed[{k}] {base:.4f} -> {new_val:.4f}")

    elif MODE == "bias":
        for k, v in list(new_shed.items()):
            try:
                base = float(v)
            except (TypeError, ValueError):
                continue
            new_val = base * BIAS_GAIN + BIAS_OFFSET
            new_shed[k] = new_val
            log(f"bias: shed[{k}] {base:.4f} -> {new_val:.4f}")

    elif MODE == "targeted":
        for k, v in list(new_shed.items()):
            if k not in CRITICAL_IDS:
                continue
            try:
                base = float(v)
            except (TypeError, ValueError):
                continue
            factor = 1.0 + random.gauss(0.0, TARGET_NOISE_STD_FRAC)
            new_val = base * factor
            new_shed[k] = new_val
            log(f"targeted: shed[{k}] {base:.4f} -> {new_val:.4f}")

    resp_obj["shed"] = new_shed
    resp_obj["_attack_mode"] = MODE
    return resp_obj


def apply_replay_before_live():
    """
    For replay mode, decide if we reuse stored response.
    Returns (use_replay, replay_line_or_none).
    """
    if MODE != "replay":
        return False, None

    with _last_response_lock:
        if _last_response_line is not None and random.random() < REPLAY_PROB:
            log("replay: using stored response instead of live one")
            return True, _last_response_line
    return False, None


def store_for_replay(resp_line_out):
    """
    Store response line for replay mode.
    """
    if MODE != "replay":
        return
    with _last_response_lock:
        globals()["_last_response_line"] = resp_line_out
        log("replay: stored live response for future use")


def apply_freeze_filter(resp_line_out):
    """
    For freeze mode, after FREEZE_AFTER_N live responses,
    always send stored one.
    """
    if MODE != "freeze":
        return resp_line_out

    global _freeze_line, _freeze_count, _freeze_frozen
    with _freeze_lock:
        if not _freeze_frozen:
            _freeze_count += 1
            if _freeze_line is None:
                _freeze_line = resp_line_out
            if _freeze_count > FREEZE_AFTER_N:
                _freeze_frozen = True
                log("freeze: now frozen, will reuse stored response from now on")
            return resp_line_out
        else:
            log("freeze: sending stored frozen response")
            return _freeze_line


def send_response_line_slow(sock, line):
    """
    Send response line to client slowly, one byte at a time, on the raw socket.
    """
    if not line.endswith("\n"):
        line = line + "\n"

    data = line.encode("utf-8")
    log(f"slow: sending {len(data)} bytes slowly")
    try:
        for i in range(len(data)):
            sock.sendall(data[i:i + 1])
            time.sleep(SLOW_BYTE_DELAY_SEC)
    except Exception as e:
        log(f"slow: error while sending bytes: {e}")


def send_response_line_rate_limited(sock, line):
    """
    Send response line to client with rate limit, in chunks.
    """
    if not line.endswith("\n"):
        line = line + "\n"

    data = line.encode("utf-8")
    log(f"rate_limit: sending {len(data)} bytes in chunks")
    try:
        offset = 0
        n = len(data)
        while offset < n:
            end = min(offset + RATE_LIMIT_CHUNK, n)
            sock.sendall(data[offset:end])
            offset = end
            time.sleep(RATE_LIMIT_TICK)
    except Exception as e:
        log(f"rate_limit: error while sending bytes: {e}")


# ---------- main handler -----------------------------------------------------


def handle_client(client_sock, client_addr):
    """
    Handle one client connection.
    One request and one response.
    """
    log(f"client connected from {client_addr}")

    # Choose upstream controller for this connection
    up_host, up_port = choose_upstream()

    try:
        server_sock = socket.create_connection((up_host, up_port), timeout=5.0)
        log(f"connected to controller at {up_host}:{up_port}")
    except Exception as e:
        log(f"cannot connect to controller: {e}")
        client_sock.close()
        return

    try:
        with client_sock, server_sock:
            cf = client_sock.makefile("rw", encoding="utf-8")
            sf = server_sock.makefile("rw", encoding="utf-8")

            # Read request from client
            line_req = cf.readline()
            if not line_req:
                log("empty request from client, closing")
                return

            log(f"raw request line from client: {line_req.strip()}")

            # Parse JSON request
            try:
                req_obj = json.loads(line_req)
            except json.JSONDecodeError as e:
                log(f"JSON decode error on request: {e}, forwarding as is")
                slow_this_conn = False
                maybe_add_delay_on_request()
                sf.write(line_req)
                sf.flush()
            else:
                # Decide slow mode for this connection
                slow_this_conn = should_use_slow(req_obj)

                # smart_drop or pass through
                should_forward, req_mod = tamper_request(req_obj)
                if not should_forward and MODE == "smart_drop":
                    log("smart_drop: not forwarding request, closing connection")
                    return

                line_out = json.dumps(req_mod) + "\n"
                maybe_add_delay_on_request()
                sf.write(line_out)
                sf.flush()
                log(f"sent request to controller: {line_out.strip()}")

            # Response handling
            use_replay, replay_line = apply_replay_before_live()

            if use_replay:
                resp_line = replay_line
                live_resp_obj = None
            else:
                resp_line = sf.readline()
                if not resp_line:
                    log("empty response from controller, closing")
                    return
                log(f"raw response line from controller: {resp_line.strip()}")
                live_resp_obj = None
                try:
                    live_resp_obj = json.loads(resp_line)
                except json.JSONDecodeError as e:
                    log(f"JSON decode error on response: {e}, forward as is")

            # If we have live JSON, tamper and maybe store for replay
            if live_resp_obj is not None:
                resp_mod_obj = tamper_response_obj(live_resp_obj)
                resp_line_out = json.dumps(resp_mod_obj) + "\n"

                # replay storage
                store_for_replay(resp_line_out)

                final_line = resp_line_out
            else:
                final_line = resp_line

            # freeze filter
            final_line = apply_freeze_filter(final_line)

            maybe_add_delay_on_response()

            # send to client with selected mode
            if MODE == "slow" or (MODE == "conditional_slow" and slow_this_conn):
                send_response_line_slow(client_sock, final_line)
            elif MODE == "rate_limit":
                send_response_line_rate_limited(client_sock, final_line)
            else:
                cf.write(final_line)
                cf.flush()
                log(f"sent response to client: {final_line.strip()}")

            # duplicate send if needed
            if MODE == "dup" and random.random() < DUP_PROB:
                log("dup: sending duplicate response line")
                cf.write(final_line)
                cf.flush()

    except Exception as e:
        log(f"error in handle_client: {e}")

    log(f"connection with {client_addr} closed")


def main():
    log(f"listening on {LISTEN_HOST}:{LISTEN_PORT}")
    log(f"main controller at {CTRL_HOST}:{CTRL_PORT_MAIN}")
    log(f"alt controller at {CTRL_HOST}:{CTRL_PORT_ALT}")
    log(f"mode={MODE}")

    wireshark_proc = start_wireshark_capture()

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((LISTEN_HOST, LISTEN_PORT))
            s.listen(5)
            while True:
                client_sock, client_addr = s.accept()
                t = threading.Thread(
                    target=handle_client,
                    args=(client_sock, client_addr),
                    daemon=True,
                )
                t.start()
    finally:
        if wireshark_proc is not None:
            try:
                wireshark_proc.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 17:08:27 2025

@author: tbehdadn
"""

# -*- coding: utf-8 -*-
"""
controller_server.py

Simple controller server that listens on localhost and TCP port 5001.
Receives JSON:
    { "lost_p": float,
      "loads": [ { "id": str, "Ppf": float, "bus": int }, ... ] }

Returns JSON:
    { "shed": { "PQ_x": float, ... } }

Logs every request and response.
"""

import json
import socket
import numpy as np

HOST = "127.0.0.1"
PORT = 5001


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


def handle_client(conn, addr):
    """
    Handle one client connection.
    """
    print(f"[controller] new connection from {addr}")
    with conn:
        f = conn.makefile("rw", encoding="utf-8")
        line = f.readline()
        if not line:
            print("[controller] empty request, closing")
            return

        print(f"[controller] received: {line.strip()}")
        req = json.loads(line)
        lost_p = float(req["lost_p"])
        loads = req["loads"]

        shed_dict = compute_load_shedding(lost_p, loads)
        resp = {"shed": shed_dict}
        resp_line = json.dumps(resp) + "\n"
        print(f"[controller] sending: {resp_line.strip()}")
        f.write(resp_line)
        f.flush()
    print(f"[controller] connection closed for {addr}")


def main():
    print(f"[controller] listening on {HOST}:{PORT}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(5)
        while True:
            conn, addr = s.accept()
            handle_client(conn, addr)


if __name__ == "__main__":
    main()

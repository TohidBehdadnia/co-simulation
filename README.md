# co-simulation

## IEC 60870-5-104 variant (c104)

This repo includes an IEC-104 version of the same “measure delay → apply load shedding at `t_trip + delay`” experiment.

Scripts:
- [system_RT_iec104.py](system_RT_iec104.py): runs the ANDES simulation and hosts an IEC-104 outstation (server).
- [controller_iec104_master.py](controller_iec104_master.py): IEC-104 master (client) that computes proportional shedding and sends setpoints.

Run (two terminals):
- Terminal A (outstation + simulation): `.venv/bin/python system_RT_iec104.py --scenario direct`
- Terminal B (controller): `.venv/bin/python controller_iec104_master.py`

> [!NOTE]
> - By default the outstation waits for a start handshake command (IOA `8999`) before transmitting measurements and starting the delay timer. This makes results independent of how long you take to start the controller.
> - Disable if needed with `IEC104_WAIT_START=0` or `--no-wait-start` on the outstation.

### MITM proxy with delay

To inject extra network delay into IEC-104, run an IEC-104-aware MITM bridge (it speaks IEC-104 as an outstation to the controller, and as a master to the real outstation).

Run (three terminals):
- Terminal A (outstation): `.venv/bin/python system_RT_iec104.py --scenario mitm --induced-delay 0.20` (listens on `127.0.0.1:2404`)
- Terminal B (MITM proxy): `.venv/bin/python mitm_proxy_iec104_delay.py --listen-port 2405 --upstream-port 2404 --shed-delay 0.20`
- Terminal C (controller via proxy): `.venv/bin/python controller_iec104_master.py --scenario mitm` (defaults to port `2405`)

Notes:
- The proxy adds delay only to the shedding setpoint commands (IOA `20000+bus`), not to monitoring values.
- If you set `IEC104_PORT`, it overrides the scenario default.

Outputs:
- Direct: `freq_iec104_direct.png`, `case_iec104_direct.npz`
- MITM: `freq_iec104_mitm_delay<X>.png`, `case_iec104_mitm_delay<X>.npz` (e.g. `delay0p20`)
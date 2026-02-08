# co-simulation

This repo contains a small ANDES-based co-simulation that measures control/communication delay and applies load shedding at the corresponding simulation time.

All frequency plots are written under the `plots/` folder (auto-created by the scripts).

## TCP JSON variant

Scripts:
- [system_RT.py](system_RT.py): runs the ANDES simulation and executes 3 scenarios (local controller, remote controller over TCP, remote controller via a TCP proxy).
- [controller_RT.py](controller_RT.py): simple TCP JSON controller (used by `system_RT.py` for the remote scenarios).

Outputs:
- Plots: `plots/freq_local.png`, `plots/freq_direct.png`, `plots/freq_proxy.png`
- Data (ignored by git): `case_local.npz`, `case_direct.npz`, `case_proxy.npz`

## IEC 60870-5-104 variant (c104)

This repo includes an IEC-104 version of the same “measure delay → apply load shedding at `t_trip + delay`” experiment.

Scripts:
- [system_RT_iec104.py](system_RT_iec104.py): runs the ANDES simulation and hosts an IEC-104 outstation (server).
- [controller_iec104_master.py](controller_iec104_master.py): IEC-104 master (client) that computes proportional shedding and sends setpoints.
- [mitm_proxy_iec104_delay.py](mitm_proxy_iec104_delay.py): IEC-104-aware MITM bridge that can delay only the shedding commands.

Shedding application modes (outstation):
- Default: `--shed-mode all_at_once` applies all shed deltas at `t_trip + delay(last command)`.
- Optional: `--shed-mode per_command` applies each bus's shed delta at its own command-arrival time (stepwise response).

To make stepwise effects clearly visible, you can intentionally stagger controller commands:
- `--inter-command-sleep-ms 50` (or env `IEC104_INTER_COMMAND_SLEEP_MS=50`).

Run (two terminals):
- Terminal A (outstation + simulation): `.venv/bin/python system_RT_iec104.py --scenario direct`
- Terminal B (controller): `.venv/bin/python controller_iec104_master.py`

Example (stepwise response):
- Terminal A: `.venv/bin/python system_RT_iec104.py --scenario direct --shed-mode per_command`
- Terminal B: `.venv/bin/python controller_iec104_master.py --scenario direct --inter-command-sleep-ms 50`

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
- Plots (direct): `plots/freq_iec104_direct_all.png` or `plots/freq_iec104_direct_percmd.png`
- Plots (MITM): `plots/freq_iec104_mitm_delay<X>_all.png` or `plots/freq_iec104_mitm_delay<X>_percmd.png`
- Data (ignored by git): `case_iec104_*.npz`

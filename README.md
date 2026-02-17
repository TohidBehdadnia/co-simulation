# co-simulation

This repo contains a small ANDES-based co-simulation that measures control/communication delay and applies load shedding at the corresponding simulation time.

All frequency plots are written under the `plots/` folder (auto-created by the scripts).

## TCP JSON variant

Scripts:
- [system_RT.py](system_RT.py): runs one selected scenario (`local`, `direct`, `proxy`) and supports repeated runs for statistical studies.
- [controller_RT.py](controller_RT.py): simple TCP JSON controller (used by `system_RT.py` for the remote scenarios).

Outputs:
- Plots: `plots/freq_<scenario>_<id>.pdf`
- Data: `results/case_<scenario>/case_<scenario>_<id>.npz`

Run examples:
- Local, 20 runs: `.venv/bin/python system_RT.py --scenario local --num-runs 20`
- Direct, 30 runs (ids 101..130): `.venv/bin/python system_RT.py --scenario direct --num-runs 30 --start-id 101`
- Proxy, 10 runs: `.venv/bin/python system_RT.py --scenario proxy --num-runs 10`

## IEC 60870-5-104 variant (c104)

This repo includes an IEC-104 version of the same “measure delay → apply load shedding at `t_trip + delay`” experiment.

Scripts:
- [system_RT_iec104.py](system_RT_iec104.py): runs one selected IEC-104 scenario and supports repeated runs for statistical studies.
- [controller_iec104_master.py](controller_iec104_master.py): IEC-104 master (client) that computes proportional shedding and sends setpoints.
- [mitm_proxy_iec104.py](mitm_proxy_iec104.py): IEC-104-aware MITM bridge that can delay or drop only the shedding commands.
- [run_iec104_scenario.py](run_iec104_scenario.py): one-command orchestrator for repeated end-to-end runs (system + controller + optional MITM).
- [run_batch_experiment.py](run_batch_experiment.py): runs the full batch (direct + multiple MITM delay ranges) and generates the aggregated comparison plot (mean ±95% CI by default; use `--band sd` or `--band both`).

`run_iec104_scenario.py` orchestrates runs **in-process** by creating Python objects for each component (system, controller, proxy) instead of spawning shell commands for each script.

Shedding application modes (outstation):
- Default: `--shed-mode per_command` applies each bus's shed delta at its own command-arrival time (stepwise response).
- Optional: `--shed-mode all_at_once` applies all shed deltas at `t_trip + delay(last command)`.

Plot behavior:
- In multi-run mode (`--num-runs > 1`), plots are disabled by default and only `.npz` data files are stored.
- Use `--plot` to force plot generation in multi-run campaigns.

To make stepwise effects clearly visible, you can intentionally stagger controller commands:
- `--inter-command-sleep-ms 50` (or env `IEC104_INTER_COMMAND_SLEEP_MS=50`).

Run (two terminals):
- Terminal A (outstation + simulation): `.venv/bin/python system_RT_iec104.py --scenario direct`
- Terminal B (controller): `.venv/bin/python controller_iec104_master.py`

Multi-run example (direct):
- Terminal A: `.venv/bin/python system_RT_iec104.py --scenario direct --num-runs 20 --start-id 1`
- Terminal B: start one controller run per outstation run.

One-command multi-run (recommended):
- Direct (20 runs): `.venv/bin/python run_iec104_scenario.py --scenario direct --num-runs 20 --start-id 1`
- MITM delay (20 runs): `.venv/bin/python run_iec104_scenario.py --scenario mitm --mitm-mode delay --num-runs 20 --proxy-shed-delay 0.20`
- MITM drop (20 runs): `.venv/bin/python run_iec104_scenario.py --scenario mitm --mitm-mode drop --num-runs 20 --proxy-command-drop-prob 0.4 --proxy-seed 98`

Batch experiments (single command):
- Delay batch (direct + 3 delay ranges), 10 runs each:
	- `.venv/bin/python run_batch_experiment.py --batch delay --num-runs 10 --out-name freq_comparison_delay_mean.pdf`
- Drop batch (direct + pdrop=0.1/0.5/0.9), 10 runs each, reuse existing direct results:
	- `.venv/bin/python run_batch_experiment.py --batch drop --num-runs 10 --reuse-direct --out-name freq_comparison_drop_mean.pdf`
- Add the combined 3D distribution plot (one subplot per scenario):
	- `.venv/bin/python run_batch_experiment.py --batch drop --only-plot --num-runs 10 --band sd --dist3d --dist3d-bins 100 --out-name freq_comparison_drop_mean.pdf`
- Plot-only regeneration from existing results (no rerun):
	- `.venv/bin/python run_batch_experiment.py --batch delay --only-plot --band ci95 --out-name freq_comparison_delay_mean.pdf`
	- `.venv/bin/python run_batch_experiment.py --batch drop --only-plot --band both --out-name freq_comparison_drop_mean.pdf`

Aggregated plots (from existing results):
- `plotter_comparator.py` overlays mean frequency-vs-time curves across scenarios (band: `--band ci95` or `--band sd`).
- `--dist3d` additionally writes a single PDF with a grid of 3D subplots (one per scenario) showing the per-time frequency distribution across runs.
- Output naming:
	- 2D comparison: `plots/<out-name>`
	- 3D distribution: `plots/<out-stem>_dist3d_all_<delay|drop|mixed>.pdf`
- Run limiting:
	- `plotter_comparator.py --max-runs N` limits how many `.npz` run files are loaded per scenario folder.
	- `run_batch_experiment.py` forwards `--num-runs` as the plot run-limit, so the aggregated plots use the latest `--num-runs` run files per scenario.

Example (drop scenarios):
```bash
.venv/bin/python plotter_comparator.py \
	case_iec104_direct_percmd="Direct" \
	case_iec104_mitm_drop_pdrop0p1_percmd="MITM drop p=0.1" \
	case_iec104_mitm_drop_pdrop0p5_percmd="MITM drop p=0.5" \
	case_iec104_mitm_drop_pdrop0p9_percmd="MITM drop p=0.9" \
	--band sd --dist3d --dist3d-bins 100 \
	--out-name tmp_drop_compare.pdf \
	--title "tmp-drop"
```

Example (delay scenarios):
```bash
.venv/bin/python plotter_comparator.py \
	case_iec104_direct_percmd="Direct" \
	case_iec104_mitm_delay_dmin0p05_dmax0p25_percmd="MITM delay 50–250ms" \
	case_iec104_mitm_delay_dmin0p3_dmax0p5_percmd="MITM delay 300–500ms" \
	case_iec104_mitm_delay_dmin0p55_dmax0p75_percmd="MITM delay 550–750ms" \
	--band sd --dist3d --dist3d-bins 100 \
	--out-name tmp_delay_compare.pdf \
	--title "tmp-delay"
```

Logs (orchestrator):
- When using `run_iec104_scenario.py`, the three components write to separate log files under `logs/run_<id>/`:
	- `system.log`, `controller.log`, and (MITM only) `proxy.log`.
	- Per-run packet capture (if enabled): `traffic.pcap` (plus `tcpdump.log` and `tcpdump_cmd.txt`).

Packet capture notes:
- `run_iec104_scenario.py` enables tcpdump by default. Disable with `--no-tcpdump`.
- Capturing packets usually requires running with sufficient privileges (e.g. `sudo` or granting `tcpdump` the needed capabilities).
- The default filter captures both IEC-104 TCP ports `2404` and `2405`. Override with `--tcpdump-filter`.

Example (stepwise response):
- Terminal A: `.venv/bin/python system_RT_iec104.py --scenario direct --shed-mode per_command`
- Terminal B: `.venv/bin/python controller_iec104_master.py --scenario direct --inter-command-sleep-ms 50`

> [!NOTE]
> - By default the outstation waits for a start handshake command (IOA `8999`) before transmitting measurements and starting the delay timer. This makes results independent of how long you take to start the controller.
> - Disable if needed with `IEC104_WAIT_START=0` or `--no-wait-start` on the outstation.

### MITM proxy with delay

To inject extra network delay into IEC-104, run an IEC-104-aware MITM bridge (it speaks IEC-104 as an outstation to the controller, and as a master to the real outstation).

Run (three terminals):
- Terminal A (outstation): `.venv/bin/python system_RT_iec104.py --scenario mitm` (listens on `127.0.0.1:2404`)
- Terminal B (MITM proxy, fixed delay): `.venv/bin/python mitm_proxy_iec104.py --listen-port 2405 --upstream-port 2404 --shed-delay 0.20`
- Terminal B (MITM proxy, random per-packet delay): `.venv/bin/python mitm_proxy_iec104.py --listen-port 2405 --upstream-port 2404 --shed-delay-min 0.05 --shed-delay-max 0.45 --seed 98`
- Terminal C (controller via proxy): `.venv/bin/python controller_iec104_master.py --scenario mitm` (defaults to port `2405`)

Notes:
- The proxy adds delay only to the shedding setpoint commands (IOA `20000+bus`), not to monitoring values.
- With `--shed-delay-min/--shed-delay-max`, each shed command gets its own delay sampled uniformly.
- When running via `run_iec104_scenario.py`, `--proxy-seed` is treated as a *base seed* and the orchestrator uses `base_seed + run_index` so runs are repeatable but not identical.
- If you set `IEC104_PORT`, it overrides the scenario default.

Outputs:
- Plots (direct): `plots/freq_iec104_direct_<all|percmd>_<id>.pdf`
- Plots (MITM): `plots/freq_iec104_mitm_<variant>_<all|percmd>_<id>.pdf`
- Data (direct): `results/case_iec104_direct_<all|percmd>/case_iec104_direct_<all|percmd>_<id>.npz`
- Data (MITM delay, fixed): `results/case_iec104_mitm_delay_d<delay>_<all|percmd>/..._<id>.npz` (example `d0p2`)
- Data (MITM delay, range): `results/case_iec104_mitm_delay_dmin<min>_dmax<max>_<all|percmd>/..._<id>.npz`
- Data (MITM drop): `results/case_iec104_mitm_drop_pdrop<prob>_<all|percmd>/..._<id>.npz` (example `pdrop0p4`)

### MITM proxy with drop

This mode randomly drops *shed command* packets in the MITM proxy (monitoring values are still forwarded).

Run (three terminals):
- Terminal A (outstation + simulation): `.venv/bin/python system_RT_iec104.py --scenario mitm --mitm-mode drop`
- Terminal B (MITM proxy, dropping commands): `.venv/bin/python mitm_proxy_iec104.py --listen-port 2405 --upstream-port 2404 --command-drop-prob 0.4 --seed 98`
- Terminal C (controller via proxy): `.venv/bin/python controller_iec104_master.py --scenario mitm`

Outstation behavior in `drop` mode:
- The outstation waits up to 20s for commands, then applies whatever commands arrived by the timeout.
- Missing (dropped) commands are treated as “no shedding” for those buses.
- If no commands arrive, it still runs the trip-only simulation (no shedding applied).

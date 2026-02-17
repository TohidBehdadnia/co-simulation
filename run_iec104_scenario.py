#!/usr/bin/env python3
import argparse
import contextlib
import multiprocessing
import os
import shlex
import shutil
import signal
import subprocess
import traceback
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import controller_iec104_master as iec104_controller
import mitm_proxy_iec104 as iec104_mitm
import system_RT_iec104 as iec104_system


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = BASE_DIR / "logs"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _redirect_stdio_to_file(log_path: Path):
    """Redirect both stdout/stderr to a log file (process-local)."""
    _ensure_dir(log_path.parent)
    f = open(log_path, "w", encoding="utf-8", buffering=1)
    return f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f)


def _proc_run_system(config: "IEC104SystemConfig", run_id: int, log_path: str) -> None:
    log_file, r_out, r_err = _redirect_stdio_to_file(Path(log_path))
    with log_file, r_out, r_err:
        try:
            component = IEC104SystemComponent(config)
            component.run_once(int(run_id))
        except BaseException:
            traceback.print_exc(file=log_file)
            log_file.flush()
            os._exit(1)


def _proc_run_controller(config: "IEC104ControllerConfig", log_path: str) -> None:
    log_file, r_out, r_err = _redirect_stdio_to_file(Path(log_path))
    with log_file, r_out, r_err:
        try:
            component = IEC104ControllerComponent(config)
            component.run_once()
        except BaseException:
            traceback.print_exc(file=log_file)
            log_file.flush()
            os._exit(1)


def _proc_run_proxy(config: "IEC104MitmConfig", stop_flag, log_path: str) -> None:
    log_file, r_out, r_err = _redirect_stdio_to_file(Path(log_path))
    with log_file, r_out, r_err:
        try:
            proxy = IEC104MitmProxyComponent(config)
            # Run proxy in-process (this process) and stop when flag is set.
            args = argparse.Namespace(
                listen_host=proxy.config.listen_host,
                listen_port=proxy.config.listen_port,
                upstream_host=proxy.config.upstream_host,
                upstream_port=proxy.config.upstream_port,
                ca=proxy.config.ca,
                tick_ms=proxy.config.tick_ms,
                command_timeout_ms=proxy.config.command_timeout_ms,
                shed_buses=proxy.config.shed_buses,
                shed_delay=proxy.config.shed_delay,
                shed_delay_min=proxy.config.shed_delay_min,
                shed_delay_max=proxy.config.shed_delay_max,
                command_drop_prob=proxy.config.command_drop_prob,
                seed=proxy.config.seed,
                log_messages=proxy.config.log_messages,
            )
            iec104_mitm.run_mitm_proxy(args, stop_event=stop_flag)
        except BaseException:
            traceback.print_exc(file=log_file)
            log_file.flush()
            os._exit(1)


@dataclass
class _TcpdumpCapture:
    proc: subprocess.Popen
    stderr_file: TextIO
    pcap_path: Path


def _build_default_tcpdump_filter(args: argparse.Namespace) -> str:
    # Capture both legs by default:
    # - controller <-> proxy (listen_port, typically 2405)
    # - proxy/direct controller <-> outstation (upstream_port, typically 2404)
    ports: set[int] = {
        int(getattr(args, "upstream_port", 2404)),
        int(getattr(args, "listen_port", 2405)),
    }
    return " or ".join([f"tcp port {p}" for p in sorted(ports)])


def _start_tcpdump_capture(args: argparse.Namespace, run_log_dir: Path) -> _TcpdumpCapture | None:
    if not bool(getattr(args, "tcpdump", True)):
        return None

    tcpdump_bin = shutil.which("tcpdump")
    if tcpdump_bin is None:
        print("[orchestrator] tcpdump not found in PATH; skipping packet capture", flush=True)
        return None

    iface = str(getattr(args, "tcpdump_interface", "any"))
    snaplen = int(getattr(args, "tcpdump_snaplen", 0))

    # Filter expression: user-provided overrides default.
    filter_expr = getattr(args, "tcpdump_filter", None)
    if filter_expr is None:
        filter_expr = _build_default_tcpdump_filter(args)
    filter_expr = str(filter_expr).strip()

    pcap_path = run_log_dir / "traffic.pcap"
    stderr_path = run_log_dir / "tcpdump.log"
    cmd_path = run_log_dir / "tcpdump_cmd.txt"

    cmd: list[str] = [
        tcpdump_bin,
        "-i",
        iface,
        "-nn",
        "-s",
        str(snaplen),
        "-U",
        "-w",
        str(pcap_path),
    ]
    if filter_expr:
        cmd.extend(shlex.split(filter_expr))

    cmd_path.write_text(" ".join([shlex.quote(x) for x in cmd]) + "\n", encoding="utf-8")

    stderr_f = open(stderr_path, "w", encoding="utf-8", buffering=1)
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=stderr_f,
            start_new_session=True,
        )
    except Exception:
        stderr_f.close()
        raise

    # If tcpdump fails immediately (missing permissions, bad iface, etc.), stop and keep the log.
    time.sleep(0.2)
    if proc.poll() is not None:
        print(
            f"[orchestrator] tcpdump exited early (code={proc.returncode}); see {stderr_path}",
            flush=True,
        )
        with contextlib.suppress(Exception):
            proc.wait(timeout=0.5)
        try:
            stderr_f.close()
        finally:
            return None

    print(f"[orchestrator] tcpdump: {pcap_path}", flush=True)
    return _TcpdumpCapture(proc=proc, stderr_file=stderr_f, pcap_path=pcap_path)


def _stop_tcpdump_capture(capture: _TcpdumpCapture | None) -> None:
    if capture is None:
        return

    try:
        if capture.proc.poll() is None:
            # Graceful stop so the pcap is finalized.
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(int(capture.proc.pid), signal.SIGINT)
            try:
                capture.proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(int(capture.proc.pid), signal.SIGTERM)
                try:
                    capture.proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError, PermissionError):
                        os.killpg(int(capture.proc.pid), signal.SIGKILL)
                    capture.proc.wait(timeout=3.0)
    finally:
        with contextlib.suppress(Exception):
            capture.stderr_file.close()


@dataclass
class IEC104SystemConfig:
    scenario: str
    mitm_mode: str
    shed_mode: str
    tick_ms: int
    wait_start: bool
    wait_start_timeout: float
    timeout: float
    num_runs: int
    plot: bool | None
    debug_timing: bool
    mitm_shed_delay: float
    mitm_shed_delay_min: float | None
    mitm_shed_delay_max: float | None
    mitm_drop_prob: float


@dataclass
class IEC104ControllerConfig:
    scenario: str
    ip: str
    port: int
    ca: int
    tick_ms: int
    inter_command_sleep_ms: float
    debug_timing: bool


@dataclass
class IEC104MitmConfig:
    listen_host: str
    listen_port: int
    upstream_host: str
    upstream_port: int
    ca: int
    tick_ms: int
    command_timeout_ms: int
    shed_buses: str
    shed_delay: float
    shed_delay_min: float | None
    shed_delay_max: float | None
    command_drop_prob: float
    seed: int | None
    log_messages: bool


class IEC104SystemComponent:
    def __init__(self, config: IEC104SystemConfig):
        self.config = config

    def run_once(self, run_id: int) -> None:
        if self.config.plot is None:
            plot_enabled = bool(int(self.config.num_runs) == 1)
        else:
            plot_enabled = bool(self.config.plot)

        args = argparse.Namespace(
            scenario=self.config.scenario,
            wait_start=self.config.wait_start,
            wait_start_timeout=self.config.wait_start_timeout,
            timeout=self.config.timeout,
            tick_ms=self.config.tick_ms,
            shed_mode=self.config.shed_mode,
            mitm_mode=self.config.mitm_mode,
            num_runs=int(self.config.num_runs),
            start_id=int(run_id),
            plot=self.config.plot,
            plot_enabled=plot_enabled,
            debug_timing=bool(self.config.debug_timing),
            mitm_shed_delay=float(self.config.mitm_shed_delay),
            mitm_shed_delay_min=self.config.mitm_shed_delay_min,
            mitm_shed_delay_max=self.config.mitm_shed_delay_max,
            mitm_drop_prob=float(self.config.mitm_drop_prob),
        )
        iec104_system.andes.config_logger(stream_level=20)
        iec104_system._run_once(args, int(run_id))


class IEC104ControllerComponent:
    def __init__(self, config: IEC104ControllerConfig):
        self.config = config

    def run_once(self) -> None:
        args = argparse.Namespace(
            scenario=self.config.scenario,
            ip=self.config.ip,
            port=self.config.port,
            ca=self.config.ca,
            tick_ms=self.config.tick_ms,
            inter_command_sleep_ms=self.config.inter_command_sleep_ms,
            debug_timing=bool(self.config.debug_timing),
        )
        iec104_controller.run_controller_once(args)


class IEC104MitmProxyComponent:
    def __init__(self, config: IEC104MitmConfig):
        self.config = config
        # With process-based orchestration, this component is executed via _proc_run_proxy.


class IEC104ScenarioOrchestrator:
    def __init__(self, args: argparse.Namespace):
        # Some callers (e.g. run_batch_experiment.py) instantiate this class directly with a
        # hand-built argparse.Namespace. Ensure new flags have sane defaults.
        if not hasattr(args, "tcpdump"):
            setattr(args, "tcpdump", True)
        if not hasattr(args, "tcpdump_interface"):
            setattr(args, "tcpdump_interface", "any")
        if not hasattr(args, "tcpdump_filter"):
            setattr(args, "tcpdump_filter", None)
        if not hasattr(args, "tcpdump_snaplen"):
            setattr(args, "tcpdump_snaplen", 0)
        if not hasattr(args, "debug_timing"):
            setattr(args, "debug_timing", bool(int(os.environ.get("IEC104_DEBUG_TIMING", "0"))))
        self.args = args

    def _controller_port(self) -> int:
        return int(self.args.listen_port) if self.args.scenario == "mitm" else int(self.args.upstream_port)

    def _build_system(self) -> IEC104SystemComponent:
        return IEC104SystemComponent(
            IEC104SystemConfig(
                scenario=self.args.scenario,
                mitm_mode=self.args.mitm_mode,
                shed_mode=self.args.shed_mode,
                tick_ms=int(self.args.tick_ms),
                wait_start=bool(self.args.wait_start),
                wait_start_timeout=float(self.args.wait_start_timeout),
                timeout=float(self.args.system_timeout),
                num_runs=int(self.args.num_runs),
                plot=self.args.plot,
                debug_timing=bool(self.args.debug_timing),
                mitm_shed_delay=float(self.args.proxy_shed_delay),
                mitm_shed_delay_min=self.args.proxy_shed_delay_min,
                mitm_shed_delay_max=self.args.proxy_shed_delay_max,
                mitm_drop_prob=float(self.args.proxy_command_drop_prob),
            )
        )

    def _build_controller(self) -> IEC104ControllerComponent:
        return IEC104ControllerComponent(
            IEC104ControllerConfig(
                scenario=self.args.scenario,
                ip=str(self.args.listen_host if self.args.scenario == "mitm" else self.args.upstream_host),
                port=self._controller_port(),
                ca=int(self.args.ca),
                tick_ms=int(self.args.tick_ms),
                inter_command_sleep_ms=float(self.args.inter_command_sleep_ms),
                debug_timing=bool(self.args.debug_timing),
            )
        )

    def _build_proxy(self) -> IEC104MitmProxyComponent:
        return IEC104MitmProxyComponent(
            IEC104MitmConfig(
                listen_host=str(self.args.listen_host),
                listen_port=int(self.args.listen_port),
                upstream_host=str(self.args.upstream_host),
                upstream_port=int(self.args.upstream_port),
                ca=int(self.args.ca),
                tick_ms=int(self.args.tick_ms),
                command_timeout_ms=int(self.args.command_timeout_ms),
                shed_buses=str(self.args.shed_buses),
                shed_delay=float(self.args.proxy_shed_delay),
                shed_delay_min=self.args.proxy_shed_delay_min,
                shed_delay_max=self.args.proxy_shed_delay_max,
                command_drop_prob=float(self.args.proxy_command_drop_prob),
                seed=self.args.proxy_seed,
                log_messages=bool(self.args.log_messages),
            )
        )

    def run(self) -> int:
        if self.args.num_runs <= 0:
            raise ValueError("--num-runs must be > 0")

        print(
            f"[orchestrator] scenario={self.args.scenario} mitm_mode={self.args.mitm_mode} "
            f"shed_mode={self.args.shed_mode} num_runs={self.args.num_runs} start_id={self.args.start_id}",
            flush=True,
        )

        ctx = multiprocessing.get_context("spawn")
        base_log_dir = Path(self.args.log_dir) if self.args.log_dir else DEFAULT_LOG_DIR
        _ensure_dir(base_log_dir)

        base_seed = self.args.proxy_seed
        for run_offset, run_id in enumerate(range(self.args.start_id, self.args.start_id + self.args.num_runs)):
            print(f"\n[orchestrator] ===== run_id={run_id} =====", flush=True)
            run_log_dir = base_log_dir / f"run_{int(run_id)}"
            _ensure_dir(run_log_dir)

            tcpdump_capture = None

            system_cfg = self._build_system().config
            controller_cfg = self._build_controller().config
            proxy_cfg = self._build_proxy().config if self.args.scenario == "mitm" else None
            if proxy_cfg is not None and base_seed is not None:
                # Important: the proxy restarts each run. If we used the exact same seed each time,
                # the injected random delays/drops would be identical across runs.
                # Derive a per-run seed so runs differ but the overall sequence is repeatable.
                proxy_cfg.seed = int(base_seed) + int(run_offset)

            system_log = str(run_log_dir / "system.log")
            controller_log = str(run_log_dir / "controller.log")
            proxy_log = str(run_log_dir / "proxy.log")

            proxy_stop = ctx.Event() if proxy_cfg is not None else None
            proxy_proc = None

            try:
                tcpdump_capture = _start_tcpdump_capture(self.args, run_log_dir)

                if proxy_cfg is not None:
                    print(f"[orchestrator] proxy log: {proxy_log}", flush=True)
                    proxy_proc = ctx.Process(
                        target=_proc_run_proxy,
                        args=(proxy_cfg, proxy_stop, proxy_log),
                        name=f"iec104-proxy-{run_id}",
                        daemon=True,
                    )
                    proxy_proc.start()
                    time.sleep(0.3)

                print(f"[orchestrator] system log: {system_log}", flush=True)
                system_proc = ctx.Process(
                    target=_proc_run_system,
                    args=(system_cfg, int(run_id), system_log),
                    name=f"iec104-system-{run_id}",
                    daemon=True,
                )
                system_proc.start()

                time.sleep(0.5)
                print(f"[orchestrator] controller log: {controller_log}", flush=True)
                controller_proc = ctx.Process(
                    target=_proc_run_controller,
                    args=(controller_cfg, controller_log),
                    name=f"iec104-controller-{run_id}",
                    daemon=True,
                )
                controller_proc.start()

                system_proc.join(timeout=float(self.args.run_timeout))
                if system_proc.is_alive():
                    system_proc.terminate()
                    system_proc.join(timeout=5.0)
                    raise TimeoutError(f"Run {run_id} exceeded --run-timeout={self.args.run_timeout}s")

                controller_proc.join(timeout=40.0)

                if int(system_proc.exitcode or 0) != 0:
                    raise RuntimeError(f"system failed (exitcode={system_proc.exitcode})")
                if int(controller_proc.exitcode or 0) != 0:
                    raise RuntimeError(f"controller failed (exitcode={controller_proc.exitcode})")

                print(f"[orchestrator] run_id={run_id} completed", flush=True)

            except KeyboardInterrupt:
                print("[orchestrator] interrupted by user", flush=True)
                if proxy_stop is not None:
                    proxy_stop.set()
                if proxy_proc is not None and proxy_proc.is_alive():
                    proxy_proc.terminate()
                    proxy_proc.join(timeout=5.0)
                return 130

            except Exception as exc:
                print(f"[orchestrator] ERROR on run_id={run_id}: {exc}", flush=True)
                return 1

            finally:
                if proxy_stop is not None:
                    proxy_stop.set()
                if proxy_proc is not None:
                    proxy_proc.join(timeout=5.0)
                    if proxy_proc.is_alive():
                        proxy_proc.terminate()
                        proxy_proc.join(timeout=5.0)

                _stop_tcpdump_capture(tcpdump_capture)

        print("\n[orchestrator] all runs completed", flush=True)
        return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run IEC-104 experiments end-to-end from one script: "
            "system + controller (+ optional MITM) for multiple run ids."
        )
    )
    parser.add_argument("--scenario", choices=["direct", "mitm"], default="direct")
    parser.add_argument("--mitm-mode", choices=["delay", "drop"], default="delay")
    parser.add_argument("--shed-mode", choices=["all_at_once", "per_command"], default="per_command")

    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument(
        "--run-timeout",
        type=float,
        default=120.0,
        help="Max seconds to wait for each system run before aborting.",
    )

    parser.add_argument(
        "--inter-command-sleep-ms",
        type=float,
        default=0.0,
        help="Forwarded to controller_iec104_master.py",
    )
    parser.add_argument(
        "--debug-timing",
        action="store_true",
        default=bool(int(os.environ.get("IEC104_DEBUG_TIMING", "0"))),
        help="Enable perf_counter-based timing logs in both system and controller.",
    )
    parser.add_argument("--tick-ms", type=int, default=50)
    parser.add_argument("--ca", type=int, default=1)
    parser.add_argument("--wait-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wait-start-timeout", type=float, default=0.0)
    parser.add_argument("--system-timeout", type=float, default=20.0)
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable plotting output from system runs. By default disabled when --num-runs > 1.",
    )
    parser.add_argument("--command-timeout-ms", type=int, default=10000)
    parser.add_argument("--shed-buses", default="2,3,4,5,6,9")
    parser.add_argument("--log-messages", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--proxy-shed-delay", type=float, default=0.0)
    parser.add_argument("--proxy-shed-delay-min", type=float, default=None)
    parser.add_argument("--proxy-shed-delay-max", type=float, default=None)
    parser.add_argument("--proxy-command-drop-prob", type=float, default=0.0)
    parser.add_argument("--proxy-seed", type=int, default=None)

    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=2405)
    parser.add_argument("--upstream-host", default="127.0.0.1")
    parser.add_argument("--upstream-port", type=int, default=2404)

    parser.add_argument(
        "--log-dir",
        default=str(DEFAULT_LOG_DIR),
        help="Directory for per-run component logs (default: ./logs).",
    )

    parser.add_argument(
        "--tcpdump",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture a per-run tcpdump into logs/run_<id>/traffic.pcap (requires tcpdump permissions).",
    )
    parser.add_argument(
        "--tcpdump-interface",
        default="any",
        help="Interface for tcpdump (default: any).",
    )
    parser.add_argument(
        "--tcpdump-filter",
        default=None,
        help=(
            "tcpdump filter expression. Default captures tcp ports used by the scenario "
            "(direct: upstream port; mitm: upstream+listen ports)."
        ),
    )
    parser.add_argument(
        "--tcpdump-snaplen",
        type=int,
        default=0,
        help="tcpdump snaplen (-s). Use 0 for full packets (default).",
    )

    return parser


def main() -> int:
    args = _build_parser().parse_args()
    orchestrator = IEC104ScenarioOrchestrator(args)
    return orchestrator.run()


if __name__ == "__main__":
    raise SystemExit(main())

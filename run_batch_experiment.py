#!/usr/bin/env python3
"""Batch runner for the IEC-104 scenarios + aggregated comparison plot(s).

Runs:
- Direct (per-command)
- MITM delay with ranges: 50–250ms, 300–500ms, 550–750ms

Then generates a comparison plot (mean ±95% CI by default).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# NOTE: Imports that indirectly require heavy optional dependencies (e.g. `andes`)
# are deferred until actually needed so `--only-plot` works in lightweight envs.


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"


def _scenario_dirs() -> list[str]:
    # Default batch (delay ranges)
    return [
        "case_iec104_direct_percmd",
        "case_iec104_mitm_delay_dmin0p05_dmax0p25_percmd",
        "case_iec104_mitm_delay_dmin0p3_dmax0p5_percmd",
        "case_iec104_mitm_delay_dmin0p55_dmax0p75_percmd",
    ]


def _scenario_dirs_drop() -> list[str]:
    return [
        "case_iec104_direct_percmd",
        "case_iec104_mitm_drop_pdrop0p1_percmd",
        "case_iec104_mitm_drop_pdrop0p5_percmd",
        "case_iec104_mitm_drop_pdrop0p9_percmd",
    ]


def _clean_results() -> None:
    for name in _scenario_dirs():
        path = RESULTS_DIR / name
        if path.exists() and path.is_dir():
            shutil.rmtree(path)


def _clean_results_drop(reuse_direct: bool) -> None:
    for name in _scenario_dirs_drop():
        if reuse_direct and name == "case_iec104_direct_percmd":
            continue
        path = RESULTS_DIR / name
        if path.exists() and path.is_dir():
            shutil.rmtree(path)


def _run_campaign(
    *,
    scenario: str,
    mitm_mode: str,
    num_runs: int,
    start_id: int,
    run_timeout: float,
    base_seed: int | None,
    delay_min: float | None = None,
    delay_max: float | None = None,
    drop_prob: float | None = None,
) -> None:
    import run_iec104_scenario

    args = argparse.Namespace(
        scenario=scenario,
        mitm_mode=mitm_mode,
        shed_mode="per_command",
        num_runs=int(num_runs),
        start_id=int(start_id),
        run_timeout=float(run_timeout),
        inter_command_sleep_ms=0.0,
        tick_ms=50,
        ca=1,
        wait_start=True,
        wait_start_timeout=0.0,
        system_timeout=20.0,
        command_timeout_ms=10000,
        shed_buses="2,3,4,5,6,9",
        log_messages=True,
        plot=None,
        listen_host="127.0.0.1",
        listen_port=2405,
        upstream_host="127.0.0.1",
        upstream_port=2404,
        log_dir=str(BASE_DIR / "logs"),
        proxy_shed_delay=0.0,
        proxy_shed_delay_min=delay_min,
        proxy_shed_delay_max=delay_max,
        proxy_command_drop_prob=0.0,
        # For drop batch
        # (used only when scenario=mitm, mitm_mode=drop)
        #
        # Note: label naming already includes this probability.
        # The orchestrator will apply per-run seed derivation.
        proxy_seed=base_seed,
    )

    if drop_prob is not None:
        args.proxy_command_drop_prob = float(drop_prob)

    orchestrator = run_iec104_scenario.IEC104ScenarioOrchestrator(args)
    rc = orchestrator.run()
    if rc != 0:
        raise SystemExit(rc)


def _build_plot_specs() -> list[str]:
    return [
        {"scenario_id": "case_iec104_direct_percmd", "label": "Direct"},
        {"scenario_id": "case_iec104_mitm_delay_dmin0p05_dmax0p25_percmd", "label": "MITM delay 50–250ms"},
        {"scenario_id": "case_iec104_mitm_delay_dmin0p3_dmax0p5_percmd", "label": "MITM delay 300–500ms"},
        {"scenario_id": "case_iec104_mitm_delay_dmin0p55_dmax0p75_percmd", "label": "MITM delay 550–750ms"},
    ]


def _build_plot_specs_drop() -> list[dict]:
    return [
        {"scenario_id": "case_iec104_direct_percmd", "label": "Direct"},
        {"scenario_id": "case_iec104_mitm_drop_pdrop0p1_percmd", "label": "MITM drop p=0.1"},
        {"scenario_id": "case_iec104_mitm_drop_pdrop0p5_percmd", "label": "MITM drop p=0.5"},
        {"scenario_id": "case_iec104_mitm_drop_pdrop0p9_percmd", "label": "MITM drop p=0.9"},
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run all IEC-104 scenarios (10 runs each) and generate aggregated plot")
    ap.add_argument("--num-runs", type=int, default=10)
    ap.add_argument("--base-seed", type=int, default=98)
    ap.add_argument("--run-timeout", type=float, default=240.0)
    ap.add_argument(
        "--batch",
        choices=["delay", "drop"],
        default="delay",
        help="Which batch to execute: delay ranges (delay) or drop probabilities (drop).",
    )
    ap.add_argument(
        "--reuse-direct",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If batch=drop, reuse existing direct results instead of re-running them.",
    )
    ap.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--only-plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip rerunning simulations and only generate the plot from existing results.",
    )
    ap.add_argument("--out-name", default="freq_comparison_mean_ci.pdf")
    ap.add_argument("--title", default="IEC-104 frequency")
    ap.add_argument(
        "--band",
        choices=["ci95", "sd", "both"],
        default="ci95",
        help="Which plot(s) to generate: 95% CI of mean (ci95), ±1 SD (sd), or both.",
    )

    args = ap.parse_args()

    if int(args.num_runs) <= 0:
        raise ValueError("--num-runs must be > 0")

    if bool(args.clean) and not bool(args.only_plot):
        if str(args.batch) == "drop":
            _clean_results_drop(reuse_direct=bool(args.reuse_direct))
        else:
            _clean_results()

    if not bool(args.only_plot):
        if str(args.batch) == "drop":
            if not bool(args.reuse_direct):
                _run_campaign(
                    scenario="direct",
                    mitm_mode="delay",
                    num_runs=int(args.num_runs),
                    start_id=2001,
                    run_timeout=float(args.run_timeout),
                    base_seed=None,
                )

            _run_campaign(
                scenario="mitm",
                mitm_mode="drop",
                num_runs=int(args.num_runs),
                start_id=3101,
                run_timeout=float(args.run_timeout),
                base_seed=int(args.base_seed),
                drop_prob=0.1,
            )
            _run_campaign(
                scenario="mitm",
                mitm_mode="drop",
                num_runs=int(args.num_runs),
                start_id=3201,
                run_timeout=float(args.run_timeout),
                base_seed=int(args.base_seed),
                drop_prob=0.5,
            )
            _run_campaign(
                scenario="mitm",
                mitm_mode="drop",
                num_runs=int(args.num_runs),
                start_id=3301,
                run_timeout=float(args.run_timeout),
                base_seed=int(args.base_seed),
                drop_prob=0.9,
            )
        else:
            # Direct
            _run_campaign(
                scenario="direct",
                mitm_mode="delay",
                num_runs=int(args.num_runs),
                start_id=2001,
                run_timeout=float(args.run_timeout),
                base_seed=None,
            )

            # MITM delay ranges
            _run_campaign(
                scenario="mitm",
                mitm_mode="delay",
                num_runs=int(args.num_runs),
                start_id=2101,
                run_timeout=float(args.run_timeout),
                base_seed=int(args.base_seed),
                delay_min=0.05,
                delay_max=0.25,
            )
            _run_campaign(
                scenario="mitm",
                mitm_mode="delay",
                num_runs=int(args.num_runs),
                start_id=2201,
                run_timeout=float(args.run_timeout),
                base_seed=int(args.base_seed),
                delay_min=0.30,
                delay_max=0.50,
            )
            _run_campaign(
                scenario="mitm",
                mitm_mode="delay",
                num_runs=int(args.num_runs),
                start_id=2301,
                run_timeout=float(args.run_timeout),
                base_seed=int(args.base_seed),
                delay_min=0.55,
                delay_max=0.75,
            )

    # Plot aggregated comparison
    import plotter_comparator

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_specs = _build_plot_specs_drop() if str(args.batch) == "drop" else _build_plot_specs()

    band_sel = str(args.band)
    bands = [band_sel] if band_sel in {"ci95", "sd"} else ["sd", "ci95"]

    out_name_raw = str(args.out_name)
    base = str(Path(out_name_raw).with_suffix(""))

    for b in bands:
        out_name = out_name_raw if band_sel != "both" else f"{base}_{b}.pdf"
        out_name = str(Path(out_name).with_suffix(".pdf"))

        # Keep user-provided titles as-is; for the default title, append the band.
        title = str(args.title)
        if title == "IEC-104 frequency":
            title = title + (" (mean ±95% CI)" if b == "ci95" else " (mean ±1 SD)")

        plot_args = argparse.Namespace(
            scenario_specs=plot_specs,
            out_name=out_name,
            title=title,
            band=b,
        )
        plotter_comparator.plot_comparison(plot_args)
        print(f"Saved plot to {PLOTS_DIR / out_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

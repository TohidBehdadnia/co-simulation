#!/usr/bin/env python3
"""Batch runner for the IEC-104 scenarios + aggregated comparison plot(s).

Runs:
- Direct (per-command)
- MITM delay with ranges: 150–350ms, 400–600ms, 650–850ms

Then generates a comparison plot (mean ±95% CI by default).
"""

from __future__ import annotations

import argparse
import os
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
        "case_iec104_mitm_delay_dmin0p15_dmax0p35_percmd",
        "case_iec104_mitm_delay_dmin0p4_dmax0p6_percmd",
        "case_iec104_mitm_delay_dmin0p65_dmax0p85_percmd",
        # Backward-compatible cleanup for older naming/ranges.
        "case_iec104_mitm_delay_dmin0p05_dmax0p25_percmd",
        "case_iec104_mitm_delay_dmin0p15_dmax0p6_percmd",
        "case_iec104_mitm_delay_dmin0p15_dmax0p85_percmd",
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

def _scenario_dirs_mod() -> list[str]:
    return [
        "case_iec104_direct_percmd",
        "case_iec104_mitm_mod_pmod0p1_percmd",
        "case_iec104_mitm_mod_pmod0p5_percmd",
        "case_iec104_mitm_mod_pmod0p9_percmd",
    ]


def _clean_results() -> None:
    for name in _scenario_dirs():
        path = RESULTS_DIR / name
        if path.exists() and path.is_dir():
            shutil.rmtree(path)


def _clean_results_drop_mod(reuse_direct: bool) -> None:
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
    debug_timing: bool,
    delay_min: float | None = None,
    delay_max: float | None = None,
    drop_prob: float | None = None,
    mod_prob: float | None = None,
    mod_factor: float | None = None,
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
        tick_ms=200,
        ca=1,
        wait_start=True,
        wait_start_timeout=0.0,
        system_timeout=20.0,
        command_timeout_ms=10000,
        shed_buses="2,3,4,5,6,9",
        log_messages=True,
        plot=None,
        debug_timing=bool(debug_timing),
        listen_host="127.0.0.1",
        listen_port=2405,
        upstream_host="127.0.0.1",
        upstream_port=2404,
        log_dir=str(BASE_DIR / "logs"),
        proxy_shed_delay=0.0,
        proxy_shed_delay_min=delay_min,
        proxy_shed_delay_max=delay_max,
        proxy_command_drop_prob=float(drop_prob) if drop_prob is not None else 0.0,
        proxy_command_mod_prob=float(mod_prob) if mod_prob is not None else 0.0,
        proxy_command_mod_factor=float(mod_factor) if mod_factor is not None else 1.0,
        # For drop batch
        # (used only when scenario=mitm, mitm_mode=drop)
        #
        # Note: label naming already includes this probability.
        # The orchestrator will apply per-run seed derivation.
        proxy_seed=base_seed,
    )

    orchestrator = run_iec104_scenario.IEC104ScenarioOrchestrator(args)
    rc = orchestrator.run()
    if rc != 0:
        raise SystemExit(rc)


def _build_plot_specs(latexEnabled: bool = False) -> list[str]:
    if not latexEnabled:
        return [
            {"scenario_id": "case_iec104_direct_percmd", "label": "Direct"},
            {"scenario_id": "case_iec104_mitm_delay_dmin0p15_dmax0p35_percmd", "label": "MITM delay 150–350ms"},
            {"scenario_id": "case_iec104_mitm_delay_dmin0p4_dmax0p6_percmd", "label": "MITM delay 400–600ms"},
            {"scenario_id": "case_iec104_mitm_delay_dmin0p65_dmax0p85_percmd", "label": "MITM delay 650–850ms"},
        ]
    else:
        return [
            {"scenario_id": "case_iec104_direct_percmd", "label": "Direct"},
            {"scenario_id": "case_iec104_mitm_delay_dmin0p15_dmax0p35_percmd", "label": "MITM delay $150–350\\mathrm{{ms}}$"},
            {"scenario_id": "case_iec104_mitm_delay_dmin0p4_dmax0p6_percmd", "label": "MITM delay $400–600\\mathrm{{ms}}$"},
            {"scenario_id": "case_iec104_mitm_delay_dmin0p65_dmax0p85_percmd", "label": "MITM delay $650–850\\mathrm{{ms}}$"},
        ]


def _build_plot_specs_drop(latexEnabled: bool = False) -> list[dict]:
    if not latexEnabled:
        return [
            {"scenario_id": "case_iec104_direct_percmd", "label": "Direct"},
            {"scenario_id": "case_iec104_mitm_drop_pdrop0p1_percmd", "label": "MITM drop p=0.1"},
            {"scenario_id": "case_iec104_mitm_drop_pdrop0p5_percmd", "label": "MITM drop p=0.5"},
            {"scenario_id": "case_iec104_mitm_drop_pdrop0p9_percmd", "label": "MITM drop p=0.9"},
        ]
    else:
        return [
            {"scenario_id": "case_iec104_direct_percmd", "label": "Direct"},
            {"scenario_id": "case_iec104_mitm_drop_pdrop0p1_percmd", "label": "MITM drop $p=0.1$"},
            {"scenario_id": "case_iec104_mitm_drop_pdrop0p5_percmd", "label": "MITM drop $p=0.5$"},
            {"scenario_id": "case_iec104_mitm_drop_pdrop0p9_percmd", "label": "MITM drop $p=0.9$"},
        ]

def _build_plot_specs_mod(latexEnabled: bool = False) -> list[dict]:
    if not latexEnabled:
        return [
            {"scenario_id": "case_iec104_direct_percmd", "label": "Direct"},
            {"scenario_id": "case_iec104_mitm_mod_pmod0p1_f1p5_percmd", "label": "MITM mod. p=0.1"},
            {"scenario_id": "case_iec104_mitm_mod_pmod0p5_f1p5_percmd", "label": "MITM mod. p=0.5"},
            {"scenario_id": "case_iec104_mitm_mod_pmod0p9_f1p5_percmd", "label": "MITM mod. p=0.9"},
        ]
    else:
        return [
            {"scenario_id": "case_iec104_direct_percmd", "label": "Direct"},
            {"scenario_id": "case_iec104_mitm_mod_pmod0p1_f1p5_percmd", "label": "MITM mod. $p=0.1$"},
            {"scenario_id": "case_iec104_mitm_mod_pmodp0p5_f1p5_percmd", "label": "MITM mod. $p=0.5$"},
            {"scenario_id": "case_iec104_mitm_mod_pmodp0p9_f1p5_percmd", "label": "MITM mod. $p=0.9$"},
        ]



def main() -> int:
    ap = argparse.ArgumentParser(description="Run all IEC-104 scenarios (10 runs each) and generate aggregated plot")
    ap.add_argument("--num-runs", type=int, default=10)
    ap.add_argument("--base-seed", type=int, default=98)
    ap.add_argument("--run-timeout", type=float, default=240.0)
    ap.add_argument(
        "--debug-timing",
        action="store_true",
        default=bool(int(os.environ.get("IEC104_DEBUG_TIMING", "0"))),
        help="Enable perf_counter-based timing logs in both system and controller.",
    )
    ap.add_argument(
        "--batch",
        choices=["delay", "drop", "mod"],
        default="delay",
        help="Which batch to execute: delay ranges (delay), drop probabilities (drop) or  modification probabilities (mod). Default is delay.",
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
    ap.add_argument("--title", default="Frequency in IEC 104 Load Shedding VS MITM attacks")
    ap.add_argument(
        "--band",
        choices=["ci95", "sd", "both"],
        default="ci95",
        help="Which plot(s) to generate: 95%% CI of mean (ci95), ±1 SD (sd), or both.",
    )
    ap.add_argument(
        "--pcap-parser",
        choices=["none", "dpkt"],
        default="none",
        help=(
            "Parser to use for optional pcap-derived plots: none (default) or dpkt."
        ),
    )

    ap.add_argument(
        "--latex",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable LaTeX text rendering in plots (PGF backend when possible). "
            "Requires a working LaTeX installation (e.g. pdflatex)."
        ),
    )

    args = ap.parse_args()

    if int(args.num_runs) <= 0:
        raise ValueError("--num-runs must be > 0")

    if bool(args.clean) and not bool(args.only_plot):
        if str(args.batch) == "drop" or str(args.batch) == "mod":
            _clean_results_drop_mod(reuse_direct=bool(args.reuse_direct))
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
                    base_seed=int(args.base_seed),
                    debug_timing=bool(args.debug_timing),
                )

            _run_campaign(
                scenario="mitm",
                mitm_mode="drop",
                num_runs=int(args.num_runs),
                start_id=3101,
                run_timeout=float(args.run_timeout),
                base_seed=int(args.base_seed),
                debug_timing=bool(args.debug_timing),
                drop_prob=0.1,
            )
            _run_campaign(
                scenario="mitm",
                mitm_mode="drop",
                num_runs=int(args.num_runs),
                start_id=3201,
                run_timeout=float(args.run_timeout),
                base_seed=int(args.base_seed),
                debug_timing=bool(args.debug_timing),
                drop_prob=0.5,
            )
            _run_campaign(
                scenario="mitm",
                mitm_mode="drop",
                num_runs=int(args.num_runs),
                start_id=3301,
                run_timeout=float(args.run_timeout),
                base_seed=int(args.base_seed),
                debug_timing=bool(args.debug_timing),
                drop_prob=0.9,
            )
        elif str(args.batch) == "mod":
            mod_factor = 1.5
            if not bool(args.reuse_direct):
                _run_campaign(
                    scenario="direct",
                    mitm_mode="delay",
                    num_runs=int(args.num_runs),
                    start_id=2001,
                    run_timeout=float(args.run_timeout),
                    base_seed=int(args.base_seed),
                    debug_timing=bool(args.debug_timing),
                )

            _run_campaign(
                scenario="mitm",
                mitm_mode="mod",
                num_runs=int(args.num_runs),
                start_id=4101,
                run_timeout=float(args.run_timeout),
                base_seed=int(args.base_seed),
                debug_timing=bool(args.debug_timing),
                mod_prob=0.1,
                mod_factor=mod_factor,
            )
            _run_campaign(
                scenario="mitm",
                mitm_mode="mod",
                num_runs=int(args.num_runs),
                start_id=4201,
                run_timeout=float(args.run_timeout),
                base_seed=int(args.base_seed),
                debug_timing=bool(args.debug_timing),
                mod_prob=0.5,
                mod_factor=mod_factor,
            )
            _run_campaign(
                scenario="mitm",
                mitm_mode="mod",
                num_runs=int(args.num_runs),
                start_id=4301,
                run_timeout=float(args.run_timeout),
                base_seed=int(args.base_seed),
                debug_timing=bool(args.debug_timing),
                mod_prob=0.9,
                mod_factor=mod_factor,
            )
        else:
            # Direct
            _run_campaign(
                scenario="direct",
                mitm_mode="delay",
                num_runs=int(args.num_runs),
                start_id=2001,
                run_timeout=float(args.run_timeout),
                base_seed=int(args.base_seed),
                debug_timing=bool(args.debug_timing),
            )

            # MITM delay ranges
            _run_campaign(
                scenario="mitm",
                mitm_mode="delay",
                num_runs=int(args.num_runs),
                start_id=2101,
                run_timeout=float(args.run_timeout),
                base_seed=int(args.base_seed),
                debug_timing=bool(args.debug_timing),
                delay_min=0.15,
                delay_max=0.35,
            )
            _run_campaign(
                scenario="mitm",
                mitm_mode="delay",
                num_runs=int(args.num_runs),
                start_id=2201,
                run_timeout=float(args.run_timeout),
                base_seed=int(args.base_seed),
                debug_timing=bool(args.debug_timing),
                delay_min=0.40,
                delay_max=0.60,
            )
            _run_campaign(
                scenario="mitm",
                mitm_mode="delay",
                num_runs=int(args.num_runs),
                start_id=2301,
                run_timeout=float(args.run_timeout),
                base_seed=int(args.base_seed),
                debug_timing=bool(args.debug_timing),
                delay_min=0.65,
                delay_max=0.85,
            )

    # Plot aggregated comparison
    if bool(args.latex):
        # plotter_comparator reads this env var before importing pyplot.
        os.environ["CO_SIM_USE_LATEX"] = "1"

    import plotter_comparator

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_specs = _build_plot_specs_mod(args.latex) if str(args.batch) == "mod" else (_build_plot_specs_drop(args.latex) if str(args.batch) == "drop" else _build_plot_specs(args.latex))

    band_sel = str(args.band)
    bands = [band_sel] if band_sel in {"ci95", "sd"} else ["sd", "ci95"]

    # If the user requested only extra plots (pcap-derived), run a single pass.
    # The extra plots don't depend on the band, and plotter_comparator will skip the 2D plot.
    only_extras = str(args.pcap_parser).lower().strip() != "none"
    if only_extras:
        bands = ["sd"]

    out_name_raw = str(args.out_name)
    base = str(Path(out_name_raw).with_suffix(""))

    for b in bands:
        out_name = out_name_raw if band_sel != "both" else f"{base}_{b}.pdf"
        out_name = str(Path(out_name).with_suffix(".pdf"))

        # Keep user-provided titles as-is; for the default title, append the band.
        title = str(args.title)
        if title == "Frequency in IEC 104 Load Shedding VS MITM attacks":
            title = title + (" (mean ±95% CI)" if b == "ci95" else " (mean ±1 SD)")

        plot_args = argparse.Namespace(
            scenario_specs=plot_specs,
            out_name=out_name,
            title=title,
            band=b,
            max_runs=int(args.num_runs),
            pcap_parser=str(args.pcap_parser),
            latex=bool(args.latex),
        )
        plotter_comparator.plot_comparison(plot_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

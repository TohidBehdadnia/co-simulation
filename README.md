# co-simulation

This repo contains an ANDES-IEC 104-based co-simulation that measures control/communication characterustucs over IEC-104 and frequency response after a contingency event and the subsequent load shedding operations are performed. 
It also allows for the simulation of alternative Man In The Middle (MITM) attack vectors, such as delay, drop, and modification of IEC-104 setpoint commands.
Moreover, it also includes a machine learning-based attack vector classification component that evaluates the performance of several classifiers in distinguishing between normal operation and different attack scenarios based on features derived from the frequency response and cyber-layer characteristics.

This README specifies the setup and commands to reproduce the results in the associated paper. The code is organized into several Python scripts, each responsible for different aspects of the simulation, data processing, and plotting.

Folders structure:
- Plots: plots/
- Per-run data: results/
- Run logs: logs/
- ML dataset: data/final_data.csv

## Setup (once)

1. Create a Python virtual environment and install dependencies:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

2. LaTeX setup (for paper-quality plots):

```bash
sudo apt-get update
sudo apt-get install -y texlive-latex-base texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra
```

3. Packet capture (mandatory for runs that store pcaps):

```bash
sudo apt-get update
sudo apt-get install -y tcpdump
```

4. Optional (pcap-derived plots for Figs. 2 and 4):

```bash
.venv/bin/pip install dpkt
```

## Reproduce results

This section provides the exact commands to reproduce each figure in the paper. Each command is designed to run the necessary simulations and generate the corresponding plots. The outputs are specified for each command, and optional flags for styling are noted. 
Follow the commands in order if you want to ensure that the the --only-plot flag is called on an existing set of results. If you want to regenerate the data, simply omit the --only-plot flag and rerun the command. 

### Figures 2-6: Frequency response and cyber-layer characteristics

- Fig. 2 (setpoint arrival times under delay attacks):

```bash
.venv/bin/python run_batch_experiment.py \
  --batch delay \
  --num-runs 100 \
  --pcap-parser dpkt \
  --latex
```

Output: plots/pcap_setpoint_arrival_rel_to_meas_delay.pdf'

- Fig. 3 (frequency comparison, MITM delay ranges):

```bash
.venv/bin/python run_batch_experiment.py \
  --batch delay \
  --num-runs 100 \
  --only-plot \
  --band sd \
  --latex \
  --out-name freq_comparison_delay_mean_sd.pdf
```

Output: plots/freq_comparison_delay_mean_sd.pdf

- Fig. 4 (setpoint counts under drop attacks):

```bash
.venv/bin/python run_batch_experiment.py \
  --batch drop \
  --num-runs 100 \
  --pcap-parser dpkt \
  --latex
```

Output: plots/pcap_setpoint_count_to_server_drop.pdf

- Fig. 5 (frequency comparison, MITM drop probabilities):

```bash
.venv/bin/python run_batch_experiment.py \
  --batch drop \
  --num-runs 100 \
  --only-plot \
  --band sd \
  --latex \
  --out-name freq_comparison_drop_mean_sd.pdf
```

Output: plots/freq_comparison_drop_mean_sd.pdf

- Fig. 6 (frequency comparison, MITM modification probabilities):

```bash
.venv/bin/python run_batch_experiment.py \
  --batch mod \
  --num-runs 100 \
  --reuse-direct \
  --band sd \
  --latex \
  --out-name freq_comparison_mod_mean_sd.pdf
```

Output: plots/freq_comparison_mod_mean_sd.pdf


### Figures 7-8: ML model classification performance

1. Build the dataset after running all IEC-104 scenarios above.

```bash
.venv/bin/python data_preparator.py
```

Output: data/final_data.csv

2. Train/evaluate and generate the classification plots.

```bash
.venv/bin/python learner_classifier.py --latex
```

Outputs:
- plots/classifier_comparison.pdf (Fig. 7)
- plots/confusion_matrices_by_model_1_2s.pdf (Fig. 8)

### Figures 9-10: Real-time attack vector classification

Run the single-server simulation to generate the real-time classification plots.

```bash
.venv/bin/python pred_time_sim.py --latex
```

Outputs:
- plots/time_granularity_distribution_sim.pdf (Fig. 9)
- plots/classifier_prediction_sim.pdf (Fig. 10)

> [!WARNING]
> Results regarding the real-time classification response time (Figs. 9-10) may vary significantly based on the hardware capabilities of the machine running the simulation. The paper's results were obtained on a Raspberry Pi 3B+, which has limited processing power and adequately reflects the hardware characteristics of RTU used in real-world scenarios. Running the same simulation on a more powerful machine (e.g., a modern laptop or desktop) may yield much faster classification times.

## Appendix: extra commands not strictly necessary

Single-run frequency plot (direct, one run):

```bash
.venv/bin/python system_RT_iec104.py --scenario direct --num-runs 1 --plot
```

Manual two-terminal run (direct):

```bash
# Terminal A
.venv/bin/python system_RT_iec104.py --scenario direct

# Terminal B
.venv/bin/python controller_iec104_master.py
```

Manual three-terminal run (MITM delay):

```bash
# Terminal A
.venv/bin/python system_RT_iec104.py --scenario mitm

# Terminal B
.venv/bin/python mitm_proxy_iec104.py --listen-port 2405 --upstream-port 2404 --shed-delay 0.20

# Terminal C
.venv/bin/python controller_iec104_master.py --scenario mitm
```

One-command multi-run orchestration:

```bash
.venv/bin/python run_iec104_scenario.py --scenario direct --num-runs 20 --start-id 1
```

Plot-only regeneration (no rerun):

```bash
.venv/bin/python run_batch_experiment.py --batch delay --only-plot --out-name tmp_delay.pdf
```

Optional plot styling:
- Add --latex to run_batch_experiment.py or plotter_comparator.py for LaTeX text rendering.
- Add --poster to increase plot font sizes.

Notes:
- By default the outstation waits for a start handshake command (IOA 8999). Disable with IEC104_WAIT_START=0 or --no-wait-start.
- Shedding modes: --shed-mode per_command (default) or --shed-mode all_at_once.

> [!NOTE]
> Pcap plots require tcpdump permissions (or equivalent capabilities) so the captures exist under logs/.

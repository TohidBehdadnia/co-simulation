# Edge-ready anomaly detection using cyber-physical consistency assessment in power systems: A co-simulation-based approach — Co-Simulation Testbed 

This repo contains an ANDES-IEC 104-based co-simulation that measures control/communication characteristics over IEC-104 and frequency response after a contingency event and the subsequent load shedding operations are performed. 
It also allows for the simulation of alternative Man In The Middle (MITM) attack vectors, such as delay, drop, and modification of IEC-104 setpoint commands.
Moreover, it also includes a machine learning-based attack vector classification component that evaluates the performance of several classifiers in distinguishing between normal operation and different attack scenarios based on features derived from the frequency response and cyber-layer characteristics.

This README specifies the setup and commands to reproduce the results presented in the paper "Edge-ready anomaly detection using cyber-physical consistency assessment in power systems: A co-simulation-based approach". 

## Authors and Contacts
- Tohid Behdadnia<sup>†</sup> &rarr; [tohid.behdadnia@kuleuven.be](mailto:tohid.behdadnia@kuleuven.be)
- Davide Savarro<sup>†</sup> &rarr; [davide.savarro@unito.it](mailto:davide.savarro@unito.it)
- Roberta Terruggia &rarr; [roberta.terruggia@rse-web.it](mailto:roberta.terruggia@rse-web.it)
- Geert Deconinck &rarr; [geert.deconinck@kuleuven.be](mailto:geert.deconinck@kuleuven.be)

<sup>†</sup> These authors contributed equally to this work and share first authorship.

## Testbed structure and organization

The code is organized into several Python scripts, each responsible for different aspects of the simulation, data processing, and plotting. 

Main folders and files structure:
```
co-simulation/
├── plots/                       // Generated plots
├── results/                     // Per-run data (e.g., frequency response, features)
├── logs/                        // Run logs and pcaps
├── data/                        // Processed datasets (e.g., final_data.csv for ML)
├── system_RT_iec104.py          // Remote Terminal Unit (RTU) implementation and co-simulation orchestration
├── controller_iec104_master.py  // Master Terminal Unit (MTU) implementation
├── mitm_proxy_iec104.py         // MITM proxy implementation to simulate multiple attack vectors
├── run_batch_experiment.py      // Script to run batch experiments for different scenarios and generate plots
├── data_preparator.py           // Script to prepare the dataset for ML classification
├── learner_classifier.py        // Script to train/evaluate ML classifiers and generate performance plots
├── requirements.txt             // Python dependencies
├── requirements.lock.txt        // Locked Python dependencies (specific versions)
...
```

## Software requirements
The following installation steps assume a Debian-based Linux distribution. The code is written in Python and relies on several libraries for simulation, data processing, and plotting. The specific dependencies are listed in the requirements.txt file.

In our experiments, we used the following software versions:
- Python: 3.10.12
- tcpdump (for packet capture): 4.99.1

## Setup (once)

1. Create and activate a Python virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
```
> [!NOTE]
> If you keep the venv activated, you can run the commands below with `python` and `pip` directly. Otherwise, keep the `.venv/bin/` prefix as shown.

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

## Replicating the results

This section provides the exact commands to reproduce each figure in the paper. Each command is designed to run the necessary simulations and generate the corresponding plots. The outputs are specified for each command, and optional flags for styling are noted. 

Follow the commands in order if you want to ensure that the the `--only-plot` flag is called on an existing set of results. If you want to regenerate the data, simply omit the `--only-plot` flag and rerun the command. 

### Figures 3-7: Frequency response and cyber-layer characteristics

- Fig. 3 (setpoint arrival times under delay attacks):

```bash
.venv/bin/python run_batch_experiment.py \
  --batch delay \
  --num-runs 100 \
  --pcap-parser dpkt \
  --latex
```

Output: `plots/pcap_setpoint_arrival_rel_to_meas_delay.pdf`

- Fig. 4 (frequency comparison, MITM delay ranges):

```bash
.venv/bin/python run_batch_experiment.py \
  --batch delay \
  --num-runs 100 \
  --only-plot \
  --band sd \
  --latex \
  --out-name freq_comparison_delay_mean_sd.pdf
```

Output: `plots/freq_comparison_delay_mean_sd.pdf`

- Fig. 5 (setpoint counts under drop attacks):

```bash
.venv/bin/python run_batch_experiment.py \
  --batch drop \
  --num-runs 100 \
  --pcap-parser dpkt \
  --latex
```

Output: plots/pcap_setpoint_count_to_server_drop.pdf

- Fig. 6 (frequency comparison, MITM drop probabilities):

```bash
.venv/bin/python run_batch_experiment.py \
  --batch drop \
  --num-runs 100 \
  --only-plot \
  --band sd \
  --latex \
  --out-name freq_comparison_drop_mean_sd.pdf
```

Output: `plots/freq_comparison_drop_mean_sd.pdf`

- Fig. 7 (frequency comparison, MITM modification probabilities):

```bash
.venv/bin/python run_batch_experiment.py \
  --batch mod \
  --num-runs 100 \
  --reuse-direct \
  --band sd \
  --latex \
  --out-name freq_comparison_mod_mean_sd.pdf
```

Output: `plots/freq_comparison_mod_mean_sd.pdf`


### Figures 8-9: ML model classification performance

Balanced dataset setup (recommended for classification):

The default workflow produces fewer normal runs than attack runs when combining all attack families.
To generate a balanced dataset with 900 normal runs and 900 attack runs (300 delay + 300 drop + 300 modification), run:

```bash
.venv/bin/python run_batch_experiment.py \
  --batch delay \
  --num-runs 100 \
  --normal-runs 900

.venv/bin/python run_batch_experiment.py \
  --batch drop \
  --num-runs 100 \
  --reuse-direct

.venv/bin/python run_batch_experiment.py \
  --batch mod \
  --num-runs 100 \
  --reuse-direct
```

1. Build the dataset after running all IEC-104 scenarios above.

```bash
.venv/bin/python data_preparator.py
```

Output: `data/final_data.csv`

2. Train/evaluate and generate the classification plots.

```bash
.venv/bin/python learner_classifier.py --latex
```

Outputs:
- `plots/classifier_comparison.pdf` (Fig. 8)
- `plots/confusion_matrices_by_model_1_2s.pdf` (Fig. 9)

### Figures 10-11: Real-time attack vector classification

Run the single-server simulation to generate the real-time classification plots.

```bash
.venv/bin/python pred_time_sim.py --latex
```

Outputs:
- `plots/time_granularity_distribution_sim.pdf` (Fig. 10)
- `plots/classifier_prediction_sim.pdf` (Fig. 11)

> [!WARNING]
> Results regarding the real-time classification response time (Figs. 10-11) may vary significantly based on the hardware capabilities of the machine running the simulation. The paper's results were obtained on a Raspberry Pi 3B+, which has limited processing power and adequately reflects the hardware characteristics of RTU used in real-world scenarios. Running the same simulation on a more powerful machine (e.g., a modern laptop or desktop) may yield much faster classification times.

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
- Add `--latex` to `run_batch_experiment.py` or `plotter_comparator.py` for LaTeX text rendering.
- Add `--poster` to increase plot font sizes.

Notes:
- By default the outstation waits for a start handshake command (IOA 8999). Disable with `IEC104_WAIT_START=0` or `--no-wait-start`.
- Shedding modes: `--shed-mode per_command` (default) or `--shed-mode all_at_once`.

> [!NOTE]
> Pcap plots require tcpdump permissions (or equivalent capabilities) so the captures exist under logs/.

## License
This repository is licensed under the GNU General Public License v3.0 (GPL-3.0). See the [LICENSE](LICENSE) file for details.

## Reference

If you use the code, models, or data in this repository in your research, please cite:

Tohid Behdadnia, Davide Savarro, Roberta Terruggia, and Geert Deconinck, "Edge-ready anomaly detection using cyber-physical consistency assessment in power systems: A co-simulation-based approach," *International Journal of Critical Infrastructure Protection*, vol. 55, article 100911, 2026. [https://doi.org/10.1016/j.ijcip.2026.100911](https://doi.org/10.1016/j.ijcip.2026.100911)

### BibTeX

```bibtex
@article{BEHDADNIA2026100911,
  title = {Edge-ready anomaly detection using cyber-physical consistency assessment in power systems: A co-simulation-based approach},
  author = {Tohid Behdadnia and Davide Savarro and Roberta Terruggia and Geert Deconinck},
  journal = {International Journal of Critical Infrastructure Protection},
  volume = {55},
  pages = {100911},
  year = {2026},
  issn = {1874-5482},
  doi = {10.1016/j.ijcip.2026.100911},
  url = {https://www.sciencedirect.com/science/article/pii/S1874548226000831},
  keywords = {Cyber-physical power systems, Anomaly detection, Edge inference, Co-simulation, IEC 60870-5-104}
}
```

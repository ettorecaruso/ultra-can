# Ultra-CAN: a lightweight blind edge-AI receiver for ISAC in the Internet of Drones

Reference implementation and full experimental pipeline of the paper

> **Ultra-CAN: A Lightweight Blind Edge AI Receiver for Integrated Sensing and Chaos-Based Communications in Internet of Drones**

One ultra-light neural receiver supports a dual-mode ISAC link. In telemetry mode it decodes
a peer's chaos-based burst blindly from the received I/Q. In radar mode the node transmits its
own burst, regenerates the chaotic reference locally from the seed it drew for that burst, and
reads the delay of the dominant echo after self-interference cancellation. The two modes are
time-multiplexed and share the same backbone, the same waveform family and one forward pass per
burst.

This repository contains the channel/scene simulator, the receivers, the training and
evaluation code, the scripts that produce every figure and table of the paper, and the exact
result tree used for the manuscript.

## Layout

```
configs/                 YAML configuration (base model/data, channel variants, experiment profiles)
src/
  data/                  chaotic maps, channel models (geometry, 3GPP TDL, two-ray), dataset generation
  models/                Ultra-CAN (Conv1D attention pool) and Ultra-CAN-QKV, baselines, classical receivers
  training/              Trainer, losses (communication + ranging)
  evaluation/            metrics, ranging utilities, statistics
  experiments/           runner.py (8 experiments) + standalone experiments
  visualization/         attention and layer-activity tools
scripts/
  data/                  dataset generation shell entry points
  figures/               scripts that build the paper figures
  tables/                one script per paper table
  diagnostics/           model-free controls (label observability, scene report, LPI)
  tools/                 latency bench, jamming-aware runner, reproduction probes,
                         TFLite export, int8 parity and MCU log capture
  style/                 shared Matplotlib style
firmware/esp32/          int8 TensorFlow Lite Micro firmware benchmarked on an ESP32-S3
notebooks/               Colab/local orchestration notebooks (see below)
results/full/            the result tree of the paper (CSV, checkpoints, provenance)
figures/                 the PDFs referenced by the paper
```

## Requirements

Python 3.10+ with TensorFlow 2.13+.

```bash
python -m pip install -r requirements.txt
```

## Reproducing the paper

Everything is driven from the repository root. The two supported entry points are the
notebooks (recommended, works on Colab) and the shell/Python scripts (any machine).

### 1. Datasets

```bash
bash scripts/data/generate_all_datasets.sh
```

Datasets are content-addressed under `data/raw/<hash>/`; reruns reuse them and the hash
depends on the channel/scene configuration, so it changes only when the channel changes.

### 2. Experiments

```bash
python src/experiments/runner.py --experiments all --mode full
```

`runner.py` accepts `--experiments` (comma separated, or `all`), `--mode {fast,full}`,
`--model`, `--scenario`, `--channel`, `--no-regen`. The registered experiments are:

| Experiment | What it produces |
|---|---|
| `ber_vs_snr` | the four received-BER scenarios (single echo, multi-echo, reduced Doppler, peers) and their checkpoints |
| `classical_receivers` | DCSK correlator, matched filter, energy detector on the noise-free echo-only link |
| `jamming` | CW / barrage / partial-band sweeps over the JSR grid |
| `jamming_interpretability` | per-layer cosine retention and the decision margin |
| `frequency_agility` | hopping versus dwell, JSR and jammer reaction |
| `channel_generalization` | the TDL/two-ray cross-channel benchmark, with and without in-domain retraining |
| `peer_estimation` | peer-aware ranging outcomes and the fail-safe trade-off |
| `final_report` | weight table and aggregated report |

Additional experiments have their own entry points:

| Experiment | Command |
|---|---|
| Unified 7-way benchmark | `python src/experiments/unified_benchmark.py --scenario k3_doppler_full` |
| Blind vs reference-aided (price of blindness) | `python src/experiments/blind_vs_oracle.py --scenario k3_doppler_full` |
| Ranging benchmark with per-burst dumps | `python src/experiments/ranging_benchmark.py --scenario k3_doppler_full` |
| Ambiguous pile (negative control) | `python src/experiments/ambiguity.py --scenario k3_ambiguity_mix` |
| Velocity from a track of bursts | `python src/experiments/velocity_track.py` (model-free profile; add `--range-source head --arch conv1d` for the learned head) |
| 5G-NR OFDM-ISAC baseline | `python src/experiments/ofdm_isac.py --scenario k3_doppler_full` |
| On-board footprint and single-burst latency | `python src/experiments/board_footprint.py` |
| SIC residual sensitivity (evaluation only) | `python scripts/diagnostics/sic_residual_sensitivity.py` |
| 200k-burst sensing evaluation (CPU) | `python scripts/diagnostics/peer_estimation_200k.py` |

### 3. Figures and tables
```bash
bash scripts/make_all.sh
```

Each script in `scripts/figures/` writes its paper PDF(s) into `figures/`; each script in
`scripts/tables/` writes the CSVs into `results/full/tables/`. Model-free controls live in
`scripts/diagnostics/` (label observability, swarm-scene report, LPI detectability).

### 4. Microcontroller deployment

The int8 firmware, the quantisation-parity checks and the measured SWaP-C row are
reproduced in four steps (an ESP32-S3 board is needed only for the last two).

```bash
# 1. export the deployable int8 models (real-burst calibration, both variants)
python scripts/tools/export_deploy_tflite.py --arch conv1d qkv

# 2. host parity checks: BER and sensing, float against int8, same 20k bursts
python scripts/tools/int8_parity.py --arch conv1d qkv --snr 5 9 13 17 21
python scripts/tools/sensing_parity.py --arch conv1d qkv --n 20000

# 3. build, flash and capture the firmware (ESP32-S3, PlatformIO)
cd firmware/esp32 && pio run -t upload && cd -
python scripts/tools/mcu_log_capture.py --seconds 45 --out results/full/mcu_bench/bench.log

# 4. turn the captured log into the paper table
python src/experiments/mcu_footprint.py --logs results/full/mcu_bench/*.log
```

Step 1 writes `models/*_deploy_int8_recal.tflite` and the C arrays consumed by the
firmware; step 2 must be repeated whenever the models are re-exported, because a
noise-calibrated representative dataset silently destroys the receiver (see the note in
`firmware/esp32/README.md`). Step 4 regenerates
`results/full/tables/mcu_footprint.csv`, which is the source of `tab:swapc` and
`tab:rates`.

### 5. Notebooks

`notebooks/` mirrors the paper section by section and can be run end to end on Colab or
locally:

- `00_Setup_and_Datasets.ipynb` — environment check and dataset generation.
- `01_Communication_Benchmark.ipynb` — `ber_vs_snr`, `classical_receivers`, unified benchmark, BER figures.
- `02_Sensing_Ranging_and_Velocity.ipynb` — ranging benchmark, ambiguity control, velocity track, OFDM-ISAC baseline, observability diagnostics.
- `03_Cross_Channel_and_Jamming.ipynb` — cross-channel generalization, jamming and interpretability, frequency agility, LPI.
- `04_Report_Figures_and_Tables.ipynb` — final report, on-board footprint, single-burst latency, all paper tables and figures.
- `orchestrator.ipynb` — prepares the environment and runs 00–04 in order.

The notebooks detect the repository root automatically, so they work both on Colab (after
cloning into `/content`) and from a local checkout.

## Results

`results/full/` contains the tree produced by the runs reported in the paper: per-scenario
`metrics.csv`, `best_model.keras` checkpoints, `config_used.yaml` provenance, and the derived
tables under `results/full/tables/`. It is the input of `scripts/make_all.sh`, so the figures
and tables can be rebuilt without retraining. The 200,000-burst CPU evaluation used for the
sensing table is archived under `results/full/peer_estimation_200k/`.

## Reproducibility notes

- Training uses a fixed seed per backbone, `105,000` training symbols and the
  "at least 100 bit errors" testing criterion with online generation.
- Operating-region metrics are the pooled BER over SNR ≥ 5 dB and the minimum SNR at which
  BER ≤ 1e-4 is reached; the per-point 1σ relative uncertainty at the 100-error rule is ≈10%,
  so differences below that level are not claimed.
- Latency numbers are a single-burst relative complexity index measured on one reference
  general-purpose CPU (single-threaded TensorFlow, static graph); they are not a measurement
  on a microcontroller.


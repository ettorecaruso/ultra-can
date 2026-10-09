# Dual-Head Ultra-CAN ISAC - Results report

This report aggregates the outputs of the runner experiments and mirrors the evaluation flow of the paper.

## Communication benchmark (operating region, SNR >= 5 dB)

| scenario | architecture | pooled BER | min SNR @1e-4 (dB) | best BER |
|---|---|---|---|---|
| k1_doppler_full | Ultra-CAN (Conv1D) | 4.8757e-04 | n/a | 2.4400e-04 |
| k1_doppler_full | Ultra-CAN (QKV) | 3.3344e-04 | n/a | 1.8667e-04 |
| k1_doppler_full | LSTM-OFDM-DCSK | 5.3398e-04 | n/a | 3.2250e-04 |
| k1_doppler_full | MC-DLCSK | 5.4259e-04 | n/a | 3.5667e-04 |
| k3_doppler_full | Ultra-CAN (Conv1D) | 0.0025 | n/a | 0.0019 |
| k3_doppler_full | Ultra-CAN (QKV) | 0.0013 | n/a | 6.6000e-04 |
| k3_doppler_full | LSTM-OFDM-DCSK | 0.0025 | n/a | 0.0020 |
| k3_doppler_full | MC-DLCSK | 0.0029 | n/a | 0.0023 |
| k3_doppler_limited | Ultra-CAN (Conv1D) | 0.0025 | n/a | 0.0018 |
| k3_doppler_limited | Ultra-CAN (QKV) | 0.0011 | n/a | 5.2000e-04 |
| k3_doppler_limited | LSTM-OFDM-DCSK | 0.0024 | n/a | 0.0019 |
| k3_doppler_limited | MC-DLCSK | 0.0031 | n/a | 0.0025 |

## Sensing: delay estimation (single-echo scenario)

| architecture | corr(tau) @top SNR | RMSE tau (samples) |
|---|---|---|
| Ultra-CAN (Conv1D) | 1.000 | 0.08 |
| Ultra-CAN (QKV) | 1.000 | 0.07 |
| LSTM-OFDM-DCSK | 1.000 | 0.07 |
| MC-DLCSK | 1.000 | 0.07 |

## Blind statistical reference receiver

| scenario | mean BER (SNR >= 5 dB) |
|---|---|
| k1_doppler_full | 0.3690 |
| k3_doppler_full | 0.3965 |
| k3_doppler_limited | 0.3965 |

## Jamming robustness and interpretability

| architecture | clean BER | BER @JSR=-2 dB (barrage) |
|---|---|---|
| Ultra-CAN (Conv1D) | 0.0048 | 0.0225 |
| Ultra-CAN (QKV) | 0.0042 | 0.0224 |
| LSTM-OFDM-DCSK | 0.0043 | 0.0212 |
| MC-DLCSK | 0.0047 | 0.0211 |

## Jamming-aware training control (QKV receiver)

| jammer | clean-trained BER @JSR=-2 dB | jamming-aware BER @JSR=-2 dB |
|---|---|---|

## Memory footprint (SWaP-C)

| architecture | parameters | footprint (KiB, float32) |
|---|---|---|
| conv1d | 26596 | 103.9 |
| qkv | 43171 | 168.6 |
| lstm | 45987 | 179.6 |
| mc_dlsk | 44755 | 174.8 |

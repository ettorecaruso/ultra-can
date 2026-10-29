from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

MIN_DB = 5.0
TARGET = 1e-4
SCENARIOS = ['k1_doppler_full', 'k3_doppler_full', 'k3_doppler_limited']
ARCHS = ['conv1d', 'qkv', 'lstm', 'mc_dlsk']
ARCH_NAMES = {'conv1d': 'Ultra-CAN (Conv1D)', 'qkv': 'Ultra-CAN (QKV)',
              'lstm': 'LSTM-OFDM-DCSK', 'mc_dlsk': 'MC-DLCSK'}

def _fmt(value: Any, digits: int = 4) -> str:
    try:
        if value is None or str(value) in ('nan', 'None'):
            return 'n/a'
        return f'{float(value):.{digits}e}' if abs(float(value)) < 1e-3 else f'{float(value):.{digits}f}'
    except (TypeError, ValueError):
        return str(value)

def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def _comm_section(results: Path) -> List[str]:
    lines = ['## Communication benchmark (operating region, SNR >= 5 dB)', '',
             '| scenario | architecture | pooled BER | min SNR @1e-4 (dB) | best BER |',
             '|---|---|---|---|---|']
    for scenario in SCENARIOS:
        for arch in ARCHS:
            df = _load_csv(results / 'ber_vs_snr' / scenario / arch / 'metrics.csv')
            if df is None:
                continue
            reg = df[df['snr_db'] >= MIN_DB]
            hit = reg[reg['ber'] <= TARGET]
            pooled = float(reg['ber'].mean()) if len(reg) else float('nan')
            min_snr = float(hit['snr_db'].min()) if len(hit) else float('nan')
            best = float(reg['ber'].min()) if len(reg) else float('nan')
            lines.append(f'| {scenario} | {ARCH_NAMES[arch]} | {_fmt(pooled)} | '
                         f'{_fmt(min_snr, 1)} | {_fmt(best)} |')
    return lines + ['']

def _sensing_section(results: Path) -> List[str]:
    lines = ['## Sensing: delay estimation (single-echo scenario)', '',
             '| architecture | corr(tau) @top SNR | RMSE tau (samples) |', '|---|---|---|']
    for arch in ARCHS:
        df = _load_csv(results / 'ber_vs_snr' / 'k1_doppler_full' / arch / 'metrics.csv')
        if df is None or not len(df):
            continue
        top = df[df['snr_db'] == df['snr_db'].max()]
        corr = float(top['corr_tau'].iloc[0])
        rmse = float(top['mse_tau'].iloc[0]) ** 0.5
        lines.append(f'| {ARCH_NAMES[arch]} | {_fmt(corr, 3)} | {_fmt(rmse, 2)} |')
    return lines + ['']

def _blind_section(results: Path) -> List[str]:
    lines = ['## Blind statistical reference receiver', '', '| scenario | mean BER (SNR >= 5 dB) |', '|---|---|']
    for scenario in SCENARIOS:
        df = _load_csv(results / 'ber_vs_snr' / scenario / 'blind_stat' / 'metrics.csv')
        if df is None:
            continue
        reg = df[df['snr_db'] >= MIN_DB]
        lines.append(f'| {scenario} | {_fmt(reg["ber"].mean(), 4)} |')
    return lines + ['']

def _jamming_section(results: Path) -> List[str]:
    lines = ['## Jamming robustness and interpretability', '',
             '| architecture | clean BER | BER @JSR=-2 dB (barrage) |', '|---|---|---|']
    for arch in ARCHS:
        df = _load_csv(results / 'jamming_interpretability' / arch / 'conditions.csv')
        if df is None:
            continue
        clean = df[df['jammer'] == 'clean']
        jm2 = df[(df['jammer'] == 'barrage') & (df['jsr_db'] == -2.0)]
        cb = float(clean['ber'].iloc[0]) if len(clean) else float('nan')
        b2 = float(jm2['ber'].iloc[0]) if len(jm2) else float('nan')
        lines.append(f'| {ARCH_NAMES[arch]} | {_fmt(cb, 4)} | {_fmt(b2, 4)} |')
    return lines + ['']

def _jamming_aware_section(results: Path) -> List[str]:
    lines = ['## Jamming-aware training control (QKV receiver)', '',
             '| jammer | clean-trained BER @JSR=-2 dB | jamming-aware BER @JSR=-2 dB |', '|---|---|---|']
    clean = _load_csv(results / 'jamming_interpretability' / 'qkv' / 'conditions.csv')
    aware = _load_csv(results / 'jamming_interpretability' / 'jamming_aware_training' / 'qkv'
                      / 'conditions.csv')
    if clean is not None and aware is not None:
        for jammer in ['cw', 'barrage', 'partial_band']:
            a = clean[(clean['jammer'] == jammer) & (clean['jsr_db'] == -2.0)]
            b = aware[(aware['jammer'] == jammer) & (aware['jsr_db'] == -2.0)]
            if len(a) and len(b):
                lines.append(f'| {jammer} | {_fmt(a["ber"].iloc[0], 4)} | {_fmt(b["ber"].iloc[0], 4)} |')
    return lines + ['']

def _swapc_section(models: Optional[Dict[str, Any]]) -> List[str]:
    lines = ['## Memory footprint (SWaP-C)', '', '| architecture | parameters | footprint (KiB, float32) |', '|---|---|---|']
    if models:
        for name, model in models.items():
            if model is None:
                continue
            params = int(model.count_params())
            lines.append(f'| {name} | {params} | {params * 4 / 1024:.1f} |')
    return lines + ['']

def generate_final_report(results_root: Path, output_path: Path,
                          models: Optional[Dict[str, Any]] = None) -> Path:
    results = Path(results_root)
    parts = ['# Ultra-CAN ISAC - Results report', '',
             'This report aggregates the outputs of the runner experiments and '
             'mirrors the evaluation flow of the paper.', '']
    parts += _comm_section(results)
    parts += _sensing_section(results)
    parts += _blind_section(results)
    parts += _jamming_section(results)
    parts += _jamming_aware_section(results)
    parts += _swapc_section(models)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(parts))
    print('final report written to', output_path)
    return output_path

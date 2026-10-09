
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / 'results' / 'full'
MIN_DB = 5.0
TARGET = 1e-4
SCENARIOS = ['k1_doppler_full', 'k3_doppler_full', 'k3_doppler_limited']
ARCHS = ['conv1d', 'qkv', 'lstm', 'mc_dlsk']

def row_for(scenario: str, arch: str) -> dict:
    path = RESULTS / 'ber_vs_snr' / scenario / arch / 'metrics.csv'
    if not path.exists():
        return {'scenario': scenario, 'arch': arch}
    df = pd.read_csv(path)
    reg = df[df['snr_db'] >= MIN_DB]
    hit = reg[reg['ber'] <= TARGET]
    return {'scenario': scenario, 'arch': arch,
            'pooled_ber': float(reg['ber'].mean()) if len(reg) else float('nan'),
            'min_snr_1e-4_db': float(hit['snr_db'].min()) if len(hit) else float('nan'),
            'best_ber': float(reg['ber'].min()) if len(reg) else float('nan')}

def main() -> None:
    rows = [row_for(s, a) for s in SCENARIOS for a in ARCHS]
    df = pd.DataFrame(rows).sort_values(['scenario', 'arch']).reset_index(drop=True)
    out = RESULTS / 'ber_vs_snr' / 'operating_region_table.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print('saved', out)

if __name__ == '__main__':
    main()

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / 'results' / 'full'
OUT = REPO / 'figures' / 'jamming_layer_retention.pdf'
MIRROR = REPO / 'results' / 'figures' / 'jamming_layer_retention.pdf'

BAR_COLOR = '#4c72b0'
PANELS = [
    ('conv1d', 'Ultra-CAN (Conv1D)',
     [('conv1', 'conv1_cos'), ('conv2', 'conv2_cos'),
      ('attn\npooling', 'attn_pool_cos')]),
    ('qkv', 'Ultra-CAN (QKV)',
     [('conv1', 'conv1_cos'), ('conv2', 'conv2_cos'),
      ('QKV\nattention', 'qkv_attention_cos')]),
    ('lstm', 'LSTM-OFDM-DCSK',
     [('LSTM\nstate', 'lstm_1_cos'), ('projection', 'lstm_proj_cos')]),
    ('mc_dlsk', 'MC-DLCSK',
     [('BiLSTM', 'mc_bilstm_1_cos'), ('projection', 'mc_proj_cos')]),
]

fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.6), sharey=True)
for ax, (key, title, bars) in zip(axes, PANELS):
    path = RESULTS / 'jamming_interpretability' / key / 'conditions.csv'
    df = pd.read_csv(path)
    row = df[(df.jammer == 'barrage') & (df.jsr_db == 10.0)].iloc[0]
    names = [b[0] for b in bars]
    vals = [float(row[b[1]]) for b in bars]
    ax.bar(range(len(vals)), vals, color=BAR_COLOR, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=10,
                zorder=4, bbox=dict(boxstyle='square,pad=0.10', fc='white',
                                    ec='none', alpha=0.85))
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_xlim(-0.6, len(vals) - 0.4)
    ax.set_ylim(0, 1.15)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=0.3)
    ax.set_title(title, fontsize=12)
    ax.tick_params(labelsize=10)
fig.suptitle('Layer-wise activation retention, barrage jamming at JSR +10 dB '
             '(cosine similarity, 1 = intact)',
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.90])
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
print('saved', OUT)
if MIRROR.parent.exists():
    fig.savefig(MIRROR)
    print('saved', MIRROR)
plt.close(fig)


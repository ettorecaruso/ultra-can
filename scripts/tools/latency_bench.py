import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import src.models.heads
import src.models.ultra_can
import src.models.ultra_can_qkv
import src.models.baselines
from src.models import ultra_can_qkv as qkv_module

BASE = REPO / 'results' / 'full' / 'ber_vs_snr' / 'k1_doppler_full'
MODELS = [
    ('Ultra-CAN (Conv1D)', 'conv1d'),
    ('Ultra-CAN-QKV', 'qkv'),
    ('LSTM-OFDM-DCSK', 'lstm'),
    ('MC-DLCSK', 'mc_dlsk'),
]

def load(name: str, folder: str):
    path = BASE / folder / 'best_model.keras'
    if name == 'Ultra-CAN-QKV':
        return tf.keras.models.load_model(
            path, compile=False,
            custom_objects={'_MultiHeadQKVAttention': qkv_module._MultiHeadQKVAttention})
    return tf.keras.models.load_model(path, compile=False)

def stats(vals):
    return {'mean_us': round(statistics.mean(vals), 1),
            'median_us': round(statistics.median(vals), 1),
            'min_us': round(min(vals), 1),
            'max_us': round(max(vals), 1),
            'stdev_us': round(statistics.pstdev(vals), 1)}

def make_unrolled(model, x):
    @tf.function
    def many(n):
        s = tf.constant(0.0, dtype=tf.float32)
        for _ in range(n):
            y = model(x, training=False)
            s += tf.reduce_sum(y['comm']) + tf.reduce_sum(y['sensing'])
        return s
    return many

TRIALS = 5
BATCH1_REPS = 10
BATCH32_REPS = 8

loaded, funcs = {}, {}
for name, folder in MODELS:
    model = load(name, folder)
    x1 = tf.constant(np.random.randn(1, 100, 3).astype(np.float32))
    x32 = tf.constant(np.random.randn(32, 100, 3).astype(np.float32))
    loaded[name] = model
    funcs[name] = (make_unrolled(model, x1), make_unrolled(model, x32))

for name, _ in MODELS:
    f1, f32 = funcs[name]
    for _ in range(5):
        f1(20)
    for _ in range(3):
        f32(5)

t1 = {name: [] for name, _ in MODELS}
t32 = {name: [] for name, _ in MODELS}
for trial in range(TRIALS):
    order = MODELS[trial % len(MODELS):] + MODELS[:trial % len(MODELS)]
    for name, _ in order:
        f1, f32 = funcs[name]
        t0 = time.perf_counter()
        for _ in range(BATCH1_REPS):
            f1(20)
        t1[name].append((time.perf_counter() - t0) / (BATCH1_REPS * 20.0) * 1e6)
        t0 = time.perf_counter()
        for _ in range(BATCH32_REPS):
            f32(5)
        t32[name].append((time.perf_counter() - t0) / (BATCH32_REPS * 5.0 * 32.0) * 1e6)

out = {}
for name, _ in MODELS:
    row = {'batch1': stats(t1[name]), 'batch32_amortized': stats(t32[name])}
    out[name] = row
    print(name)
    print('  batch=1 (static unroll) us:', row['batch1'])
    print('  batch=32 amortised/burst us:', row['batch32_amortized'])
    del loaded[name]

dest = (Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / 'results' / 'full') \
    / 'final_report' / 'latency_results.json'
dest.parent.mkdir(parents=True, exist_ok=True)
with open(dest, 'w') as fp:
    json.dump(out, fp, indent=2)
print('saved', dest)

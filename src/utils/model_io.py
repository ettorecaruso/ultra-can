
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import tensorflow as tf

logger = logging.getLogger(__name__)

def load_model(model_path: Union[str, Path]) -> tf.keras.Model:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    from src.models.heads import (_attention_pool, _dca_mf_profile,
                                  _position_feature, _slice_received)
    from src.models.ultra_can_qkv import _MultiHeadQKVAttention
    from src.training.losses import comm_ce_loss, mse_sensing_loss

    logger.debug("Loading model from %s with custom objects", model_path)
    return tf.keras.models.load_model(
        str(model_path),
        custom_objects={
            "_MultiHeadQKVAttention": _MultiHeadQKVAttention,
            "_attention_pool": _attention_pool,
            "_position_feature": _position_feature,
            "_slice_received": _slice_received,
            "_dca_mf_profile": _dca_mf_profile,
            "comm_ce_loss": comm_ce_loss,
            "mse_sensing_loss": mse_sensing_loss,
        },
        compile=False,
    )

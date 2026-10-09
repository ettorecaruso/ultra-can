
from __future__ import annotations

import json
import logging
import os
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import tensorflow as tf

from src.training.losses import (
    comm_ce_loss,
    comm_ce_loss_factory,
    mse_sensing_loss_factory,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

class Trainer:

    def __init__(
        self,
        config: Dict[str, Any],
        model: tf.keras.Model,
        train_ds: tf.data.Dataset,
        val_ds: Optional[tf.data.Dataset] = None,
    ) -> None:
        
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dict, got: {type(config).__name__}")
        if not isinstance(model, tf.keras.Model):
            raise TypeError(f"model must be a tf.keras.Model, got: {type(model).__name__}")
        if not isinstance(train_ds, tf.data.Dataset):
            raise TypeError(f"train_ds must be a tf.data.Dataset, got: {type(train_ds).__name__}")
        if val_ds is not None and not isinstance(val_ds, tf.data.Dataset):
            raise TypeError(f"val_ds must be a tf.data.Dataset or None, got: {type(val_ds).__name__}")

        self.config = config
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds

        self.training_cfg = config.get("training")
        if not isinstance(self.training_cfg, dict):
            raise ValueError("config['training'] missing or not a dict")

        self.lambda_mse = self.training_cfg.get("lambda_mse")
        if self.lambda_mse is None:
            raise ValueError("missing 'training.lambda_mse' key in config")
        if not isinstance(self.lambda_mse, (int, float)):
            raise ValueError(f"training.lambda_mse must be a finite number, got: {self.lambda_mse!r}")
        self.lambda_mse = float(self.lambda_mse)
        if not math.isfinite(self.lambda_mse):
            raise ValueError(f"training.lambda_mse must be a finite number, got: {self.lambda_mse!r}")

        self.sensing_component_weights = None
        self._validate_sensing_component_weights()

        logger.info("Trainer initialized: lambda_mse = %s", self.lambda_mse)

        try:
            if hasattr(train_ds, "cardinality") and train_ds.cardinality() == 0:
                raise ValueError("train_ds is empty (cardinality 0)")
            for _ in train_ds.take(1):
                break
            else:
                raise ValueError("train_ds is empty (no elements)")
        except Exception as e:
            raise ValueError(f"train_ds invalid or empty: {e}")

        if val_ds is not None:
            try:
                for _ in val_ds.take(1):
                    break
                else:
                    logger.warning("val_ds is empty: early stopping and validation will be disabled")
                    self.val_ds = None
            except Exception:
                logger.warning("val_ds is not iterable or empty: early stopping will be disabled")
                self.val_ds = None
        else:
            logger.info("val_ds not provided: early stopping and validation disabled")

    def _validate_sensing_component_weights(self) -> None:
        
        weights = self.training_cfg.get("sensing_component_weights")
        if weights is None:
            return
        if not isinstance(weights, dict):
            raise ValueError(
                "config['training']['sensing_component_weights'] must be a dict "
                "keyed by the sensing outputs"
            )
        from src.models.heads import sensing_output_units

        names = ("delay", "doppler")[: sensing_output_units(self.config)]
        values: List[float] = []
        for name in names:
            value = weights.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"training.sensing_component_weights.{name} must be a number, "
                    f"got: {value!r}"
                )
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(
                    f"training.sensing_component_weights.{name} must be finite "
                    f"and >= 0, got: {value!r}"
                )
            values.append(float(value))
        if all(value <= 0.0 for value in values):
            raise ValueError("sensing_component_weights must not be all zero")
        self.sensing_component_weights = values
        logger.info("Sensing component weights: %s", self.sensing_component_weights)

    def _validate_loss_weights(self) -> None:
        
        loss_weights = self.training_cfg.get("loss_weights")
        if not isinstance(loss_weights, dict):
            raise ValueError("config['training']['loss_weights'] missing or not a dict")

        comm_weight = loss_weights.get("comm")
        sensing_weight = loss_weights.get("sensing")

        if comm_weight != 1.0:
            logger.warning(
                "loss_weights.comm = %s != 1.0, forced to 1.0 (Eq. (14))",
                comm_weight,
            )
            loss_weights["comm"] = 1.0

        if sensing_weight != self.lambda_mse:
            logger.warning(
                "loss_weights.sensing = %s != lambda_mse = %s, aligning to lambda_mse",
                sensing_weight,
                self.lambda_mse,
            )
            loss_weights["sensing"] = self.lambda_mse

        self.training_cfg["loss_weights"] = loss_weights
        logger.debug("loss_weights aligned: comm=%.1f, sensing=%.3f",
                     loss_weights["comm"], loss_weights["sensing"])

    def _build_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        
        callbacks: List[tf.keras.callbacks.Callback] = []

        if self.val_ds is not None:
            monitor_metric = str(self.training_cfg.get(
                "early_stopping_monitor", "val_loss"
            ))
            if monitor_metric not in ("val_loss", "val_comm_loss"):
                raise ValueError(
                    f"training.early_stopping_monitor must be 'val_loss' or "
                    f"'val_comm_loss', got: {monitor_metric!r}"
                )
        else:
            monitor_metric = "loss"

        patience = self.training_cfg.get("early_stopping_patience", 0)
        if self.val_ds is not None and patience > 0:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor=monitor_metric,
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            )
            callbacks.append(early_stopping)
            logger.debug(
                "EarlyStopping configured with patience=%d (monitor=%s)",
                patience, monitor_metric,
            )
        else:
            logger.info("EarlyStopping disabled (val_ds missing or patience=%d)", patience)

        run_output_dir = (self.config.get("general") or {}).get("run_output_dir")
        if run_output_dir:
            checkpoint_path = str(Path(str(run_output_dir)) / "logs" / "best_weights.weights.h5")
        else:
            checkpoint_path = self.training_cfg.get("checkpoint_path")
        if checkpoint_path and not checkpoint_path.endswith(".weights.h5"):
            checkpoint_path = str(Path(checkpoint_path).with_suffix(".weights.h5"))
        if not checkpoint_path:
            logger.warning("training.checkpoint_path not set; checkpoint will not be saved")
        else:
            ckpt_dir = Path(checkpoint_path).parent
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_best_only=True,
                save_weights_only=True,
                monitor=monitor_metric,
                mode="min",
                verbose=1,
            )
            callbacks.append(model_checkpoint)
            logger.debug("ModelCheckpoint configured on %s (monitor=%s)", checkpoint_path, monitor_metric)

        log_dir = self._get_log_dir()
        csv_path = log_dir / "history.csv"
        csv_logger = tf.keras.callbacks.CSVLogger(
            filename=str(csv_path),
            separator=",",
            append=False,
        )
        callbacks.append(csv_logger)
        logger.debug("CSVLogger configured on %s", csv_path)

        if bool(self.training_cfg.get("tensorboard", True)):
            tensorboard_dir = log_dir / "tensorboard"
            tensorboard = tf.keras.callbacks.TensorBoard(
                log_dir=str(tensorboard_dir),
                histogram_freq=1,
                write_graph=True,
                write_images=False,
            )
            callbacks.append(tensorboard)
            logger.debug("TensorBoard configured on %s", tensorboard_dir)

        lr_schedule = str(self.training_cfg.get("lr_schedule", "none")).lower()
        if lr_schedule == "cosine":
            epochs_total = max(1, int(self.training_cfg.get("epochs", 5)))
            lr_max = float(self.training_cfg.get("learning_rate", 0.001))
            lr_min = lr_max * 0.1
            warmup_epochs = max(1, int(epochs_total * 0.1))

            def _cosine_lr(epoch: int, current_lr: float) -> float:
                del current_lr
                if epoch < warmup_epochs:
                    return lr_max * (epoch + 1) / warmup_epochs
                frac = (epoch - warmup_epochs) / max(1, epochs_total - warmup_epochs)
                return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * frac))

            callbacks.append(tf.keras.callbacks.LearningRateScheduler(_cosine_lr, verbose=0))
            logger.debug(
                "LearningRateScheduler cosine: lr_max=%.5f, lr_min=%.5f, warmup=%d epochs",
                lr_max, lr_min, warmup_epochs,
            )

        logger.info("Callbacks configured: %d", len(callbacks))
        return callbacks

    def restore_best_weights(self) -> bool:
        
        run_output_dir = (self.config.get("general") or {}).get("run_output_dir")
        if run_output_dir:
            ckpt = Path(str(run_output_dir)) / "logs" / "best_weights.weights.h5"
        else:
            ckpt_raw = self.training_cfg.get("checkpoint_path")
            ckpt = Path(str(ckpt_raw)).with_suffix(".weights.h5") if ckpt_raw else None
        if ckpt is None:
            logger.warning(
                "restore_best_weights: checkpoint not configured "
                "(missing general.run_output_dir and training.checkpoint_path); "
                "model stays at the last epoch"
            )
            return False
        if not ckpt.is_file():
            logger.warning(
                "restore_best_weights: checkpoint not found in %s; the "
                "model stays at the last epoch (ModelCheckpoint disabled?)",
                ckpt,
            )
            return False
        self.model.load_weights(str(ckpt))
        logger.info("Best weights reloaded from %s before evaluation/saving", ckpt)
        return True

    def _get_log_dir(self) -> Path:
        
        general = self.config.get("general") or {}
        run_output_dir = general.get("run_output_dir")
        if run_output_dir:
            log_dir = Path(str(run_output_dir)) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            return log_dir

        experiment_name = general.get("experiment_name", "ultra_can_isac")
        repo_root = Path(__file__).resolve().parents[2]
        log_dir = repo_root / "results" / experiment_name / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def verify_finite_gradients(self) -> None:
        
        if self.train_ds is None:
            logger.warning("verify_finite_gradients: train_ds is None, cannot verify")
            return

        try:
            for features, labels in self.train_ds.take(1):
                batch_features, batch_labels = features, labels
                break
            else:
                logger.warning("verify_finite_gradients: train_ds is empty, skipping check")
                return
        except Exception as e:
            raise RuntimeError(f"Unable to fetch a batch from train_ds: {e}")

        sensing_loss_fn = mse_sensing_loss_factory(
            float(self.training_cfg.get("sensing_range_penalty", 0.0)),
            self.sensing_component_weights,
        )
        with tf.GradientTape() as tape:
            predictions = self.model(batch_features, training=True)
            loss = (
                comm_ce_loss(batch_labels["comm"], predictions["comm"])
                + self.lambda_mse
                * sensing_loss_fn(batch_labels["sensing"], predictions["sensing"])
            )

        gradients = tape.gradient(loss, self.model.trainable_variables)

        if gradients is None:
            raise RuntimeError("tape.gradient returned None (no gradient computed)")

        for idx, grad in enumerate(gradients):
            if grad is None:
                continue
            try:
                tf.debugging.assert_all_finite(grad, f"gradient {idx} contains NaN/Inf")
            except tf.errors.InvalidArgumentError as e:
                logger.error("Non-finite gradient detected for parameter %d", idx)
                raise ValueError(f"Non-finite gradient for parameter {idx}: {e}") from e

        logger.debug("Finite gradients verified (%d gradients)", len(gradients))

    def save_artifacts(self, history: tf.keras.callbacks.History) -> None:
        
        if not isinstance(history, tf.keras.callbacks.History):
            raise TypeError(f"history must be a tf.keras.callbacks.History, got: {type(history).__name__}")

        log_dir = self._get_log_dir()

        history_df = pd.DataFrame(history.history)
        history_path = log_dir / "history.csv"
        history_df.to_csv(history_path, index=False)
        logger.info("History saved to %s", history_path)

        metadata = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "tensorflow_version": tf.__version__,
            "config_snapshot_path": str(log_dir / "config_used.yaml"),
            "lambda_mse": self.lambda_mse,
            "sensing_range_penalty": self.training_cfg.get("sensing_range_penalty", 0.0),
            "sensing_component_weights": self.sensing_component_weights,
            "epochs": self.training_cfg.get("epochs"),
            "batch_size": self.training_cfg.get("batch_size"),
            "learning_rate": self.training_cfg.get("learning_rate"),
            "early_stopping_patience": self.training_cfg.get("early_stopping_patience", 0),
            "model_name": self.model.name if hasattr(self.model, "name") else "unknown",
            "train_batches": int(self.train_ds.cardinality().numpy()) if hasattr(self.train_ds, "cardinality") else None,
        }
        metadata_path = log_dir / "run_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Metadata saved to %s", metadata_path)

    def train(self) -> tf.keras.callbacks.History:
        
        self._validate_loss_weights()

        learning_rate = self.training_cfg.get("learning_rate", 0.001)
        optimizer_name = self.training_cfg.get("optimizer", "adam")
        if optimizer_name.lower() == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            logger.warning("Optimizer %s not supported, using Adam with lr=%f", optimizer_name, learning_rate)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        loss_weights = self.training_cfg["loss_weights"]
        label_smoothing = float(self.training_cfg.get("label_smoothing", 0.0))
        comm_loss_fn = comm_ce_loss_factory(label_smoothing) if label_smoothing > 0.0 else comm_ce_loss
        sensing_range_penalty = float(self.training_cfg.get("sensing_range_penalty", 0.0))
        sensing_loss_fn = mse_sensing_loss_factory(
            sensing_range_penalty, self.sensing_component_weights
        )
        self.model.compile(
            optimizer=optimizer,
            loss={
                "comm": comm_loss_fn,
                "sensing": sensing_loss_fn,
            },
            loss_weights=loss_weights,
        )
        logger.info(
            "Model compiled: loss_weights = %s, lr=%f, optimizer=%s, "
            "label_smoothing=%s, lr_schedule=%s, sensing_range_penalty=%s, "
            "sensing_component_weights=%s",
            loss_weights,
            learning_rate,
            optimizer_name,
            label_smoothing,
            self.training_cfg.get("lr_schedule", "none"),
            sensing_range_penalty,
            self.sensing_component_weights,
        )

        try:
            self.verify_finite_gradients()
        except Exception as e:
            logger.error("Gradient verification failed: %s", e)
            raise RuntimeError(f"Gradient verification failed, training aborted: {e}") from e

        callbacks = self._build_callbacks()

        epochs = self.training_cfg.get("epochs", 5)
        verbose = 1 if logger.isEnabledFor(logging.INFO) else 0

        logger.info("Starting training: epochs=%d, batch_size=%d",
                    epochs, self.training_cfg.get("batch_size", "N/D"))

        try:
            history = self.model.fit(
                self.train_ds,
                validation_data=self.val_ds,
                epochs=epochs,
                callbacks=callbacks,
                verbose=verbose,
            )
        except Exception as e:
            logger.error("Error during model.fit: %s", e)
            raise RuntimeError(f"Training failed: {e}") from e

        self.save_artifacts(history)

        best_val_loss = min(history.history.get("val_loss", [float("inf")]))
        best_loss = min(history.history.get("loss", [float("inf")]))
        val_loss_str = "N/D" if best_val_loss == float("inf") else f"{best_val_loss:.6f}"
        loss_str = "N/D" if best_loss == float("inf") else f"{best_loss:.6f}"
        logger.info(
            "Training completed in %d epochs. Best val_loss = %s, best loss = %s",
            len(history.history.get("loss", [])),
            val_loss_str,
            loss_str,
        )

        return history


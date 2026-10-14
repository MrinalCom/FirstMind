"""Unified MLOps Experiment Tracker for Weights & Biases and MLflow.

This module provides a unified telemetry interface for tracking Reinforcement Learning
and fine-tuning experiments across both Weights & Biases (W&B) and MLflow.
Supports metric logging, model artifact versioning, parameter registration,
and context-managed lifecycle execution.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# NOTE: Before running experiment tracking with Weights & Biases in Colab or terminal:
# Call `wandb.login(key="YOUR_WANDB_API_KEY")` or export `WANDB_API_KEY="your-api-key"`
# in your shell environment before initializing ExperimentTracker.
try:
    import wandb
except ImportError:
    wandb = None

try:
    import mlflow
except ImportError:
    mlflow = None


class ExperimentTracker:
    """Unified experiment tracking wrapper for W&B and MLflow.

    Provides synchronous or dual logging of hyperparameters, scalars, artifacts,
    and checkpoint registries for GEO policy training.

    Usage:
        >>> with ExperimentTracker(project_name="firstmind", run_name="grpo-run-1", config=cfg) as tracker:
        >>>     tracker.log_metrics({"reward": 0.85, "rank": 1}, step=10)
    """

    def __init__(
        self,
        project_name: str,
        run_name: str,
        config: Optional[Dict[str, Any]] = None,
        use_wandb: bool = True,
        use_mlflow: bool = False,
    ) -> None:
        """Initialize tracker configuration.

        Args:
            project_name: Top-level project or experiment namespace.
            run_name: Specific run identifier.
            config: Hyperparameters and environment configuration dict.
            use_wandb: Enable Weights & Biases logging (default: True).
            use_mlflow: Enable MLflow logging (default: False).
        """
        self.project_name = project_name
        self.run_name = run_name
        self.config = dict(config or {})
        self.use_wandb = use_wandb
        self.use_mlflow = use_mlflow

        self._wandb_run: Optional[Any] = None
        self._mlflow_run: Optional[Any] = None
        self._is_active: bool = False

    def start(self) -> None:
        """Initialize and start logging sessions in W&B and/or MLflow."""
        if self._is_active:
            return

        # Initialize Weights & Biases
        if self.use_wandb:
            if wandb is not None:
                try:
                    self._wandb_run = wandb.init(
                        project=self.project_name,
                        name=self.run_name,
                        config=self.config,
                        reinit=True,
                    )
                    logger.info(f"Initialized W&B run: {self.run_name} in project {self.project_name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize W&B ({e}). Continuing in offline mode.")
            else:
                logger.warning("wandb is not installed. Skipping W&B tracking.")

        # Initialize MLflow
        if self.use_mlflow:
            if mlflow is not None:
                try:
                    mlflow.set_experiment(self.project_name)
                    self._mlflow_run = mlflow.start_run(run_name=self.run_name)
                    # Flatten or filter config parameters for MLflow logging
                    flat_params = {
                        str(k): str(v)
                        for k, v in self.config.items()
                        if not isinstance(v, (dict, list))
                    }
                    if flat_params:
                        mlflow.log_params(flat_params)
                    logger.info(f"Initialized MLflow run: {self.run_name} in project {self.project_name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize MLflow ({e}). Continuing without MLflow.")
            else:
                logger.warning("mlflow is not installed. Skipping MLflow tracking.")

        self._is_active = True

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log numeric evaluation and training metrics at a given step.

        Args:
            metrics: Dictionary of metric names and numeric values.
            step: Current step / episode number.
        """
        if not self._is_active:
            self.start()

        clean_metrics = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }

        # Log to W&B
        if self.use_wandb and wandb is not None and self._wandb_run is not None:
            try:
                wandb.log(clean_metrics, step=int(step))
            except Exception as e:
                logger.debug(f"Error logging to W&B: {e}")

        # Log to MLflow
        if self.use_mlflow and mlflow is not None and self._mlflow_run is not None:
            try:
                mlflow.log_metrics(clean_metrics, step=int(step))
            except Exception as e:
                logger.debug(f"Error logging to MLflow: {e}")

    def log_artifact(self, filepath: str, artifact_name: str) -> None:
        """Upload and version a file or model directory as a tracking artifact.

        Args:
            filepath: Local filesystem path to file or folder.
            artifact_name: Logical artifact name.
        """
        if not os.path.exists(filepath):
            logger.warning(f"Artifact path '{filepath}' does not exist. Skipping artifact upload.")
            return

        if not self._is_active:
            self.start()

        # Log to W&B Artifacts
        if self.use_wandb and wandb is not None and self._wandb_run is not None:
            try:
                artifact = wandb.Artifact(name=artifact_name, type="model_checkpoint")
                if os.path.isdir(filepath):
                    artifact.add_dir(filepath)
                else:
                    artifact.add_file(filepath)
                self._wandb_run.log_artifact(artifact)
                logger.info(f"Logged artifact '{artifact_name}' to W&B.")
            except Exception as e:
                logger.warning(f"Failed to log artifact to W&B: {e}")

        # Log to MLflow Artifacts
        if self.use_mlflow and mlflow is not None and self._mlflow_run is not None:
            try:
                if os.path.isdir(filepath):
                    mlflow.log_artifacts(filepath, artifact_path=artifact_name)
                else:
                    mlflow.log_artifact(filepath, artifact_path=artifact_name)
                logger.info(f"Logged artifact '{artifact_name}' to MLflow.")
            except Exception as e:
                logger.warning(f"Failed to log artifact to MLflow: {e}")

    def log_model_info(
        self,
        model_id: str,
        adapter_path: str,
        params_count: int,
    ) -> None:
        """Register model architectural and checkpoint metadata.

        Args:
            model_id: Base foundation model identifier.
            adapter_path: Path to LoRA adapter safetensors.
            params_count: Total trainable parameters count.
        """
        info = {
            "model_id": model_id,
            "adapter_path": adapter_path,
            "trainable_parameters": int(params_count),
        }
        if self.use_wandb and wandb is not None and self._wandb_run is not None:
            self._wandb_run.summary.update(info)

        if self.use_mlflow and mlflow is not None and self._mlflow_run is not None:
            flat_info = {k: str(v) for k, v in info.items()}
            mlflow.log_params(flat_info)

    def finish(self) -> None:
        """Gracefully close and sync active W&B and MLflow runs."""
        if not self._is_active:
            return

        if self.use_wandb and wandb is not None and self._wandb_run is not None:
            try:
                wandb.finish()
                logger.info("W&B session successfully closed.")
            except Exception as e:
                logger.debug(f"Error closing W&B session: {e}")
            self._wandb_run = None

        if self.use_mlflow and mlflow is not None and self._mlflow_run is not None:
            try:
                mlflow.end_run()
                logger.info("MLflow session successfully closed.")
            except Exception as e:
                logger.debug(f"Error ending MLflow run: {e}")
            self._mlflow_run = None

        self._is_active = False

    def __enter__(self) -> "ExperimentTracker":
        """Enter context manager, starting tracker session."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager, cleanly closing tracker sessions."""
        self.finish()

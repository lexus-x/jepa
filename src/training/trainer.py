"""Main training loop for VL-JEPA.

Supports:
    - Distributed Data Parallel (DDP) multi-GPU training
    - Mixed precision (AMP bf16)
    - Gradient clipping
    - Checkpoint save/load/resume
    - W&B logging
    - Validation loop
    - Cosine LR scheduler with warmup
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler
from torch import Tensor

from ..encoder.vjepa2_wrapper import VJEPA2Encoder
from ..encoder.language_adapter import LanguageAdapter
from ..flow.velocity_field import VelocityField
from ..flow.geodesic_flow import GeodesicFlowMatcher
from .losses import (
    FlowMatchingLoss,
    TimestepSampler,
    geodesic_distance_metric,
    compute_action_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Optimization
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    total_steps: int = 100_000
    grad_clip_norm: float = 1.0

    # Batch / data
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"

    # Logging
    log_interval: int = 50
    eval_interval: int = 1000
    save_interval: int = 5000
    use_wandb: bool = True
    wandb_project: str = "vl-jepa"
    wandb_run_name: Optional[str] = None

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None

    # Model
    freeze_encoder: bool = True
    unfreeze_last_n: int = 2
    visual_dim: int = 1024
    proprio_dim: int = 7
    hidden_dim: int = 512
    velocity_layers: int = 6

    # Flow matching
    sigma_min: float = 0.001
    sigma_max: float = 0.5
    beta_alpha: float = 1.5
    beta_beta: float = 1.0

    # DDP
    local_rank: int = -1
    world_size: int = 1


class VLJEPATrainer:
    """Main trainer for VL-JEPA.

    Orchestrates:
        1. Model construction (encoder, language adapter, velocity field, flow matcher)
        2. Optimizer & scheduler setup
        3. Training loop with AMP + gradient clipping
        4. Validation loop
        5. Checkpoint management
        6. W&B logging

    Args:
        config: Training configuration.
        train_dataset: Training dataset.
        val_dataset: Optional validation dataset.
    """

    def __init__(
        self,
        config: TrainingConfig,
        train_dataset: Optional[Any] = None,
        val_dataset: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # DDP setup
        self.distributed = config.world_size > 1
        self.local_rank = config.local_rank
        self.is_main = self.local_rank in (-1, 0)

        if self.distributed:
            self._setup_ddp()

        self.device = self._get_device()

        # Build models
        self._build_models()

        # Build optimizer & scheduler
        self._build_optimizer()

        # Loss & flow matcher
        self.loss_fn = FlowMatchingLoss().to(self.device)
        self.flow_matcher = GeodesicFlowMatcher(
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
            beta_alpha=config.beta_alpha,
            beta_beta=config.beta_beta,
        ).to(self.device)

        # Timestep sampler
        self.timestep_sampler = TimestepSampler(
            alpha=config.beta_alpha,
            beta=config.beta_beta,
            device=self.device,
        )

        # AMP
        self.use_amp = config.use_amp and torch.cuda.is_available()
        self.amp_dtype = getattr(torch, config.amp_dtype, torch.bfloat16)
        self.scaler = GradScaler(enabled=self.use_amp and self.amp_dtype == torch.float16)

        # W&B
        self._wandb_run = None
        if config.use_wandb and self.is_main:
            self._setup_wandb()

        # State
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        # Resume
        if config.resume_from:
            self._load_checkpoint(config.resume_from)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_ddp(self) -> None:
        """Initialize distributed training."""
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(self.local_rank)

    def _get_device(self) -> torch.device:
        """Get the appropriate device."""
        if torch.cuda.is_available():
            if self.distributed:
                return torch.device(f"cuda:{self.local_rank}")
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_models(self) -> None:
        """Build encoder, language adapter, and velocity field."""
        cfg = self.config

        # Visual encoder
        self.encoder = VJEPA2Encoder(
            device=self.device,
            freeze=cfg.freeze_encoder,
            unfreeze_last_n=cfg.unfreeze_last_n,
        )

        # Language adapter
        self.language_adapter = LanguageAdapter(
            visual_dim=cfg.visual_dim,
        )

        # Velocity field
        self.velocity_field = VelocityField(
            visual_dim=cfg.visual_dim,
            proprio_dim=cfg.proprio_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.velocity_layers,
        ).to(self.device)

        # DDP wrap
        if self.distributed:
            self.velocity_field = DDP(
                self.velocity_field,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )

    def _build_optimizer(self) -> None:
        """Build optimizer and cosine LR scheduler with warmup."""
        cfg = self.config

        # Collect trainable parameters
        params = []
        params.extend(self.velocity_field.parameters())
        params.extend(self.language_adapter.parameters())
        if not cfg.freeze_encoder or cfg.unfreeze_last_n > 0:
            params.extend(
                p for p in self.encoder.parameters() if p.requires_grad
            )

        self.optimizer = torch.optim.AdamW(
            params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
        )

        # Cosine schedule with linear warmup
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_scheduler_fn=self._cosine_warmup_schedule,
        )

    def _cosine_warmup_schedule(self, step: int) -> float:
        """Cosine decay with linear warmup."""
        cfg = self.config
        if step < cfg.warmup_steps:
            return step / max(cfg.warmup_steps, 1)
        progress = (step - cfg.warmup_steps) / max(cfg.total_steps - cfg.warmup_steps, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())

    def _setup_wandb(self) -> None:
        """Initialize W&B logging."""
        try:
            import wandb
            self._wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=vars(self.config),
            )
        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")
            self.config.use_wandb = False

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> dict[str, float]:
        """Run the full training loop.

        Returns:
            metrics: Final training metrics.
        """
        cfg = self.config

        # Build data loaders
        train_loader = self._make_loader(self.train_dataset, shuffle=True)
        val_loader = self._make_loader(self.val_dataset, shuffle=False) if self.val_dataset else None

        # Set models to train mode
        self.velocity_field.train()
        self.language_adapter.train()
        if not cfg.freeze_encoder or cfg.unfreeze_last_n > 0:
            self.encoder.train()

        logger.info("Starting training at step %d", self.global_step)
        metrics: dict[str, float] = {}

        while self.global_step < cfg.total_steps:
            self.epoch += 1

            if self.distributed and hasattr(train_loader, "sampler"):
                train_loader.sampler.set_epoch(self.epoch)

            for batch in train_loader:
                if self.global_step >= cfg.total_steps:
                    break

                metrics = self._train_step(batch)
                self.global_step += 1
                self.scheduler.step()

                # Logging
                if self.global_step % cfg.log_interval == 0 and self.is_main:
                    self._log_metrics(metrics, "train")

                # Validation
                if val_loader and self.global_step % cfg.eval_interval == 0:
                    val_metrics = self._validate(val_loader)
                    if self.is_main:
                        self._log_metrics(val_metrics, "val")
                        if val_metrics["loss"] < self.best_val_loss:
                            self.best_val_loss = val_metrics["loss"]
                            self._save_checkpoint("best")

                # Checkpoint
                if self.global_step % cfg.save_interval == 0 and self.is_main:
                    self._save_checkpoint(f"step_{self.global_step}")

        # Final save
        if self.is_main:
            self._save_checkpoint("final")

        return metrics

    def _train_step(self, batch: dict) -> dict[str, float]:
        """Execute a single training step.

        Args:
            batch: Dictionary with keys:
                - "images": [B, 3, T, H, W] RGB frames
                - "instructions": list of B text strings
                - "actions": [B, 4, 4] SE(3) action poses
                - "proprioception": [B, D_proprio] proprioceptive state

        Returns:
            metrics: Dictionary of scalar metrics.
        """
        cfg = self.config
        self.optimizer.zero_grad(set_to_none=True)

        images = batch["images"].to(self.device)
        actions = batch["actions"].to(self.device)
        instructions = batch["instructions"]
        proprio = batch.get("proprioception")
        if proprio is not None:
            proprio = proprio.to(self.device)

        with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            # Encode visual features
            with torch.no_grad() if cfg.freeze_encoder and cfg.unfreeze_last_n == 0 else torch.enable_grad():
                visual_tokens = self.encoder(images)  # [B, N, D]
                visual_features = visual_tokens.mean(dim=1)  # [B, D]

            # Language conditioning
            lang_features = self.language_adapter(instructions, visual_tokens, self.device)

            # Fuse: add language to visual
            conditioning = visual_features + lang_features  # [B, D]

            # Sample timesteps
            t = self.timestep_sampler.sample(actions.shape[0])

            # Flow matching forward
            loss, pred_vel, target_vel = self.flow_matcher(
                data_poses=actions,
                velocity_field=lambda x_t, t_val, **kw: self.velocity_field(
                    x_t, t_val, conditioning, proprio
                ),
                visual_features=conditioning,
                proprioception=proprio,
                timesteps=t,
            )

        # Backward
        self.scaler.scale(loss).backward()

        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(
            self._trainable_params(), cfg.grad_clip_norm
        )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "loss": loss.item(),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> dict[str, float]:
        """Run validation loop.

        Args:
            val_loader: Validation data loader.

        Returns:
            metrics: Aggregated validation metrics.
        """
        cfg = self.config
        self.velocity_field.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            images = batch["images"].to(self.device)
            actions = batch["actions"].to(self.device)
            instructions = batch["instructions"]
            proprio = batch.get("proprioception")
            if proprio is not None:
                proprio = proprio.to(self.device)

            with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                visual_tokens = self.encoder(images)
                visual_features = visual_tokens.mean(dim=1)
                lang_features = self.language_adapter(instructions, visual_tokens, self.device)
                conditioning = visual_features + lang_features

                t = self.timestep_sampler.sample(actions.shape[0])
                loss, _, _ = self.flow_matcher(
                    data_poses=actions,
                    velocity_field=lambda x_t, t_val, **kw: self.velocity_field(
                        x_t, t_val, conditioning, proprio
                    ),
                    visual_features=conditioning,
                    proprioception=proprio,
                    timesteps=t,
                )

            total_loss += loss.item()
            num_batches += 1

        self.velocity_field.train()

        avg_loss = total_loss / max(num_batches, 1)
        return {"loss": avg_loss}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, tag: str) -> None:
        """Save training checkpoint.

        Args:
            tag: Checkpoint identifier (e.g., "best", "step_5000", "final").
        """
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"checkpoint_{tag}.pt"

        # Unwrap DDP
        velocity_state = (
            self.velocity_field.module.state_dict()
            if isinstance(self.velocity_field, DDP)
            else self.velocity_field.state_dict()
        )

        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "velocity_field": velocity_state,
            "language_adapter": self.language_adapter.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": vars(self.config),
        }

        # Optionally save encoder (if fine-tuning)
        if not self.config.freeze_encoder or self.config.unfreeze_last_n > 0:
            state["encoder"] = self.encoder.state_dict()

        torch.save(state, path)
        logger.info("Saved checkpoint: %s", path)

    def _load_checkpoint(self, path: str) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        logger.info("Loading checkpoint: %s", path)
        state = torch.load(path, map_location=self.device)

        self.global_step = state["global_step"]
        self.epoch = state["epoch"]
        self.best_val_loss = state["best_val_loss"]

        # Load model weights
        vf = (
            self.velocity_field.module
            if isinstance(self.velocity_field, DDP)
            else self.velocity_field
        )
        vf.load_state_dict(state["velocity_field"])
        self.language_adapter.load_state_dict(state["language_adapter"])

        if "encoder" in state:
            self.encoder.load_state_dict(state["encoder"])

        # Load optimizer & scheduler
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])

        logger.info("Resumed from step %d, epoch %d", self.global_step, self.epoch)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_loader(
        self,
        dataset: Optional[Any],
        shuffle: bool,
    ) -> Optional[DataLoader]:
        """Build a DataLoader with optional DDP sampler."""
        if dataset is None:
            return None

        sampler = None
        if self.distributed:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True,
        )

    def _trainable_params(self) -> list[nn.Parameter]:
        """Collect all trainable parameters."""
        params = list(self.velocity_field.parameters())
        params.extend(self.language_adapter.parameters())
        params.extend(p for p in self.encoder.parameters() if p.requires_grad)
        return params

    def _log_metrics(self, metrics: dict[str, float], prefix: str) -> None:
        """Log metrics to console and W&B."""
        step = self.global_step
        msg_parts = [f"step={step}"]
        for k, v in metrics.items():
            msg_parts.append(f"{prefix}/{k}={v:.6f}")
        logger.info(" | ".join(msg_parts))

        if self._wandb_run is not None:
            import wandb
            wandb.log(
                {f"{prefix}/{k}": v for k, v in metrics.items()},
                step=step,
            )

    def cleanup(self) -> None:
        """Cleanup distributed training resources."""
        if self.distributed:
            dist.destroy_process_group()
        if self._wandb_run is not None:
            self._wandb_run.finish()

"""Smoke tests: model creation, forward pass, output shapes, checkpoint save/load."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Minimal model for smoke testing (mirrors the expected architecture)
# ---------------------------------------------------------------------------


class _MiniVLAConfig:
    """Minimal config that mirrors hydra defaults."""

    def __init__(self):
        self.hidden_dim = 128
        self.num_heads = 4
        self.num_layers = 2
        self.image_size = 224
        self.patch_size = 16
        self.num_patches = (self.image_size // self.patch_size) ** 2  # 196
        self.action_dim = 7  # 6-DoF + gripper
        self.action_horizon = 4
        self.vocab_size = 32000
        self.max_text_len = 64


class _MiniEncoder(nn.Module):
    """Minimal vision encoder (patch embedding + transformer)."""

    def __init__(self, config: _MiniVLAConfig):
        super().__init__()
        self.patch_embed = nn.Linear(
            3 * config.patch_size * config.patch_size,
            config.hidden_dim,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (B, num_patches, patch_dim)
        Returns:
            cls_out: (B, hidden_dim)
        """
        B = patches.shape[0]
        x = self.patch_embed(patches)  # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, N+1, D)
        x = self.transformer(x)
        return x[:, 0]  # CLS token


class _MiniFlowHead(nn.Module):
    """Minimal flow-matching action head."""

    def __init__(self, config: _MiniVLAConfig):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.action_embed = nn.Linear(config.action_dim, config.hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        action_noisy: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, D) encoder output.
            action_noisy: (B, H, A) noisy action chunk.
            t: (B,) time in [0, 1].
        Returns:
            velocity: (B, H, A) predicted velocity.
        """
        B, H, A = action_noisy.shape
        t_emb = self.time_embed(t.unsqueeze(-1))  # (B, D)
        # Repeat features for each horizon step
        feat_expanded = features.unsqueeze(1).expand(-1, H, -1)  # (B, H, D)
        a_emb = self.action_embed(action_noisy)  # (B, H, D)
        inp = torch.cat([feat_expanded + t_emb.unsqueeze(1), a_emb], dim=-1)  # (B, H, 2D)
        return self.net(inp)


class _MiniVLA(nn.Module):
    """Minimal VLA model for smoke testing."""

    def __init__(self, config: _MiniVLAConfig):
        super().__init__()
        self.config = config
        self.encoder = _MiniEncoder(config)
        self.flow_head = _MiniFlowHead(config)

    def forward(
        self,
        patches: torch.Tensor,
        action_noisy: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        features = self.encoder(patches)
        return self.flow_head(features, action_noisy, t)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestModelCreation:
    """Model should instantiate without errors."""

    def test_config_creation(self):
        config = _MiniVLAConfig()
        assert config.hidden_dim == 128
        assert config.num_patches == 196
        assert config.action_dim == 7

    def test_model_creation(self):
        config = _MiniVLAConfig()
        model = _MiniVLA(config)
        assert isinstance(model, nn.Module)

    def test_model_parameter_count(self):
        config = _MiniVLAConfig()
        model = _MiniVLA(config)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0, "Model has no parameters"
        assert n_params < 10_000_000, f"Smoke model too large: {n_params} params"


class TestForwardPass:
    """Forward pass should produce correct output shapes."""

    def test_forward_shape(self):
        config = _MiniVLAConfig()
        model = _MiniVLA(config)
        model.eval()

        B = 4
        patches = torch.randn(B, config.num_patches, 3 * config.patch_size ** 2)
        action_noisy = torch.randn(B, config.action_horizon, config.action_dim)
        t = torch.rand(B)

        with torch.no_grad():
            velocity = model(patches, action_noisy, t)

        assert velocity.shape == (B, config.action_horizon, config.action_dim)

    def test_forward_finite(self):
        config = _MiniVLAConfig()
        model = _MiniVLA(config)
        model.eval()

        patches = torch.randn(2, config.num_patches, 3 * config.patch_size ** 2)
        action = torch.randn(2, config.action_horizon, config.action_dim)
        t = torch.rand(2)

        with torch.no_grad():
            out = model(patches, action, t)

        assert torch.isfinite(out).all(), "Forward pass produced NaN/Inf"

    def test_forward_single_sample(self):
        config = _MiniVLAConfig()
        model = _MiniVLA(config)
        model.eval()

        patches = torch.randn(1, config.num_patches, 3 * config.patch_size ** 2)
        action = torch.randn(1, config.action_horizon, config.action_dim)
        t = torch.tensor([0.5])

        with torch.no_grad():
            out = model(patches, action, t)

        assert out.shape == (1, config.action_horizon, config.action_dim)

    def test_different_times(self):
        """Model should handle different time values."""
        config = _MiniVLAConfig()
        model = _MiniVLA(config)
        model.eval()

        patches = torch.randn(2, config.num_patches, 3 * config.patch_size ** 2)
        action = torch.randn(2, config.action_horizon, config.action_dim)

        for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            t = torch.tensor([t_val, t_val])
            with torch.no_grad():
                out = model(patches, action, t)
            assert out.shape == (2, config.action_horizon, config.action_dim)


class TestCheckpointSaveLoad:
    """Checkpoint save/load should preserve model state."""

    def test_save_load_roundtrip(self):
        config = _MiniVLAConfig()
        model = _MiniVLA(config)
        model.eval()

        # Generate a reference output
        patches = torch.randn(2, config.num_patches, 3 * config.patch_size ** 2)
        action = torch.randn(2, config.action_horizon, config.action_dim)
        t = torch.tensor([0.5, 0.3])

        with torch.no_grad():
            out_before = model(patches, action, t)

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": vars(config),
            }, path)

            # Load into fresh model
            model2 = _MiniVLA(config)
            checkpoint = torch.load(path, weights_only=False)
            model2.load_state_dict(checkpoint["model_state_dict"])
            model2.eval()

            with torch.no_grad():
                out_after = model2(patches, action, t)

        torch.testing.assert_close(out_before, out_after, atol=1e-6, rtol=1e-6)

    def test_checkpoint_file_exists(self):
        config = _MiniVLAConfig()
        model = _MiniVLA(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            torch.save(model.state_dict(), path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_load_missing_key_raises(self):
        """Loading a checkpoint with missing keys should raise an error."""
        config = _MiniVLAConfig()
        model = _MiniVLA(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.pt")
            # Save only encoder weights
            torch.save({"encoder.cls_token": model.encoder.cls_token}, path)
            checkpoint = torch.load(path, weights_only=False)
            with pytest.raises(RuntimeError):
                model.load_state_dict(checkpoint, strict=True)

    def test_checkpoint_overwrite(self):
        """Saving to same path should overwrite cleanly."""
        config = _MiniVLAConfig()
        model = _MiniVLA(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            # Save twice
            torch.save(model.state_dict(), path)
            torch.save(model.state_dict(), path)
            loaded = torch.load(path, weights_only=False)
            model.load_state_dict(loaded)  # should not raise

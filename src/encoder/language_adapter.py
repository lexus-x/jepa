"""Language conditioning adapter for VL-JEPA.

Encodes natural language instructions into a conditioning vector that is
compatible with the visual token space of V-JEPA 2.

Upgraded from the original all-MiniLM-L6-v2 (22M) to support stronger
language backbones that can actually ground spatial instructions:

    - "all-mpnet-base-v2" (109M): good default, 768-dim
    - "intfloat/multilingual-e5-large-instruct" (560M): strong multilingual
    - "Qwen/Qwen2.5-1.5B" or similar: for complex spatial reasoning

Architecture:
    1. Frozen language encoder → token-level embeddings [B, S, D_lang]
    2. Cross-attention pooling: language tokens attend to visual tokens
    3. Spatial reasoning MLP that encodes relative positions
    4. Projection to visual space with residual connection
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialPositionalEncoding(nn.Module):
    """2D sinusoidal positional encoding for visual patch grid.

    Injects explicit (row, col) coordinates so the spatial reasoning
    module knows WHERE each token is in the image, enabling it to
    ground spatial language ("left of", "above", "behind").

    Args:
        dim: Output dimension (must match feature dim).
        grid_size: Spatial grid size (e.g. 16 for 256/16 patches).
        num_temporal: Number of temporal patches.
    """

    def __init__(self, dim: int, grid_size: int = 16, num_temporal: int = 1) -> None:
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size
        self.num_temporal = num_temporal

        # Project (row, col, t) coordinates → dim
        self.coord_proj = nn.Sequential(
            nn.Linear(3, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, dim),
        )
        nn.init.zeros_(self.coord_proj[-1].weight)
        nn.init.zeros_(self.coord_proj[-1].bias)

    def forward(self, visual_tokens: Tensor) -> Tensor:
        """Add spatial position encoding to visual tokens.

        Args:
            visual_tokens: [B, N, D] where N = num_temporal * grid_size^2.

        Returns:
            encoded: [B, N, D] with spatial information added.
        """
        B, N, D = visual_tokens.shape
        device, dtype = visual_tokens.device, visual_tokens.dtype
        gs = self.grid_size

        # Build coordinate grid: (t, row, col) normalized to [0, 1]
        coords = []
        for t_idx in range(self.num_temporal):
            for r in range(gs):
                for c in range(gs):
                    coords.append([
                        t_idx / max(self.num_temporal - 1, 1),
                        r / max(gs - 1, 1),
                        c / max(gs - 1, 1),
                    ])

        coord_tensor = torch.tensor(coords, device=device, dtype=dtype)  # [N, 3]
        coord_tensor = coord_tensor.unsqueeze(0).expand(B, -1, -1)  # [B, N, 3]

        pos_enc = self.coord_proj(coord_tensor)  # [B, N, D]
        return visual_tokens + pos_enc


class SpatialReasoningModule(nn.Module):
    """Process spatial relationships between language and visual tokens.

    Takes cross-attended features and applies a lightweight transformer
    to reason about spatial concepts like "left", "above", "behind".

    The visual tokens are injected with 2D grid coordinates BEFORE
    cross-attention, so the spatial reasoning has access to explicit
    position information — not just learned positional biases.

    Args:
        dim: Feature dimension.
        num_heads: Attention heads.
        num_layers: Transformer layers.
        dropout: Dropout probability.
        grid_size: Spatial grid size for position encoding.
        num_temporal: Number of temporal patches.
    """

    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        grid_size: int = 16,
        num_temporal: int = 1,
    ) -> None:
        super().__init__()

        # 2D spatial position encoding for visual tokens
        self.spatial_pos_enc = SpatialPositionalEncoding(
            dim=dim, grid_size=grid_size, num_temporal=num_temporal,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: Tensor,
        visual_tokens: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply spatial reasoning.

        Args:
            x: [B, S, D] cross-attended language features.
            visual_tokens: [B, N, D] original visual tokens (for position injection).
                If provided, spatial position encoding is added before cross-attention.
            mask: [B, S] optional padding mask.

        Returns:
            y: [B, S, D] spatially-refined features.
        """
        return self.norm(self.transformer(x, src_key_padding_mask=mask))


class LanguageAdapter(nn.Module):
    """Project language embeddings into the V-JEPA 2 visual token space.

    Architecture:
        1. Frozen language encoder → token-level embeddings [B, S, D_lang]
        2. Cross-attention: language tokens query visual tokens
        3. Spatial reasoning transformer
        4. Pool + project to [B, D_visual]

    Supports two backends:
        - sentence-transformers: fast, smaller models
        - transformers: full LLM backbones for complex spatial reasoning

    Args:
        lang_model_name: Language model identifier.
            Recommended: "sentence-transformers/all-mpnet-base-v2" (109M, fast)
            or "Qwen/Qwen2.5-1.5B-Instruct" (1.5B, strong spatial reasoning)
        visual_dim: Target visual token dimension (1024 for V-JEPA 2).
        hidden_dim: Intermediate dimension.
        num_cross_heads: Number of cross-attention heads.
        num_spatial_layers: Layers in the spatial reasoning module.
        dropout: Dropout probability.
        max_seq_len: Maximum language sequence length.
    """

    def __init__(
        self,
        lang_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        visual_dim: int = 1024,
        hidden_dim: int = 768,
        num_cross_heads: int = 8,
        num_spatial_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self._lang_model_name = lang_model_name

        # Lazy-loaded language model
        self._lang_model = None
        self._lang_dim: int | None = None
        self._use_token_level = False  # True for transformers, False for sentence-transformers

        # Cross-attention: language queries attend to visual keys/values
        self._cross_attn: nn.Module | None = None
        self._num_cross_heads = num_cross_heads

        # Spatial reasoning
        self._spatial_reasoning: nn.Module | None = None
        self._num_spatial_layers = num_spatial_layers

        # Output projection
        self._output_proj: nn.Module | None = None
        self._dropout = dropout

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def _detect_backend(self) -> str:
        """Detect whether to use sentence-transformers or transformers."""
        name = self._lang_model_name.lower()
        # Large models or models with "instruct" → use transformers
        if any(kw in name for kw in ["qwen", "llama", "mistral", "phi", "gemma", "instruct"]):
            return "transformers"
        return "sentence_transformers"

    def _ensure_lang_model(self, device: torch.device) -> None:
        """Lazy-load the language model."""
        if self._lang_model is not None:
            return

        backend = self._detect_backend()

        if backend == "transformers":
            self._init_transformers_model(device)
        else:
            self._init_sentence_transformers_model(device)

    def _init_sentence_transformers_model(self, device: torch.device) -> None:
        """Load a sentence-transformers model with token-level output."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers required. Install via: pip install sentence-transformers"
            )

        self._lang_model = SentenceTransformer(
            self._lang_model_name, device=str(device)
        )
        self._lang_model.eval()
        for p in self._lang_model.parameters():
            p.requires_grad = False

        # Infer dimension
        with torch.no_grad():
            dummy = self._lang_model.encode(
                ["hello world"],
                convert_to_tensor=True,
                device=str(device),
                output_value="token_embeddings",
            )
            # dummy is a list of tensors (one per input)
            if isinstance(dummy, list):
                self._lang_dim = dummy[0].shape[-1]
            else:
                self._lang_dim = dummy.shape[-1]

        self._use_token_level = False

    def _init_transformers_model(self, device: torch.device) -> None:
        """Load a full transformers model for token-level embeddings."""
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers required. Install via: pip install transformers"
            )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._lang_model_name, trust_remote_code=True
        )
        self._lang_model = AutoModel.from_pretrained(
            self._lang_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(device)
        self._lang_model.eval()
        for p in self._lang_model.parameters():
            p.requires_grad = False

        self._lang_dim = self._lang_model.config.hidden_size
        self._use_token_level = True

    def _ensure_modules(self, device: torch.device, dtype: torch.dtype) -> None:
        """Build projection and attention modules on first use."""
        if self._cross_attn is not None:
            return

        assert self._lang_dim is not None, "Call _ensure_lang_model first"

        # Project language dim to visual dim
        self.lang_proj = nn.Sequential(
            nn.Linear(self._lang_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.visual_dim),
        ).to(device=device, dtype=dtype)

        # Cross-attention: language queries → visual keys/values
        self._cross_attn = nn.MultiheadAttention(
            embed_dim=self.visual_dim,
            num_heads=self._num_cross_heads,
            dropout=self._dropout,
            batch_first=True,
        ).to(device=device, dtype=dtype)

        # Spatial reasoning
        self._spatial_reasoning = SpatialReasoningModule(
            dim=self.visual_dim,
            num_heads=self._num_cross_heads,
            num_layers=self._num_spatial_layers,
            dropout=self._dropout,
        ).to(device=device, dtype=dtype)

        # Output: pool + project
        self._output_proj = nn.Sequential(
            nn.Linear(self.visual_dim, self.visual_dim),
            nn.LayerNorm(self.visual_dim),
        ).to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_language(
        self,
        instructions: list[str],
        device: torch.device,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Encode instructions to token-level embeddings.

        Args:
            instructions: Batch of text strings.
            device: Target device.

        Returns:
            embeddings: [B, S, D_lang] token embeddings.
            mask: [B, S] padding mask (True = padded).
        """
        self._ensure_lang_model(device)

        if self._use_token_level:
            return self._encode_with_transformers(instructions, device)
        else:
            return self._encode_with_st(instructions, device)

    def _encode_with_st(
        self,
        instructions: list[str],
        device: torch.device,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Encode with sentence-transformers (token-level)."""
        assert self._lang_model is not None

        with torch.no_grad():
            # Get token-level embeddings
            token_embs = self._lang_model.encode(
                instructions,
                convert_to_tensor=True,
                device=str(device),
                output_value="token_embeddings",
                padding=True,
                truncation=True,
                max_length=self.max_seq_len,
            )

        # sentence-transformers returns a list of tensors, pad to same length
        if isinstance(token_embs, list):
            max_len = max(t.shape[0] for t in token_embs)
            padded = []
            masks = []
            for t in token_embs:
                pad_len = max_len - t.shape[0]
                if pad_len > 0:
                    padded.append(F.pad(t, (0, 0, 0, pad_len)))
                    masks.append(
                        torch.cat([torch.zeros(t.shape[0]), torch.ones(pad_len)]).to(device)
                    )
                else:
                    padded.append(t)
                    masks.append(torch.zeros(t.shape[0], device=device))
            return torch.stack(padded).float(), torch.stack(masks).bool()

        # Single tensor [B, S, D]
        return token_embs.float(), None

    def _encode_with_transformers(
        self,
        instructions: list[str],
        device: torch.device,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Encode with a full transformers model."""
        assert self._lang_model is not None and self._tokenizer is not None

        tokens = self._tokenizer(
            instructions,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = self._lang_model(**tokens)
            embeddings = outputs.last_hidden_state  # [B, S, D]

        mask = tokens.attention_mask.logical_not()  # True = padded
        return embeddings.float(), mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        instructions: list[str],
        visual_tokens: Tensor,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Encode instructions and project into visual token space.

        Args:
            instructions: Batch of natural language strings.
            visual_tokens: [B, N, D_visual] visual tokens from V-JEPA 2.
            device: Target device.

        Returns:
            conditioning: [B, D_visual] language conditioning vector.
        """
        if device is None:
            device = visual_tokens.device

        self._ensure_modules(device, torch.float32)

        # Encode language → [B, S, D_lang]
        lang_embs, lang_mask = self.encode_language(instructions, device)

        # Project to visual dim → [B, S, D_visual]
        lang_proj = self.lang_proj(lang_embs)

        # Add spatial position encoding to visual tokens before cross-attention
        visual_spatial = self._spatial_reasoning.spatial_pos_enc(visual_tokens)

        # Cross-attention: language queries attend to spatially-encoded visual tokens
        attn_out, _ = self._cross_attn(
            query=lang_proj,
            key=visual_spatial,
            value=visual_spatial,
        )  # [B, S, D_visual]

        # Residual connection
        attended = lang_proj + attn_out

        # Spatial reasoning: refine cross-attended features
        refined = self._spatial_reasoning(attended, mask=lang_mask)  # [B, S, D_visual]

        # Pool: mean over non-padded tokens
        if lang_mask is not None:
            # lang_mask: True = padded → invert for valid mask
            valid = (~lang_mask).unsqueeze(-1).float()  # [B, S, 1]
            pooled = (refined * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-8)
        else:
            pooled = refined.mean(dim=1)

        # Final projection
        conditioning = self._output_proj(pooled)  # [B, D_visual]

        return conditioning

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"lang_model={self._lang_model_name}, "
            f"visual_dim={self.visual_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"cross_heads={self._num_cross_heads}, "
            f"spatial_layers={self._num_spatial_layers}"
        )

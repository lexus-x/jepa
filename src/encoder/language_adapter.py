"""Language conditioning adapter for VL-JEPA.

Encodes natural language instructions into a conditioning vector that is
compatible with the visual token space of V-JEPA 2.  Uses a frozen
sentence-transformers backbone followed by a learnable MLP / cross-attention
adapter that projects language embeddings into the 1024-dim visual space.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageAdapter(nn.Module):
    """Project language embeddings into the V-JEPA 2 visual token space.

    Architecture:
        1. Frozen sentence-transformers encoder → [B, D_lang]
        2. Learnable MLP with residual connections → [B, D_visual]

    Args:
        lang_model_name: sentence-transformers model identifier.
        visual_dim: Target visual token dimension (1024 for V-JEPA 2).
        hidden_dim: Intermediate MLP dimension.
        num_layers: Number of MLP layers.
        dropout: Dropout probability.
        use_cross_attention: If True, use cross-attention instead of plain MLP.
        num_cross_heads: Number of cross-attention heads (only if use_cross_attention).
    """

    def __init__(
        self,
        lang_model_name: str = "all-MiniLM-L6-v2",
        visual_dim: int = 1024,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_cross_attention: bool = False,
        num_cross_heads: int = 8,
    ) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.use_cross_attention = use_cross_attention

        # Lazy-loaded sentence-transformers (avoids import at module level)
        self._lang_model_name = lang_model_name
        self._lang_model = None  # loaded on first use
        self._lang_dim: int | None = None

        # Projection MLP (built after we know lang_dim)
        self._mlp: nn.Module | None = None
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._dropout = dropout

        # Cross-attention branch (optional)
        self._cross_attn: nn.Module | None = None
        self._num_cross_heads = num_cross_heads

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def _ensure_lang_model(self, device: torch.device) -> None:
        """Lazy-load the sentence-transformers model."""
        if self._lang_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for LanguageAdapter. "
                "Install via: pip install sentence-transformers"
            ) from exc

        self._lang_model = SentenceTransformer(
            self._lang_model_name, device=str(device)
        )
        # Freeze
        self._lang_model.eval()
        for p in self._lang_model.parameters():
            p.requires_grad = False

        # Infer language dimension by encoding a dummy sentence
        with torch.no_grad():
            dummy = self._lang_model.encode(
                ["hello"], convert_to_tensor=True, device=str(device)
            )
        self._lang_dim = dummy.shape[-1]

    def _ensure_projection(self, device: torch.device, dtype: torch.dtype) -> None:
        """Build the projection MLP / cross-attention on first use."""
        if self._mlp is not None:
            return

        assert self._lang_dim is not None, "Call _ensure_lang_model first"

        layers: list[nn.Module] = []
        in_dim = self._lang_dim
        for i in range(self._num_layers):
            out_dim = self.visual_dim if i == self._num_layers - 1 else self._hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < self._num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(self._dropout))
            in_dim = out_dim
        self._mlp = nn.Sequential(*layers).to(device=device, dtype=dtype)

        if self.use_cross_attention:
            self._cross_attn = nn.MultiheadAttention(
                embed_dim=self.visual_dim,
                num_heads=self._num_cross_heads,
                dropout=self._dropout,
                batch_first=True,
            ).to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_language(self, instructions: list[str], device: torch.device) -> torch.Tensor:
        """Encode natural language instructions to embeddings.

        Args:
            instructions: List of text strings.
            device: Target device.

        Returns:
            Language embeddings [B, D_lang].
        """
        self._ensure_lang_model(device)
        assert self._lang_model is not None
        with torch.no_grad():
            embeddings = self._lang_model.encode(
                instructions,
                convert_to_tensor=True,
                device=str(device),
                show_progress_bar=False,
            )
        # sentence-transformers may return [B, D] or [D]
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        return embeddings.float()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        instructions: list[str],
        visual_tokens: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Encode instructions and project into visual token space.

        Args:
            instructions: Batch of natural language strings.
            visual_tokens: Optional [B, N, D_visual] for cross-attention pooling.
                If provided and ``use_cross_attention=True``, language features
                attend to visual tokens.
            device: Target device (inferred from visual_tokens if provided).

        Returns:
            Conditioning vector [B, D_visual].
        """
        if device is None:
            if visual_tokens is not None:
                device = visual_tokens.device
            else:
                device = torch.device("cpu")

        self._ensure_projection(device, torch.float32)

        # Encode language → [B, D_lang]
        lang_emb = self.encode_language(instructions, device)

        # Project → [B, D_visual]
        assert self._mlp is not None
        projected = self._mlp(lang_emb)  # [B, D_visual]

        if self.use_cross_attention and visual_tokens is not None and self._cross_attn is not None:
            # Cross-attention: language query attends to visual keys/values
            # projected → [B, 1, D_visual] as query
            query = projected.unsqueeze(1)
            # visual_tokens → [B, N, D_visual]
            attn_out, _ = self._cross_attn(query, visual_tokens, visual_tokens)
            projected = attn_out.squeeze(1) + projected  # residual

        return projected

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        parts = [f"lang_model={self._lang_model_name}", f"visual_dim={self.visual_dim}"]
        if self.use_cross_attention:
            parts.append(f"cross_attn_heads={self._num_cross_heads}")
        return ", ".join(parts)

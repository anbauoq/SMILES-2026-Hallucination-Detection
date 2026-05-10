"""
aggregation.py — Token aggregation strategy and feature extraction.

Implements response_mean_late_concat: mean-pool of response-only tokens
across late transformer layers 17–24, concatenated into a 7168-dim vector.

When use_geometric=True, 15 additional geometric descriptors are appended
(8 SVD singular values + 7 layer-to-layer cosine similarities) → 7183 dims.

Response token positions are precomputed at module import time by
tokenizing every sample in dataset.csv and test.csv with the same
Qwen/Qwen2.5-0.5B tokenizer and the same prompt-preserving truncation
logic used during hidden-state extraction.  A module-level cache maps each
call to aggregate() to the correct sample's response positions.
"""

from __future__ import annotations

import os

import pandas as pd
import torch

from model import MAX_LENGTH

# Late transformer layers — Qwen2.5-0.5B has 24 transformer layers (+ 1 embedding).
_LATE_LAYERS = list(range(17, 25))   # 8 layers × 896 dims = 7 168


# ---------------------------------------------------------------------------
# Prompt-length cache
# ---------------------------------------------------------------------------

class _PromptLengthCache:
    """Precomputed per-sample prompt-token counts, consumed in call order."""

    def __init__(self) -> None:
        self._lengths: list[int] = []
        self._call_idx: int = 0

    def extend(self, lengths: list[int]) -> None:
        self._lengths.extend(lengths)

    def next_prompt_len(self) -> int | None:
        """Return the prompt length for the current call and advance the counter.

        Returns ``None`` when the precomputed list is exhausted (fallback path).
        """
        if self._call_idx < len(self._lengths):
            length = self._lengths[self._call_idx]
        else:
            length = None
        self._call_idx += 1
        return length


_prompt_cache = _PromptLengthCache()


# ---------------------------------------------------------------------------
# Precompute per-sample prompt lengths at module import time
# ---------------------------------------------------------------------------

def _compute_prompt_len(p_ids: list[int], r_ids: list[int], max_length: int) -> int:
    """Number of prompt tokens kept after response-first truncation.

    The full response is always preserved; the prompt is trimmed from the left
    if ``len(p_ids) + len(r_ids) > max_length``.  Returns 0 when the response
    alone fills or exceeds the budget.
    """
    if len(r_ids) >= max_length:
        return 0
    return min(len(p_ids), max_length - len(r_ids))


def _build_prompt_lengths(csv_path: str, tokenizer, max_length: int) -> list[int]:
    """Return the number of prompt tokens kept for each row in *csv_path*."""
    df = pd.read_csv(csv_path)
    lengths: list[int] = []
    for _, row in df.iterrows():
        p_ids = tokenizer(str(row["prompt"]),   add_special_tokens=False)["input_ids"]
        r_ids = tokenizer(str(row["response"]), add_special_tokens=False)["input_ids"]
        lengths.append(_compute_prompt_len(p_ids, r_ids, max_length))
    return lengths


try:
    from transformers import AutoTokenizer as _AT
    _tok = _AT.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    if _tok.pad_token is None:
        _tok.pad_token = _tok.eos_token
    for _csv in ("./data/dataset.csv", "./data/test.csv"):
        if os.path.exists(_csv):
            _prompt_cache.extend(_build_prompt_lengths(_csv, _tok, MAX_LENGTH))
    del _tok, _AT
except Exception:
    pass  # fall back to last-token pooling if tokenizer load fails


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_resp_positions(
    attention_mask: torch.Tensor,
    prompt_len: int | None,
) -> torch.Tensor:
    """Return a 1-D index tensor of response token positions."""
    real_pos = attention_mask.nonzero(as_tuple=False).squeeze(1)
    if real_pos.numel() == 0:
        real_pos = torch.tensor([attention_mask.shape[0] - 1])
    if prompt_len is not None:
        resp_pos = real_pos[prompt_len:]
        if resp_pos.numel() == 0:
            resp_pos = real_pos[-1:]
    else:
        resp_pos = real_pos[-1:]
    return resp_pos


def _aggregate_at_positions(
    hidden_states: torch.Tensor,
    resp_pos: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool response tokens at each late layer → ``(7168,)`` feature vector."""
    vecs = [hidden_states[layer][resp_pos].mean(dim=0) for layer in _LATE_LAYERS]
    return torch.cat(vecs, dim=0).float()


def _geometric_at_positions(
    hidden_states: torch.Tensor,
    resp_pos: torch.Tensor,
) -> torch.Tensor:
    """Geometric descriptors of the layer-trajectory across late transformer layers.

    Computes two complementary features for the mean-pooled response representation:

    1. SVD singular values of the ``(n_late_layers × hidden_dim)`` trajectory matrix.
       Captures how much of the representation space is explored across layers 17–24.
       Shape: ``(8,)``.

    2. Layer-to-layer cosine similarities between adjacent layer mean-vectors.
       Captures how rapidly the representation changes between consecutive layers.
       Shape: ``(7,)``.

    Total output shape: ``(15,)``.
    """
    trajectory = torch.stack(
        [hidden_states[layer][resp_pos].mean(dim=0) for layer in _LATE_LAYERS]
    ).float()                                             # (8, 896)

    singular_values = torch.linalg.svdvals(trajectory)   # (8,)

    norms = trajectory.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = trajectory / norms                           # (8, 896)
    cosine_sims = (normed[:-1] * normed[1:]).sum(dim=1)  # (7,)

    return torch.cat([singular_values, cosine_sims], dim=0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool response tokens across late layers → 7168-dim feature vector.

    Args:
        hidden_states:  ``(n_layers, seq_len, hidden_dim)``
        attention_mask: ``(seq_len,)`` — 1 for real tokens, 0 for padding.

    Returns:
        1-D float tensor of shape ``(8 * 896,)`` = ``(7168,)``.
    """
    resp_pos = _get_resp_positions(attention_mask, _prompt_cache.next_prompt_len())
    return _aggregate_at_positions(hidden_states, resp_pos)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """SVD singular values + cosine similarities → 15-dim feature vector.

    Args:
        hidden_states:  ``(n_layers, seq_len, hidden_dim)``
        attention_mask: ``(seq_len,)`` — 1 for real tokens, 0 for padding.

    Returns:
        1-D float tensor of shape ``(15,)``.
    """
    resp_pos = _get_resp_positions(attention_mask, _prompt_cache.next_prompt_len())
    return _geometric_at_positions(hidden_states, resp_pos)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states into a fixed-size feature vector.

    Resolves response token positions once and shares them between the
    aggregation and geometric feature paths, consuming exactly one cache
    entry per sample regardless of ``use_geometric``.

    Args:
        hidden_states:  ``(n_layers, seq_len, hidden_dim)``
        attention_mask: ``(seq_len,)`` — 1 for real tokens, 0 for padding.
        use_geometric:  If ``True``, appends 15 geometric features → 7183 dims.

    Returns:
        1-D float tensor of shape ``(7168,)`` or ``(7183,)``.
    """
    resp_pos = _get_resp_positions(attention_mask, _prompt_cache.next_prompt_len())
    agg_features = _aggregate_at_positions(hidden_states, resp_pos)
    if use_geometric:
        geo_features = _geometric_at_positions(hidden_states, resp_pos)
        return torch.cat([agg_features, geo_features], dim=0)
    return agg_features

"""Orthogonal basis changes shared by NVFP4 query and key operands."""

from __future__ import annotations

import math

import torch

BLOCK = 64


def hadamard(dim: int, device: torch.device | str) -> torch.Tensor:
    """Return the normalized Sylvester Hadamard matrix of order ``dim``.

    The construction intentionally matches the offline rotation search.  The
    canonical matrix is FP32; serving may keep a lower-precision execution
    view so Q and K remain in the BF16 contract their quantizers require.
    """
    if type(dim) is not int or dim <= 0 or dim & (dim - 1):
        raise ValueError(
            f"Hadamard dimension must be a positive power of two, got {dim!r}"
        )

    matrix = torch.ones(1, 1, dtype=torch.float32, device=device)
    while matrix.shape[0] < dim:
        matrix = torch.cat(
            [
                torch.cat([matrix, matrix], dim=1),
                torch.cat([matrix, -matrix], dim=1),
            ],
            dim=0,
        )
    return matrix / math.sqrt(dim)


def block_hadamard_64(
    dim: int, device: torch.device | str
) -> torch.Tensor:
    """Hadamard rotations confined to the active ``roped64`` blocks.

    Each block groups channels with their RoPE partners, then applies a
    normalized H64 matrix within that group.
    """
    if type(dim) is not int or dim <= 0 or dim % BLOCK:
        raise ValueError(
            f"block_hadamard_64 needs a positive multiple of {BLOCK}, got "
            f"{dim!r}"
        )
    low = torch.arange(dim // 2, device=device).reshape(-1, BLOCK // 2)
    blocks = torch.cat((low, low + dim // 2), dim=1)
    if blocks.shape != (dim // BLOCK, BLOCK):
        raise ValueError(
            f"roped64 blocking for dim={dim} has shape {tuple(blocks.shape)}"
        )
    local = hadamard(BLOCK, device)
    rotation = torch.zeros(dim, dim, dtype=torch.float32, device=device)
    for channels in blocks:
        rotation[channels[:, None], channels[None, :]] = local
    return rotation

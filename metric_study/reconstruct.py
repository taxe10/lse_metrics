"""Load a trained VAE checkpoint and run deterministic reconstructions (plan §2
"Real reconstructions", §11 latents).

Imports ``vae.py`` and ``YParams.py`` from the training repo
(``cvae_model_for_lse/train``) rather than duplicating the model definition, per the
plan's rule to reuse existing utilities. No checkpoint exists on this machine yet
(plan §9 Q1); every function here degrades gracefully (returns None / raises a
clear, catchable error) so the rest of the pipeline can run without one.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TrainingRepoNotFoundError(RuntimeError):
    """Raised when the cvae_model_for_lse training repo can't be located/imported."""


def _import_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise TrainingRepoNotFoundError(f"Could not load module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_training_repo(train_dir: str | Path):
    """Import ``vae`` and ``YParams`` modules from the training repo's ``train/`` directory.

    Args:
        train_dir: Path to ``cvae_model_for_lse/train`` (contains ``vae.py``,
            ``YParams.py``).

    Returns:
        (vae_module, yparams_module).

    Raises:
        TrainingRepoNotFoundError: If the expected files aren't found.
    """
    train_dir = Path(train_dir)
    vae_path = train_dir / "vae.py"
    yparams_path = train_dir / "YParams.py"
    if not vae_path.exists() or not yparams_path.exists():
        raise TrainingRepoNotFoundError(
            f"Expected vae.py and YParams.py under {train_dir}; found "
            f"vae.py={vae_path.exists()}, YParams.py={yparams_path.exists()}"
        )
    yparams_mod = _import_from_path("_training_repo_yparams", yparams_path)
    vae_mod = _import_from_path("_training_repo_vae", vae_path)
    return vae_mod, yparams_mod


@dataclass
class LoadedCheckpoint:
    """A trained model ready for inference.

    Attributes:
        model: The ``ConvVAE`` in eval mode, on ``device``.
        latent_dim: Latent dimension it was built with.
        image_size: (H, W) it was trained at (always (512, 512) per the current repo).
        device: torch device the model lives on.
        checkpoint_path: Path the weights were loaded from.
    """

    model: torch.nn.Module
    latent_dim: int
    image_size: tuple[int, int]
    device: torch.device
    checkpoint_path: Path


def load_checkpoint(
    checkpoint_path: str | Path,
    yaml_config_path: str | Path,
    config_section: str,
    train_dir: str | Path,
    device: str | torch.device = "cpu",
) -> LoadedCheckpoint:
    """Build the model from the training YAML and load trained weights (plan §2).

    Args:
        checkpoint_path: Path to a ``.ckpt`` file written by
            ``Trainer.save_checkpoint``/``save_best_checkpoint``
            (``train/training_2.py:390-407``): a dict with ``model_state_dict``.
        yaml_config_path: The training run's YAML config (e.g. ``default_500.yaml``).
        config_section: Section name within that YAML (e.g. ``"default"``).
        train_dir: Path to the training repo's ``train/`` directory.
        device: Device to load the model onto.

    Returns:
        A ``LoadedCheckpoint`` with the model in eval mode.

    Raises:
        TrainingRepoNotFoundError: If the training repo can't be imported.
        FileNotFoundError: If ``checkpoint_path`` doesn't exist.
    """
    vae_mod, yparams_mod = load_training_repo(train_dir)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    params = yparams_mod.YParams(str(yaml_config_path), config_section)
    device_t = torch.device(device)
    model = vae_mod.ConvVAE(latent_dim=params.latent_dim, image_size=params.image_size).to(device_t)

    state = torch.load(checkpoint_path, map_location=device_t, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    image_size = tuple(params.image_size) if isinstance(params.image_size, (list, tuple)) else (params.image_size,) * 2
    logger.info("Loaded checkpoint %s (latent_dim=%d, image_size=%s)", checkpoint_path, params.latent_dim, image_size)
    return LoadedCheckpoint(
        model=model, latent_dim=params.latent_dim, image_size=image_size, device=device_t, checkpoint_path=checkpoint_path
    )


@torch.no_grad()
def reconstruct_deterministic(loaded: LoadedCheckpoint, images: torch.Tensor) -> dict[str, Any]:
    """Encode->mu->decode (no sampling) for a batch of model-space images (plan §2, §11).

    Args:
        loaded: Result of :func:`load_checkpoint`.
        images: (B, 1, H, W) tensor in [0, 1], matching ``loaded.image_size``.

    Returns:
        Dict with ``recon`` (B, 1, H, W), ``mu`` (B, latent_dim), ``logvar`` (B, latent_dim).
    """
    if tuple(images.shape[-2:]) != tuple(loaded.image_size):
        raise ValueError(f"images shape {tuple(images.shape[-2:])} != model image_size {loaded.image_size}")
    images = images.to(loaded.device)
    mu, logvar = loaded.model.encode(images)
    recon = loaded.model.decode(mu)  # deterministic: decode mu directly, no reparameterize sampling
    return {"recon": recon.cpu(), "mu": mu.cpu(), "logvar": logvar.cpu()}


def find_latest_checkpoint(checkpoint_dir: str | Path, prefer_best: bool = True) -> Path | None:
    """Mirror ``Trainer.load_checkpoint``'s selection logic (``training_2.py:409-422``):
    newest-by-mtime ``*.ckpt`` file, or ``best.ckpt`` if ``prefer_best`` and it exists.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    if prefer_best and (checkpoint_dir / "best.ckpt").exists():
        return checkpoint_dir / "best.ckpt"
    ckpts = sorted(checkpoint_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0] if ckpts else None

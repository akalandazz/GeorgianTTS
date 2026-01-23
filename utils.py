"""
Shared utilities for Georgian TTS project
"""

import sys
import numpy as np


def check_pytorch_version(exit_on_failure=True):
    """
    Verify PyTorch version meets security requirements (2.6.0+).

    Args:
        exit_on_failure: If True, exits program on failure. If False, returns bool.

    Returns:
        bool: True if version is secure (only when exit_on_failure=False)
    """
    import torch

    version_parts = torch.__version__.split('.')[:2]
    try:
        torch_version = tuple(map(int, version_parts))
    except ValueError:
        # Handle versions like "2.6.0+cu124"
        torch_version = tuple(int(p.split('+')[0]) for p in version_parts)

    if torch_version < (2, 6):
        if exit_on_failure:
            print("\n" + "=" * 70)
            print("CRITICAL SECURITY ERROR")
            print("=" * 70)
            print(f"PyTorch version {torch.__version__} is NOT SECURE!")
            print("\nA serious vulnerability exists in torch.load for PyTorch < 2.6.0")
            print("This vulnerability can allow malicious code execution.")
            print("\n" + "=" * 70)
            print("REQUIRED ACTION:")
            print("=" * 70)
            print("Upgrade PyTorch to version 2.6.0 or later:")
            print("\n  pip uninstall torch torchvision torchaudio")
            print("  pip install torch>=2.6.0 torchvision torchaudio")
            print("\nSee SECURITY_NOTICE.md for more information.")
            print("=" * 70)
            sys.exit(1)
        return False

    if exit_on_failure:
        print(f"PyTorch {torch.__version__} (secure version)")
    return True


def load_speaker_embeddings(index=None):
    """
    Load speaker embeddings for SpeechT5 from CMU Arctic dataset.

    Args:
        index: Speaker embedding index. Defaults to config.SPEAKER_EMBEDDING_INDEX

    Returns:
        torch.Tensor: Speaker embedding tensor of shape (1, 512)
    """
    import torch
    from datasets import load_dataset
    import config

    if index is None:
        index = config.SPEAKER_EMBEDDING_INDEX

    try:
        print("\nLoading speaker embeddings...")
        embeddings_dataset = load_dataset(
            config.SPEAKER_EMBEDDING_DATASET,
            split="validation"
        )
        speaker_embedding = torch.tensor(embeddings_dataset[index]["xvector"]).unsqueeze(0)
        print("Loaded speaker embeddings from CMU Arctic dataset")
        return speaker_embedding
    except Exception as e:
        print(f"Warning: Could not load speaker embeddings: {e}")
        print("Creating dummy embeddings...")
        return torch.zeros(1, 512)


def normalize_audio(audio_array):
    """
    Normalize audio to [-1, 1] range.

    Args:
        audio_array: numpy array of audio samples

    Returns:
        numpy.ndarray: Normalized audio array
    """
    if not isinstance(audio_array, np.ndarray):
        audio_array = np.array(audio_array)

    max_val = np.max(np.abs(audio_array))
    if max_val > 0:
        return audio_array / max_val
    return audio_array


def load_metadata_csv(filepath):
    """
    Load metadata CSV with configured separator.

    Args:
        filepath: Path to the CSV file

    Returns:
        pandas.DataFrame: Loaded dataframe
    """
    import pandas as pd
    import config

    return pd.read_csv(filepath, sep=config.CSV_SEPARATOR)

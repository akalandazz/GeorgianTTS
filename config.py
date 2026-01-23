"""
Configuration file for TTS model fine-tuning
Adjust these parameters based on your data and hardware
"""

import os

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.csv")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Pre-trained model from Hugging Face
MODEL_NAME = "microsoft/speecht5_tts"
# Alternative models:
# MODEL_NAME = "facebook/mms-tts-kat"  # If available for Georgian
# MODEL_NAME = "coqui/XTTS-v2"  # For voice cloning (more advanced)

# ============================================================================
# AUDIO PROCESSING
# ============================================================================
SAMPLE_RATE = 16000  # Target sample rate (Hz)
MAX_AUDIO_LENGTH = 20  # Maximum audio length in seconds
MIN_AUDIO_LENGTH = 1   # Minimum audio length in seconds
AUDIO_FORMAT = "wav"   # Audio file format

# ============================================================================
# DATA PROCESSING
# ============================================================================
TRAIN_TEST_SPLIT = 0.9  # 90% train, 10% validation
MAX_TEXT_LENGTH = 200   # Maximum characters in text
MIN_TEXT_LENGTH = 10    # Minimum characters in text
RANDOM_SEED = 42        # For reproducibility

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
# Batch sizes - adjust based on your GPU memory
BATCH_SIZE = 4           # Reduce to 2 or 1 if out of memory
EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = BATCH_SIZE * this

# Training duration
NUM_EPOCHS = 50          # Total number of training epochs
MAX_STEPS = -1          # Set to positive number to limit total steps

# Optimizer settings
LEARNING_RATE = 1e-5    # Start conservative, increase if training is slow
WEIGHT_DECAY = 0.01     # L2 regularization
WARMUP_STEPS = 500      # Learning rate warmup

# ============================================================================
# CHECKPOINT & LOGGING
# ============================================================================
SAVE_STEPS = 500           # Save checkpoint every N steps
EVAL_STEPS = 500           # Evaluate every N steps
LOGGING_STEPS = 100        # Log metrics every N steps
SAVE_TOTAL_LIMIT = 5       # Keep only last N checkpoints
LOAD_BEST_MODEL_AT_END = True  # Load best model after training

# ============================================================================
# MIXED PRECISION & OPTIMIZATION
# ============================================================================
FP16 = True  # Use mixed precision training (faster on modern GPUs)
DATALOADER_NUM_WORKERS = 4  # Parallel data loading (adjust based on CPU)

# ============================================================================
# VOCODER CONFIGURATION (for SpeechT5)
# ============================================================================
VOCODER_NAME = "microsoft/speecht5_hifigan"  # Converts mel-spectrograms to audio

# ============================================================================
# EARLY STOPPING (Optional)
# ============================================================================
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 5  # Stop if no improvement for N evaluations
EARLY_STOPPING_THRESHOLD = 0.01  # Minimum change to qualify as improvement

# ============================================================================
# LANGUAGE SPECIFIC
# ============================================================================
LANGUAGE = "ka"  # Georgian language code
TEXT_CLEANER = "basic"  # Options: basic, advanced, custom

# ============================================================================
# SPEAKER EMBEDDINGS
# ============================================================================
SPEAKER_EMBEDDING_INDEX = 7306  # Index in CMU Arctic dataset
SPEAKER_EMBEDDING_DATASET = "Matthijs/cmu-arctic-xvectors"

# ============================================================================
# DATA FORMAT
# ============================================================================
CSV_SEPARATOR = "|"  # Metadata file separator

# ============================================================================
# OUTPUT PATTERNS
# ============================================================================
BATCH_OUTPUT_DIR = "batch_outputs"

# ============================================================================
# ADVANCED OPTIONS
# ============================================================================
USE_CACHE = True  # Cache processed datasets
OVERWRITE_CACHE = False  # Set to True to reprocess data
PUSH_TO_HUB = False  # Upload model to Hugging Face Hub after training
HF_USERNAME = ""  # Your Hugging Face username (if PUSH_TO_HUB=True)

# ============================================================================
# HARDWARE SETTINGS
# ============================================================================
# Automatically detect and use GPU if available
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

def print_config_summary():
    """Print configuration summary. Call explicitly when needed."""
    print(f"Configuration loaded:")
    print(f"  Device: {DEVICE}")
    print(f"  Number of GPUs: {NUM_GPUS}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")


if __name__ == "__main__":
    print_config_summary()

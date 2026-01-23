# Georgian TTS Fine-tuning

Fine-tune Microsoft SpeechT5 for Georgian text-to-speech using your own voice data.

> **Security:** PyTorch 2.6.0+ is required. See [SECURITY_NOTICE.md](SECURITY_NOTICE.md).

## Quick Start

```bash
# 1. Setup environment
conda create -n ka-tts python=3.10 -y
conda activate ka-tts
conda install -c conda-forge ffmpeg=8
pip install -r requirements.txt

# 2. GPU support (choose your CUDA version)
pip install torch>=2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Verify system
python check_system.py

# 4. Initialize directories
python setup.py

# 5. Add your data to data/audio/ and create data/metadata.csv

# 6. Process data
python prepare_data.py

# 7. Train
python trainer.py

# 8. Generate speech
python inference.py --text "ეს არის ტესტი"
```

## Data Format

Place WAV files in `data/audio/` and create `data/metadata.csv`:

```
filename|text
sample_001.wav|ეს არის ქართული ტექსტი
sample_002.wav|საქართველო არის ქვეყანა
```

**Requirements:**
- WAV format, 1-20 seconds per clip
- Minimum 30 minutes total (2+ hours recommended)
- Clear speech, minimal background noise

## Project Structure

```
├── data/audio/         # Your WAV files
├── data/metadata.csv   # Transcriptions
├── processed_data/     # Preprocessed data (auto-generated)
├── checkpoints/        # Training checkpoints (auto-generated)
├── output/             # Final model (auto-generated)
├── logs/               # TensorBoard logs (auto-generated)
├── config.py           # Configuration
├── utils.py            # Shared utilities
├── prepare_data.py     # Data preprocessing
├── trainer.py          # Training script
├── inference.py        # Generate speech
├── monitor.py          # Training monitor
├── check_system.py     # System verification
└── setup.py            # Directory initialization
```

## Configuration

Edit `config.py` to adjust settings:

```python
# Adjust for GPU memory
BATCH_SIZE = 4                    # Reduce to 2 or 1 if OOM
GRADIENT_ACCUMULATION_STEPS = 4   # Increase when reducing batch size
FP16 = True                       # Mixed precision for faster training

# Training
LEARNING_RATE = 1e-5
NUM_EPOCHS = 50

# Audio
SAMPLE_RATE = 16000
```

## Training

```bash
# Start training
python trainer.py

# Monitor with TensorBoard
tensorboard --logdir logs/

# Check status
python monitor.py
```

## Inference

```bash
# Single text
python inference.py --text "გამარჯობა"

# Interactive mode
python inference.py --interactive

# Batch processing
python inference.py --batch texts.txt
```

**Python API:**
```python
from inference import generate_speech

generate_speech(
    text="ეს არის ტესტი",
    model_path="output/",
    output_path="speech.wav"
)
```

## Troubleshooting

**Out of Memory:**
```python
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
FP16 = True
```

**Poor Quality:**
- Check audio quality and transcription accuracy
- Train for more epochs
- Try lower learning rate: `LEARNING_RATE = 5e-6`

**No GPU Detected:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Hardware Requirements

| | Minimum | Recommended |
|---|---------|-------------|
| GPU | 8GB VRAM | 16GB+ VRAM |
| RAM | 16GB | 32GB |
| Storage | 10GB | 50GB |

## License

This project uses Microsoft SpeechT5 (MIT), Hugging Face Transformers (Apache 2.0), and PyTorch (BSD).

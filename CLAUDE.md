# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Georgian TTS fine-tuning pipeline using Microsoft SpeechT5. Converts Georgian text to speech by fine-tuning a pre-trained model on custom voice data.

## Essential Commands

```bash
# Setup
conda create -n ka-tts python=3.10 -y
conda activate ka-tts
conda install -c conda-forge ffmpeg=8
pip install -r requirements.txt
pip install torch>=2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify and initialize
python check_system.py
python setup.py

# Pipeline
python prepare_data.py              # Preprocess data
python trainer.py                   # Train model
python inference.py --text "ტესტი"  # Generate speech
python inference.py --interactive   # Interactive mode

# Monitoring
python monitor.py
tensorboard --logdir logs/
```

## Architecture

**Pipeline Flow:**
1. `prepare_data.py` - Validates audio, resamples to 16kHz, cleans text, splits train/val (90/10)
2. `trainer.py` - Fine-tunes SpeechT5 with speaker embeddings using Seq2SeqTrainer
3. `inference.py` - Loads model + HiFi-GAN vocoder to generate speech

**Key Components:**
- `utils.py` - Shared utilities (version check, speaker embeddings, audio normalization)
- `config.py` - Central configuration for all parameters
- `TTSDataset` class in trainer.py - Handles dataset loading and mel spectrogram conversion
- Speaker embeddings from `Matthijs/cmu-arctic-xvectors` dataset

**Data Format:**
```
filename|text
audio_001.wav|ეს არის ქართული ტექსტი
```

## Configuration

All settings in `config.py`. Key parameters:
- `BATCH_SIZE` - Default 4, reduce for low VRAM
- `GRADIENT_ACCUMULATION_STEPS` - Increase when reducing batch size
- `SPEAKER_EMBEDDING_INDEX` - CMU Arctic speaker index (7306)
- `CSV_SEPARATOR` - Metadata separator (|)
- `SAMPLE_RATE` - 16000 Hz (fixed for SpeechT5)

## Security

PyTorch 2.6.0+ mandatory due to `torch.load` vulnerability. All scripts enforce this via `utils.check_pytorch_version()`.

## Directory Structure

- `data/audio/` - Input WAV files
- `data/metadata.csv` - Transcriptions
- `processed_data/` - Preprocessed dataset with train.csv/val.csv
- `checkpoints/` - Training checkpoints
- `output/` - Final trained model
- `logs/` - TensorBoard logs

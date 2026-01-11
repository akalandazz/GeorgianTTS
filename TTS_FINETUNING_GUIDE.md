# Complete Guide: Fine-tuning TTS Model for Georgian Language

## Overview
This guide will help you fine-tune a Text-to-Speech model using your local Georgian voice data. We'll use a pre-trained model from Hugging Face and adapt it to your voice.

## Recommended Model
**XTTS-v2** or **VITS** - Both are excellent for voice cloning and support multiple languages.

For simplicity, we'll use **Microsoft SpeechT5** as it's well-documented and production-ready.

## Project Structure
```
tts-georgian-finetune/
├── data/
│   ├── audio/              # Your .wav audio files
│   │   ├── sample_001.wav
│   │   ├── sample_002.wav
│   │   └── ...
│   └── metadata.csv        # Text transcriptions
├── processed_data/         # Preprocessed datasets (auto-generated)
├── checkpoints/            # Model checkpoints during training
├── output/                 # Final trained model
├── logs/                   # Training logs
├── config.py              # Configuration file
├── prepare_data.py        # Data preprocessing script
├── trainer.py             # Main training script
├── inference.py           # Testing the trained model
└── requirements.txt       # Python dependencies
```

## Prerequisites

### 1. Data Requirements
- **Audio files**: WAV format, 16kHz or 22kHz sample rate, mono
- **Duration**: At least 30 minutes of clean speech (1-2 hours recommended)
- **Quality**: Clear recordings, minimal background noise
- **Transcriptions**: Accurate Georgian text matching each audio file

### 2. Hardware Requirements
- **GPU**: NVIDIA GPU with at least 8GB VRAM (16GB+ recommended)
- **RAM**: 16GB+ system RAM
- **Storage**: 10GB+ free space

## Step-by-Step Setup

### Step 1: Install Dependencies
See `requirements.txt` for all required packages.

### Step 2: Prepare Your Data
Your `metadata.csv` should look like:
```
filename|text
sample_001.wav|ეს არის ქართული ტექსტი
sample_002.wav|საქართველო არის ქვეყანა
```

### Step 3: Configure Training
Edit `config.py` with your preferences.

### Step 4: Preprocess Data
Run: `python prepare_data.py`

### Step 5: Train the Model
Run: `python trainer.py`

### Step 6: Test the Model
Run: `python inference.py --text "თქვენი ტექსტი აქ"`

## Training Tips

1. **Start Small**: Begin with 100-200 samples to verify everything works
2. **Monitor Loss**: Training loss should decrease steadily
3. **Regular Checkpoints**: Save checkpoints every 500-1000 steps
4. **Validation**: Use 10-15% of data for validation
5. **Learning Rate**: Start with 1e-5, adjust if needed
6. **Batch Size**: Adjust based on your GPU memory

## Expected Training Time
- **Small dataset (30 min)**: 2-4 hours
- **Medium dataset (2 hours)**: 8-12 hours
- **Large dataset (5+ hours)**: 24-48 hours

## Troubleshooting

### Out of Memory Error
- Reduce `batch_size` in config.py
- Reduce `max_audio_length`
- Use gradient accumulation

### Poor Quality Output
- Check audio quality of training data
- Increase training epochs
- Verify transcriptions are accurate
- Ensure consistent audio format

### Model Not Learning
- Check learning rate (try 5e-5 or 1e-4)
- Verify data preprocessing worked correctly
- Ensure GPU is being used
- Check for data normalization issues

## Next Steps After Training

1. **Evaluate**: Test with various text inputs
2. **Fine-tune**: Adjust hyperparameters if needed
3. **Export**: Save model in production format
4. **Deploy**: Integrate into your application

## Additional Resources

- Hugging Face TTS Documentation: https://huggingface.co/docs/transformers/model_doc/speecht5
- SpeechT5 Paper: https://arxiv.org/abs/2110.07205
- Audio Processing Guide: https://librosa.org/doc/latest/

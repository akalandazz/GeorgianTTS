# Georgian TTS Fine-tuning Project

Fine-tune a Text-to-Speech model for Georgian language using your own voice data.

## 📋 Overview

This project provides a complete pipeline to:
- Preprocess Georgian audio and text data
- Fine-tune a pre-trained TTS model (SpeechT5)
- Generate natural-sounding Georgian speech from text

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare your data (see structure below)
# 3. Run preprocessing
python prepare_data.py

# 4. Start training
python trainer.py

# 5. Test the model
python inference.py --text "ეს არის ტესტი"
```

For detailed instructions, see [QUICKSTART.md](QUICKSTART.md)

## 📁 Project Structure

```
tts-georgian-finetune/
├── data/
│   ├── audio/              # Your .wav files
│   └── metadata.csv        # Transcriptions
├── processed_data/         # Preprocessed data (auto-generated)
├── checkpoints/            # Training checkpoints (auto-generated)
├── output/                 # Final model (auto-generated)
├── logs/                   # Training logs (auto-generated)
├── config.py              # Configuration settings
├── prepare_data.py        # Data preprocessing
├── trainer.py             # Training script
├── inference.py           # Generate speech
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## 📊 Data Requirements

### Audio Files
- Format: WAV
- Sample rate: 16kHz (will be resampled automatically)
- Duration: 1-20 seconds per clip
- Quality: Clear speech, minimal background noise
- Total: Minimum 30 minutes (1-2 hours recommended)

### Metadata Format
CSV file with pipe separator: `filename|text`

Example:
```csv
filename|text
audio_001.wav|ეს არის ქართული ტექსტი
audio_002.wav|საქართველო არის ქვეყანა
```

See [metadata_example.csv](metadata_example.csv) for a complete example.

## ⚙️ Configuration

Edit `config.py` to customize:
- Model architecture
- Training hyperparameters
- Data processing settings
- Hardware utilization

Key settings:
```python
BATCH_SIZE = 4              # Adjust based on GPU memory
LEARNING_RATE = 1e-5        # Learning rate
NUM_EPOCHS = 50             # Training epochs
SAMPLE_RATE = 16000         # Audio sample rate
```

## 🎓 Training

### Monitor Training
```bash
tensorboard --logdir logs/
```

### Training Tips
- Start with a small dataset to verify everything works
- Monitor loss - it should decrease steadily
- Save checkpoints regularly (configured in config.py)
- Test early checkpoints - sometimes they sound better

### Expected Training Time
- Small dataset (30 min): 2-4 hours
- Medium dataset (2 hours): 8-12 hours
- Large dataset (5+ hours): 24-48 hours

## 🎤 Inference

### Single Text
```bash
python inference.py --text "გამარჯობა"
```

### Interactive Mode
```bash
python inference.py --interactive
```

### Batch Processing
Create a text file with one sentence per line:
```bash
python inference.py --batch texts.txt
```

### Python API
```python
from inference import generate_speech

generate_speech(
    text="ეს არის ტესტი",
    model_path="output/",
    output_path="speech.wav"
)
```

## 🔧 Troubleshooting

### Out of Memory
Reduce batch size in `config.py`:
```python
BATCH_SIZE = 2  # or even 1
GRADIENT_ACCUMULATION_STEPS = 8
```

### Poor Audio Quality
- Check training data quality
- Verify transcriptions are accurate
- Train for more epochs
- Try different learning rates

### Slow Training
- Increase batch size (if GPU allows)
- Reduce number of dataloader workers
- Enable FP16 training

See [TTS_FINETUNING_GUIDE.md](TTS_FINETUNING_GUIDE.md) for detailed troubleshooting.

## 📚 Documentation

- [Quick Start Guide](QUICKSTART.md) - Get started in 5 minutes
- [Complete Guide](TTS_FINETUNING_GUIDE.md) - Comprehensive documentation
- [Example Metadata](metadata_example.csv) - Sample data format

## 🛠️ Technical Details

### Model Architecture
- Base model: Microsoft SpeechT5
- Fine-tuning approach: Full model fine-tuning
- Vocoder: HiFi-GAN

### Dependencies
- PyTorch >= 2.0.0
- Transformers >= 4.35.0
- Librosa >= 0.10.0
- See [requirements.txt](requirements.txt) for complete list

## 📝 Scripts Overview

| Script | Purpose |
|--------|---------|
| `config.py` | Configuration settings |
| `prepare_data.py` | Preprocess audio and text |
| `trainer.py` | Train the TTS model |
| `inference.py` | Generate speech from text |

## 🎯 Best Practices

1. **Data Quality**
   - Use consistent recording equipment
   - Record in a quiet environment
   - Ensure clear pronunciation
   - Match text exactly to audio

2. **Training**
   - Start with small dataset to test pipeline
   - Monitor training metrics regularly
   - Save multiple checkpoints
   - Test during training, not just at the end

3. **Inference**
   - Test with diverse texts
   - Compare different checkpoints
   - Adjust inference parameters if needed

## 🤝 Contributing

This is a template project. Feel free to:
- Modify for other languages
- Try different base models
- Improve preprocessing
- Add new features

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- Microsoft for SpeechT5 model
- Hugging Face for the Transformers library
- The open-source community

## 📮 Support

For issues and questions:
- Check the troubleshooting section
- Review the documentation
- Examine the example files

---

**Note:** This is a template for fine-tuning TTS models. Adjust configurations based on your specific requirements and hardware capabilities.

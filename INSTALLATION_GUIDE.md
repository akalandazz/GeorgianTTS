# 🎙️ Georgian TTS Training - Complete Project Package

## 📦 What's Included

This package contains everything you need to train a Text-to-Speech model for Georgian language:

### Core Scripts
- **`trainer.py`** - Main training script
- **`prepare_data.py`** - Data preprocessing
- **`inference.py`** - Generate speech from text
- **`config.py`** - Configuration settings

### Utilities
- **`setup.py`** - Initialize project structure
- **`monitor.py`** - Monitor training progress
- **`check_system.py`** - Verify system requirements

### Documentation
- **`README.md`** - Project overview
- **`QUICKSTART.md`** - Quick start guide
- **`TTS_FINETUNING_GUIDE.md`** - Comprehensive guide
- **`INSTALLATION_GUIDE.md`** - This file

### Dependencies
- **`requirements.txt`** - Python package dependencies
- **`metadata_example.csv`** - Example data format

## 🚀 Installation Steps

### Step 1: System Requirements

**Minimum:**
- Python 3.8+
- 16GB RAM
- 10GB free disk space
- NVIDIA GPU with 8GB VRAM (recommended)

**Recommended:**
- Python 3.10+
- 32GB RAM
- 50GB free disk space
- NVIDIA GPU with 16GB+ VRAM

### Step 2: Install Python Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# For GPU support (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For GPU support (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Verify Installation

```bash
python check_system.py
```

This will verify:
- ✓ Python version
- ✓ Required packages
- ✓ GPU availability
- ✓ Disk space
- ✓ System readiness

### Step 4: Setup Project Structure

```bash
python setup.py
```

This creates:
```
tts-georgian-finetune/
├── data/audio/          # Place your audio files here
├── processed_data/      # Auto-generated
├── checkpoints/         # Auto-generated
├── output/             # Auto-generated
└── logs/               # Auto-generated
```

## 📊 Data Preparation

### Audio Requirements
- **Format:** WAV (mono)
- **Sample Rate:** 16kHz or 22kHz (will be resampled)
- **Duration:** 1-20 seconds per clip
- **Quality:** Clear speech, minimal noise
- **Quantity:** 
  - Minimum: 30 minutes (200-300 clips)
  - Recommended: 2 hours (800-1000 clips)
  - Optimal: 5+ hours (2000+ clips)

### Creating metadata.csv

1. Place all `.wav` files in `data/audio/`
2. Create `data/metadata.csv` with format:

```csv
filename|text
sample_001.wav|გამარჯობა ეს არის პირველი ნიმუში
sample_002.wav|საქართველო არის ძალიან ლამაზი ქვეყანა
sample_003.wav|მე მიყვარს ქართული ენა და კულტურა
```

**Important:**
- Use pipe `|` as separator (not comma)
- No header row
- Text must match audio exactly
- Use UTF-8 encoding

### Preprocessing

```bash
python prepare_data.py
```

This will:
1. Validate all audio files
2. Resample to target sample rate
3. Normalize audio levels
4. Clean text transcriptions
5. Split into train/validation sets
6. Save processed data

## 🎯 Training

### Basic Training

```bash
python trainer.py
```

### Customizing Training

Edit `config.py` before training:

```python
# Adjust for your GPU memory
BATCH_SIZE = 4              # Reduce if out of memory
LEARNING_RATE = 1e-5        # Adjust learning speed
NUM_EPOCHS = 50             # More = better quality

# For smaller GPU (8GB or less)
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
FP16 = True
```

### Monitoring Training

**Option 1: Training Monitor**
```bash
python monitor.py
```

**Option 2: TensorBoard**
```bash
tensorboard --logdir logs/
# Open browser: http://localhost:6006
```

**Option 3: Watch GPU**
```bash
watch -n 1 nvidia-smi
```

## 🎤 Using Your Trained Model

### Single Text

```bash
python inference.py --text "გამარჯობა"
```

### Interactive Mode

```bash
python inference.py --interactive
```

Then type any Georgian text to generate speech.

### Batch Processing

Create `texts.txt`:
```
ეს არის პირველი წინადადება
ეს არის მეორე წინადადება
ეს არის მესამე წინადადება
```

Run:
```bash
python inference.py --batch texts.txt
```

### Python API

```python
from inference import generate_speech

# Generate speech
generate_speech(
    text="ეს არის ტესტი",
    model_path="output/",
    output_path="my_speech.wav"
)
```

## 🔧 Troubleshooting

### "CUDA out of memory"

**Solution 1:** Reduce batch size
```python
# In config.py
BATCH_SIZE = 2  # or 1
GRADIENT_ACCUMULATION_STEPS = 8
```

**Solution 2:** Use mixed precision
```python
# In config.py
FP16 = True
```

### "Audio file not found"

**Check:**
- Files are in `data/audio/`
- Filenames in metadata.csv match exactly
- File format is WAV
- No typos in filenames

### "Poor quality output"

**Try:**
1. Check training data quality
2. Verify transcriptions are accurate
3. Train for more epochs
4. Reduce learning rate: `LEARNING_RATE = 5e-6`
5. Use more training data

### "Training too slow"

**On CPU:**
- Training on CPU is 100x slower
- Use Google Colab or cloud GPU

**On GPU:**
- Increase batch size if memory allows
- Reduce `DATALOADER_NUM_WORKERS`
- Ensure GPU is actually being used: `python -c "import torch; print(torch.cuda.is_available())"`

### "Package installation fails"

```bash
# Update pip first
pip install --upgrade pip

# Install with verbose output
pip install -r requirements.txt -v

# For specific package errors
pip install package_name --no-cache-dir
```

## 📈 Expected Results

### Training Time
| Dataset Size | GPU | Time |
|-------------|-----|------|
| 30 min | RTX 3060 (12GB) | 2-4 hours |
| 2 hours | RTX 3060 (12GB) | 8-12 hours |
| 5 hours | RTX 4090 (24GB) | 12-24 hours |

### Quality Metrics
- **Good:** Clear speech, minor artifacts
- **Excellent:** Natural-sounding, minimal artifacts
- **Production:** Indistinguishable from recordings

Achieving "Excellent" typically requires:
- 2+ hours of clean audio
- Accurate transcriptions
- 30-50 training epochs
- Proper hyperparameter tuning

## 🎓 Tips for Best Results

### Data Quality
1. Record in quiet environment
2. Use consistent microphone
3. Clear pronunciation
4. Natural speaking pace
5. Diverse sentence structures

### Training Strategy
1. Start small (100 samples) to test
2. Monitor loss regularly
3. Save multiple checkpoints
4. Test during training
5. Compare checkpoint outputs

### Hyperparameter Tuning
1. Start with default settings
2. Adjust one parameter at a time
3. If loss plateaus: reduce learning rate
4. If loss unstable: reduce learning rate
5. If training too slow: increase learning rate

## 📚 Additional Resources

### Hugging Face
- [SpeechT5 Documentation](https://huggingface.co/docs/transformers/model_doc/speecht5)
- [TTS Task Guide](https://huggingface.co/docs/transformers/tasks/text-to-speech)

### Audio Processing
- [Librosa Tutorial](https://librosa.org/doc/latest/tutorial.html)
- [Audio Data Preprocessing](https://pytorch.org/audio/stable/tutorials.html)

### Model Training
- [Fine-tuning Guide](https://huggingface.co/docs/transformers/training)
- [Hyperparameter Search](https://huggingface.co/docs/transformers/hpo_train)

## 🆘 Getting Help

### Before Asking for Help

1. Run `python check_system.py`
2. Check error messages carefully
3. Review troubleshooting section
4. Read relevant documentation

### Include in Help Requests

1. Error message (full traceback)
2. System specifications
3. Python/package versions
4. What you've already tried
5. Relevant config settings

## 📝 Next Steps

1. ✅ Verify installation: `python check_system.py`
2. ✅ Setup project: `python setup.py`
3. ✅ Prepare data: `python prepare_data.py`
4. ✅ Start training: `python trainer.py`
5. ✅ Monitor progress: `python monitor.py`
6. ✅ Test model: `python inference.py --interactive`

## 🎉 Success Checklist

- [ ] Python 3.8+ installed
- [ ] GPU detected (optional but recommended)
- [ ] All packages installed
- [ ] Project structure created
- [ ] Audio files in `data/audio/`
- [ ] `metadata.csv` created
- [ ] Data preprocessed successfully
- [ ] Training started without errors
- [ ] Checkpoints being saved
- [ ] Loss decreasing over time
- [ ] Generated speech sounds good

## 📄 License & Credits

This project uses:
- Microsoft SpeechT5 (MIT License)
- Hugging Face Transformers (Apache 2.0)
- PyTorch (BSD License)

---

**Good luck with your Georgian TTS model training!** 🚀

For questions or issues, review the documentation or check the troubleshooting section.

# Quick Start Guide - Georgian TTS Training

## 🚀 Setup (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

**⚠️ SECURITY IMPORTANT:** Ensure PyTorch 2.6+ is installed (required for security fix):
```bash
pip install torch>=2.6.0 torchvision torchaudio
```

### 2. Prepare Your Data Structure
```
data/
├── audio/
│   ├── sample_001.wav
│   ├── sample_002.wav
│   └── ...
└── metadata.csv
```

### 3. Create metadata.csv
Format: `filename|text` (pipe-separated)

```csv
filename|text
sample_001.wav|ეს არის პირველი ნიმუში
sample_002.wav|საქართველო ჩემი სამშობლოა
sample_003.wav|გამარჯობა რა გაკეთონ
```

**Important:** 
- Audio files must be WAV format
- One line per audio file
- Text must be in Georgian
- No header row needed

## 📝 Step-by-Step Training

### Step 1: Prepare Data
```bash
python prepare_data.py
```

This will:
- ✅ Validate all audio files
- ✅ Resample to 16kHz
- ✅ Clean and normalize text
- ✅ Split into train/validation sets
- ✅ Create processed_data/ folder

### Step 2: Train Model
```bash
python trainer.py
```

Training will:
- Load SpeechT5 model
- Fine-tune on your Georgian data
- Save checkpoints every 500 steps
- Save final model to output/

**Expected time:** 2-12 hours depending on data size

### Step 3: Test Your Model
```bash
# Single text
python inference.py --text "ეს არის ტესტი"

# Interactive mode
python inference.py --interactive

# Batch processing
python inference.py --batch texts.txt
```

## ⚙️ Configuration

Edit `config.py` to adjust:

### Common adjustments:
```python
BATCH_SIZE = 4          # Reduce if out of memory
LEARNING_RATE = 1e-5    # Increase if training too slow
NUM_EPOCHS = 50         # More epochs = better quality
SAMPLE_RATE = 16000     # Audio sample rate
```

### GPU Memory Issues?
```python
BATCH_SIZE = 2  # or 1
GRADIENT_ACCUMULATION_STEPS = 8  # Increase this
FP16 = True  # Enable mixed precision
```

## 📊 Monitoring Training

### TensorBoard (Real-time monitoring)
```bash
tensorboard --logdir logs/
```
Open browser: http://localhost:6006

### Look for:
- ✅ Loss decreasing over time
- ✅ Validation loss following training loss
- ⚠️ If validation loss increases: reduce learning rate or stop training

## 🎯 Data Quality Tips

### Good Audio:
✅ Clear speech, minimal background noise
✅ Consistent volume levels
✅ Native Georgian speaker
✅ Natural speaking pace
✅ 1-20 seconds per clip

### Good Text:
✅ Matches audio exactly
✅ Proper Georgian spelling
✅ Natural sentences
✅ No special symbols or numbers (write them out)

### How Much Data?
- **Minimum:** 30 minutes (200-300 samples)
- **Good:** 2 hours (800-1000 samples)  
- **Excellent:** 5+ hours (2000+ samples)

## 🐛 Troubleshooting

### "CUDA out of memory"
```python
# In config.py
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
```

### "Audio file not found"
- Check that audio files are in `data/audio/`
- Verify filenames in metadata.csv match exactly
- Ensure WAV format

### "Poor quality output"
- Train for more epochs
- Check training data quality
- Ensure transcriptions are accurate
- Try lower learning rate: `LEARNING_RATE = 5e-6`

### "Training very slow"
- Increase batch size if GPU memory allows
- Reduce audio quality: `SAMPLE_RATE = 16000`
- Use fewer workers: `DATALOADER_NUM_WORKERS = 2`

## 📁 File Outputs

After training, you'll have:
```
├── processed_data/     # Preprocessed dataset
├── checkpoints/        # Training checkpoints
├── output/            # Final trained model ⭐
├── logs/              # Training logs
└── output_001.wav     # Generated speech samples
```

## 🎤 Using Your Trained Model

### Python API
```python
from inference import generate_speech

generate_speech(
    text="ეს არის ტესტი",
    model_path="output/",
    output_path="my_speech.wav"
)
```

### Command Line
```bash
python inference.py --text "გამარჯობა" --output hello.wav
```

## 🔄 Next Steps

1. **Test quality:** Generate speech with various texts
2. **Iterate:** If quality is poor, gather more data and retrain
3. **Fine-tune:** Adjust hyperparameters in config.py
4. **Deploy:** Integrate into your application
5. **Share:** (Optional) Upload to Hugging Face Hub

## 💡 Tips for Best Results

1. **Use consistent recording setup** for all audio
2. **Include variety** in text (questions, statements, emotions)
3. **Start small** (100 samples) to verify pipeline works
4. **Monitor training** using TensorBoard
5. **Save checkpoints** - earlier checkpoints might sound better
6. **Test regularly** during training

## 📚 Need Help?

- Check the full guide: `TTS_FINETUNING_GUIDE.md`
- Review config options: `config.py`
- Inspect your data: `python prepare_data.py`

## ✨ Example Workflow

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Prepare data
python prepare_data.py

# 3. Train (takes several hours)
python trainer.py

# 4. Test
python inference.py --interactive

# 5. Monitor
tensorboard --logdir logs/
```

Good luck with your Georgian TTS model! 🎉

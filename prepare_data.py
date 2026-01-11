"""
Data Preparation Script for TTS Fine-tuning
This script preprocesses your audio files and transcriptions
"""

import os
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
from pathlib import Path
import config

def clean_text(text):
    """Clean and normalize Georgian text"""
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters if needed (customize for Georgian)
    # Georgian alphabet: ა-ჰ
    # Keep only Georgian letters, spaces, and basic punctuation
    
    return text.strip()

def validate_audio(audio_path, min_length=config.MIN_AUDIO_LENGTH, 
                   max_length=config.MAX_AUDIO_LENGTH):
    """Check if audio file meets requirements"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        duration = len(audio) / sr
        
        # Check duration
        if duration < min_length or duration > max_length:
            return False, f"Duration {duration:.2f}s out of range"
        
        # Check for silence (very low energy)
        if np.mean(np.abs(audio)) < 0.001:
            return False, "Audio too quiet or silent"
        
        return True, "OK"
    except Exception as e:
        return False, f"Error: {str(e)}"

def resample_audio(audio_path, output_path, target_sr=config.SAMPLE_RATE):
    """Resample audio to target sample rate"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Resample if necessary
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        # Save
        sf.write(output_path, audio, target_sr)
        return True
    except Exception as e:
        print(f"Error resampling {audio_path}: {e}")
        return False

def prepare_dataset():
    """Main function to prepare the dataset"""
    
    print("=" * 70)
    print("TTS Data Preparation")
    print("=" * 70)
    
    # Create necessary directories
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    processed_audio_dir = os.path.join(config.PROCESSED_DATA_DIR, "audio")
    os.makedirs(processed_audio_dir, exist_ok=True)
    
    # Load metadata
    print(f"\nLoading metadata from: {config.METADATA_FILE}")
    if not os.path.exists(config.METADATA_FILE):
        print(f"ERROR: Metadata file not found at {config.METADATA_FILE}")
        print("\nCreate a metadata.csv file with columns: filename|text")
        print("Example:")
        print("sample_001.wav|ეს არის ქართული ტექსტი")
        print("sample_002.wav|საქართველო არის ქვეყანა")
        return
    
    df = pd.read_csv(config.METADATA_FILE, sep='|')
    print(f"Found {len(df)} entries in metadata")
    
    # Validate and process each entry
    valid_entries = []
    invalid_count = 0
    
    print("\nValidating and processing audio files...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['filename']
        text = row['text']
        
        # Input and output paths
        input_path = os.path.join(config.AUDIO_DIR, filename)
        output_filename = f"processed_{filename}"
        output_path = os.path.join(processed_audio_dir, output_filename)
        
        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"  Warning: Audio file not found: {input_path}")
            invalid_count += 1
            continue
        
        # Validate audio
        is_valid, message = validate_audio(input_path)
        if not is_valid:
            print(f"  Skipping {filename}: {message}")
            invalid_count += 1
            continue
        
        # Clean text
        cleaned_text = clean_text(text)
        
        # Check text length
        if len(cleaned_text) < config.MIN_TEXT_LENGTH or \
           len(cleaned_text) > config.MAX_TEXT_LENGTH:
            print(f"  Skipping {filename}: Text length {len(cleaned_text)} out of range")
            invalid_count += 1
            continue
        
        # Resample and normalize audio
        if resample_audio(input_path, output_path):
            valid_entries.append({
                'filename': output_filename,
                'text': cleaned_text,
                'original_filename': filename
            })
        else:
            invalid_count += 1
    
    # Save processed metadata
    if len(valid_entries) == 0:
        print("\nERROR: No valid entries found!")
        return
    
    processed_df = pd.DataFrame(valid_entries)
    processed_metadata_path = os.path.join(config.PROCESSED_DATA_DIR, "metadata.csv")
    processed_df.to_csv(processed_metadata_path, sep='|', index=False)
    
    # Split into train and validation
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        processed_df, 
        test_size=1-config.TRAIN_TEST_SPLIT,
        random_state=config.RANDOM_SEED
    )
    
    train_df.to_csv(
        os.path.join(config.PROCESSED_DATA_DIR, "train.csv"),
        sep='|', index=False
    )
    val_df.to_csv(
        os.path.join(config.PROCESSED_DATA_DIR, "val.csv"),
        sep='|', index=False
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("Data Preparation Complete!")
    print("=" * 70)
    print(f"Total entries processed: {len(processed_df)}")
    print(f"Invalid/skipped entries: {invalid_count}")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"\nProcessed files saved to: {config.PROCESSED_DATA_DIR}")
    print(f"Sample rate: {config.SAMPLE_RATE} Hz")
    print("\nYou can now run: python trainer.py")
    print("=" * 70)

if __name__ == "__main__":
    prepare_dataset()

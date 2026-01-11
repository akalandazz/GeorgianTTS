#!/usr/bin/env python3
"""
Project Setup Script
Run this once to set up the project structure
"""

import os
from pathlib import Path

def create_directory_structure():
    """Create all necessary directories"""
    
    directories = [
        "data/audio",
        "processed_data",
        "checkpoints",
        "output",
        "logs",
        "batch_outputs"
    ]
    
    print("Creating project directory structure...")
    print("=" * 70)
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}/")
    
    print("=" * 70)
    print("\nDirectory structure created successfully!")

def create_example_metadata():
    """Create example metadata file if it doesn't exist"""
    
    metadata_path = "data/metadata.csv"
    
    if os.path.exists(metadata_path):
        print(f"\n⚠️  metadata.csv already exists at {metadata_path}")
        print("   Skipping example creation to avoid overwriting.")
        return
    
    example_content = """filename|text
sample_001.wav|გამარჯობა ეს არის პირველი ნიმუში
sample_002.wav|საქართველო არის ძალიან ლამაზი ქვეყანა
sample_003.wav|მე მიყვარს ქართული ენა და კულტურა
sample_004.wav|თბილისი არის საქართველოს დედაქალაქი
sample_005.wav|როგორ ხარ დღეს ძალიან კარგად ვარ"""
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write(example_content)
    
    print(f"\n✓ Created example metadata.csv at {metadata_path}")
    print("  Replace this with your actual data before training!")

def check_dependencies():
    """Check if key dependencies are installed"""
    
    print("\nChecking dependencies...")
    print("=" * 70)
    
    dependencies = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('datasets', 'Datasets'),
        ('librosa', 'Librosa'),
        ('soundfile', 'SoundFile'),
    ]
    
    missing = []
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✓ {name} is installed")
        except ImportError:
            print(f"✗ {name} is NOT installed")
            missing.append(name)
    
    print("=" * 70)
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("   Install them with: pip install -r requirements.txt")
    else:
        print("\n✓ All key dependencies are installed!")

def print_next_steps():
    """Print next steps for the user"""
    
    print("\n" + "=" * 70)
    print("Setup Complete! Next Steps:")
    print("=" * 70)
    print("\n1. Prepare your data:")
    print("   - Place .wav audio files in: data/audio/")
    print("   - Update metadata.csv with your transcriptions")
    print("   - Format: filename|text (pipe-separated)")
    
    print("\n2. Preprocess your data:")
    print("   python prepare_data.py")
    
    print("\n3. Start training:")
    print("   python trainer.py")
    
    print("\n4. Test your model:")
    print('   python inference.py --text "ეს არის ტესტი"')
    
    print("\n5. Monitor training (optional):")
    print("   tensorboard --logdir logs/")
    
    print("\n" + "=" * 70)
    print("For detailed instructions, see:")
    print("  - QUICKSTART.md (quick guide)")
    print("  - TTS_FINETUNING_GUIDE.md (complete guide)")
    print("  - README.md (project overview)")
    print("=" * 70)

def main():
    print("\n" + "=" * 70)
    print("Georgian TTS Project Setup")
    print("=" * 70)
    
    # Create directories
    create_directory_structure()
    
    # Create example metadata
    create_example_metadata()
    
    # Check dependencies
    check_dependencies()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()

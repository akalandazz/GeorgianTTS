#!/usr/bin/env python3
"""
Installation Verification Script
Checks if your system is ready for TTS training
"""

import sys
import subprocess

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro}")
        print("  Required: Python 3.8 or higher")
        return False

def check_pip():
    """Check if pip is available"""
    print("\nChecking pip...")
    try:
        import pip
        print(f"✓ pip is available")
        return True
    except ImportError:
        print("✗ pip not found")
        return False

def check_gpu():
    """Check for NVIDIA GPU and CUDA"""
    print("\nChecking GPU/CUDA...")
    try:
        import torch
        
        # Check PyTorch version for security
        torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        if torch_version < (2, 6):
            print(f"✗ PyTorch {torch.__version__} - SECURITY RISK!")
            print("  ⚠️  CRITICAL: PyTorch 2.6+ required due to torch.load vulnerability")
            print("  Upgrade with: pip install torch>=2.6.0")
            return False
        else:
            print(f"✓ PyTorch {torch.__version__} (secure version)")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  PyTorch CUDA: {torch.cuda.is_available()}")
            
            # Check memory
            props = torch.cuda.get_device_properties(0)
            total_memory_gb = props.total_memory / 1e9
            print(f"  GPU Memory: {total_memory_gb:.2f} GB")
            
            if total_memory_gb < 8:
                print("  ⚠️  Warning: Less than 8GB GPU memory")
                print("     You may need to reduce batch size")
            
            return True
        else:
            print("✗ CUDA not available")
            print("  Training will be VERY slow on CPU")
            print("  Consider using a GPU or cloud service")
            return False
    except ImportError:
        print("✗ PyTorch not installed yet")
        return None

def check_packages():
    """Check if required packages are installed"""
    print("\nChecking required packages...")
    
    packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'datasets': 'Datasets',
        'librosa': 'Librosa',
        'soundfile': 'SoundFile',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'tqdm': 'tqdm',
        'accelerate': 'Accelerate',
        'tensorboard': 'TensorBoard'
    }
    
    all_installed = True
    missing = []
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (not installed)")
            missing.append(package)
            all_installed = False
    
    if not all_installed:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("\nTo install all requirements:")
        print("  pip install -r requirements.txt")
    
    return all_installed

def check_disk_space():
    """Check available disk space"""
    print("\nChecking disk space...")
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        
        free_gb = free / (1024**3)
        print(f"  Free space: {free_gb:.2f} GB")
        
        if free_gb < 10:
            print("  ⚠️  Warning: Less than 10GB free")
            print("     You may run out of space during training")
            return False
        else:
            print("  ✓ Sufficient disk space")
            return True
    except Exception as e:
        print(f"  Could not check disk space: {e}")
        return None

def check_ffmpeg():
    """Check if ffmpeg is installed (for audio processing)"""
    print("\nChecking ffmpeg...")
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("  ✓ ffmpeg is installed")
            return True
        else:
            print("  ✗ ffmpeg not working properly")
            return False
    except FileNotFoundError:
        print("  ⚠️  ffmpeg not found (optional)")
        print("     Install with: sudo apt install ffmpeg")
        return None
    except Exception as e:
        print(f"  Could not check ffmpeg: {e}")
        return None

def print_summary(results):
    """Print summary of checks"""
    print("\n" + "=" * 70)
    print("INSTALLATION SUMMARY")
    print("=" * 70)
    
    all_ok = all(r is not False for r in results.values())
    
    if all_ok:
        print("✓ Your system is ready for TTS training!")
        print("\nNext steps:")
        print("1. Run setup: python setup.py")
        print("2. Prepare data: python prepare_data.py")
        print("3. Start training: python trainer.py")
    else:
        print("⚠️  Some issues were found:")
        
        if not results['python']:
            print("  • Upgrade Python to 3.8+")
        if not results['pip']:
            print("  • Install pip")
        if results['packages'] is False:
            print("  • Install missing packages: pip install -r requirements.txt")
        if results['gpu'] is False:
            print("  • GPU not available (training will be slow)")
        if results['disk'] is False:
            print("  • Free up disk space (need 10GB+)")
        
        print("\nFix these issues before training.")
    
    print("=" * 70)

def main():
    print("=" * 70)
    print("TTS TRAINING - SYSTEM CHECK")
    print("=" * 70)
    
    results = {
        'python': check_python_version(),
        'pip': check_pip(),
        'gpu': check_gpu(),
        'packages': check_packages(),
        'disk': check_disk_space(),
        'ffmpeg': check_ffmpeg()
    }
    
    print_summary(results)

if __name__ == "__main__":
    main()

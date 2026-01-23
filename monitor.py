#!/usr/bin/env python3
"""
Training Monitor Utility
View training progress and statistics
"""

import os
import json
import glob
from pathlib import Path
import config

def parse_tensorboard_logs():
    """Parse TensorBoard event files to show training progress"""
    
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        log_dir = config.LOG_DIR
        
        if not os.path.exists(log_dir):
            print(f"Log directory not found: {log_dir}")
            return None
        
        # Find event files
        event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
        
        if not event_files:
            print(f"No TensorBoard event files found in {log_dir}")
            return None
        
        # Load the most recent event file
        event_file = max(event_files, key=os.path.getctime)
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        
        # Get available tags
        tags = ea.Tags()
        
        stats = {}
        
        # Parse loss metrics
        if 'scalars' in tags:
            for tag in tags['scalars']:
                events = ea.Scalars(tag)
                if events:
                    stats[tag] = {
                        'current': events[-1].value,
                        'min': min(e.value for e in events),
                        'max': max(e.value for e in events),
                        'steps': len(events)
                    }
        
        return stats
    
    except ImportError:
        print("TensorBoard not installed. Install with: pip install tensorboard")
        return None
    except Exception as e:
        print(f"Error parsing logs: {e}")
        return None

def check_checkpoints():
    """Check saved checkpoints"""
    
    checkpoint_dir = config.CHECKPOINT_DIR
    
    if not os.path.exists(checkpoint_dir):
        return []
    
    # Find all checkpoint directories
    checkpoints = []
    for item in os.listdir(checkpoint_dir):
        item_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(item_path) and item.startswith('checkpoint-'):
            try:
                step = int(item.split('-')[1])
                size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, _, filenames in os.walk(item_path)
                    for filename in filenames
                )
                checkpoints.append({
                    'name': item,
                    'step': step,
                    'size_mb': size / (1024 * 1024),
                    'path': item_path
                })
            except:
                pass
    
    return sorted(checkpoints, key=lambda x: x['step'])

def check_output_model():
    """Check if final model exists"""
    
    output_dir = config.OUTPUT_DIR
    
    if not os.path.exists(output_dir):
        return None
    
    # Check for model files
    model_files = ['config.json', 'pytorch_model.bin']
    has_model = all(
        os.path.exists(os.path.join(output_dir, f))
        for f in model_files
    )
    
    if has_model:
        size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, _, filenames in os.walk(output_dir)
            for filename in filenames
        )
        return {
            'exists': True,
            'size_mb': size / (1024 * 1024),
            'path': output_dir
        }
    
    return None

def display_status():
    """Display comprehensive training status"""
    
    print("\n" + "=" * 70)
    print("TTS Training Status Monitor")
    print("=" * 70)
    
    # Check data
    print("\n📊 DATA STATUS")
    print("-" * 70)
    
    processed_data = config.PROCESSED_DATA_DIR
    if os.path.exists(processed_data):
        train_file = os.path.join(processed_data, "train.csv")
        val_file = os.path.join(processed_data, "val.csv")
        
        if os.path.exists(train_file) and os.path.exists(val_file):
            import pandas as pd
            train_df = pd.read_csv(train_file, sep=config.CSV_SEPARATOR)
            val_df = pd.read_csv(val_file, sep=config.CSV_SEPARATOR)
            
            print(f"✓ Data preprocessed")
            print(f"  Training samples: {len(train_df)}")
            print(f"  Validation samples: {len(val_df)}")
            print(f"  Total samples: {len(train_df) + len(val_df)}")
        else:
            print("✗ Data not preprocessed yet")
            print("  Run: python prepare_data.py")
    else:
        print("✗ No processed data found")
        print("  Run: python prepare_data.py")
    
    # Check training progress
    print("\n🎯 TRAINING PROGRESS")
    print("-" * 70)
    
    stats = parse_tensorboard_logs()
    if stats:
        print("Training metrics (from TensorBoard logs):")
        for metric, values in stats.items():
            print(f"\n  {metric}:")
            print(f"    Current: {values['current']:.6f}")
            print(f"    Min: {values['min']:.6f}")
            print(f"    Max: {values['max']:.6f}")
            print(f"    Steps recorded: {values['steps']}")
    else:
        print("No training logs found")
        print("Training hasn't started yet or logs are not available")
    
    # Check checkpoints
    print("\n💾 CHECKPOINTS")
    print("-" * 70)
    
    checkpoints = check_checkpoints()
    if checkpoints:
        print(f"Found {len(checkpoints)} checkpoint(s):")
        for cp in checkpoints[-5:]:  # Show last 5
            print(f"  • Step {cp['step']}: {cp['size_mb']:.1f} MB")
        if len(checkpoints) > 5:
            print(f"  ... and {len(checkpoints) - 5} more")
    else:
        print("No checkpoints found yet")
    
    # Check final model
    print("\n✨ FINAL MODEL")
    print("-" * 70)
    
    model = check_output_model()
    if model:
        print(f"✓ Final model exists")
        print(f"  Location: {model['path']}")
        print(f"  Size: {model['size_mb']:.1f} MB")
        print("\n  Ready for inference!")
        print('  Test with: python inference.py --text "ტესტი"')
    else:
        print("Final model not found")
        print("Training may still be in progress")
    
    print("\n" + "=" * 70)
    print("Monitoring Commands:")
    print("=" * 70)
    print("  View this status: python monitor.py")
    print("  TensorBoard: tensorboard --logdir logs/")
    print("  Watch GPU: watch -n 1 nvidia-smi")
    print("=" * 70)

def main():
    display_status()

if __name__ == "__main__":
    main()

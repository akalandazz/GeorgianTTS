"""
Main Training Script for TTS Model Fine-tuning
This script handles the complete training pipeline
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import Dataset, Audio, load_dataset as hf_load_dataset
from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import config

# CRITICAL: Check PyTorch version for security
def check_pytorch_version():
    """Verify PyTorch version meets security requirements"""
    torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    if torch_version < (2, 6):
        print("\n" + "=" * 70)
        print("❌ CRITICAL SECURITY ERROR")
        print("=" * 70)
        print(f"PyTorch version {torch.__version__} is NOT SECURE!")
        print("\nA serious vulnerability exists in torch.load for PyTorch < 2.6.0")
        print("This vulnerability can allow malicious code execution.")
        print("\n" + "=" * 70)
        print("REQUIRED ACTION:")
        print("=" * 70)
        print("Upgrade PyTorch to version 2.6.0 or later:")
        print("\n  pip uninstall torch torchvision torchaudio")
        print("  pip install torch>=2.6.0 torchvision torchaudio")
        print("\nSee SECURITY_NOTICE.md for more information.")
        print("=" * 70)
        sys.exit(1)
    print(f"✓ PyTorch {torch.__version__} (secure version)")

# Run security check immediately
check_pytorch_version()

# Set random seeds for reproducibility
torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

print(f"Using device: {config.DEVICE}")
if config.DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


def load_speaker_embeddings():
    """Load speaker embeddings for SpeechT5"""
    try:
        print("\nLoading speaker embeddings...")
        embeddings_dataset = hf_load_dataset(
            "Matthijs/cmu-arctic-xvectors",
            split="validation"
        )
        # Use a consistent speaker embedding
        speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        print("✓ Loaded speaker embeddings from CMU Arctic dataset")
        return speaker_embedding
    except Exception as e:
        print(f"Warning: Could not load speaker embeddings: {e}")
        print("Creating dummy embeddings...")
        # Create dummy embeddings as fallback
        return torch.zeros(1, 512)


class TTSDataset:
    """Handle dataset loading and preprocessing"""
    
    def __init__(self, processor, speaker_embeddings):
        self.processor = processor
        self.speaker_embeddings = speaker_embeddings
        self.audio_dir = os.path.join(config.PROCESSED_DATA_DIR, "audio")
    
    def load_data(self, split="train"):
        """Load train or validation data"""
        csv_file = os.path.join(config.PROCESSED_DATA_DIR, f"{split}.csv")
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Data file not found: {csv_file}")
        
        df = pd.read_csv(csv_file, sep='|')
        
        # Create dataset dictionary
        data_dict = {
            'audio': [os.path.join(self.audio_dir, f) for f in df['filename']],
            'text': df['text'].tolist()
        }
        
        # Create HuggingFace Dataset
        dataset = Dataset.from_dict(data_dict)
        
        # Cast audio column to Audio type
        dataset = dataset.cast_column("audio", Audio(sampling_rate=config.SAMPLE_RATE))
        
        print(f"Loaded {len(dataset)} samples for {split}")
        return dataset
    
    def prepare_dataset(self, batch):
        """Preprocess a batch of data"""
        # Extract audio
        audio = batch["audio"]
        
        # Tokenize text
        text_inputs = self.processor.tokenizer(
            batch["text"],
            padding=False,
            truncation=True,
            max_length=config.MAX_TEXT_LENGTH
        )
        
        # Process audio to mel spectrogram
        audio_array = audio["array"]
        
        # Normalize audio
        if audio_array.max() > 0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Extract mel spectrogram
        mel_spec = self.processor.feature_extractor(
            audio_array,
            sampling_rate=audio["sampling_rate"],
            return_tensors="np"
        ).input_values[0]
        
        # Prepare output
        batch["input_ids"] = text_inputs["input_ids"]
        batch["labels"] = mel_spec.T  # Transpose to [time, mel_bins]
        batch["speaker_embeddings"] = self.speaker_embeddings[0].numpy()
        
        return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Custom data collator for TTS with proper padding"""
    
    processor: Any
    
    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, torch.Tensor]:
        # Extract input_ids
        input_ids = [{"input_ids": torch.tensor(feature["input_ids"])} for feature in features]
        
        # Pad input_ids using tokenizer
        batch = self.processor.tokenizer.pad(
            input_ids,
            return_tensors="pt",
            padding=True
        )
        
        # Process labels (mel spectrograms)
        # Find max length
        label_lengths = [feature["labels"].shape[0] for feature in features]
        max_label_length = max(label_lengths)
        
        # Get mel bins dimension
        mel_bins = features[0]["labels"].shape[1]
        
        # Pad labels manually
        labels = []
        labels_attention_mask = []
        
        for feature in features:
            label = feature["labels"]
            label_length = label.shape[0]
            
            # Pad to max length
            pad_length = max_label_length - label_length
            if pad_length > 0:
                padding = np.zeros((pad_length, mel_bins))
                label = np.concatenate([label, padding], axis=0)
            
            labels.append(label)
            
            # Create attention mask
            attention_mask = np.ones(max_label_length)
            attention_mask[label_length:] = 0
            labels_attention_mask.append(attention_mask)
        
        # Convert to tensors
        labels = torch.tensor(np.array(labels), dtype=torch.float32)
        labels_attention_mask = torch.tensor(np.array(labels_attention_mask), dtype=torch.long)
        
        # Mask padded positions with -100
        labels = labels.masked_fill(
            labels_attention_mask.unsqueeze(-1) == 0, -100.0
        )
        
        batch["labels"] = labels
        
        # Add speaker embeddings (all same for fine-tuning)
        speaker_embeddings = torch.tensor(
            np.array([feature["speaker_embeddings"] for feature in features]),
            dtype=torch.float32
        )
        batch["speaker_embeddings"] = speaker_embeddings
        
        return batch


def compute_metrics(pred):
    """Compute evaluation metrics"""
    # For TTS, we primarily monitor loss
    # Can add more sophisticated metrics here
    return {}


def train():
    """Main training function"""
    
    print("=" * 70)
    print("Starting TTS Model Training")
    print("=" * 70)
    
    # Create output directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Load speaker embeddings
    speaker_embeddings = load_speaker_embeddings()
    
    # Load processor and model
    print(f"\nLoading model: {config.MODEL_NAME}")
    processor = SpeechT5Processor.from_pretrained(config.MODEL_NAME)
    model = SpeechT5ForTextToSpeech.from_pretrained(config.MODEL_NAME)
    
    # Move model to device
    model = model.to(config.DEVICE)
    
    # Load datasets
    print("\nLoading datasets...")
    dataset_loader = TTSDataset(processor, speaker_embeddings)
    train_dataset = dataset_loader.load_data("train")
    val_dataset = dataset_loader.load_data("val")
    
    # Preprocess datasets
    print("\nPreprocessing datasets...")
    train_dataset = train_dataset.map(
        dataset_loader.prepare_dataset,
        remove_columns=train_dataset.column_names,
        desc="Preprocessing training data"
    )
    
    val_dataset = val_dataset.map(
        dataset_loader.prepare_dataset,
        remove_columns=val_dataset.column_names,
        desc="Preprocessing validation data"
    )
    
    # Create data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.CHECKPOINT_DIR,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        num_train_epochs=config.NUM_EPOCHS,
        max_steps=config.MAX_STEPS,
        warmup_steps=config.WARMUP_STEPS,
        logging_dir=config.LOG_DIR,
        logging_steps=config.LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_steps=config.SAVE_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=config.FP16 and torch.cuda.is_available(),
        dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
        push_to_hub=config.PUSH_TO_HUB,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        label_names=["labels"],
        predict_with_generate=False,  # Not using generate during training
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )
    
    # Start training
    print("\n" + "=" * 70)
    print("Training Configuration:")
    print("=" * 70)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Gradient accumulation: {config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Device: {config.DEVICE}")
    print(f"Mixed precision (FP16): {config.FP16 and torch.cuda.is_available()}")
    print("=" * 70)
    print("\nStarting training...\n")
    
    # Train
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(config.OUTPUT_DIR)
    processor.save_pretrained(config.OUTPUT_DIR)
    
    # Save speaker embeddings for inference
    torch.save(
        speaker_embeddings,
        os.path.join(config.OUTPUT_DIR, "speaker_embeddings.pt")
    )
    print(f"✓ Saved speaker embeddings")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Model saved to: {config.OUTPUT_DIR}")
    print(f"Checkpoints saved to: {config.CHECKPOINT_DIR}")
    print(f"Logs saved to: {config.LOG_DIR}")
    print("\nTo test your model, run:")
    print('python inference.py --text "თქვენი ტექსტი აქ"')
    print("=" * 70)


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()

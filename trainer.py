"""
Main Training Script for TTS Model Fine-tuning
This script handles the complete training pipeline
"""

import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import Dataset, Audio
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

# Set random seeds for reproducibility
torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

print(f"Using device: {config.DEVICE}")
if config.DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


class TTSDataset:
    """Handle dataset loading and preprocessing"""
    
    def __init__(self, processor):
        self.processor = processor
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
        text_inputs = self.processor(
            text=batch["text"],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Process audio to get mel spectrogram
        audio_inputs = self.processor(
            audio=audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        )
        
        # Add labels (mel spectrogram is both input and target for TTS)
        batch["labels"] = audio_inputs["input_values"][0]
        batch["input_ids"] = text_inputs["input_ids"][0]
        
        return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Custom data collator for TTS"""
    
    processor: Any
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Separate input_ids and labels
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        
        # Pad input_ids
        batch = self.processor.pad(
            input_ids,
            return_tensors="pt",
            padding=True
        )
        
        # Pad labels
        labels_batch = self.processor.pad(
            label_features,
            return_tensors="pt",
            padding=True
        )
        
        # Replace padding with -100 to ignore in loss
        labels = labels_batch["input_values"]
        labels = labels.masked_fill(
            labels_batch.attention_mask.unsqueeze(-1).eq(0), -100.0
        )
        
        batch["labels"] = labels
        
        return batch


def compute_metrics(pred):
    """Compute evaluation metrics"""
    # For TTS, we typically look at the loss
    # You can add more sophisticated metrics here
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
    
    # Load processor and model
    print(f"\nLoading model: {config.MODEL_NAME}")
    processor = SpeechT5Processor.from_pretrained(config.MODEL_NAME)
    model = SpeechT5ForTextToSpeech.from_pretrained(config.MODEL_NAME)
    
    # Move model to device
    model = model.to(config.DEVICE)
    
    # Load datasets
    print("\nLoading datasets...")
    dataset_loader = TTSDataset(processor)
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
        fp16=config.FP16,
        dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
        push_to_hub=config.PUSH_TO_HUB,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        label_names=["labels"],
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
    print("=" * 70)
    print("\nStarting training...\n")
    
    # Train
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(config.OUTPUT_DIR)
    processor.save_pretrained(config.OUTPUT_DIR)
    
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

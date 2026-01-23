"""
Inference Script for Trained TTS Model
Use this to test your fine-tuned model
"""

import os
import torch
import soundfile as sf
import argparse
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import config
from utils import check_pytorch_version, load_speaker_embeddings

# Run security check immediately
check_pytorch_version()

def generate_speech(text, model_path=None, output_path="output.wav"):
    """
    Generate speech from text using the trained model
    
    Args:
        text: Georgian text to convert to speech
        model_path: Path to trained model (uses config.OUTPUT_DIR if None)
        output_path: Where to save the generated audio
    """
    if model_path is None:
        model_path = config.OUTPUT_DIR
    
    print(f"Loading model from: {model_path}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using: python trainer.py")
        return
    
    # Load processor and model
    processor = SpeechT5Processor.from_pretrained(model_path)
    model = SpeechT5ForTextToSpeech.from_pretrained(model_path)
    
    # Load vocoder (converts mel spectrogram to audio)
    vocoder = SpeechT5HifiGan.from_pretrained(config.VOCODER_NAME)
    
    # Move to device
    device = config.DEVICE
    model = model.to(device)
    vocoder = vocoder.to(device)
    
    # Load speaker embeddings
    speaker_embeddings = load_speaker_embeddings()
    speaker_embeddings = speaker_embeddings.to(device)
    
    print(f"\nGenerating speech for text: '{text}'")
    
    # Tokenize text
    inputs = processor(text=text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    # Generate speech
    with torch.no_grad():
        spectrogram = model.generate_speech(input_ids, speaker_embeddings)
        
        # Convert spectrogram to waveform
        speech = vocoder(spectrogram)
    
    # Save audio
    speech = speech.cpu().numpy()
    sf.write(output_path, speech, samplerate=config.SAMPLE_RATE)

    print(f"Audio saved to: {output_path}")
    print(f"Duration: {len(speech)/config.SAMPLE_RATE:.2f} seconds")
    
    return output_path

def interactive_mode(model_path=None):
    """Run in interactive mode - keep generating speech from user input"""
    print("\n" + "=" * 70)
    print("Interactive TTS Mode")
    print("=" * 70)
    print("Enter Georgian text to generate speech (or 'quit' to exit)")
    print("=" * 70 + "\n")
    
    counter = 1
    while True:
        text = input("\nEnter text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not text:
            print("Please enter some text")
            continue
        
        output_file = f"output_{counter:03d}.wav"
        try:
            generate_speech(text, model_path, output_file)
            counter += 1
        except Exception as e:
            print(f"Error generating speech: {e}")

def batch_mode(input_file, model_path=None, output_dir=None):
    """Generate speech for multiple texts from a file"""
    if output_dir is None:
        output_dir = config.BATCH_OUTPUT_DIR

    print(f"\nBatch mode: processing {input_file}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(texts)} texts to process")
    
    # Generate speech for each text
    for i, text in enumerate(texts, 1):
        output_file = os.path.join(output_dir, f"output_{i:03d}.wav")
        print(f"\n[{i}/{len(texts)}] Processing: {text[:50]}...")
        
        try:
            generate_speech(text, model_path, output_file)
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\nBatch processing complete. Files saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="TTS Inference Script")
    parser.add_argument(
        "--text",
        type=str,
        help="Text to convert to speech"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model (default: from config)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output audio file path"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="Process multiple texts from a file (one per line)"
    )
    
    args = parser.parse_args()
    
    # Interactive mode
    if args.interactive:
        interactive_mode(args.model)
    
    # Batch mode
    elif args.batch:
        batch_mode(args.batch, args.model)
    
    # Single text mode
    elif args.text:
        generate_speech(args.text, args.model, args.output)
    
    # No arguments - show usage
    else:
        print("\nTTS Inference Script")
        print("=" * 70)
        print("\nUsage examples:")
        print('  python inference.py --text "ეს არის ტესტი"')
        print('  python inference.py --text "თქვენი ტექსტი" --output my_audio.wav')
        print("  python inference.py --interactive")
        print("  python inference.py --batch texts.txt")
        print("\nFor more options: python inference.py --help")
        print("=" * 70)

if __name__ == "__main__":
    main()

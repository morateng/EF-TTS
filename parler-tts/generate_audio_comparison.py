#!/usr/bin/env python3

import torch
import torchaudio
import os
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from parler_tts.vector_utils import VectorLoader


def generate_with_vectors(
    model, tokenizer, vector_loader, 
    style_caption: str, text: str, output_path: str
):
    """Generate audio using precomputed vectors."""
    
    print(f"Generating with vectors: '{style_caption}' -> '{text}'")
    
    # Get vectors
    description_vectors, tokens, attributes = vector_loader.get_vectors_for_caption(style_caption)
    print(f"  Attributes: {attributes}")
    print(f"  Vector shape: {description_vectors.shape}")
    
    # Prepare inputs
    prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids
    
    # Configure model for vector mode
    model.config.use_precomputed_vectors = True
    model.config.precomputed_vector_dim = 1024
    
    # Generate with longer sequences for better audio
    with torch.no_grad():
        generation = model.generate(
            prompt_input_ids=prompt_input_ids,
            precomputed_vectors=description_vectors.unsqueeze(0),
            description_tokens=[tokens],
            attention_mask=torch.ones((1, description_vectors.shape[0])),
            do_sample=True,
            temperature=0.8,
            max_new_tokens=1000,  # Longer generation
            num_return_sequences=1,
            pad_token_id=model.config.pad_token_id,
        )
    
    # Decode to audio
    audio_arr = model.audio_encoder.decode(generation, audio_scales=[None]).audio_values
    
    # Save audio
    sample_rate = model.audio_encoder.config.sampling_rate
    torchaudio.save(output_path, audio_arr.squeeze(0).cpu(), sample_rate)
    print(f"  ✓ Saved to: {output_path}")
    return audio_arr


def generate_with_original(
    model, tokenizer, 
    style_caption: str, text: str, output_path: str
):
    """Generate audio using original text encoder."""
    
    print(f"Generating with original: '{style_caption}' -> '{text}'")
    
    # Prepare inputs
    description_input_ids = tokenizer(style_caption, return_tensors="pt").input_ids
    prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids
    
    # Configure model for original mode
    model.config.use_precomputed_vectors = False
    
    # Generate
    with torch.no_grad():
        generation = model.generate(
            input_ids=description_input_ids,
            prompt_input_ids=prompt_input_ids,
            do_sample=True,
            temperature=0.8,
            max_new_tokens=1000,  # Longer generation
            num_return_sequences=1,
            pad_token_id=model.config.pad_token_id,
        )
    
    # Decode to audio
    audio_arr = model.audio_encoder.decode(generation, audio_scales=[None]).audio_values
    
    # Save audio
    sample_rate = model.audio_encoder.config.sampling_rate
    torchaudio.save(output_path, audio_arr.squeeze(0).cpu(), sample_rate)
    print(f"  ✓ Saved to: {output_path}")
    return audio_arr


def main():
    """Generate multiple audio samples for comparison."""
    
    print("Loading model and setup...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-v1",
        torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
    vector_loader = VectorLoader("/Users/morateng/Code/EF-TTS/parler-tts")
    
    # Create output directory
    os.makedirs("audio_samples", exist_ok=True)
    
    # Test scenarios
    test_cases = [
        {
            "style": "female American quickly medium clean",
            "text": "Hello, this is a test of the precomputed vector system.",
            "name": "female_american_quick"
        },
        {
            "style": "male British slowly low animated",
            "text": "Good morning, how are you doing today?",
            "name": "male_british_slow"
        },
        {
            "style": "female Japanese moderate high clean", 
            "text": "Welcome to our amazing technology demonstration.",
            "name": "female_japanese_moderate"
        }
    ]
    
    print("\n" + "="*60)
    print("GENERATING AUDIO SAMPLES")
    print("="*60)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {case['name']} ---")
        
        # Generate with vectors
        try:
            vector_path = f"audio_samples/{case['name']}_vectors.wav"
            audio_vectors = generate_with_vectors(
                model, tokenizer, vector_loader,
                case['style'], case['text'], vector_path
            )
        except Exception as e:
            print(f"  ✗ Vector generation failed: {e}")
            audio_vectors = None
        
        # Generate with original
        try:
            original_path = f"audio_samples/{case['name']}_original.wav"
            audio_original = generate_with_original(
                model, tokenizer,
                case['style'], case['text'], original_path
            )
        except Exception as e:
            print(f"  ✗ Original generation failed: {e}")
            audio_original = None
        
        # Compare if both succeeded
        if audio_vectors is not None and audio_original is not None:
            if audio_vectors.shape == audio_original.shape:
                similarity = torch.cosine_similarity(
                    audio_vectors.flatten(),
                    audio_original.flatten(),
                    dim=0
                )
                print(f"  Audio similarity: {similarity.item():.4f}")
            else:
                print(f"  Different audio shapes: {audio_vectors.shape} vs {audio_original.shape}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("Generated audio files in ./audio_samples/:")
    
    if os.path.exists("audio_samples"):
        for filename in sorted(os.listdir("audio_samples")):
            if filename.endswith('.wav'):
                filepath = os.path.join("audio_samples", filename)
                print(f"  - {filename}")
                
        print(f"\nTotal files: {len([f for f in os.listdir('audio_samples') if f.endswith('.wav')])}")
        print("\nYou can now listen to these files to compare vector-based vs original audio generation!")
    
    print("\n✓ Audio generation completed!")


if __name__ == "__main__":
    main()
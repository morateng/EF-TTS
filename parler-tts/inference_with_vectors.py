#!/usr/bin/env python3

import torch
import torchaudio
from transformers import AutoTokenizer, AutoProcessor
from parler_tts import ParlerTTSForConditionalGeneration
from parler_tts.vector_utils import VectorLoader


def inference_with_precomputed_vectors(
    model_name: str = "parler-tts/parler-tts-mini-v1",
    style_caption: str = "A female voice with American accent speaks quickly at a medium pitch",
    text: str = "Hello, this is a test of the precomputed vector system.",
    output_path: str = "output_with_vectors.wav",
    vector_base_path: str = "/Users/morateng/Code/EF-TTS/parler-tts"
):
    """Generate audio using precomputed vectors instead of text encoder."""
    
    print(f"Loading model: {model_name}")
    # Load model and tokenizer
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Initialize vector loader
    print("Initializing vector loader...")
    vector_loader = VectorLoader(vector_base_path)
    
    # Get precomputed vectors for the style caption
    print(f"Processing style caption: '{style_caption}'")
    description_vectors, tokens, attributes = vector_loader.get_vectors_for_caption(style_caption)
    print(f"  Found attributes: {attributes}")
    print(f"  Vector shape: {description_vectors.shape}")
    
    # Tokenize the text prompt
    print(f"Processing text: '{text}'")
    prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids
    
    # Prepare inputs for the model
    inputs = {
        "prompt_input_ids": prompt_input_ids,
        "precomputed_vectors": description_vectors.unsqueeze(0),  # Add batch dimension
        "description_tokens": [tokens],  # For PEFT if enabled
        "attention_mask": torch.ones((1, description_vectors.shape[0])),  # Attention mask for vectors
    }
    
    # Update model config to use precomputed vectors
    model.config.use_precomputed_vectors = True
    model.config.precomputed_vector_dim = description_vectors.shape[-1]
    
    print("Generating audio...")
    
    # Generate audio
    with torch.no_grad():
        generation = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.8,
            max_length=2000,
        )
    
    # Decode audio
    print("Decoding audio...")
    audio_arr = model.audio_encoder.decode(generation, audio_scales=[None]).audio_values
    
    # Save audio
    print(f"Saving audio to: {output_path}")
    torchaudio.save(output_path, audio_arr.squeeze(0).cpu(), sample_rate=model.audio_encoder.config.sampling_rate)
    
    print("✓ Audio generation completed successfully!")
    return audio_arr


def inference_with_original_encoder(
    model_name: str = "parler-tts/parler-tts-mini-v1",
    style_caption: str = "A female voice with American accent speaks quickly at a medium pitch",
    text: str = "Hello, this is a test of the precomputed vector system.",
    output_path: str = "output_original.wav"
):
    """Generate audio using original text encoder for comparison."""
    
    print(f"Loading model for original inference: {model_name}")
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize inputs
    description_input_ids = tokenizer(style_caption, return_tensors="pt").input_ids
    prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids
    
    print("Generating audio with original encoder...")
    
    # Generate audio
    with torch.no_grad():
        generation = model.generate(
            input_ids=description_input_ids,
            prompt_input_ids=prompt_input_ids,
            do_sample=True,
            temperature=0.8,
            max_length=2000,
        )
    
    # Decode audio
    print("Decoding audio...")
    audio_arr = model.audio_encoder.decode(generation, audio_scales=[None]).audio_values
    
    # Save audio
    print(f"Saving audio to: {output_path}")
    torchaudio.save(output_path, audio_arr.squeeze(0).cpu(), sample_rate=model.audio_encoder.config.sampling_rate)
    
    print("✓ Original audio generation completed successfully!")
    return audio_arr


def main():
    """Test both inference methods."""
    
    style_caption = "A female voice with American accent speaks quickly at a medium pitch"
    text = "Hello, this is a test of the precomputed vector system."
    
    print("="*60)
    print("TESTING PRECOMPUTED VECTOR INFERENCE")
    print("="*60)
    
    try:
        audio_vectors = inference_with_precomputed_vectors(
            style_caption=style_caption,
            text=text,
            output_path="vector_output.wav"
        )
        print("✓ Vector-based inference successful!")
    except Exception as e:
        print(f"✗ Vector-based inference failed: {e}")
        audio_vectors = None
    
    print("\n" + "="*60)
    print("TESTING ORIGINAL ENCODER INFERENCE")
    print("="*60)
    
    try:
        audio_original = inference_with_original_encoder(
            style_caption=style_caption,
            text=text,
            output_path="original_output.wav"
        )
        print("✓ Original inference successful!")
    except Exception as e:
        print(f"✗ Original inference failed: {e}")
        audio_original = None
    
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    if audio_vectors is not None and audio_original is not None:
        print(f"Vector output shape: {audio_vectors.shape}")
        print(f"Original output shape: {audio_original.shape}")
        
        # Simple similarity check
        if audio_vectors.shape == audio_original.shape:
            similarity = torch.cosine_similarity(
                audio_vectors.flatten(), 
                audio_original.flatten(), 
                dim=0
            )
            print(f"Cosine similarity: {similarity:.4f}")
        else:
            print("Different shapes - cannot compute similarity")
    
    print("\nFiles generated:")
    print("- vector_output.wav (using precomputed vectors)")
    print("- original_output.wav (using text encoder)")


if __name__ == "__main__":
    main()
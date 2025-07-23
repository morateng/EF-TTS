#!/usr/bin/env python3

import torch
import torchaudio
import os
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from parler_tts.vector_utils import VectorLoader


def simple_audio_generation():
    """Simple audio generation test."""
    
    print("Loading model...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-v1",
        torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
    
    # Create output directory
    os.makedirs("audio_samples", exist_ok=True)
    
    # Simple test case
    style_caption = "A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very close-sounding environment with very clear audio quality."
    text = "Hello world"
    
    print(f"Style: {style_caption}")
    print(f"Text: {text}")
    
    # Tokenize
    description_input_ids = tokenizer(style_caption, return_tensors="pt").input_ids
    prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids
    
    print(f"Description tokens: {description_input_ids.shape}")
    print(f"Prompt tokens: {prompt_input_ids.shape}")
    
    # Generate audio
    print("Generating audio...")
    with torch.no_grad():
        generation = model.generate(
            input_ids=description_input_ids,
            prompt_input_ids=prompt_input_ids,
            max_new_tokens=200,
            do_sample=True,
            temperature=1.0,
        )
    
    print(f"Generation shape: {generation.shape}")
    
    # Decode audio
    print("Decoding audio...")
    # Generation shape should be (batch_size * num_codebooks, seq_len), but it's coming as (batch_size, total_length)
    # We need to reshape properly
    batch_size = generation.shape[0]  # Should be 1
    total_len = generation.shape[1]
    num_codebooks = model.decoder.num_codebooks
    
    # The total length should be divisible by num_codebooks, if not, truncate
    seq_len = total_len // num_codebooks
    adjusted_total_len = seq_len * num_codebooks
    
    print(f"Original total length: {total_len}, Adjusted: {adjusted_total_len}")
    
    if adjusted_total_len < total_len:
        print(f"Truncating from {total_len} to {adjusted_total_len}")
        generation = generation[:, :adjusted_total_len]
    
    # Reshape to (batch_size, num_codebooks, seq_len)
    generation_reshaped = generation.view(batch_size, num_codebooks, seq_len)
    print(f"Reshaped generation: {generation_reshaped.shape}")
    
    # Add frame dimension: (frames=1, batch_size, num_codebooks, seq_len)
    generation_with_frame = generation_reshaped.unsqueeze(0)
    print(f"With frame dimension: {generation_with_frame.shape}")
    
    audio_arr = model.audio_encoder.decode(generation_with_frame, audio_scales=[None]).audio_values
    print(f"Audio shape: {audio_arr.shape}")
    
    # Save
    output_path = "audio_samples/simple_test.wav"
    sample_rate = model.audio_encoder.config.sampling_rate
    torchaudio.save(output_path, audio_arr.squeeze(0).cpu(), sample_rate)
    
    print(f"✓ Audio saved to: {output_path}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {audio_arr.shape[-1] / sample_rate:.2f} seconds")


if __name__ == "__main__":
    simple_audio_generation()
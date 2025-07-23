#!/usr/bin/env python3

import torch
import torchaudio
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from parler_tts.vector_utils import VectorLoader


def simple_vector_inference():
    """Simple test to verify vector loading works with the model."""
    
    print("Loading model...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-v1", 
        torch_dtype=torch.float32  # Use float32 for CPU
    )
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
    
    print("Loading vectors...")
    vector_loader = VectorLoader("/Users/morateng/Code/EF-TTS/parler-tts")
    
    style_caption = "female American quickly medium"
    text = "Hello world"
    
    print(f"Style caption: {style_caption}")
    print(f"Text: {text}")
    
    # Get vectors
    description_vectors, tokens, attributes = vector_loader.get_vectors_for_caption(style_caption)
    print(f"Vector shape: {description_vectors.shape}")
    print(f"Attributes: {attributes}")
    
    # Prepare inputs
    prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids
    
    # Update model config
    model.config.use_precomputed_vectors = True
    model.config.precomputed_vector_dim = 1024
    
    print("Running forward pass...")
    
    # Prepare decoder input_ids (start token) - shape should be (bsz * num_codebooks, seq_len)
    pad_token_id = model.config.pad_token_id
    num_codebooks = model.decoder.num_codebooks
    decoder_input_ids = torch.ones((1 * num_codebooks, 1), dtype=torch.long) * pad_token_id
    
    with torch.no_grad():
        try:
            # Just test the forward pass, not generation
            outputs = model(
                prompt_input_ids=prompt_input_ids,
                decoder_input_ids=decoder_input_ids,
                precomputed_vectors=description_vectors.unsqueeze(0),
                description_tokens=[tokens],
                attention_mask=torch.ones((1, description_vectors.shape[0])),
            )
            print(f"✓ Forward pass successful!")
            print(f"  Logits shape: {outputs.logits.shape}")
            
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    simple_vector_inference()
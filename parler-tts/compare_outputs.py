#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from parler_tts.vector_utils import VectorLoader


def compare_vector_vs_original():
    """Compare outputs from vector-based vs original text encoder."""
    
    print("Loading model and tokenizer...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-v1", 
        torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
    vector_loader = VectorLoader("/Users/morateng/Code/EF-TTS/parler-tts")
    
    # Test data
    style_caption = "female American quickly medium"
    text = "Hello world"
    
    # Common decoder setup
    pad_token_id = model.config.pad_token_id
    num_codebooks = model.decoder.num_codebooks
    decoder_input_ids = torch.ones((1 * num_codebooks, 1), dtype=torch.long) * pad_token_id
    prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids
    
    print(f"Style caption: '{style_caption}'")
    print(f"Text: '{text}'")
    print()
    
    with torch.no_grad():
        # Test 1: Vector-based inference
        print("=== VECTOR-BASED INFERENCE ===")
        description_vectors, tokens, attributes = vector_loader.get_vectors_for_caption(style_caption)
        print(f"Vector shape: {description_vectors.shape}")
        print(f"Attributes: {attributes}")
        
        model.config.use_precomputed_vectors = True
        model.config.precomputed_vector_dim = 1024
        
        outputs_vector = model(
            prompt_input_ids=prompt_input_ids,
            decoder_input_ids=decoder_input_ids,
            precomputed_vectors=description_vectors.unsqueeze(0),
            description_tokens=[tokens],
            attention_mask=torch.ones((1, description_vectors.shape[0])),
        )
        
        print(f"Output logits shape: {outputs_vector.logits.shape}")
        print(f"Max logit value: {outputs_vector.logits.max().item():.4f}")
        print(f"Min logit value: {outputs_vector.logits.min().item():.4f}")
        print(f"Mean logit value: {outputs_vector.logits.mean().item():.4f}")
        print()
        
        # Test 2: Original text encoder
        print("=== ORIGINAL TEXT ENCODER ===")
        description_input_ids = tokenizer(style_caption, return_tensors="pt").input_ids
        print(f"Description input shape: {description_input_ids.shape}")
        
        model.config.use_precomputed_vectors = False  # Disable vector mode
        
        outputs_original = model(
            input_ids=description_input_ids,
            prompt_input_ids=prompt_input_ids,
            decoder_input_ids=decoder_input_ids,
        )
        
        print(f"Output logits shape: {outputs_original.logits.shape}")
        print(f"Max logit value: {outputs_original.logits.max().item():.4f}")
        print(f"Min logit value: {outputs_original.logits.min().item():.4f}")
        print(f"Mean logit value: {outputs_original.logits.mean().item():.4f}")
        print()
        
        # Comparison
        print("=== COMPARISON ===")
        if outputs_vector.logits.shape == outputs_original.logits.shape:
            # Calculate similarity
            cosine_sim = torch.cosine_similarity(
                outputs_vector.logits.flatten(),
                outputs_original.logits.flatten(),
                dim=0
            )
            
            # Calculate MSE
            mse = torch.nn.functional.mse_loss(
                outputs_vector.logits,
                outputs_original.logits
            )
            
            # Calculate max absolute difference
            max_diff = (outputs_vector.logits - outputs_original.logits).abs().max()
            
            print(f"Cosine similarity: {cosine_sim.item():.6f}")
            print(f"MSE: {mse.item():.6f}")
            print(f"Max absolute difference: {max_diff.item():.6f}")
            
            # Sample a few logits for inspection
            print(f"Vector logits sample: {outputs_vector.logits[0, 0, :5].numpy()}")
            print(f"Original logits sample: {outputs_original.logits[0, 0, :5].numpy()}")
            
        else:
            print(f"Different shapes: {outputs_vector.logits.shape} vs {outputs_original.logits.shape}")
    
    print("\n✓ Comparison completed successfully!")


if __name__ == "__main__":
    compare_vector_vs_original()
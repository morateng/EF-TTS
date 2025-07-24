#!/usr/bin/env python3
"""
Simple test script to verify vector training pipeline works
"""

import torch
import os
import sys
sys.path.append('.')

from parler_tts import ParlerTTSForConditionalGeneration
from parler_tts.vector_utils import VectorLoader
from parler_tts.peft_modules import PrecomputedVectorPEFT
from transformers import AutoTokenizer

def test_vector_training():
    print("🚀 Testing vector training pipeline...")
    
    # Setup
    device = torch.device('cpu')
    print(f"📱 Device: {device}")
    
    # Load components
    print("📦 Loading model and tokenizer...")
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1")
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
    
    # Configure for vector mode
    model.config.use_precomputed_vectors = True
    model.config.precomputed_vector_dim = 1024
    
    print("🔧 Loading vector loader...")
    vector_loader = VectorLoader("./")
    
    # Test vector loading
    print("🧪 Testing vector loading...")
    test_caption = "female American quickly medium clean"
    vectors, tokens, attributes = vector_loader.get_vectors_for_caption(test_caption)
    print(f"   Caption: {test_caption}")
    print(f"   Tokens: {tokens}")
    print(f"   Vector shape: {vectors.shape}")
    print(f"   Attributes: {attributes}")
    
    # Setup PEFT module
    print("⚙️  Setting up PEFT module...")
    attribute_values = {
        "gender": ["male", "female"],
        "pitch": ["high", "low", "medium"],
        "speed": ["slowly", "quickly", "moderate"],
        "accent": ["American", "British", "Japanese"],  # Limited for test
        "modulation": ["monoton", "animated"],
        "quality": ["clean", "noisy"]
    }
    
    peft_module = PrecomputedVectorPEFT(
        vector_dim=1024,
        num_vectors=None,  # Dynamic
        attribute_values=attribute_values,
        lora_rank=8,
        vae_latent_dim=64
    )
    
    # Test PEFT processing
    print("🔄 Testing PEFT processing...")
    processed_vectors, combined_loss = peft_module(vectors.unsqueeze(0), tokens, attributes)
    print(f"   Input shape: {vectors.shape}")
    print(f"   Processed shape: {processed_vectors.shape}")
    print(f"   Combined loss: {combined_loss.item():.4f}")
    
    # Test model forward pass
    print("🎯 Testing model forward pass...")
    prompt_text = "Hello world"
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    
    try:
        with torch.no_grad():
            # Test generation (just a few tokens to verify it works)
            generation = model.generate(
                prompt_input_ids=prompt_ids,
                precomputed_vectors=processed_vectors,
                attention_mask=torch.ones((1, processed_vectors.shape[1])),
                max_new_tokens=10,  # Very small for quick test
                do_sample=False
            )
            print(f"   Generation shape: {generation.shape}")
            print("✅ Model forward pass successful!")
    
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        return False
    
    print("\n🎉 Vector training pipeline test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_vector_training()
    if success:
        print("\n✨ Ready for actual training!")
    else:
        print("\n💥 Issues found - check the errors above")
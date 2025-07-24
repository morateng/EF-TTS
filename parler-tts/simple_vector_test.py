#!/usr/bin/env python3
"""
Simple test to verify basic vector loading and model forward pass works
"""

import torch
import sys
sys.path.append('.')

from parler_tts import ParlerTTSForConditionalGeneration
from parler_tts.vector_utils import VectorLoader
from transformers import AutoTokenizer

def simple_test():
    print("🚀 Simple vector test...")
    
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
    
    # Test model forward pass with raw vectors (no PEFT)
    print("🎯 Testing model forward pass with raw vectors...")
    prompt_text = "Hello world"
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    
    try:
        with torch.no_grad():
            # Test generation (just a few tokens to verify it works)
            generation = model.generate(
                prompt_input_ids=prompt_ids,
                precomputed_vectors=vectors.unsqueeze(0),
                attention_mask=torch.ones((1, vectors.shape[0])),
                max_new_tokens=5,  # Very small for quick test
                do_sample=False
            )
            print(f"   Generation shape: {generation.shape}")
            print("✅ Model forward pass successful!")
            return True
    
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        return False

if __name__ == "__main__":
    success = simple_test()
    if success:
        print("\n🎉 Basic vector pipeline works!")
        print("✨ You can now proceed with PEFT training!")
    else:
        print("\n💥 Basic vector pipeline failed")
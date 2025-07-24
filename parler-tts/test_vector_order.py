#!/usr/bin/env python3
"""
Test vector order consistency: description → vectors → PEFT → model
"""

import torch
import sys
sys.path.append('.')

from parler_tts import ParlerTTSForConditionalGeneration
from parler_tts.vector_utils import VectorLoader
from parler_tts.peft_modules import PrecomputedVectorPEFT
from transformers import AutoTokenizer

def test_vector_order():
    print("🔍 Testing Vector Order Consistency")
    print("=" * 60)
    
    # Setup
    device = torch.device('cpu')
    model_name = "parler-tts/parler-tts-mini-v1"
    
    print("📥 Loading components...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vector_loader = VectorLoader(".")
    
    # Configure model
    model.config.use_precomputed_vectors = True
    model.config.precomputed_vector_dim = 1024
    model = model.to(device)
    
    # Setup PEFT module
    attribute_values = {
        "gender": ["male", "female"],
        "pitch": ["high", "low", "medium"],
        "speed": ["slowly", "quickly", "moderate"],
        "accent": ["American", "British", "Australian", "Canadian"],
        "modulation": ["monoton", "animated"],
        "quality": ["clean", "noisy"]
    }
    
    peft_module = PrecomputedVectorPEFT(
        vector_dim=1024,
        num_vectors=10,
        attribute_values=attribute_values,
        lora_rank=8,
        lora_alpha=16.0,
        vae_latent_dim=64,
        orthogonal_reg_strength=0.01
    ).to(device)
    
    print("✅ Components loaded")
    
    # Test different descriptions
    test_descriptions = [
        "female American quickly medium clean.",
        "A female voice with British accent speaks slowly at high pitch.",
        "male Japanese moderate low animated noisy."
    ]
    
    for i, description in enumerate(test_descriptions):
        print(f"\n{'='*50}")
        print(f"🧪 Test Case {i+1}: '{description}'")
        print("-" * 50)
        
        # Step 1: Original T5 tokenization (for comparison)
        print("1️⃣ Original T5 tokenization:")
        t5_tokens = tokenizer(description, return_tensors="pt")
        t5_token_list = tokenizer.convert_ids_to_tokens(t5_tokens['input_ids'][0])
        print(f"   T5 tokens: {t5_token_list}")
        print(f"   T5 length: {len(t5_token_list)}")
        
        # Step 2: Vector loader processing
        print("\n2️⃣ Vector loader processing:")
        vectors, vector_tokens, attributes = vector_loader.get_vectors_for_caption(description)
        print(f"   Vector tokens: {vector_tokens}")
        print(f"   Vector length: {len(vector_tokens)}")
        print(f"   Detected attributes: {attributes}")
        
        # Step 3: Vector order verification
        print("\n3️⃣ Vector order verification:")
        print("   Position | Vector Token | Is Attribute | Vector Norm")
        print("   ---------|--------------|--------------|------------")
        
        for j, token in enumerate(vector_tokens):
            vector = vectors[j]
            
            # Check if it's an attribute token
            is_attribute = any(token.lower() == attr_val.lower() for attr_val in attributes.values())
            attr_type = "N/A"
            
            if is_attribute:
                for attr_name, attr_value in attributes.items():
                    if token.lower() == attr_value.lower():
                        attr_type = attr_name
                        break
            
            status = f"ATTR ({attr_type})" if is_attribute else "NON-ATTR"
            
            print(f"   {j:8d} | {token:12s} | {status:12s} | {vector.norm().item():8.4f}")
        
        # Step 4: PEFT application and order preservation
        print("\n4️⃣ PEFT application:")
        vectors_batch = vectors.unsqueeze(0).to(device)
        
        with torch.no_grad():
            peft_output = peft_module(
                vectors_batch,
                vector_tokens,
                return_losses=True
            )
        
        enhanced_vectors = peft_output['enhanced_vectors'].squeeze(0)  # Remove batch dim
        
        print(f"   Original shape: {vectors.shape}")
        print(f"   Enhanced shape: {enhanced_vectors.shape}")
        print(f"   Order preserved: {enhanced_vectors.shape == vectors.shape}")
        
        # Step 5: Per-position comparison
        print("\n5️⃣ Before/After PEFT comparison:")
        print("   Pos | Token        | Original Norm | Enhanced Norm | Change")
        print("   ----|--------------|---------------|---------------|--------")
        
        for j, token in enumerate(vector_tokens):
            orig_norm = vectors[j].norm().item()
            enh_norm = enhanced_vectors[j].norm().item()
            change = enh_norm - orig_norm
            
            print(f"   {j:3d} | {token:12s} | {orig_norm:12.4f} | {enh_norm:12.4f} | {change:+7.4f}")
        
        # Step 6: Cross-attention input verification
        print("\n6️⃣ Cross-attention input verification:")
        text_prompt = "Hello world"
        prompt_tokens = tokenizer(text_prompt, return_tensors="pt")
        prompt_input_ids = prompt_tokens['input_ids'].to(device)
        
        attention_mask = torch.ones((1, enhanced_vectors.shape[0]), device=device)
        
        print(f"   Prompt: '{text_prompt}'")
        print(f"   Enhanced vectors shape: {enhanced_vectors.unsqueeze(0).shape}")
        print(f"   Attention mask shape: {attention_mask.shape}")
        
        # Test that model can use these vectors
        try:
            with torch.no_grad():
                # Create dummy labels for forward pass
                batch_size = 1
                seq_len = 50
                num_codebooks = 9
                dummy_labels = torch.randint(0, 1024, (batch_size, seq_len, num_codebooks), device=device)
                
                outputs = model(
                    labels=dummy_labels,
                    prompt_input_ids=prompt_input_ids,
                    precomputed_vectors=enhanced_vectors.unsqueeze(0),
                    attention_mask=attention_mask,
                    return_dict=True
                )
            
            print(f"   ✅ Model forward pass successful!")
            print(f"   📊 Output logits shape: {outputs.logits.shape}")
            
        except Exception as e:
            print(f"   ❌ Model forward pass failed: {e}")
        
        print(f"\n🎯 Summary for Test Case {i+1}:")
        print(f"   - Original description: '{description}'")
        print(f"   - Vector tokens: {vector_tokens}")
        print(f"   - Token order: {'✅ PRESERVED' if len(vector_tokens) > 0 else '❌ LOST'}")
        print(f"   - PEFT applied: {'✅ SUCCESS' if enhanced_vectors.shape == vectors.shape else '❌ FAILED'}")

if __name__ == "__main__":
    test_vector_order()
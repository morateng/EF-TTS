#!/usr/bin/env python3
"""
Compare T5 encoder outputs with precomputed vectors
"""

import torch
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from parler_tts.vector_utils import VectorLoader
import numpy as np

def compare_outputs():
    print("🔍 Comparing T5 Encoder vs Precomputed Vectors")
    
    # Load model and tokenizers
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1")
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
    vector_loader = VectorLoader("/home/user/Code/EF-TTS/parler-tts")
    
    # Test descriptions
    test_descriptions = [
        "female American quickly medium clean",
        "male British slowly high noisy",
        "female Japanese moderate low animated"
    ]
    
    for desc in test_descriptions:
        print(f"\n📝 Testing: '{desc}'")
        
        # Method 1: T5 Encoder (Original)
        inputs = tokenizer(desc, return_tensors="pt", padding=True)
        with torch.no_grad():
            encoder_outputs = model.text_encoder(**inputs)
            t5_vectors = encoder_outputs.last_hidden_state  # [1, seq_len, 1024]
        
        # Method 2: Precomputed Vectors (Our approach)
        vectors, tokens, attributes = vector_loader.get_vectors_for_caption(desc)
        precomputed_vectors = vectors.unsqueeze(0)  # [1, seq_len, 1024]
        
        # Compare shapes
        print(f"  T5 shape: {t5_vectors.shape}")
        print(f"  Precomputed shape: {precomputed_vectors.shape}")
        print(f"  T5 tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
        print(f"  Our tokens: {tokens}")
        
        # Compute similarity if shapes match
        if t5_vectors.shape[1] == precomputed_vectors.shape[1]:
            # Cosine similarity
            t5_flat = t5_vectors.view(-1)
            precomp_flat = precomputed_vectors.view(-1)
            cos_sim = torch.nn.functional.cosine_similarity(t5_flat, precomp_flat, dim=0)
            
            # MSE
            mse = torch.nn.functional.mse_loss(t5_vectors, precomputed_vectors)
            
            print(f"  ✅ Cosine Similarity: {cos_sim.item():.4f}")
            print(f"  📊 MSE: {mse.item():.6f}")
            
            if cos_sim < 0.8:
                print(f"  ⚠️  LOW SIMILARITY! This might explain the generation issues.")
        else:
            print(f"  ❌ Shape mismatch! Cannot compare directly.")
            print(f"     This is definitely causing issues.")

def test_generation():
    print("\n🎵 Testing Generation with both approaches")
    
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1")
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
    vector_loader = VectorLoader("/home/user/Code/EF-TTS/parler-tts")
    
    desc = "female American quickly medium clean"
    prompt = "Hello world"
    
    # Tokenize inputs
    desc_inputs = tokenizer(desc, return_tensors="pt")
    prompt_inputs = tokenizer(prompt, return_tensors="pt")
    
    print(f"📝 Description: {desc}")
    print(f"🎤 Prompt: {prompt}")
    
    # Method 1: Original T5 encoder approach
    print("\n1️⃣ Original T5 Encoder Generation:")
    try:
        with torch.no_grad():
            original_output = model.generate(
                input_ids=desc_inputs['input_ids'],
                prompt_input_ids=prompt_inputs['input_ids'],
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7
            )
        print(f"   ✅ Generated {original_output.shape} tokens")
        
        # Convert to audio array (like README)
        audio_array = original_output.cpu().numpy().squeeze()
        
        import soundfile as sf
        sf.write("original_t5_output.wav", audio_array, model.config.sampling_rate)
        print(f"   🎵 Saved audio: original_t5_output.wav")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Method 2: Precomputed vectors approach  
    print("\n2️⃣ Precomputed Vectors Generation:")
    try:
        vectors, tokens, attributes = vector_loader.get_vectors_for_caption(desc)
        
        # Use same approach as successful experiment - encoder_outputs parameter
        from transformers.modeling_outputs import BaseModelOutput
        encoder_hidden_states = vectors.unsqueeze(0)  # [1, seq_len, dim]
        enc_out = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        
        with torch.no_grad():
            vector_output = model.generate(
                encoder_outputs=enc_out,
                prompt_input_ids=prompt_inputs['input_ids'],
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7
            )
        print(f"   ✅ Generated {vector_output.shape} tokens")
        
        # Convert to audio array (like README)
        audio_array = vector_output.cpu().numpy().squeeze()
        
        import soundfile as sf
        sf.write("precomputed_vector_output.wav", audio_array, model.config.sampling_rate)
        print(f"   🎵 Saved audio: precomputed_vector_output.wav")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        
    # Method 3: Generate fresh T5 vectors and compare
    print("\n3️⃣ Fresh T5 Vector Generation:")
    try:
        with torch.no_grad():
            # Generate fresh T5 vectors
            encoder_outputs = model.text_encoder(**desc_inputs)
            fresh_vectors = encoder_outputs.last_hidden_state
            
            # Use fresh vectors with encoder_outputs approach
            fresh_enc_out = BaseModelOutput(last_hidden_state=fresh_vectors)
            
            fresh_output = model.generate(
                encoder_outputs=fresh_enc_out,
                prompt_input_ids=prompt_inputs['input_ids'],
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7
            )
        print(f"   ✅ Generated {fresh_output.shape} tokens")
        
        # Convert to audio array (like README)
        audio_array = fresh_output.cpu().numpy().squeeze()
        
        import soundfile as sf
        sf.write("fresh_t5_vector_output.wav", audio_array, model.config.sampling_rate)
        print(f"   🎵 Saved audio: fresh_t5_vector_output.wav")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    compare_outputs()
    test_generation()
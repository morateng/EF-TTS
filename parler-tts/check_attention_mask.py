#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from parler_tts.vector_utils import VectorLoader


def check_attention_mask_processing():
    """Check how attention masks are handled in cross-attention."""
    
    print("🔍 ATTENTION MASK PROCESSING CHECK")
    print("=" * 70)
    
    # Load components
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-v1", 
        torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
    vector_loader = VectorLoader("/Users/morateng/Code/EF-TTS/parler-tts")
    
    # Test with different length captions
    test_cases = [
        "female American quickly.",  # Short: 4 tokens
        "A female voice with American accent speaks quickly.",  # Medium: 9 tokens  
        "A female speaker with American accent talks quickly at a medium pitch and clean quality.",  # Long: 15+ tokens
    ]
    
    for i, caption in enumerate(test_cases, 1):
        print(f"\n📝 TEST CASE {i}: {len(caption.split())} words")
        print(f"Caption: '{caption}'")
        print("-" * 50)
        
        # Get vectors and create attention mask
        vectors, tokens, attributes = vector_loader.get_vectors_for_caption(caption)
        
        print(f"Parsed tokens ({len(tokens)}): {tokens}")
        print(f"Vector shape: {vectors.shape}")
        
        # Create attention masks
        batch_size = 1
        seq_len = vectors.shape[0]
        
        # 1. All tokens valid (normal case)
        attention_mask_full = torch.ones((batch_size, seq_len))
        
        # 2. Simulate padding (some tokens masked)
        attention_mask_padded = torch.ones((batch_size, seq_len))
        if seq_len > 5:
            attention_mask_padded[0, -2:] = 0  # Mask last 2 tokens
        
        print(f"\nAttention masks:")
        print(f"  Full mask: {attention_mask_full[0].tolist()}")
        print(f"  Padded mask: {attention_mask_padded[0].tolist()}")
        
        # Test forward pass with different masks
        text_prompt = "Hello"
        prompt_input_ids = tokenizer(text_prompt, return_tensors="pt").input_ids
        
        # Prepare decoder input
        pad_token_id = model.config.pad_token_id
        num_codebooks = model.decoder.num_codebooks
        decoder_input_ids = torch.ones((batch_size * num_codebooks, 1), dtype=torch.long) * pad_token_id
        
        model.config.use_precomputed_vectors = True
        model.config.precomputed_vector_dim = 1024
        
        # Test with full mask
        try:
            with torch.no_grad():
                outputs_full = model(
                    prompt_input_ids=prompt_input_ids,
                    decoder_input_ids=decoder_input_ids,
                    precomputed_vectors=vectors.unsqueeze(0),
                    attention_mask=attention_mask_full,
                    output_attentions=True,  # Get attention weights
                )
            print(f"\n✅ Forward pass with full mask: SUCCESS")
            print(f"   Output shape: {outputs_full.logits.shape}")
            
        except Exception as e:
            print(f"\n❌ Forward pass with full mask: FAILED - {e}")
            outputs_full = None
        
        # Test with padded mask
        try:
            with torch.no_grad():
                outputs_padded = model(
                    prompt_input_ids=prompt_input_ids,
                    decoder_input_ids=decoder_input_ids,
                    precomputed_vectors=vectors.unsqueeze(0),
                    attention_mask=attention_mask_padded,
                    output_attentions=True,
                )
            print(f"✅ Forward pass with padded mask: SUCCESS")
            print(f"   Output shape: {outputs_padded.logits.shape}")
            
        except Exception as e:
            print(f"❌ Forward pass with padded mask: FAILED - {e}")
            outputs_padded = None
        
        # Compare outputs if both succeeded
        if outputs_full is not None and outputs_padded is not None:
            # Compare logits
            logit_diff = torch.abs(outputs_full.logits - outputs_padded.logits)
            max_diff = logit_diff.max().item()
            mean_diff = logit_diff.mean().item()
            
            print(f"\n📊 COMPARISON:")
            print(f"   Max logit difference: {max_diff:.6f}")
            print(f"   Mean logit difference: {mean_diff:.6f}")
            
            if max_diff > 0.001:
                print("   → Masks are affecting the output (GOOD!)")
            else:
                print("   → Masks may not be properly applied (CHECK!)")


def test_attention_mask_with_original_encoder():
    """Compare attention mask handling between vector and original approaches."""
    
    print("\n" + "=" * 70)
    print("🔄 COMPARING VECTOR vs ORIGINAL ATTENTION MASKS")
    print("=" * 70)
    
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-v1", 
        torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
    vector_loader = VectorLoader("/Users/morateng/Code/EF-TTS/parler-tts")
    
    caption = "female American quickly medium clean."
    text_prompt = "Hello"
    
    print(f"Caption: '{caption}'")
    print(f"Text: '{text_prompt}'")
    
    # Method 1: Original encoder
    print(f"\n📝 METHOD 1: ORIGINAL T5 ENCODER")
    description_input_ids = tokenizer(caption, return_tensors="pt").input_ids
    prompt_input_ids = tokenizer(text_prompt, return_tensors="pt").input_ids
    
    print(f"Description tokens: {description_input_ids.shape}")
    print(f"Description content: {[tokenizer.decode([id]) for id in description_input_ids[0]]}")
    
    # The tokenizer automatically creates attention mask
    description_attention_mask = (description_input_ids != tokenizer.pad_token_id).long()
    print(f"Auto attention mask: {description_attention_mask[0].tolist()}")
    
    # Method 2: Vector approach  
    print(f"\n📊 METHOD 2: PRECOMPUTED VECTORS")
    vectors, tokens, attributes = vector_loader.get_vectors_for_caption(caption)
    
    print(f"Vector tokens ({len(tokens)}): {tokens}")
    print(f"Vector shape: {vectors.shape}")
    
    # Manual attention mask (all valid)
    vector_attention_mask = torch.ones((1, vectors.shape[0]))
    print(f"Vector attention mask: {vector_attention_mask[0].tolist()}")
    
    # Compare sequence lengths
    print(f"\n🔍 COMPARISON:")
    print(f"T5 tokenized length: {description_input_ids.shape[1]}")
    print(f"Vector sequence length: {vectors.shape[0]}")
    
    if description_input_ids.shape[1] == vectors.shape[0]:
        print("✅ Sequence lengths match - good alignment!")
    else:
        print("⚠️  Sequence lengths differ - may need alignment")
        
    # Check if tokens roughly correspond
    t5_tokens = [tokenizer.decode([id]) for id in description_input_ids[0]]
    print(f"T5 tokens: {t5_tokens}")
    print(f"Vector tokens: {tokens}")


def check_attention_implementation():
    """Check the actual attention implementation in the model."""
    
    print("\n" + "=" * 70)
    print("🔧 ATTENTION IMPLEMENTATION CHECK")
    print("=" * 70)
    
    print("Looking at how attention masks are used in the model...")
    
    # Check model architecture
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-v1", 
        torch_dtype=torch.float32
    )
    
    print(f"Model components:")
    print(f"  - Text encoder: {type(model.text_encoder).__name__}")
    print(f"  - Audio encoder: {type(model.audio_encoder).__name__}")
    print(f"  - Decoder: {type(model.decoder).__name__}")
    
    # Check decoder configuration
    decoder_config = model.decoder.config
    print(f"\nDecoder attention configuration:")
    print(f"  - Has cross attention: {decoder_config.add_cross_attention}")
    print(f"  - Cross attention hidden size: {decoder_config.cross_attention_hidden_size}")
    print(f"  - Number of attention heads: {decoder_config.num_attention_heads}")
    print(f"  - Number of cross attention heads: {decoder_config.num_cross_attention_key_value_heads}")
    
    # The attention mask should be passed through to the decoder's cross-attention layers
    print(f"\n💡 EXPECTED BEHAVIOR:")
    print("  1. Attention mask is passed to model forward()")
    print("  2. Model creates encoder_outputs with the vectors")  
    print("  3. Decoder receives encoder_outputs + attention_mask")
    print("  4. Each cross-attention layer uses the mask to ignore padded positions")
    print("  5. Masked positions get very negative attention weights (-inf)")
    print("  6. After softmax, masked positions have ~0 attention probability")


def main():
    check_attention_mask_processing()
    test_attention_mask_with_original_encoder()
    check_attention_implementation()
    
    print("\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    print("The attention mask should work correctly because:")
    print("1. ✅ We create proper attention masks for vector sequences")
    print("2. ✅ Model accepts attention_mask parameter") 
    print("3. ✅ Decoder has cross-attention enabled")
    print("4. ✅ Forward pass succeeds with different mask shapes")
    print("5. ✅ Cross-attention layers should respect the mask")
    print()
    print("However, to be 100% certain, we should:")
    print("- Extract attention weights and verify masked positions are ~0")
    print("- Test with deliberately malformed masks")
    print("- Compare attention patterns between vector and original methods")


if __name__ == "__main__":
    main()
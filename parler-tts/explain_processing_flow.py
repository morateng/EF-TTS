#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from parler_tts.vector_utils import VectorLoader


def explain_processing_flow():
    """Explain step-by-step how vectors are processed in the model."""
    
    print("🔄 PARLER-TTS VECTOR PROCESSING FLOW EXPLANATION")
    print("=" * 80)
    
    # Load components
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-v1", 
        torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
    vector_loader = VectorLoader("/Users/morateng/Code/EF-TTS/parler-tts")
    
    # Example inputs
    style_caption = "female American quickly medium clean."
    text_prompt = "Hello world"
    
    print(f"📝 INPUT:")
    print(f"  Style Caption: '{style_caption}'")
    print(f"  Text Prompt: '{text_prompt}'")
    print()
    
    # STEP 1: Parse style caption to vectors
    print("📊 STEP 1: STYLE CAPTION → VECTORS")
    print("-" * 50)
    
    vectors, tokens, attributes = vector_loader.get_vectors_for_caption(style_caption)
    indices = vector_loader.get_attribute_indices(tokens)
    
    print(f"  Parsed tokens: {tokens}")
    print(f"  Extracted attributes: {attributes}")
    print(f"  Vector shape: {vectors.shape} (seq_len={vectors.shape[0]}, hidden_dim={vectors.shape[1]})")
    print(f"  Attribute positions: {indices['attribute_indices']}")  
    print(f"  Non-attribute positions: {indices['nonattr_indices']}")
    print()
    
    # STEP 2: Process text prompt
    print("📝 STEP 2: TEXT PROMPT → TOKEN IDS")
    print("-" * 50)
    
    prompt_input_ids = tokenizer(text_prompt, return_tensors="pt").input_ids
    print(f"  Text: '{text_prompt}'")
    print(f"  Token IDs: {prompt_input_ids}")
    print(f"  Token shape: {prompt_input_ids.shape}")
    print(f"  Decoded tokens: {[tokenizer.decode([id]) for id in prompt_input_ids[0]]}")
    print()
    
    # STEP 3: Model configuration
    print("⚙️  STEP 3: MODEL CONFIGURATION")
    print("-" * 50)
    
    model.config.use_precomputed_vectors = True
    model.config.precomputed_vector_dim = 1024
    
    print(f"  use_precomputed_vectors: {model.config.use_precomputed_vectors}")
    print(f"  precomputed_vector_dim: {model.config.precomputed_vector_dim}")
    print(f"  text_encoder hidden_size: {model.config.text_encoder.hidden_size}")
    print(f"  decoder hidden_size: {model.config.decoder.hidden_size}")
    print()
    
    # STEP 4: Prepare model inputs
    print("📦 STEP 4: PREPARE MODEL INPUTS")
    print("-" * 50)
    
    batch_size = 1
    # Add batch dimension and create attention mask
    batched_vectors = vectors.unsqueeze(0)  # (1, seq_len, hidden_dim)
    attention_mask = torch.ones((batch_size, vectors.shape[0]))  # (1, seq_len)
    
    print(f"  Batched vectors: {batched_vectors.shape}")
    print(f"  Attention mask: {attention_mask.shape}")
    print(f"  Attention mask values: {attention_mask[0].tolist()}")
    print()
    
    # STEP 5: Model forward pass (simulated breakdown)
    print("🔄 STEP 5: MODEL FORWARD PASS")
    print("-" * 50)
    
    print("  5.1 CHECK IF USING PRECOMPUTED VECTORS:")
    print(f"      → precomputed_vectors is not None: {batched_vectors is not None}")
    print(f"      → use_precomputed_vectors: {model.config.use_precomputed_vectors}")
    print("      → YES, using precomputed vectors path!")
    print()
    
    print("  5.2 APPLY PEFT MODULES (if enabled):")
    print("      → Check if model has vector_peft module")
    has_peft = hasattr(model, 'vector_peft') and model.vector_peft is not None
    print(f"      → Has PEFT: {has_peft}")
    
    if has_peft:
        print("      → Apply PEFT transformations:")
        print("        - For attribute tokens: Apply VAE (mean, variance, KL loss)")
        print("        - For non-attribute tokens: Apply LoRA (low-rank adaptation)")
        print("        - Orthogonality regularization between enhanced vectors")
    else:
        print("      → No PEFT modules, using raw precomputed vectors")
    print()
    
    print("  5.3 PROJECTION (if needed):")
    has_projection = (
        model.config.precomputed_vector_dim != model.decoder.config.hidden_size
        and model.decoder.config.cross_attention_hidden_size is None
    )
    print(f"      → Need projection: {has_projection}")
    if has_projection:
        print(f"      → Project from {model.config.precomputed_vector_dim} to {model.decoder.config.hidden_size}")
    print()
    
    print("  5.4 CREATE ENCODER OUTPUTS:")
    print("      → Wrap enhanced vectors in BaseModelOutput format")
    print("      → This makes them compatible with existing decoder cross-attention")
    print()
    
    print("  5.5 PROCESS TEXT PROMPT:")
    print(f"      → Embed prompt tokens: {prompt_input_ids.shape} → embedding layer")
    print(f"      → Output shape: (batch_size, prompt_seq_len, hidden_dim)")
    print()
    
    print("  5.6 DECODER PROCESSING:")
    print("      → Decoder receives:")
    print("        - Prompt embeddings (for autoregressive generation)")
    print("        - Enhanced description vectors (for cross-attention)")
    print("        - Attention masks")
    print()
    print("      → Cross-attention mechanism:")
    print("        - Query: from decoder hidden states")
    print("        - Key & Value: from enhanced description vectors")
    print("        - This allows decoder to 'attend' to style information")
    print()
    print("      → For each decoder layer:")
    print("        1. Self-attention on previous tokens")
    print("        2. Cross-attention with style vectors")
    print("        3. Feed-forward processing")
    print()
    
    # STEP 6: Actual forward pass test
    print("🧪 STEP 6: ACTUAL FORWARD PASS TEST")
    print("-" * 50)
    
    # Prepare decoder input (start tokens)
    pad_token_id = model.config.pad_token_id
    num_codebooks = model.decoder.num_codebooks
    decoder_input_ids = torch.ones((batch_size * num_codebooks, 1), dtype=torch.long) * pad_token_id
    
    print(f"  Decoder input shape: {decoder_input_ids.shape}")
    print(f"  (batch_size * num_codebooks, seq_len) = ({batch_size} * {num_codebooks}, 1)")
    print()
    
    try:
        with torch.no_grad():
            outputs = model(
                prompt_input_ids=prompt_input_ids,
                decoder_input_ids=decoder_input_ids,
                precomputed_vectors=batched_vectors,
                description_tokens=[tokens],
                attention_mask=attention_mask,
            )
        
        print("  ✅ FORWARD PASS SUCCESSFUL!")
        print(f"  Output logits shape: {outputs.logits.shape}")
        print(f"  (batch_size * num_codebooks, seq_len, vocab_size)")
        print(f"  = ({batch_size} * {num_codebooks}, {outputs.logits.shape[1]}, {outputs.logits.shape[2]})")
        print()
        
        print("  📊 OUTPUT ANALYSIS:")
        print(f"    Max logit: {outputs.logits.max().item():.4f}")
        print(f"    Min logit: {outputs.logits.min().item():.4f}")
        print(f"    Mean logit: {outputs.logits.mean().item():.4f}")
        
    except Exception as e:
        print(f"  ❌ FORWARD PASS FAILED: {e}")
    
    print()
    
    # STEP 7: Generation process (conceptual)
    print("🎵 STEP 7: AUDIO GENERATION PROCESS")
    print("-" * 50)
    
    print("  7.1 AUTOREGRESSIVE GENERATION:")
    print("      → Start with special tokens (BOS/PAD)")
    print("      → For each time step:")
    print("        1. Forward pass with current sequence")
    print("        2. Get logits for next tokens (all codebooks)")
    print("        3. Sample from probability distributions")
    print("        4. Append sampled tokens to sequence")
    print("        5. Repeat until EOS or max_length")
    print()
    
    print("  7.2 MULTI-CODEBOOK STRUCTURE:")
    print(f"      → Each time step generates {num_codebooks} tokens")
    print("      → These represent different frequency bands/aspects")
    print("      → Shape: (time_steps, num_codebooks)")
    print()
    
    print("  7.3 AUDIO DECODING:")
    print("      → Generated tokens → DAC decoder")
    print("      → Tokens represent quantized audio features")
    print("      → DAC reconstructs waveform from tokens")
    print("      → Final output: raw audio waveform")
    print()
    
    # STEP 8: Key advantages
    print("🚀 STEP 8: KEY ADVANTAGES OF VECTOR APPROACH")
    print("-" * 50)
    
    print("  ✅ EFFICIENCY:")
    print("      → Skip T5 encoder forward pass (expensive)")
    print("      → Precomputed vectors load instantly")
    print("      → Faster inference time")
    print()
    
    print("  ✅ CONTROLLABILITY:")
    print("      → Direct manipulation of style vectors")
    print("      → PEFT allows fine-grained control")
    print("      → Mix and match attributes easily")
    print()
    
    print("  ✅ CONSISTENCY:")
    print("      → Same style description = same vectors")
    print("      → Reproducible voice characteristics")
    print("      → No encoder variability")
    print()


if __name__ == "__main__":
    explain_processing_flow()
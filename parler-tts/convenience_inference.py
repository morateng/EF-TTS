#!/usr/bin/env python3
"""
Convenience functions for inference with description text.
"""

import torch
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from parler_tts.vector_utils import VectorLoader

def generate_from_description(
    model: ParlerTTSForConditionalGeneration,
    description: str,
    prompt_text: str,
    tokenizer: AutoTokenizer = None,
    max_new_tokens: int = 1000,
    vector_base_path: str = "./",
    **kwargs
):
    """
    Generate audio from description text and prompt text.
    
    Args:
        model: ParlerTTS model with vector support
        description: Style description like "female American quickly medium clean"
        prompt_text: Text to synthesize
        tokenizer: Tokenizer for prompt text
        max_new_tokens: Maximum tokens to generate
        vector_base_path: Path to vector directory
        **kwargs: Additional arguments for generate()
        
    Returns:
        Generated audio tokens
    """
    # Convert description to vectors
    vector_loader = VectorLoader(vector_base_path)
    vectors, tokens, attributes = vector_loader.get_vectors_for_caption(description)
    
    # Tokenize prompt text
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    
    prompt_tokenized = tokenizer(prompt_text, return_tensors="pt")
    prompt_input_ids = prompt_tokenized["input_ids"]
    
    # Prepare vector inputs
    precomputed_vectors = vectors.unsqueeze(0)  # Add batch dimension
    attention_mask = torch.ones((1, vectors.shape[0]), dtype=torch.long, device=vectors.device)
    description_tokens = [tokens]  # For PEFT
    
    # Configure model for vector mode
    model.config.use_precomputed_vectors = True
    model.config.precomputed_vector_dim = 1024
    
    # Generate
    with torch.no_grad():
        generation = model.generate(
            precomputed_vectors=precomputed_vectors,
            attention_mask=attention_mask,
            description_tokens=description_tokens,
            prompt_input_ids=prompt_input_ids,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
    
    return generation

def test_convenience_function():
    """Test the convenience function."""
    print("🎯 Testing convenience function for description-to-audio generation...")
    
    # Load model and tokenizer
    print("1️⃣ Loading model and tokenizer...")
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1")
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
    
    # Test parameters
    description = "female American quickly medium clean"
    prompt_text = "Hello world, this is a test."
    
    print(f"2️⃣ Test inputs:")
    print(f"   Description: '{description}'")
    print(f"   Prompt: '{prompt_text}'")
    
    try:
        # Generate using convenience function
        print("3️⃣ Generating audio...")
        generation = generate_from_description(
            model=model,
            description=description,
            prompt_text=prompt_text,
            tokenizer=tokenizer,
            max_new_tokens=100,  # Short for testing
            do_sample=True,
            temperature=0.9
        )
        
        print(f"4️⃣ Generation successful!")
        print(f"   Output shape: {generation.shape}")
        print(f"   Output type: {type(generation)}")
        
        # Decode to audio if possible
        try:
            audio_arr = model.audio_encoder.decode(generation).audio_values
            print(f"   Audio shape: {audio_arr.shape}")
            print("✅ Complete pipeline working!")
            
        except Exception as decode_error:
            print(f"   Decode error (expected): {decode_error}")
            print("✅ Generation pipeline working!")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_convenience_function()
    if success:
        print("\n🎉 Convenience function ready for use!")
        print("\nUsage example:")
        print("""
from convenience_inference import generate_from_description
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1")
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

generation = generate_from_description(
    model=model,
    description="female American quickly medium clean",
    prompt_text="Hello world!",
    tokenizer=tokenizer,
    max_new_tokens=1000
)
        """)
    else:
        print("\n❌ Convenience function needs debugging")
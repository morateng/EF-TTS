#!/usr/bin/env python3

import torch
import numpy as np
from parler_tts.vector_utils import VectorLoader


def show_vector_concatenation():
    """Demonstrate how style captions are converted to concatenated vectors."""
    
    print("Loading vector loader...")
    vector_loader = VectorLoader("/Users/morateng/Code/EF-TTS/parler-tts")
    
    # Test different style captions
    test_captions = [
        "female American quickly medium clean.",
        "A female voice with American accent speaks quickly at a medium pitch and a clean quality.",
        "male British slowly low animated.",
        "A male speaker with British accent talks slowly in a low pitch with animated modulation.",
        "female Japanese moderate high noisy."
    ]
    
    for i, caption in enumerate(test_captions, 1):
        print("=" * 80)
        print(f"TEST {i}: '{caption}'")
        print("=" * 80)
        
        try:
            # Parse and get vectors
            vectors, tokens, attributes = vector_loader.get_vectors_for_caption(caption)
            indices = vector_loader.get_attribute_indices(tokens)
            
            print(f"📝 PARSED TOKENS: {tokens}")
            print(f"🎯 FOUND ATTRIBUTES: {attributes}")
            print(f"📊 FINAL VECTOR SHAPE: {vectors.shape}")
            print()
            
            # Show token-by-token breakdown
            print("🔍 TOKEN-BY-TOKEN BREAKDOWN:")
            print("-" * 50)
            for j, token in enumerate(tokens):
                try:
                    token_vector = vector_loader.load_vector_for_token(token)
                    is_attribute = token in vector_loader.token_to_attribute
                    
                    if is_attribute:
                        attr_type, attr_value = vector_loader.token_to_attribute[token]
                        print(f"  {j:2d}. '{token}' -> ATTRIBUTE ({attr_type}: {attr_value}) | shape: {token_vector.shape}")
                    else:
                        print(f"  {j:2d}. '{token}' -> NON-ATTRIBUTE | shape: {token_vector.shape}")
                        
                    # Show first few values of the vector
                    preview = token_vector.flatten()[:5].numpy()
                    print(f"      Preview: [{preview[0]:.3f}, {preview[1]:.3f}, {preview[2]:.3f}, {preview[3]:.3f}, {preview[4]:.3f}]")
                    
                except Exception as e:
                    print(f"  {j:2d}. '{token}' -> ERROR: {e}")
                    
            print()
            
            # Show concatenation process
            print("🔗 CONCATENATION PROCESS:")
            print("-" * 50)
            individual_vectors = []
            for j, token in enumerate(tokens):
                try:
                    token_vector = vector_loader.load_vector_for_token(token)
                    individual_vectors.append(token_vector)
                    print(f"  Step {j+1}: Added '{token}' | Running shape: ({len(individual_vectors)}, {token_vector.shape[0]})")
                except:
                    pass
            
            if individual_vectors:
                final_concat = torch.stack(individual_vectors, dim=0)
                print(f"  Final: torch.stack() -> {final_concat.shape}")
                
                # Verify it matches our method
                assert torch.equal(final_concat, vectors), "Concatenation mismatch!"
                print("  ✅ Concatenation verified!")
            
            print()
            
            # Show attribute vs non-attribute split
            print("🏷️  ATTRIBUTE vs NON-ATTRIBUTE SPLIT:")
            print("-" * 50)
            print(f"  Attribute indices: {indices['attribute_indices']}")
            print(f"  Non-attribute indices: {indices['nonattr_indices']}")
            
            if indices['attribute_indices']:
                attr_vectors = vectors[indices['attribute_indices']]
                print(f"  Attribute vectors shape: {attr_vectors.shape}")
                print(f"  Attribute tokens: {[tokens[i] for i in indices['attribute_indices']]}")
            
            if indices['nonattr_indices']:
                nonattr_vectors = vectors[indices['nonattr_indices']]
                print(f"  Non-attribute vectors shape: {nonattr_vectors.shape}")
                print(f"  Non-attribute tokens: {[tokens[i] for i in indices['nonattr_indices']]}")
                
        except Exception as e:
            print(f"❌ Error processing caption: {e}")
        
        print()


def show_vector_statistics():
    """Show statistics about the loaded vectors."""
    
    print("📈 VECTOR STATISTICS")
    print("=" * 50)
    
    vector_loader = VectorLoader("/Users/morateng/Code/EF-TTS/parler-tts")
    
    # Test a few different types
    test_tokens = {
        "Attributes": ["female", "male", "American", "British", "quickly", "slowly", "high", "low", "clean", "noisy"],
        "Non-attributes": ["voice", "with", "accent", "speaks", "at", "pitch", "quality", "a"]
    }
    
    for category, tokens in test_tokens.items():
        print(f"\n🏷️  {category.upper()}:")
        print("-" * 30)
        
        for token in tokens:
            try:
                vector = vector_loader.load_vector_for_token(token)
                print(f"  '{token}':")
                print(f"    Shape: {vector.shape}")
                print(f"    Mean: {vector.mean().item():.4f}")
                print(f"    Std: {vector.std().item():.4f}")
                print(f"    Min: {vector.min().item():.4f}")
                print(f"    Max: {vector.max().item():.4f}")
                
                # Show some actual values
                sample = vector.flatten()[:10].numpy()
                print(f"    Sample: {sample}")
                print()
                
            except Exception as e:
                print(f"  '{token}': ERROR - {e}")


def compare_similar_captions():
    """Compare similar captions to see differences."""
    
    print("🔄 COMPARING SIMILAR CAPTIONS")
    print("=" * 50)
    
    vector_loader = VectorLoader("/Users/morateng/Code/EF-TTS/parler-tts")
    
    # Similar captions with small differences
    comparisons = [
        ("female American quickly", "male American quickly"),
        ("female American quickly medium", "female British quickly medium"),
        ("female American quickly medium clean", "female American slowly medium clean")
    ]
    
    for caption1, caption2 in comparisons:
        print(f"\nComparing:")
        print(f"  A: '{caption1}'")
        print(f"  B: '{caption2}'")
        print("-" * 40)
        
        try:
            vectors1, tokens1, attr1 = vector_loader.get_vectors_for_caption(caption1)
            vectors2, tokens2, attr2 = vector_loader.get_vectors_for_caption(caption2)
            
            print(f"  Tokens A: {tokens1}")
            print(f"  Tokens B: {tokens2}")
            print(f"  Attributes A: {attr1}")
            print(f"  Attributes B: {attr2}")
            
            print(f"  Vector shape A: {vectors1.shape}")
            print(f"  Vector shape B: {vectors2.shape}")
            
            # If same length, compute similarity
            if vectors1.shape == vectors2.shape:
                similarity = torch.cosine_similarity(
                    vectors1.flatten(), 
                    vectors2.flatten(), 
                    dim=0
                )
                print(f"  Cosine similarity: {similarity.item():.6f}")
                
                # Find which tokens are different
                different_positions = []
                for i, (t1, t2) in enumerate(zip(tokens1, tokens2)):
                    if t1 != t2:
                        different_positions.append((i, t1, t2))
                
                if different_positions:
                    print(f"  Different tokens at positions: {different_positions}")
            else:
                print(f"  Different shapes, cannot compare directly")
                
        except Exception as e:
            print(f"  Error: {e}")


def main():
    """Run all demonstrations."""
    show_vector_concatenation()
    print("\n" + "=" * 100 + "\n")
    show_vector_statistics()
    print("\n" + "=" * 100 + "\n")
    compare_similar_captions()


if __name__ == "__main__":
    main()
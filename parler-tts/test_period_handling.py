#!/usr/bin/env python3
"""
Test period handling in vector loader
"""

import sys
sys.path.append('.')

from parler_tts.vector_utils import VectorLoader

def test_period_handling():
    print("🔍 Testing Period Handling")
    print("=" * 40)
    
    vector_loader = VectorLoader(".")
    
    test_cases = [
        "female American quickly medium clean",      # No period
        "female American quickly medium clean.",     # With period
        "A female voice with British accent.",       # Sentence with period
        "male Japanese moderate low animated noisy"  # No period
    ]
    
    for i, description in enumerate(test_cases):
        print(f"\n📝 Test Case {i+1}: '{description}'")
        
        vectors, tokens, attributes = vector_loader.get_vectors_for_caption(description)
        
        print(f"   🏷️  Tokens: {tokens}")
        print(f"   📊 Token count: {len(tokens)}")
        
        # Count periods
        period_count = tokens.count('.')
        print(f"   🔶 Period count: {period_count}")
        
        # Check if it ends with . and <_s>
        if len(tokens) >= 2:
            ends_correctly = tokens[-2] == '.' and tokens[-1] == '<_s>'
            print(f"   ✅ Ends with '. <_s>': {ends_correctly}")
        else:
            print(f"   ❌ Too few tokens")

if __name__ == "__main__":
    test_period_handling()
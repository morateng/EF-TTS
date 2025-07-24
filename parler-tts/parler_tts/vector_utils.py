import os
import re
import torch
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class VectorLoader:
    """Utility class to load and concatenate precomputed vectors based on style captions."""
    
    def __init__(self, vector_base_path: str):
        self.vector_base_path = Path(vector_base_path) / "vectors"
        
        # Define attribute mappings
        self.attribute_values = {
            "gender": ["male", "female"],
            "pitch": ["high", "low", "medium"],
            "speed": ["slowly", "quickly", "moderate"],
            "accent": [
                "Slovenia", "Chinese", "German", "Irish", "Scottish", "Russian", "Polish",
                "Canadian", "Turkish", "Czech", "Jamaica", "Italian", "Swiss", "Brazilian", "Brooklyn",
                "Finnish", "Japanese", "Filipino", "British",
                "Indian", "Australian", "Norwegian", "English",
                "Romania", "Spanish", "Croatia", "Swedish", "Colombia", "French", "American",
                "Mexican", "Portuguese", "Dominic", "Welsh", "Nigeria",
                "Chile", "Belgia", "Dutch", "Jordan", "Serbia", "Ukrainian", "Mandarin"
            ],
            "modulation": ["monoton", "animated"],
            "quality": ["clean", "noisy"]
        }
        
        # Create reverse mapping for quick lookup
        self.token_to_attribute = {}
        for attr, values in self.attribute_values.items():
            for value in values:
                self.token_to_attribute[value.lower()] = (attr, value)
    
    def parse_style_caption(self, caption: str) -> Tuple[List[str], Dict[str, str]]:
        """
        Parse style caption and extract tokens and attribute mappings.
        
        Args:
            caption: Style caption text like "A female voice with American accent speaks quickly"
            
        Returns:
            tokens: List of all tokens in order (including . and </s> at the end)
            attributes: Dict mapping attribute type to value found in caption
        """
        # Tokenize the caption (simple whitespace split for now)
        tokens = caption.lower().strip().split()
        
        # Clean tokens (remove punctuation but keep track of it)
        cleaned_tokens = []
        has_period = False
        
        for token in tokens:
            # Check if token ends with period
            if token.endswith('.'):
                has_period = True
                # Remove punctuation except period, but keep the word
                cleaned_token = re.sub(r'[^\w]', '', token)
                if cleaned_token:
                    cleaned_tokens.append(cleaned_token)
                # Add period as separate token
                cleaned_tokens.append('.')
            else:
                # Remove punctuation but keep the word
                cleaned_token = re.sub(r'[^\w]', '', token)
                if cleaned_token:
                    cleaned_tokens.append(cleaned_token)
        
        # Always add </s> token at the end
        cleaned_tokens.append('<_s>')
        
        # Find attribute values in the tokens (excluding </s>)
        attributes = {}
        for token in cleaned_tokens[:-1]:  # Exclude last token (</s>)
            if token in self.token_to_attribute:
                attr_type, attr_value = self.token_to_attribute[token]
                attributes[attr_type] = attr_value
                
        return cleaned_tokens, attributes
    
    def load_vector_for_token(self, token: str) -> torch.Tensor:
        """Load precomputed vector for a given token."""
        # First check if it's an attribute token
        if token in self.token_to_attribute:
            attr_type, attr_value = self.token_to_attribute[token]
            vector_path = self.vector_base_path / attr_type / f"{attr_value}.pt"
        else:
            # It's a non-attribute token
            vector_path = self.vector_base_path / "nonattr_tokens" / f"{token}.pt"
        
        if vector_path.exists():
            return torch.load(vector_path, map_location='cpu')
        else:
            # If specific token not found, try common alternatives
            alternatives = [
                self.vector_base_path / "nonattr_tokens" / f"▁{token}.pt",  # with space prefix
                self.vector_base_path / "nonattr_tokens" / f"{token}.pt",
            ]
            
            # Special handling for period token
            if token == '.':
                alternatives.append(self.vector_base_path / "nonattr_tokens" / "▁.pt")
            
            # Special handling for </s> token  
            if token == '<_s>':
                alternatives.append(self.vector_base_path / "nonattr_tokens" / "<_s>.pt")
            
            for alt_path in alternatives:
                if alt_path.exists():
                    return torch.load(alt_path, map_location='cpu')
            
            raise FileNotFoundError(f"Vector not found for token: {token}")
    
    def get_vectors_for_caption(self, caption: str) -> Tuple[torch.Tensor, List[str], Dict[str, str]]:
        """
        Get concatenated vectors for a style caption.
        
        Args:
            caption: Style caption text
            
        Returns:
            vectors: Concatenated tensor of shape (seq_len, hidden_dim)
            tokens: List of tokens in order
            attributes: Dict of found attributes
        """
        tokens, attributes = self.parse_style_caption(caption)
        
        vectors = []
        for token in tokens:
            try:
                vector = self.load_vector_for_token(token)
                vectors.append(vector)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                # Could use a default/unknown vector here
                continue
        
        if not vectors:
            raise ValueError(f"No vectors found for caption: {caption}")
            
        # Concatenate along sequence dimension
        concatenated_vectors = torch.stack(vectors, dim=0)  # (seq_len, hidden_dim)
        
        return concatenated_vectors, tokens, attributes
    
    def get_attribute_indices(self, tokens: List[str]) -> Dict[str, List[int]]:
        """
        Get indices of attribute tokens vs non-attribute tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Dict with 'attribute_indices' and 'nonattr_indices' keys
        """
        attribute_indices = []
        nonattr_indices = []
        
        for i, token in enumerate(tokens):
            if token in self.token_to_attribute:
                attribute_indices.append(i)
            else:
                nonattr_indices.append(i)
        
        return {
            'attribute_indices': attribute_indices,
            'nonattr_indices': nonattr_indices
        }
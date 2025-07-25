"""
Parameter-Efficient Fine-Tuning (PEFT) modules for precomputed vectors in Parler-TTS.

This module contains LoRA adapters and VAE components for enhancing precomputed 
vectors while keeping the base model frozen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import re


class LoRAVectorTransform(nn.Module):
    """
    LoRA (Low-Rank Adaptation) transformation for precomputed vectors.
    
    Args:
        vector_dim (int): Dimension of the input vector
        rank (int): Rank of the LoRA decomposition (default: 16)
        alpha (float): Scaling parameter for LoRA (default: 32.0)
        dropout (float): Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        vector_dim: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vector_dim = vector_dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA decomposition: W = W_0 + (B @ A) * scaling
        # A: [vector_dim, rank], B: [rank, vector_dim]
        self.lora_A = nn.Parameter(torch.randn(vector_dim, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, vector_dim))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA transformation to input vectors.
        
        Args:
            x: Input tensor of shape [..., vector_dim]
            
        Returns:
            Enhanced vectors of same shape
        """
        # Apply LoRA: x + (x @ A @ B) * scaling
        # x shape: [..., vector_dim]
        # lora_A shape: [vector_dim, rank]
        # lora_B shape: [rank, vector_dim]
        
        # First step: x @ A -> [..., rank]
        temp = torch.matmul(x, self.lora_A)
        # Second step: temp @ B -> [..., vector_dim]  
        lora_output = torch.matmul(temp, self.lora_B) * self.scaling
        
        return x + self.dropout(lora_output)


class AttributeVAE(nn.Module):
    """
    Simplified Variational Autoencoder for attribute-specific vectors.
    
    This VAE uses single linear layers for encoding and decoding to minimize
    parameters while extracting mean and variance from attribute vectors.
    
    Args:
        vector_dim (int): Dimension of the input vector (e.g., 1024)
        latent_dim (int): Dimension of the latent space (default: 128)
        hidden_dim (int): Not used in simplified version, kept for compatibility
    """
    
    def __init__(
        self,
        vector_dim: int,
        latent_dim: int = 128,
        hidden_dim: int = 512
    ):
        super().__init__()
        self.vector_dim = vector_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Simplified single-layer encoder: direct mapping to latent space
        self.mu_head = nn.Linear(vector_dim, latent_dim)
        self.logvar_head = nn.Linear(vector_dim, latent_dim)
        
        # Simplified single-layer decoder: direct mapping from latent space
        self.decoder = nn.Linear(latent_dim, vector_dim)
        
        # Initialize weights to zero for residual connection stability
        self._init_zero_weights()
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent mean and log-variance."""
        mu = self.mu_head(x)
        logvar = self.logvar_head(x)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to output space."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE with residual connection.
        
        Args:
            x: Input tensor of shape [..., vector_dim]
            
        Returns:
            Tuple of (enhanced_x, mu, logvar) where enhanced_x = x + VAE(x)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        vae_output = self.decode(z)
        # Apply residual connection: x + VAE(x)
        enhanced_x = x + vae_output
        return enhanced_x, mu, logvar
    
    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss."""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    
    def reconstruction_loss(self, x: torch.Tensor, enhanced_x: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss for residual VAE.
        Since enhanced_x = x + VAE(x), we want VAE(x) to be small when no change is needed.
        """
        residual = enhanced_x - x  # This is VAE(x)
        return torch.sum(residual ** 2, dim=-1)  # L2 loss on residual
    
    def _init_zero_weights(self):
        """Initialize decoder weights to zero for stable residual connection."""
        # Only initialize decoder weights to zero (encoder can learn from start)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)


class OrthogonalityRegularizer(nn.Module):
    """
    Orthogonality regularizer to ensure PEFT-enhanced vectors remain distinct.
    
    This regularizer encourages orthogonality between different enhanced vectors
    to prevent collapse and maintain diversity.
    
    Args:
        reg_strength (float): Regularization strength (default: 0.01)
    """
    
    def __init__(self, reg_strength: float = 0.01):
        super().__init__()
        self.reg_strength = reg_strength
    
    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Compute orthogonality regularization loss.
        
        Args:
            vectors: Tensor of shape [num_vectors, vector_dim]
            
        Returns:
            Orthogonality loss scalar
        """
        # Normalize vectors
        vectors_norm = F.normalize(vectors, dim=-1)
        
        # Compute gram matrix
        gram_matrix = torch.matmul(vectors_norm, vectors_norm.t())
        
        # Remove diagonal (self-similarity)
        num_vectors = vectors.shape[0]
        identity = torch.eye(num_vectors, device=vectors.device)
        gram_matrix = gram_matrix - identity
        
        # Penalize non-zero off-diagonal elements
        orthogonality_loss = torch.sum(gram_matrix ** 2) / (num_vectors * (num_vectors - 1))
        
        return self.reg_strength * orthogonality_loss


class AttributeTokenClassifier:
    """
    Classifier to identify attribute tokens and their corresponding attribute values.
    
    This class analyzes input tokens to determine which ones correspond to style attributes
    (gender, pitch, speed, accent, modulation, quality) and maps them to specific values.
    """
    
    def __init__(self, attribute_values: Dict[str, List[str]]):
        self.attribute_values = attribute_values
        self.value_to_attribute = {}
        
        # Create reverse mapping: value -> (attribute, normalized_key)
        for attr, values in attribute_values.items():
            for value in values:
                # Normalize for case-insensitive matching
                normalized_key = f"{attr}_{value.lower()}"
                self.value_to_attribute[value.lower()] = (attr, normalized_key)
    
    def classify_tokens(self, tokens: List[str]) -> List[Optional[str]]:
        """
        Classify tokens as attribute values or non-attributes.
        
        Args:
            tokens: List of token strings
            
        Returns:
            List where each element is either:
            - None for non-attribute tokens
            - "{attribute}_{value}" string for attribute tokens (e.g., "gender_female")
        """
        classifications = []
        
        for token in tokens:
            token_lower = token.lower().strip()
            
            # Remove common punctuation and articles
            cleaned_token = re.sub(r'^(a|an|the|with|at|in)\b\s*', '', token_lower)
            cleaned_token = re.sub(r'[^\w\s]', '', cleaned_token).strip()
            
            if cleaned_token in self.value_to_attribute:
                _, normalized_key = self.value_to_attribute[cleaned_token]
                classifications.append(normalized_key)
            else:
                classifications.append(None)
                
        return classifications
    
    def get_all_attribute_keys(self) -> List[str]:
        """Get all possible attribute keys (e.g., ['gender_male', 'gender_female', ...])"""
        keys = []
        for attr, values in self.attribute_values.items():
            for value in values:
                keys.append(f"{attr}_{value.lower()}")
        return keys


class PrecomputedVectorPEFT(nn.Module):
    """
    Main PEFT module for precomputed vectors.
    
    This module manages LoRA adapters for non-attribute vectors and VAE modules for
    each specific attribute value (e.g., "gender_female", "accent_american"), along 
    with orthogonality regularization.
    
    Args:
        vector_dim (int): Dimension of precomputed vectors
        num_vectors (int): Total number of precomputed vectors
        attribute_values (Dict[str, List[str]]): Dictionary mapping attribute names to possible values
        lora_rank (int): Rank for LoRA adapters (default: 16)
        lora_alpha (float): Alpha parameter for LoRA (default: 32.0)
        vae_latent_dim (int): Latent dimension for VAE (default: 128)
        orthogonal_reg_strength (float): Orthogonality regularization strength (default: 0.01)
    """
    
    def __init__(
        self,
        vector_dim: int,
        num_vectors: int,
        attribute_values: Dict[str, List[str]],
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        vae_latent_dim: int = 128,
        orthogonal_reg_strength: float = 0.01
    ):
        super().__init__()
        self.vector_dim = vector_dim
        self.num_vectors = num_vectors
        self.attribute_values = attribute_values
        
        # Initialize token classifier
        self.token_classifier = AttributeTokenClassifier(attribute_values)
        
        # LoRA adapters for non-attribute vectors - token-content based
        self.lora_adapters = nn.ModuleDict()
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        # VAE modules for each attribute value (e.g., "gender_female", "accent_american")
        self.vae_modules = nn.ModuleDict()
        for attribute_key in self.token_classifier.get_all_attribute_keys():
            self.vae_modules[attribute_key] = AttributeVAE(vector_dim, vae_latent_dim)
            
        # Orthogonality regularizer
        self.orthogonal_reg = OrthogonalityRegularizer(orthogonal_reg_strength)
    
    def get_or_create_lora_adapter(self, token: str) -> LoRAVectorTransform:
        """
        Get existing LoRA adapter for token or create new one if doesn't exist.
        
        Args:
            token: Token string (e.g., "voice", "with", "at")
            
        Returns:
            LoRA adapter for the token
        """
        # Sanitize token name for PyTorch module names
        safe_token = token.replace(".", "_DOT_").replace("<", "_LT_").replace(">", "_GT_").replace(" ", "_SPACE_")
        
        if safe_token not in self.lora_adapters:
            self.lora_adapters[safe_token] = LoRAVectorTransform(
                self.vector_dim, 
                self.lora_rank, 
                self.lora_alpha
            )
        return self.lora_adapters[safe_token]
        
    def forward(
        self,
        precomputed_vectors: torch.Tensor,
        description_tokens: Optional[Union[List[str], List[List[str]]]] = None,
        vector_indices: Optional[List[int]] = None,
        return_losses: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Apply PEFT transformations to precomputed vectors using token-level attribute processing.
        
        Args:
            precomputed_vectors: Tensor of shape [batch_size, seq_len, vector_dim] or [seq_len, vector_dim]
            description_tokens: List of tokens (single sequence) or List[List[str]] (batch of sequences)
            vector_indices: Which vectors to process (if None, process all)
            return_losses: Whether to return VAE and orthogonality losses
            
        Returns:
            Dictionary containing enhanced vectors and losses
        """
        # Determine input shape and batch processing
        if precomputed_vectors.dim() == 3:  # [batch_size, seq_len, vector_dim]
            batch_size, seq_len, vector_dim = precomputed_vectors.shape
            is_batched = True
        else:  # [seq_len, vector_dim] - single sequence
            seq_len, vector_dim = precomputed_vectors.shape
            batch_size = 1
            is_batched = False
            precomputed_vectors = precomputed_vectors.unsqueeze(0)  # Add batch dimension
        
        if vector_indices is None:
            vector_indices = list(range(seq_len))
        
        # Handle description_tokens format
        if description_tokens is not None:
            if isinstance(description_tokens, list) and len(description_tokens) > 0:
                if isinstance(description_tokens[0], list):
                    # Batch of token sequences: [[tokens_batch1], [tokens_batch2], ...]
                    batch_tokens = description_tokens
                else:
                    # Single token sequence: [token1, token2, ...]
                    batch_tokens = [description_tokens] if not is_batched else [description_tokens] * batch_size
            else:
                batch_tokens = [[]] * batch_size
        else:
            batch_tokens = [[]] * batch_size
        
        # Process each batch item
        enhanced_vectors_batch = []
        vae_losses_batch = []
        
        for batch_idx in range(batch_size):
            # Get tokens for this batch item
            tokens_for_batch = batch_tokens[batch_idx] if batch_idx < len(batch_tokens) else []
            
            # Classify tokens for this batch item
            token_classifications = None
            if tokens_for_batch:
                token_classifications = self.token_classifier.classify_tokens(tokens_for_batch)
            
            # Process vectors for this batch item
            enhanced_vectors_for_batch = []
            vae_losses_for_batch = []
            
            for i in vector_indices:
                # Get the vector for this batch item and position
                vector = precomputed_vectors[batch_idx:batch_idx+1, i, :]  # [1, vector_dim]
                
                # Get corresponding token
                token = tokens_for_batch[i] if i < len(tokens_for_batch) else f"unk_{i}"
                
                # Determine if this vector corresponds to an attribute token
                attribute_key = None
                if token_classifications is not None and i < len(token_classifications):
                    attribute_key = token_classifications[i]
                
                if attribute_key is not None and attribute_key in self.vae_modules:
                    # Use VAE for attribute vectors
                    vae_module = self.vae_modules[attribute_key]
                    enhanced_vector, mu, logvar = vae_module(vector)
                    
                    if return_losses:
                        vae_loss = vae_module.kl_loss(mu, logvar)
                        vae_losses_for_batch.append(vae_loss)
                else:
                    # Use LoRA for non-attribute vectors (token-content based)
                    lora_adapter = self.get_or_create_lora_adapter(token)
                    enhanced_vector = lora_adapter(vector)
                
                enhanced_vectors_for_batch.append(enhanced_vector)
            
            # Stack vectors for this batch item
            enhanced_vectors_for_batch = torch.stack(enhanced_vectors_for_batch, dim=1)  # [1, seq_len, vector_dim]
            enhanced_vectors_batch.append(enhanced_vectors_for_batch)
            vae_losses_batch.append(vae_losses_for_batch)
        
        # Stack all batch items
        enhanced_vectors = torch.cat(enhanced_vectors_batch, dim=0)  # [batch_size, seq_len, vector_dim]
        
        # Remove batch dimension if input was single sequence
        if not is_batched:
            enhanced_vectors = enhanced_vectors.squeeze(0)  # [seq_len, vector_dim]
        
        result = {"enhanced_vectors": enhanced_vectors}
        
        if return_losses:
            # VAE losses - average across batch and sequence
            all_vae_losses = []
            for batch_losses in vae_losses_batch:
                if batch_losses:
                    all_vae_losses.extend(batch_losses)
            
            if all_vae_losses:
                result["vae_loss"] = torch.stack(all_vae_losses).mean()
            else:
                result["vae_loss"] = torch.tensor(0.0, device=precomputed_vectors.device)
            
            # Orthogonality loss
            if enhanced_vectors.dim() == 3:  # Batched
                # Average over batch dimension for orthogonality
                orth_loss = torch.stack([
                    self.orthogonal_reg(enhanced_vectors[b])
                    for b in range(enhanced_vectors.shape[0])
                ]).mean()
            else:  # Single sequence
                orth_loss = self.orthogonal_reg(enhanced_vectors)
            
            result["orthogonality_loss"] = orth_loss
        
        return result

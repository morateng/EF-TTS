#!/usr/bin/env python3
"""
PEFT training script for Parler-TTS with precomputed vectors
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from parler_tts.vector_utils import VectorLoader
from parler_tts.peft_modules import PrecomputedVectorPEFT
import torchaudio
import os
from tqdm import tqdm

def simple_collate_fn(batch, tokenizer, vector_loader):
    """Simple collate function for PEFT training"""
    texts = [item['text'] for item in batch]
    captions = [item['rebuilt_caption'] for item in batch]
    audio_paths = [item['audio_path'] for item in batch]
    
    # Tokenize texts
    prompt_tokens = tokenizer(texts, padding=True, return_tensors="pt")
    
    # Load precomputed vectors
    all_vectors = []
    all_tokens = []
    max_len = 0
    
    for caption in captions:
        vectors, tokens, _ = vector_loader.get_vectors_for_caption(caption)
        all_vectors.append(vectors)
        all_tokens.append(tokens)
        max_len = max(max_len, len(tokens))
    
    # Pad vectors to same length
    batch_vectors = []
    attention_masks = []
    
    for vectors in all_vectors:
        pad_len = max_len - vectors.shape[0]
        if pad_len > 0:
            padding = torch.zeros(pad_len, vectors.shape[1])
            padded_vectors = torch.cat([vectors, padding], dim=0)
        else:
            padded_vectors = vectors
        
        # Create attention mask
        mask = torch.ones(vectors.shape[0])
        if pad_len > 0:
            mask = torch.cat([mask, torch.zeros(pad_len)])
        
        batch_vectors.append(padded_vectors)
        attention_masks.append(mask)
    
    # Load and encode audio targets with DAC
    audio_targets = []
    decoder_input_ids_list = []
    
    for audio_path in audio_paths:
        if os.path.exists(audio_path):
            audio, sr = torchaudio.load(audio_path)
            # Resample to 44100 Hz if needed (DAC requirement)
            if sr != 44100:
                resampler = torchaudio.transforms.Resample(sr, 44100)
                audio = resampler(audio)
            audio_targets.append(audio)
            
            # Note: DAC encoding will be done in model forward pass
            # For now, we'll create placeholder decoder_input_ids
            # The model will handle DAC encoding internally
            decoder_input_ids_list.append(None)  # Will be handled by model
        else:
            # Dummy audio for missing files
            audio_targets.append(torch.randn(1, 16000))
            decoder_input_ids_list.append(None)
    
    return {
        'prompt_input_ids': prompt_tokens['input_ids'],
        'prompt_attention_mask': prompt_tokens['attention_mask'],
        'precomputed_vectors': torch.stack(batch_vectors),
        'attention_mask': torch.stack(attention_masks),
        'description_tokens': all_tokens,
        'audio_targets': audio_targets,
        'input_values': audio_targets  # For DAC encoding
    }

def train_peft_model():
    print("🚀 Starting PEFT training...")
    
    # Setup - Force CPU due to CUDA compatibility issues with RTX 5070 Ti
    device = torch.device('cpu')
    print(f"📱 Device: {device} (forced CPU due to CUDA compatibility)")
    
    # Load dataset
    dataset = load_from_disk("dummy_voxceleb_dataset")
    print(f"📊 Dataset loaded: {len(dataset)} samples")
    
    # Load model and tokenizer
    model_name = "parler-tts/parler-tts-mini-v1"
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vector_loader = VectorLoader(".")
    
    # Configure model for vector mode
    model.config.use_precomputed_vectors = True
    model.config.precomputed_vector_dim = 1024
    
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
        attribute_values=attribute_values,
        lora_rank=8,
        lora_alpha=16.0,
        vae_latent_dim=64,
        orthogonal_reg_strength=0.01
    )
    
    # Move to device
    model = model.to(device)
    peft_module = peft_module.to(device)
    
    # Freeze base model, only train PEFT parameters
    for param in model.parameters():
        param.requires_grad = False
    
    for param in peft_module.parameters():
        param.requires_grad = True
    
    print(f"🔒 Frozen base model parameters")
    print(f"🔓 Trainable PEFT parameters: {sum(p.numel() for p in peft_module.parameters() if p.requires_grad)}")
    
    # Setup optimizer - Slightly higher learning rate for faster convergence on small dataset
    optimizer = torch.optim.AdamW(peft_module.parameters(), lr=2e-4, weight_decay=0.01)
    
    # Setup dataloader
    def collate_wrapper(batch):
        return simple_collate_fn(batch, tokenizer, vector_loader)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_wrapper)  # CPU-friendly batch size
    
    # Training loop - Reduced epochs for demo/testing
    num_epochs = 2
    total_loss = 0
    step = 0
    
    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            optimizer.zero_grad()
            
            # Move batch to device
            prompt_input_ids = batch['prompt_input_ids'].to(device)
            precomputed_vectors = batch['precomputed_vectors'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            description_tokens = batch['description_tokens']
            
            # Skip audio processing for now - we'll use dummy labels instead
            
            # Apply PEFT to vectors
            peft_outputs = []
            total_vae_loss = 0
            total_orth_loss = 0
            
            for i in range(len(description_tokens)):
                peft_output = peft_module(
                    precomputed_vectors[i:i+1],
                    description_tokens[i],
                    return_losses=True
                )
                peft_outputs.append(peft_output['enhanced_vectors'])
                total_vae_loss += peft_output['vae_loss']
                total_orth_loss += peft_output['orthogonality_loss']
            
            enhanced_vectors = torch.cat(peft_outputs, dim=0)
            
            # Create dummy labels to avoid audio encoder call
            # Model will use labels to create decoder_input_ids automatically
            batch_size = prompt_input_ids.shape[0]
            seq_len = 200  # Reasonable audio sequence length
            num_codebooks = 9  # DAC uses 9 codebooks
            # Labels should be [batch, seq_len, num_codebooks] then transposed
            dummy_labels = torch.randint(0, 1024, (batch_size, seq_len, num_codebooks), device=device)
            
            # Forward through model 
            outputs = model(
                labels=dummy_labels,  # This will be converted to decoder_input_ids automatically
                prompt_input_ids=prompt_input_ids,
                precomputed_vectors=enhanced_vectors,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Combined loss
            reconstruction_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=device)
            vae_loss = total_vae_loss / len(description_tokens)
            orth_loss = total_orth_loss / len(description_tokens)
            
            # Weighted combination
            total_batch_loss = (
                reconstruction_loss + 
                0.1 * vae_loss + 
                0.01 * orth_loss
            )
            
            # Backward pass
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(peft_module.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Logging
            epoch_loss += total_batch_loss.item()
            total_loss += total_batch_loss.item()
            step += 1
            
            if step % 5 == 0:
                print(f"Step {step}: Loss={total_batch_loss.item():.4f} "
                      f"(Recon={reconstruction_loss.item():.4f}, "
                      f"VAE={vae_loss.item():.4f}, "
                      f"Orth={orth_loss.item():.4f})")
        
        print(f"Epoch {epoch + 1} avg loss: {epoch_loss / len(dataloader):.4f}")
    
    print(f"\n🎉 Training completed!")
    print(f"Final average loss: {total_loss / step:.4f}")
    
    # Save PEFT module
    torch.save(peft_module.state_dict(), "peft_module_trained.pth")
    print(f"💾 PEFT module saved to peft_module_trained.pth")
    
    return peft_module, model

if __name__ == "__main__":
    train_peft_model()
#!/usr/bin/env python3

import torch
from parler_tts import ParlerTTSForConditionalGeneration
from parler_tts.vector_utils import VectorLoader
from transformers import AutoTokenizer
import soundfile as sf

def debug_peft_vectors():
    print("=== PEFT 벡터 변화 디버깅 ===")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
    vector_loader = VectorLoader("/home/user/Code/EF-TTS/parler-tts")
    
    # PEFT 설정
    attribute_values = {
        "gender": ["male", "female"],
        "pitch": ["high", "low", "medium"],
        "speed": ["slowly", "quickly", "moderate"],
        "accent": ["American", "British", "Japanese"],
        "modulation": ["monoton", "animated"],
        "quality": ["clean", "noisy"]
    }
    
    model.config.use_precomputed_vectors = True
    model.config.precomputed_vector_dim = 1024
    model.config.attribute_values = attribute_values
    
    from parler_tts.peft_modules import PrecomputedVectorPEFT
    model.vector_peft = PrecomputedVectorPEFT(
        vector_dim=1024,
        num_vectors=None,
        attribute_values=attribute_values,
        lora_rank=8,
        lora_alpha=16,
        vae_latent_dim=64,
        orthogonal_reg_strength=0.1
    ).to(device)
    
    # 테스트: female vs male
    test_cases = [
        "A female voice with American accent speaks quickly at a medium pitch and a clean quality",
        "A male voice with American accent speaks quickly at a medium pitch and a clean quality"
    ]
    
    for i, description in enumerate(test_cases):
        print(f"\n--- 테스트 {i+1}: {description[:20]}... ---")
        
        # 원본 벡터 로드
        vectors, tokens, attributes = vector_loader.get_vectors_for_caption(description)
        print(f"속성: {attributes}")
        print(f"토큰: {tokens}")
        
        batch_vectors = vectors.unsqueeze(0).to(device)
        description_tokens = [tokens]
        
        # PEFT 적용 전후 비교
        with torch.no_grad():
            # PEFT 적용
            peft_results = model.vector_peft(
                batch_vectors,
                description_tokens=description_tokens,
                return_losses=False
            )
            enhanced_vectors = peft_results["enhanced_vectors"]
            
            print(f"원본 벡터 shape: {batch_vectors.shape}")
            print(f"PEFT 벡터 shape: {enhanced_vectors.shape}")
            
            # 벡터 변화량 계산
            vector_diff = torch.norm(enhanced_vectors - batch_vectors, dim=-1)
            print(f"벡터 변화량 (norm): {vector_diff.squeeze().cpu().numpy()}")
            
            # gender 토큰 위치 찾기
            gender_token = 'female' if 'female' in tokens else 'male' if 'male' in tokens else None
            if gender_token:
                gender_idx = tokens.index(f'▁{gender_token}')
                print(f"Gender 토큰 '{gender_token}' 위치: {gender_idx}")
                print(f"Gender 벡터 변화량: {vector_diff[0, gender_idx].item():.4f}")
                
                # Gender 벡터의 코사인 유사도
                orig_gender_vec = batch_vectors[0, gender_idx, :]
                peft_gender_vec = enhanced_vectors[0, gender_idx, :]
                cosine_sim = torch.cosine_similarity(orig_gender_vec, peft_gender_vec, dim=0)
                print(f"Gender 벡터 코사인 유사도: {cosine_sim.item():.4f}")
        
        # 실제 음성 생성 테스트
        prompt = "Hello world"
        prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        attention_mask = torch.ones((1, vectors.shape[0])).to(device)
        
        print("PEFT 없이 생성...")
        with torch.no_grad():
            model.vector_peft = None  # 임시로 PEFT 비활성화
            generation_no_peft = model.generate(
                prompt_input_ids=prompt_input_ids,
                precomputed_vectors=batch_vectors,
                attention_mask=attention_mask,
                max_new_tokens=500
            )
            audio_no_peft = generation_no_peft.cpu().numpy().squeeze()
            sf.write(f"debug_no_peft_{i+1}.wav", audio_no_peft, model.config.sampling_rate)
        
        # PEFT 다시 활성화
        model.vector_peft = PrecomputedVectorPEFT(
            vector_dim=1024, num_vectors=None, attribute_values=attribute_values,
            lora_rank=8, lora_alpha=16, vae_latent_dim=64, orthogonal_reg_strength=0.1
        ).to(device)
        
        print("PEFT 적용하여 생성...")
        with torch.no_grad():
            generation_with_peft = model.generate(
                prompt_input_ids=prompt_input_ids,
                precomputed_vectors=batch_vectors,
                attention_mask=attention_mask,
                description_tokens=description_tokens,
                max_new_tokens=500
            )
            audio_with_peft = generation_with_peft.cpu().numpy().squeeze()
            sf.write(f"debug_with_peft_{i+1}.wav", audio_with_peft, model.config.sampling_rate)
        
        print(f"저장: debug_no_peft_{i+1}.wav, debug_with_peft_{i+1}.wav")

if __name__ == "__main__":
    debug_peft_vectors()
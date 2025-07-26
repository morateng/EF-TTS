#!/usr/bin/env python3

import torch
from parler_tts import ParlerTTSForConditionalGeneration
from parler_tts.vector_utils import VectorLoader
from transformers import AutoTokenizer
import soundfile as sf

def test_peft_generation():
    print("=== PEFT 적용 음성 생성 테스트 ===")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
    vector_loader = VectorLoader("/home/user/Code/EF-TTS/parler-tts")
    
    print(f"디바이스: {device}")
    
    # PEFT에 필요한 attribute_values 정의
    attribute_values = {
        "gender": ["male", "female"],
        "pitch": ["high", "low", "medium"],
        "speed": ["slowly", "quickly", "moderate"],
        "accent": [
            "American", "British", "Japanese", "German", "French", "Italian", 
            "Spanish", "Chinese", "Russian", "Australian", "Canadian"
        ],
        "modulation": ["monoton", "animated"],
        "quality": ["clean", "noisy"]
    }
    
    # 모델 config에 PEFT 설정 추가
    model.config.use_precomputed_vectors = True
    model.config.precomputed_vector_dim = 1024
    model.config.num_precomputed_vectors = None  # Dynamic
    model.config.attribute_values = attribute_values
    model.config.lora_rank = 8
    model.config.lora_alpha = 16
    model.config.vae_latent_dim = 64
    model.config.orthogonal_reg_strength = 0.1
    
    # PEFT 모듈 수동 초기화
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
    
    print("✅ PEFT 모듈 초기화 완료")
    print(f"PEFT 모듈: {type(model.vector_peft).__name__}")
    
    # 테스트 케이스들
    test_cases = [
        {
            "description": "A female voice with American accent speaks quickly at a medium pitch and a clean quality",
            "prompt": "Hello, this is a test with PEFT applied."
        },
        {
            "description": "A male voice with British accent speaks slowly at a high pitch and a clean quality", 
            "prompt": "The weather is nice today."
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        description = test_case["description"]
        prompt = test_case["prompt"]
        
        print(f"\n--- PEFT 테스트 {i+1} ---")
        print(f"스타일: {description}")
        print(f"텍스트: {prompt}")
        
        try:
            # 벡터와 토큰 정보 준비
            vectors, tokens, attributes = vector_loader.get_vectors_for_caption(description)
            print(f"벡터 길이: {vectors.shape[0]}, 속성: {attributes}")
            
            # 모델 입력 준비
            prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            batch_vectors = vectors.unsqueeze(0).to(device)
            attention_mask = torch.ones((1, vectors.shape[0])).to(device)
            
            # PEFT를 위한 추가 정보 (attribute_indices는 자동 계산됨)
            description_tokens = [tokens]
            
            print(f"토큰들: {tokens}")
            
            # PEFT 적용하여 음성 생성
            print("🎵 PEFT 적용하여 음성 생성 중...")
            with torch.no_grad():
                generation = model.generate(
                    prompt_input_ids=prompt_input_ids,
                    precomputed_vectors=batch_vectors,
                    attention_mask=attention_mask,
                    description_tokens=description_tokens,
                    max_new_tokens=1000,
                    do_sample=True,
                    temperature=1.0
                )
            
            # 오디오 변환 및 저장
            audio_arr = generation.cpu().numpy().squeeze()
            filename = f"peft_generation_{i+1}.wav"
            sf.write(filename, audio_arr, model.config.sampling_rate)
            
            duration = len(audio_arr) / model.config.sampling_rate
            print(f"✅ 저장: {filename} ({duration:.2f}초)")
            
        except Exception as e:
            print(f"❌ 에러: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n=== PEFT 테스트 완료 ===")

if __name__ == "__main__":
    test_peft_generation()
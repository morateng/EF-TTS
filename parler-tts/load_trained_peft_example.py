#!/usr/bin/env python3

import torch
from parler_tts import ParlerTTSForConditionalGeneration
from parler_tts.vector_utils import VectorLoader
from parler_tts.peft_modules import PrecomputedVectorPEFT
from transformers import AutoTokenizer
import soundfile as sf

def load_trained_peft_model(peft_checkpoint_path):
    """학습된 PEFT 모듈을 포함한 모델 로드"""
    
    # 1단계: 기본 모델 로드
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
    vector_loader = VectorLoader("/home/user/Code/EF-TTS/parler-tts")
    
    # 2단계: PEFT 체크포인트 로드
    if torch.cuda.is_available():
        checkpoint = torch.load(peft_checkpoint_path)
    else:
        checkpoint = torch.load(peft_checkpoint_path, map_location='cpu')
    
    print("체크포인트 내용:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    
    # 3단계: 모델 설정 업데이트
    model.config.use_precomputed_vectors = True
    model.config.precomputed_vector_dim = 1024
    model.config.attribute_values = checkpoint['attribute_values']
    model.config.lora_rank = checkpoint['lora_rank']
    model.config.lora_alpha = checkpoint.get('lora_alpha', 16)
    model.config.vae_latent_dim = checkpoint['vae_latent_dim']
    model.config.orthogonal_reg_strength = checkpoint.get('orthogonal_reg_strength', 0.1)
    
    # 4단계: PEFT 모듈 초기화
    model.vector_peft = PrecomputedVectorPEFT(
        vector_dim=1024,
        num_vectors=None,
        attribute_values=checkpoint['attribute_values'],
        lora_rank=checkpoint['lora_rank'],
        lora_alpha=checkpoint.get('lora_alpha', 16),
        vae_latent_dim=checkpoint['vae_latent_dim'],
        orthogonal_reg_strength=checkpoint.get('orthogonal_reg_strength', 0.1)
    ).to(device)
    
    # 5단계: 학습된 가중치 로드
    model.vector_peft.load_state_dict(checkpoint['peft_state_dict'])
    model.vector_peft.eval()
    
    print("✅ 학습된 PEFT 모듈 로드 완료!")
    
    return model, tokenizer, vector_loader

def test_trained_peft(peft_checkpoint_path):
    """학습된 PEFT로 음성 생성 테스트"""
    
    try:
        model, tokenizer, vector_loader = load_trained_peft_model(peft_checkpoint_path)
        
        # 테스트 케이스
        test_cases = [
            {
                "description": "A female voice with American accent speaks quickly at a medium pitch and a clean quality",
                "prompt": "Hello, this is trained PEFT speaking."
            },
            {
                "description": "A male voice with British accent speaks slowly at a high pitch and a clean quality", 
                "prompt": "The trained model should now work correctly."
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            description = test_case["description"]
            prompt = test_case["prompt"]
            
            print(f"\n--- 학습된 PEFT 테스트 {i+1} ---")
            print(f"스타일: {description}")
            print(f"텍스트: {prompt}")
            
            # 벡터 준비
            vectors, tokens, attributes = vector_loader.get_vectors_for_caption(description)
            print(f"속성: {attributes}")
            
            # 모델 입력 준비
            device = next(model.parameters()).device
            prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            batch_vectors = vectors.unsqueeze(0).to(device)
            attention_mask = torch.ones((1, vectors.shape[0])).to(device)
            description_tokens = [tokens]
            
            # 학습된 PEFT로 음성 생성
            print("🎵 학습된 PEFT로 음성 생성 중...")
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
            
            # 오디오 저장
            audio_arr = generation.cpu().numpy().squeeze()
            filename = f"trained_peft_generation_{i+1}.wav"
            sf.write(filename, audio_arr, model.config.sampling_rate)
            
            duration = len(audio_arr) / model.config.sampling_rate
            print(f"✅ 저장: {filename} ({duration:.2f}초)")
            
    except FileNotFoundError:
        print("❌ PEFT 체크포인트 파일을 찾을 수 없습니다.")
        print("먼저 훈련을 완료해서 PEFT 가중치를 저장하세요.")
    except Exception as e:
        print(f"❌ 에러: {e}")

if __name__ == "__main__":
    # 예시 사용법
    peft_checkpoint_path = "peft_checkpoint.pth"  # 실제 경로로 수정
    test_trained_peft(peft_checkpoint_path)
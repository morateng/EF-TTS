#!/usr/bin/env python3

import torch
from parler_tts import ParlerTTSForConditionalGeneration
from parler_tts.vector_utils import VectorLoader
from transformers import AutoTokenizer
import soundfile as sf

def test_untrained_model():
    print("=== 학습되지 않은 모델로 벡터 기반 음성 생성 테스트 ===")
    
    # 모델과 토크나이저 로드
    print("모델 로딩 중...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
    vector_loader = VectorLoader("/home/user/Code/EF-TTS/parler-tts")
    
    print(f"사용 디바이스: {device}")
    
    # 벡터 모드 설정
    model.config.use_precomputed_vectors = True
    model.config.precomputed_vector_dim = 1024
    
    # 자연스러운 문장 형태의 스타일 설명들
    test_cases = [
        {
            "description": "A female voice with American accent speaks quickly at a medium pitch and a clean quality",
            "prompt": "Hello, how are you doing today?"
        },
        {
            "description": "A male voice with British accent speaks slowly at a high pitch and a clean quality", 
            "prompt": "This is a test of our vector based speech synthesis system"
        },
        {
            "description": "A female voice with Japanese accent speaks at a moderate speed with a low pitch and a noisy quality",
            "prompt": "The weather is nice today, isn't it?"
        }
    ]
    
    print("\n=== 음성 생성 시작 ===")
    
    for i, test_case in enumerate(test_cases):
        description = test_case["description"]
        prompt = test_case["prompt"]
        
        print(f"\n--- 테스트 {i+1} ---")
        print(f"스타일: {description}")
        print(f"텍스트: {prompt}")
        
        try:
            # 벡터 준비
            vectors, tokens, attributes = vector_loader.get_vectors_for_caption(description)
            print(f"벡터 시퀀스 길이: {vectors.shape[0]}")
            print(f"토큰들: {tokens}")
            print(f"속성 토큰들: {attributes}")
            
            # 입력 준비
            prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            batch_vectors = vectors.unsqueeze(0).to(device)  # [1, seq_len, 1024]
            attention_mask = torch.ones((1, vectors.shape[0])).to(device)
            
            print(f"입력 벡터 shape: {batch_vectors.shape}")
            print(f"프롬프트 토큰 shape: {prompt_input_ids.shape}")
            
            # 음성 생성 (README.md 방식)
            print("음성 생성 중...")
            with torch.no_grad():
                generation = model.generate(
                    prompt_input_ids=prompt_input_ids,
                    precomputed_vectors=batch_vectors,
                    attention_mask=attention_mask,
                    max_new_tokens=1000,
                    do_sample=True,
                    temperature=1.0
                )
            
            print(f"생성된 결과 shape: {generation.shape}")
            
            # README.md 방식으로 오디오 변환
            audio_arr = generation.cpu().numpy().squeeze()
            print(f"오디오 배열 shape: {audio_arr.shape}")
            
            # 오디오 저장
            filename = f"untrained_test_{i+1}.wav"
            sf.write(filename, audio_arr, model.config.sampling_rate)
            print(f"오디오 저장: {filename}")
            
            # 오디오 길이 정보
            duration = len(audio_arr) / model.config.sampling_rate
            print(f"오디오 길이: {duration:.2f}초")
            print(f"샘플링 레이트: {model.config.sampling_rate} Hz")
            
        except Exception as e:
            print(f"에러 발생: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n=== 테스트 완료 ===")
    print("생성된 오디오 파일들:")
    for i in range(len(test_cases)):
        print(f"  - untrained_test_{i+1}.wav")

if __name__ == "__main__":
    test_untrained_model()
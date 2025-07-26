import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, set_seed
import soundfile as sf
from transformers.modeling_outputs import BaseModelOutput

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
# model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

prompt = "Hey how are you doing today? Let's have a conversation about the weather."
description = "A male voice with British accent at a medium pitch and a noisy quality."
input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# --- 벡터 기반 description 인코딩 ---
import os

def get_mean_vector_for_token(token_str, vector_dirs):
    # Try multiple possible filenames
    possible_names = [
        token_str,  # Original token
        token_str.replace("▁", ""),  # Without prefix
        f"▁{token_str}",  # With prefix
        token_str.replace("</s>", "<_s>")  # Handle end token
    ]
    
    for folder in vector_dirs:
        for name in possible_names:
            path = os.path.join("vectors", folder, f"{name}.pt")
            if os.path.exists(path):
                data = torch.load(path)
                # Handle different data formats
                if isinstance(data, dict):
                    if 'vector' in data:
                        return data['vector'].to(device)
                    elif 'mean_vector' in data:
                        return data['mean_vector'].to(device)
                    else:
                        # Take first tensor value if it's a dict
                        return list(data.values())[0].to(device)
                else:
                    return data.to(device)
    
    raise ValueError(f"Token vector not found for: {token_str} (tried: {possible_names})")

# Step 1: 토큰화 및 토큰 문자열 추출
input_ids_flat = tokenizer(description, return_tensors="pt").input_ids[0]
token_strs = tokenizer.convert_ids_to_tokens(input_ids_flat)

# Step 2: 디렉토리 우선순위
vector_dirs = ["gender", "pitch", "speed", "accent", "modulation", "quality", "nonattr_tokens"]

# Step 3: 벡터 불러오기 및 연결
vectors = []
for token_str in token_strs:
    # Keep original token (function will handle different formats)
    vec = get_mean_vector_for_token(token_str, vector_dirs)
    vectors.append(vec)

encoder_hidden_states = torch.stack(vectors).unsqueeze(0)  # [1, seq_len, dim]
enc_out = BaseModelOutput(last_hidden_state=encoder_hidden_states)


generation = model.generate(encoder_outputs=enc_out, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("vec_con_out2.wav", audio_arr, model.config.sampling_rate)
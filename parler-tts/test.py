import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("🔄 Loading model...")
# Use existing model for testing (PEFT functionality is built-in but not enabled by default)
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
print("✅ Model loaded successfully!")

# For now, we'll test the traditional way since we don't have a trained PEFT model yet
print("🎵 Generating audio with traditional method...")

prompt = "Hey, how are you doing today?"
description = "A female speaker with a clear voice delivers speech at moderate speed."

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Generate audio using traditional method
generation = model.generate(
    input_ids=input_ids,
    prompt_input_ids=prompt_input_ids,
    max_length=1000
)

audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)
print(f"✅ Audio saved as 'parler_tts_out.wav' ({len(audio_arr)} samples, {model.config.sampling_rate} Hz)")

# Demonstrate precomputed vector extraction for future use
print("\n📤 Extracting precomputed vectors for future PEFT use...")
with torch.no_grad():
    encoder_outputs = model.text_encoder(input_ids=input_ids)
    precomputed_vectors = encoder_outputs.last_hidden_state

print(f"✅ Extracted precomputed vectors: {precomputed_vectors.shape}")
print(f"   These vectors can be used with a trained PEFT model in the future!")
print(f"   Vector dimension: {precomputed_vectors.shape[-1]}")
print(f"   Sequence length: {precomputed_vectors.shape[1]}")

# Save the vectors for future use
torch.save(precomputed_vectors, "precomputed_vectors.pt")
print("💾 Precomputed vectors saved as 'precomputed_vectors.pt'")
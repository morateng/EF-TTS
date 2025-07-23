import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
from datasets import load_dataset
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
attribute_values = {
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

encoder = model.text_encoder

# Load 100 reconstructed captions from the Hugging Face dataset
dataset = load_dataset(
    "morateng/CapTTS-SFT-voxceleb-cleaned", split="train"
)
# Collect 500 captions containing 'female' and 500 containing 'male'
all_caps = dataset['rebuilt_caption']
# female_caps = [c for c in all_caps if 'female' in c.lower()][:100]
# male_caps = [c for c in all_caps if 'male' in c.lower()][:100]

# description = female_caps + male_caps
# print(description)

# # Compute and save mean vector for 'female' via per-example encoding
# female_vecs = []

# for cap in female_caps:
#     # tokenize single caption
#     inputs = tokenizer(cap, return_tensors="pt").to(device)
#     # encode
#     outputs = encoder(**inputs)
#     hidden = outputs.last_hidden_state  # [1, seq_len, hidden_dim]
#     # identify token positions for 'female'
#     token_ids = inputs.input_ids[0]
#     target_id = tokenizer.convert_tokens_to_ids('female')
#     mask = token_ids == target_id
#     if mask.any():
#         # average hidden states at 'female' positions
#         cap_vector = hidden[0][mask].mean(dim=0)
#     else:
#         # fallback to mean over all tokens
#         cap_vector = hidden.mean(dim=1).squeeze(0)
#     female_vecs.append(cap_vector)
# # stack and average over all caps
# female_stack = torch.stack(female_vecs, dim=0)  # [N_female, hidden_dim]
# female_mean_vector = female_stack.mean(dim=0)   # [hidden_dim]
# # save
# torch.save(female_mean_vector.cpu(), "female_mean_vector.pt")
# print(f"Saved female mean vector of shape {female_mean_vector.shape}")

# # Compute and save mean vector for 'male' via per-example encoding
# male_vecs = []
# for cap in male_caps:
#     inputs = tokenizer(cap, return_tensors="pt").to(device)
#     outputs = encoder(**inputs)
#     hidden = outputs.last_hidden_state
#     token_ids = inputs.input_ids[0]
#     target_id = tokenizer.convert_tokens_to_ids('male')
#     mask = token_ids == target_id
#     if mask.any():
#         cap_vector = hidden[0][mask].mean(dim=0)
#     else:
#         cap_vector = hidden.mean(dim=1).squeeze(0)
#     male_vecs.append(cap_vector)
# male_stack = torch.stack(male_vecs, dim=0)
# male_mean_vector = male_stack.mean(dim=0)
# torch.save(male_mean_vector.cpu(), "male_mean_vector.pt")
# print(f"Saved male mean vector of shape {male_mean_vector.shape}")

# Compute and save mean vectors for each pitch attribute
# for pitch in attribute_values['pitch']:
#     # collect captions containing the pitch keyword
#     pitch_caps = [c for c in all_caps if pitch in c.lower()][:100]
#     pitch_vecs = []
#     for cap in pitch_caps:
#         inputs = tokenizer(cap, return_tensors="pt").to(device)
#         outputs = encoder(**inputs)
#         hidden = outputs.last_hidden_state
#         token_ids = inputs.input_ids[0]
#         target_id = tokenizer.convert_tokens_to_ids(pitch)
#         mask = token_ids == target_id
#         if mask.any():
#             cap_vector = hidden[0][mask].mean(dim=0)
#         else:
#             cap_vector = hidden.mean(dim=1).squeeze(0)
#         pitch_vecs.append(cap_vector)
#     pitch_stack = torch.stack(pitch_vecs, dim=0)
#     pitch_mean_vector = pitch_stack.mean(dim=0)
#     torch.save(pitch_mean_vector.cpu(), f"{pitch}.pt")
#     print(f"Saved {pitch} mean vector of shape {pitch_mean_vector.shape}")

# Compute and save mean vectors for each speed attribute
# for speed in attribute_values['speed']:
#     # collect captions containing the speed keyword
#     speed_caps = [c for c in all_caps if speed in c.lower()][:100]
#     speed_vecs = []
#     for cap in speed_caps:
#         inputs = tokenizer(cap, return_tensors="pt").to(device)
#         outputs = encoder(**inputs)
#         hidden = outputs.last_hidden_state
#         token_ids = inputs.input_ids[0]
#         target_id = tokenizer.convert_tokens_to_ids(speed)
#         mask = token_ids == target_id
#         if mask.any():
#             cap_vector = hidden[0][mask].mean(dim=0)
#         else:
#             cap_vector = hidden.mean(dim=1).squeeze(0)
#         speed_vecs.append(cap_vector)
#     speed_stack = torch.stack(speed_vecs, dim=0)
#     speed_mean_vector = speed_stack.mean(dim=0)
#     torch.save(speed_mean_vector.cpu(), f"{speed}.pt")
#     print(f"Saved {speed} mean vector of shape {speed_mean_vector.shape}")

# Ensure output directory for accent mean vectors exists
# if not os.path.exists("accent"):
#     os.makedirs("accent")

# # Compute and save mean vectors for each accent attribute
# for accent in attribute_values['accent']:
#     # collect captions containing the accent keyword (case-insensitive)
#     acc_lower = accent.lower()
#     accent_caps = [c for c in all_caps if acc_lower in c.lower()][:100]
#     accent_vecs = []
#     for cap in accent_caps:
#         inputs = tokenizer(cap, return_tensors="pt").to(device)
#         outputs = encoder(**inputs)
#         hidden = outputs.last_hidden_state
#         token_ids = inputs.input_ids[0]
#         target_id = tokenizer.convert_tokens_to_ids(acc_lower)
#         mask = token_ids == target_id
#         if mask.any():
#             cap_vector = hidden[0][mask].mean(dim=0)
#         else:
#             cap_vector = hidden.mean(dim=1).squeeze(0)
#         accent_vecs.append(cap_vector)
#     accent_stack = torch.stack(accent_vecs, dim=0)
#     accent_mean_vector = accent_stack.mean(dim=0)
#     torch.save(accent_mean_vector.cpu(), f"accent/{accent}.pt")
#     print(f"Saved accent '{accent}' mean vector of shape {accent_mean_vector.shape}")


# # Ensure modulation output directory exists
# if not os.path.exists("modulation"):
#     os.makedirs("modulation")
# # Compute and save mean vectors for each modulation attribute
# for modulation in attribute_values['modulation']:
#     mod_lower = modulation.lower()
#     modulation_caps = [c for c in all_caps if mod_lower in c.lower()][:100]
#     modulation_vecs = []
#     for cap in modulation_caps:
#         inputs = tokenizer(cap, return_tensors="pt").to(device)
#         outputs = encoder(**inputs)
#         hidden = outputs.last_hidden_state
#         token_ids = inputs.input_ids[0]
#         target_id = tokenizer.convert_tokens_to_ids(mod_lower)
#         mask = token_ids == target_id
#         if mask.any():
#             cap_vector = hidden[0][mask].mean(dim=0)
#         else:
#             cap_vector = hidden.mean(dim=1).squeeze(0)
#         modulation_vecs.append(cap_vector)
#     modulation_stack = torch.stack(modulation_vecs, dim=0)
#     modulation_mean_vector = modulation_stack.mean(dim=0)
#     torch.save(modulation_mean_vector.cpu(), f"modulation/{modulation}.pt")
#     print(f"Saved modulation '{modulation}' mean vector of shape {modulation_mean_vector.shape}")


# # Ensure quality output directory exists
# if not os.path.exists("quality"):
#     os.makedirs("quality")
# # Compute and save mean vectors for each quality attribute
# for quality in attribute_values['quality']:
#     qual_lower = quality.lower()
#     quality_caps = [c for c in all_caps if qual_lower in c.lower()][:100]
#     quality_vecs = []
#     for cap in quality_caps:
#         inputs = tokenizer(cap, return_tensors="pt").to(device)
#         outputs = encoder(**inputs)
#         hidden = outputs.last_hidden_state
#         token_ids = inputs.input_ids[0]
#         target_id = tokenizer.convert_tokens_to_ids(qual_lower)
#         mask = token_ids == target_id
#         if mask.any():
#             cap_vector = hidden[0][mask].mean(dim=0)
#         else:
#             cap_vector = hidden.mean(dim=1).squeeze(0)
#         quality_vecs.append(cap_vector)
#     quality_stack = torch.stack(quality_vecs, dim=0)
#     quality_mean_vector = quality_stack.mean(dim=0)
#     torch.save(quality_mean_vector.cpu(), f"quality/{quality}.pt")
#     print(f"Saved quality '{quality}' mean vector of shape {quality_mean_vector.shape}")

# Compute and save mean vectors for non-attribute tokens
phrase_templates = {
        "gender":     lambda v: f"A {v} voice",
        "accent":     lambda v: f"with {v} accent",
        "speed":      lambda v: f"speaks {v}",
        "pitch":      lambda v: f"at a {v} pitch",
        "modulation": lambda v: f"with {v} modulation",
        "quality":    lambda v: f"and a {v} quality",
    }
descriptions = [
    "A male voice with Slovenia accent speaks slowly at a high pitch with monoton modulation and a clean quality.",
    "A female voice with Chinese accent speaks quickly at a low pitch with animated modulation and a noisy quality.",
    "A male voice with German accent speaks moderate at a medium pitch with monoton modulation and a clean quality.",
    "A female voice with Irish accent speaks slowly at a low pitch with animated modulation and a noisy quality.",
    "A male voice with Scottish accent speaks quickly at a high pitch with monoton modulation and a clean quality.",
    "A female voice with Russian accent speaks moderate at a medium pitch with animated modulation and a noisy quality.",
    "A male voice with Polish accent speaks slowly at a medium pitch with monoton modulation and a clean quality.",
    "A female voice with Canadian accent speaks quickly at a low pitch with animated modulation and a noisy quality.",
    "A male voice with Turkish accent speaks moderate at a high pitch with monoton modulation and a clean quality.",
    "A female voice with Czech accent speaks slowly at a medium pitch with animated modulation and a noisy quality.",
    "A male voice with Jamaica accent speaks quickly at a low pitch with monoton modulation and a clean quality.",
    "A female voice with Italian accent speaks moderate at a high pitch with animated modulation and a noisy quality.",
    "A male voice with Swiss accent speaks slowly at a medium pitch with monoton modulation and a clean quality.",
    "A female voice with Brazilian accent speaks quickly at a low pitch with animated modulation and a noisy quality.",
    "A male voice with Brooklyn accent speaks moderate at a high pitch with monoton modulation and a clean quality.",
    "A female voice with Finnish accent speaks slowly at a low pitch with animated modulation and a noisy quality.",
    "A male voice with Japanese accent speaks quickly at a medium pitch with monoton modulation and a clean quality.",
    "A female voice with Filipino accent speaks moderate at a high pitch with animated modulation and a noisy quality.",
    "A male voice with British accent speaks slowly at a low pitch with monoton modulation and a clean quality.",
    "A female voice with Indian accent speaks quickly at a medium pitch with animated modulation and a noisy quality.",
    "A male voice with Australian accent speaks moderate at a high pitch with monoton modulation and a clean quality.",
    "A female voice with Norwegian accent speaks slowly at a low pitch with animated modulation and a noisy quality.",
    "A male voice with English accent speaks quickly at a high pitch with monoton modulation and a clean quality.",
    "A female voice with Romania accent speaks moderate at a medium pitch with animated modulation and a noisy quality.",
    "A male voice with Spanish accent speaks slowly at a low pitch with monoton modulation and a clean quality.",
    "A female voice with Croatia accent speaks quickly at a high pitch with animated modulation and a noisy quality.",
    "A male voice with Swedish accent speaks moderate at a medium pitch with monoton modulation and a clean quality.",
    "A female voice with Colombia accent speaks slowly at a low pitch with animated modulation and a noisy quality.",
    "A male voice with French accent speaks quickly at a high pitch with monoton modulation and a clean quality.",
    "A female voice with American accent speaks moderate at a medium pitch with animated modulation and a noisy quality.",
    "A male voice with Mexican accent speaks slowly at a low pitch with monoton modulation and a clean quality.",
    "A female voice with Portuguese accent speaks quickly at a high pitch with animated modulation and a noisy quality.",
    "A male voice with Dominic accent speaks moderate at a medium pitch with monoton modulation and a clean quality.",
    "A female voice with Welsh accent speaks slowly at a low pitch with animated modulation and a noisy quality.",
    "A male voice with Nigeria accent speaks quickly at a high pitch with monoton modulation and a clean quality.",
    "A female voice with Chile accent speaks moderate at a medium pitch with animated modulation and a noisy quality.",
    "A male voice with Belgia accent speaks slowly at a low pitch with monoton modulation and a clean quality.",
    "A female voice with Dutch accent speaks quickly at a high pitch with animated modulation and a noisy quality.",
    "A male voice with Jordan accent speaks moderate at a medium pitch with monoton modulation and a clean quality.",
    "A female voice with Serbia accent speaks slowly at a low pitch with animated modulation and a noisy quality.",
    "A male voice with Ukrainian accent speaks quickly at a high pitch with monoton modulation and a clean quality.",
    "A female voice with Mandarin accent speaks moderate at a medium pitch with animated modulation and a noisy quality.",
    "A male voice with Chinese accent speaks slowly at a low pitch with animated modulation and a clean quality.",
    "A female voice with German accent speaks quickly at a high pitch with monoton modulation and a noisy quality.",
    "A male voice with Japanese accent speaks moderate at a medium pitch with animated modulation and a clean quality.",
    "A female voice with British accent speaks slowly at a low pitch with monoton modulation and a noisy quality.",
    "A male voice with American accent speaks quickly at a high pitch with animated modulation and a clean quality.",
    "A female voice with Australian accent speaks moderate at a medium pitch with monoton modulation and a clean quality.",
    "A male voice with Spanish accent speaks slowly at a low pitch with animated modulation and a clean quality.",
    "A female voice with Italian accent speaks quickly at a high pitch with monoton modulation and a noisy quality."
]
tok = descriptions[0]
# Inspect tokenization of the first description
inputs = tokenizer(tok, return_tensors="pt", truncation=True).to(device)
token_ids = inputs.input_ids[0].tolist()
tokens = tokenizer.convert_ids_to_tokens(token_ids)
print("tok:", tok)
print("Token IDs:", token_ids)
print("Tokens:   ", tokens)

# #
# # List of non-attribute tokens to process
# non_attr_tokens = ["A", "voice", "with", "speaks", "at", "and", "quality"]
# # Directory to save non-attribute token mean vectors
# nonattr_dir = "nonattr_tokens"
# if not os.path.exists(nonattr_dir):
#     os.makedirs(nonattr_dir)

# # Compute mean vector for each non-attribute token
# for token in non_attr_tokens:
#     # ID of this token
#     token_id = tokenizer.convert_tokens_to_ids(token.lower())
#     per_desc_vecs = []
#     for desc in descriptions:
#         # tokenize and encode
#         inputs = tokenizer(desc, return_tensors="pt", truncation=True).to(device)
#         outputs = encoder(**inputs)
#         hidden = outputs.last_hidden_state  # [1, seq_len, hidden_dim]
#         ids = inputs.input_ids[0]          # [seq_len]
#         # mask positions matching the token
#         mask = ids == token_id
#         if mask.any():
#             # average hidden states at these positions
#             vec = hidden[0][mask].mean(dim=0)
#         else:
#             # skip if this description doesn't contain the token
#             continue
#         per_desc_vecs.append(vec)
#     if not per_desc_vecs:
#         print(f"No occurrences of token '{token}' found in any description, skipping.")
#         continue
#     # stack and average over all descriptions
#     stack = torch.stack(per_desc_vecs, dim=0)  # [num_descs_with_token, hidden_dim]
#     mean_vec = stack.mean(dim=0)               # [hidden_dim]
#     # save vector
#     torch.save(mean_vec.cpu(), os.path.join(nonattr_dir, f"{token}.pt"))
#     print(f"Saved non-attribute token '{token}' mean vector with shape {mean_vec.shape}")
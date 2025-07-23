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

# Compute and save mean vectors for each non-attribute token across descriptions
nonattr_dir = "nonattr_tokens"
if not os.path.exists(nonattr_dir):
    os.makedirs(nonattr_dir)

# List of token IDs to aggregate
token_ids_list = [71, 2249, 28, 5820, 12192, 44, 3, 9, 6242, 28, 7246, 257, 11, 3, 9, 463, 5, 1]

# Prepare dictionary to collect per-token hidden states
token_hidden_dict = {tid: [] for tid in token_ids_list}

# Collect matching hidden states across all descriptions
for desc in descriptions:
    # Tokenize and encode the description
    inputs = tokenizer(desc, return_tensors="pt", truncation=True).to(device)
    outputs = encoder(**inputs)
    hidden = outputs.last_hidden_state[0]  # [seq_len, hidden_dim]
    ids = inputs.input_ids[0]
    
    for tid in token_ids_list:
        mask = (ids == tid)
        if mask.any():
            token_hidden_dict[tid].append(hidden[mask])

# Compute and save mean vector for each token ID
for tid, state_list in token_hidden_dict.items():
    if state_list:
        all_states = torch.cat(state_list, dim=0)  # [total_matches, hidden_dim]
        mean_vector = all_states.mean(dim=0)       # [hidden_dim]
        token_str = tokenizer.convert_ids_to_tokens([tid])[0].replace("/", "_")  # '/'는 파일명으로 부적절하므로 대체
        save_path = os.path.join(nonattr_dir, f"{token_str}.pt")
        torch.save(mean_vector.cpu(), save_path)
        print(f"Saved mean vector for token '{token_str}' (ID {tid}) to {save_path}")
    else:
        print(f"No matching token {tid} found in any description.")
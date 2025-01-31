import torch
import pandas as pd
import numpy as np
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

dataset_name = "Jinyan1/COLING_2025_MGT_multingual"
dataset = load_dataset(dataset_name)

models = ["utter-project/EuroLLM-1.7B","ai-forever/mGPT"]
limited_languages = {"en", "zh"}  # Limit English and Chinese
excluded_languages = {"it"}  # Exclude Italian

def get_model(model_id):
    """Load model and tokenizer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = "float16" if torch.cuda.is_available() else "float32"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if precision == "float16" else torch.float32,
        device_map="auto",
        offload_folder="./offload",
        low_cpu_mem_usage=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    max_length = 1024 if "gpt2" in model_id else 2048  # Adjust max_length based on model
    return model, tokenizer, max_length

def calculate_perplexity(text, model, tokenizer, max_length, stride=512):
    """Calculate perplexity of a given text."""
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
        
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    
    ppl = int(torch.exp(torch.stack(nlls).sum() / end_loc))
    return ppl

def process_language(lang, dataset, model_id):
    """Process a specific language and compute perplexities."""
    if lang in excluded_languages:
        return
    
    model, tokenizer, max_length = get_model(model_id)
    lang_data = dataset.filter(lambda example: example["lang"] == lang)
    sample_size = 5000 if lang in limited_languages else len(lang_data)
    lang_data = lang_data.select(range(min(sample_size, len(lang_data))))
    
    results = []
    for i, example in enumerate(lang_data, start=1):
        text, label = example["text"], example["label"]
        if len(text) >= 20:  # Enforce minimum length
            ppl = calculate_perplexity(text, model, tokenizer, max_length)
            results.append((text, label, ppl))
        
        if i % 200 == 0:
            print(f"Processed {i} texts for language {lang} ({model_id})")
    
    save_path = f"perplexity_data_coling/{model_id}/{lang}.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pd.DataFrame(results, columns=["text", "label", "perplexity"]).to_csv(save_path, index=False)
    print(f"Saved results to {save_path}")

for model_id in models:
    for lang in set(dataset["train"]["lang"]):  # Iterate over unique languages
        process_language(lang, dataset["train"], model_id)

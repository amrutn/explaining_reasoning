import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import gc
import json
import os

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-32B"
NUM_WIKI_SAMPLES = 5000         
MAX_SEQ_LEN = 1024            
MAX_ANALYSIS_LEN = 100          # Analyze first 100 tokens of answer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_FILE = "entropy_results.json"

print(f"Running on device: {DEVICE}")

# -----------------------------------------------------------------------------
# 1. LOAD MODEL AND TOKENIZER
# -----------------------------------------------------------------------------

def load_tokenizer():
    print(f"Loading tokenizer: {MODEL_NAME}...")
    return AutoTokenizer.from_pretrained(MODEL_NAME)

def load_model():
    print(f"Loading model: {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None
    )
    model.eval()
    return model

# -----------------------------------------------------------------------------
# 2. DATA LOADING & FILTERING
# -----------------------------------------------------------------------------

def get_gsm8k_data(n=None):
    print("Loading GSM8K...")
    ds = load_dataset("gsm8k", "main", split="train")
    data_items = []
    
    selection = ds if n is None else ds.select(range(min(n, len(ds))))
        
    for ex in selection:
        prompt = ex['question']
        # We append the answer to measure entropy of generating it
        full = f"{ex['question']}\n{ex['answer']}"
        data_items.append({"prompt": prompt, "full": full})
        
    return data_items

def get_wikitext_data(n=100, min_char_len=200):
    print("Loading WikiText-2 (Normal Text)...")
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        data_items = []

        # Filter for non-empty lines and meaningful content
        valid_indices = [i for i, x in enumerate(ds['text']) if len(x.strip()) > min_char_len and not x.strip().startswith('=')]

        count = min(n, len(valid_indices))
        selected_indices = valid_indices[:count]

        for idx in selected_indices:
            text = ds[idx]['text'].strip()

            # No prompt for WikiText initially
            prompt = ""
            full = text

            data_items.append({"prompt": prompt, "full": full})

        return data_items
    except Exception as e:
        print(f"Failed to load WikiText: {e}")
        return []

def filter_data_by_length(data_items, tokenizer, min_answer_len):
    filtered_items = []
    print(f"Filtering data (keeping answer len >= {min_answer_len})...")
    
    for item in tqdm(data_items):
        full_ids = tokenizer.encode(item['full'], add_special_tokens=True)
        prompt_ids = tokenizer.encode(item['prompt'], add_special_tokens=True)
        
        # Calculate answer length
        answer_len = max(0, len(full_ids) - len(prompt_ids))
        
        if answer_len >= min_answer_len:
            filtered_items.append(item)
            
    print(f"Kept {len(filtered_items)} / {len(data_items)} samples.")
    return filtered_items

# -----------------------------------------------------------------------------
# 3. ENTROPY COMPUTATION
# -----------------------------------------------------------------------------

def calculate_entropy(logits):
    """
    Calculates Shannon entropy from logits.
    H(x) = - sum(p(x) * log(p(x)))
    """
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)
    # Compute log probabilities (log_softmax is more numerically stable)
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # Entropy = - sum(p * log p)
    # We use sum(dim=-1) to sum over the vocabulary dimension
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

def compute_sequential_entropy(model, tokenizer, data_items, dataset_name, fixed_start_idx=None):
    """
    Computes the mean entropy for each step k (1 to MAX_ANALYSIS_LEN).
    If fixed_start_idx is provided, uses that as the starting logit index for analysis.
    """
    # Initialize an array to store sum of entropies for each step
    # We will average later.
    entropy_sums = np.zeros(MAX_ANALYSIS_LEN)
    counts = np.zeros(MAX_ANALYSIS_LEN)
    
    print(f"Computing sequential entropy for {dataset_name}...")
    
    for item in tqdm(data_items):
        full_text = item['full']
        prompt_text = item['prompt']
        
        # Tokenize
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(model.device)
        input_ids = inputs.input_ids
        
        # Determine where the analysis starts
        if fixed_start_idx is not None:
            # Use fixed start index if provided (e.g., for WikiText aligned to GSM8K prompt len)
            start_idx = fixed_start_idx
        elif dataset_name == "WikiText":
            # Fallback for WikiText just in case
            start_idx = 10 
        else:
            # For GSM8K, start exactly after the prompt
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
            start_idx = len(prompt_ids) - 1 # -1 because logit at idx predicts token at idx+1
        
        # Ensure we don't go out of bounds
        seq_len = input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits # Shape: (1, seq_len, vocab_size)
            
        # Calculate entropy for the whole sequence on GPU
        # Squeeze batch dim -> (seq_len, vocab_size)
        seq_entropy = calculate_entropy(logits.squeeze(0)) 
        
        # Extract the relevant window (answer tokens)
        # We want to predict tokens: start_idx+1, start_idx+2...
        # The logits responsible for this are at: start_idx, start_idx+1...
        
        for k in range(MAX_ANALYSIS_LEN):
            logit_idx = start_idx + k
            
            # Check bounds
            if logit_idx < seq_len - 1:
                val = seq_entropy[logit_idx].item()
                entropy_sums[k] += val
                counts[k] += 1
        
        del inputs, outputs, logits, seq_entropy
        torch.cuda.empty_cache()

    # Calculate means, avoiding division by zero
    mean_entropies = []
    steps = []
    
    for k in range(MAX_ANALYSIS_LEN):
        if counts[k] > 0:
            mean_entropies.append(entropy_sums[k] / counts[k])
            steps.append(k + 1)
            
    return steps, mean_entropies

def compute_direct_prediction_entropy(model, tokenizer, gsm8k_items):
    """
    Computes the entropy of the FIRST token generated when asking for a direct answer.
    Prompt format: Question + " Answer:"
    """
    print("Computing Direct Prediction Entropy...")
    
    total_entropy = 0
    count = 0
    
    # We can process in small batches or one by one. One by one is safer for memory.
    for item in tqdm(gsm8k_items):
        # Construct Direct Prompt
        direct_prompt = item['prompt'] + " Only output the answer."
        
        inputs = tokenizer(direct_prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # We want the entropy of the NEXT token prediction.
            # This is the logit of the very last input token.
            last_token_logits = outputs.logits[0, -1, :] 
            
            entropy = calculate_entropy(last_token_logits).item()
            total_entropy += entropy
            count += 1
            
        del inputs, outputs
        torch.cuda.empty_cache()
        
    return total_entropy / count if count > 0 else 0

# -----------------------------------------------------------------------------
# 4. PLOTTING
# -----------------------------------------------------------------------------

def plot_entropy_results(results):
    print("Generating Entropy Plot...")
    
    # Update global font sizes for consistent look
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10 
    })

    # Use the same figure size as the previous script (5x3)
    fig, ax = plt.subplots(figsize=(4,2.5))
    
    # Define specific styles to match reference
    styles = {
        'GSM8K': {'color': '#003366'},
        'WikiText': {'color': '#5dade2'}
    }
    
    # Plot GSM8K
    if "GSM8K" in results and isinstance(results["GSM8K"], dict):
        style = styles['GSM8K']
        ax.plot(
            results["GSM8K"]["steps"], 
            results["GSM8K"]["entropy"], 
            label="GSM8K", 
            color=style['color'], 
            markersize=4, # Slightly smaller markers for dense entropy plots
            linestyle='-', 
            linewidth=2.0, 
            alpha=1.0
        )
        
    # Plot WikiText
    if "WikiText" in results and isinstance(results["WikiText"], dict):
        style = styles['WikiText']
        ax.plot(
            results["WikiText"]["steps"], 
            results["WikiText"]["entropy"], 
            label="WikiText", 
            color=style['color'], 
            markersize=4,
            linestyle='-', 
            linewidth=2.0, 
            alpha=1.0
        )
        
    # Plot Direct Baseline
    if "Direct_Baseline" in results:
        direct_val = results["Direct_Baseline"]
        ax.axhline(
            y=direct_val, 
            color='#003366', # Match GSM8K color
            linestyle='--', 
            linewidth=1.5, 
            label=f"GSM8k Direct"
        )

    ax.set_xlabel("Text Length (Tokens)")
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel("Entropy (nats)")
    
    ax.legend(loc='center right', frameon=False, handlelength=1.5, bbox_to_anchor=(1.0, .3))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = "entropy_over_time.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    tokenizer = load_tokenizer()
    
    filtered_datasets = {}
    
    # --- 1. GSM8K Processing First ---
    # We must process GSM8K first to get the average prompt length stats
    print("\n--- Processing GSM8K ---")
    gsm8k_items = get_gsm8k_data(None) # Load all
    gsm8k_filtered = filter_data_by_length(gsm8k_items, tokenizer, min_answer_len=MAX_ANALYSIS_LEN)
    filtered_datasets["GSM8K"] = gsm8k_filtered
    
    # Compute Average Prompt Length from GSM8K
    gsm8k_avg_prompt_len = 0
    if len(gsm8k_filtered) > 0:
        print("Computing average GSM8K prompt length...")
        p_lengths = []
        for item in gsm8k_filtered:
            p_ids = tokenizer.encode(item['prompt'], add_special_tokens=True)
            p_lengths.append(len(p_ids))
        gsm8k_avg_prompt_len = np.mean(p_lengths)
        print(f"Average GSM8K Prompt Length: {gsm8k_avg_prompt_len:.4f}")
    
    # --- 2. WikiText Processing ---
    print("\n--- Processing WikiText ---")
    # Threshold: length must be > 100 + avg_prompt_len
    wikitext_min_token_len = 100 + int(gsm8k_avg_prompt_len)
    
    wikitext_items = get_wikitext_data(NUM_WIKI_SAMPLES, min_char_len=wikitext_min_token_len*3)
    # Filter strictly by token length (for WikiText prompt is "", answer is full text)
    wikitext_filtered = filter_data_by_length(wikitext_items, tokenizer, min_answer_len=wikitext_min_token_len)
    filtered_datasets["WikiText"] = wikitext_filtered

    results = {}
    
    if os.path.exists(RESULTS_FILE):
        print(f"Loading existing results from {RESULTS_FILE}")
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)
            
    # Save Metadata
    results["gsm8k_avg_prompt_len"] = float(gsm8k_avg_prompt_len)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    model = load_model()
    
    # 3. Run Analysis
    
    # GSM8K Sequential
    if "GSM8K" not in results and "GSM8K" in filtered_datasets:
        steps, entropies = compute_sequential_entropy(model, tokenizer, filtered_datasets["GSM8K"], "GSM8K")
        results["GSM8K"] = {"steps": steps, "entropy": entropies}
        with open(RESULTS_FILE, 'w') as f: json.dump(results, f, indent=2)

    # WikiText Sequential
    if "WikiText" not in results and "WikiText" in filtered_datasets:
        # Override start index to match GSM8K avg prompt length
        # We start analyzing FROM the avg_prompt_length token.
        # This means we need the logit at index (avg_prompt_length - 1).
        # We assume the first `gsm8k_avg_prompt_len` tokens act as the "prompt".
        start_idx = int(gsm8k_avg_prompt_len) - 1
        print(f"WikiText: Starting analysis from logit index {start_idx} (simulating prompt len {int(gsm8k_avg_prompt_len)})")
        
        steps, entropies = compute_sequential_entropy(
            model, tokenizer, filtered_datasets["WikiText"], "WikiText", fixed_start_idx=start_idx
        )
        results["WikiText"] = {"steps": steps, "entropy": entropies}
        with open(RESULTS_FILE, 'w') as f: json.dump(results, f, indent=2)
        
    # Direct Prediction Baseline
    if "Direct_Baseline" not in results and "GSM8K" in filtered_datasets:
        # Use the same filtered GSM8K set for fairness
        direct_entropy = compute_direct_prediction_entropy(model, tokenizer, filtered_datasets["GSM8K"])
        results["Direct_Baseline"] = direct_entropy
        print(f"Direct Prediction Entropy: {direct_entropy}")
        with open(RESULTS_FILE, 'w') as f: json.dump(results, f, indent=2)
        
    # 4. Plot
    plot_entropy_results(results)

if __name__ == "__main__":
    main()
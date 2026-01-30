import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.decomposition import PCA
from tqdm import tqdm
import gc
import json
import os

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-32B"

NUM_WIKI_SAMPLES = 5000         
MAX_SEQ_LEN = 512               
PCA_VARIANCE_THRESHOLD = 0.80   
STEP_SIZE = 1                   
MAX_ANALYSIS_LEN = 100          
RESULTS_FILE = "pca_results.json"

# device_map="auto" will handle device placement, but we keep this for input tensors
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running with configuration for: {DEVICE}")

# -----------------------------------------------------------------------------
# 1. & 2. LOAD MODEL AND DATA
# -----------------------------------------------------------------------------

def load_tokenizer():
    print(f"Loading tokenizer: {MODEL_NAME}...")
    return AutoTokenizer.from_pretrained(MODEL_NAME)

def load_model():
    print(f"Loading model: {MODEL_NAME}...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16,  # Load in native half-precision
        device_map="auto",          # Splits model across all 6 GPUs
        output_hidden_states=True,  
        trust_remote_code=True      
    )
    
    model.eval()
    return model

def get_gsm8k_data(n=None):
    print("Loading GSM8K...")
    ds = load_dataset("gsm8k", "main", split="train")
    
    data_items = []
    
    # Use full dataset if n is None
    if n is None:
        print(f"Using full GSM8K dataset ({len(ds)} samples)")
        selection = ds
    else:
        print(f"Sampling {n} from GSM8K")
        selection = ds.select(range(min(n, len(ds))))
        
    for ex in selection:
        prompt = ex['question']
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
# 3. COMPUTE EMBEDDINGS
# -----------------------------------------------------------------------------

def compute_embeddings(model, tokenizer, data_items):
    last_layer_embeddings_list = []
    prompt_lengths_list = []
    
    print("Computing embeddings...")
    for item in tqdm(data_items):
        full_text = item['full']
        prompt_text = item['prompt']
        
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
        prompt_len = len(prompt_ids)
        
        # Ensure input is on the same device as the model (model.device handles device_map="auto")
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # We only need the last layer for the requested analysis
        # Cast to float32 before numpy conversion to ensure stability if compute was fp16
        last_hidden = outputs.hidden_states[-1].squeeze(0).float().cpu().numpy()
        
        last_layer_embeddings_list.append(last_hidden)
        prompt_lengths_list.append(prompt_len)
        
        del inputs, outputs
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return last_layer_embeddings_list, prompt_lengths_list

def compute_last_token_embeddings(model, tokenizer, texts):
    embeddings = []
    print("Computing embeddings for Direct Answer baseline...")
    for text in tqdm(texts):
        # Use model.device for inputs
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Last layer, last token
        last_hidden = outputs.hidden_states[-1][:, -1, :].squeeze(0).float().cpu().numpy()
        embeddings.append(last_hidden)
        del inputs, outputs
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return embeddings

# -----------------------------------------------------------------------------
# 4. SINGLE TOKEN PCA ANALYSIS (STATE)
# -----------------------------------------------------------------------------

def run_last_token_pca_analysis(embeddings_list, prompt_lengths):
    if not embeddings_list:
        return [], []
        
    results_k = []
    results_components = []
    
    print("Running PCA Analysis on Single Last Token (State)...")
    
    range_vals = range(STEP_SIZE, MAX_ANALYSIS_LEN + 1, STEP_SIZE)
    
    for k in tqdm(range_vals):
        stacked_vectors = []
        
        for i, emb in enumerate(embeddings_list):
            p_len = prompt_lengths[i]
            target_idx = p_len + k - 1
            
            if emb.shape[0] > target_idx:
                token_vector = emb[target_idx, :] 
                stacked_vectors.append(token_vector)
        
        if len(stacked_vectors) < 2:
            continue
            
        X = np.array(stacked_vectors)
        
        n_components = min(X.shape[0], X.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(X)
        
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        idx = np.searchsorted(cumulative_variance, PCA_VARIANCE_THRESHOLD)
        num_components_needed = int(idx + 1)
        
        results_k.append(k)
        results_components.append(num_components_needed)
        
    return results_k, results_components

def run_static_pca_analysis(embeddings):
    if len(embeddings) < 2:
        return 0
    X = np.array(embeddings)
    n_components = min(X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(X)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    idx = np.searchsorted(cumulative_variance, PCA_VARIANCE_THRESHOLD)
    return int(idx + 1)

# -----------------------------------------------------------------------------
# 5. PLOTTING HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def plot_single_token_results(all_results):
    print("Generating Single Token Plot...")
    
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10
    })

    fig, ax = plt.subplots(figsize=(4, 2.5)) # Increased size slightly for visibility
    lines = []
    
    styles = {
        'GSM8K': {'color': '#003366', 'marker': 'o'},
        'WikiText': {'color': '#5dade2', 'marker': 'x'}
    }

    for ds_name in all_results.keys():
        # Skip metadata fields like the average prompt length
        if not isinstance(all_results[ds_name], dict):
            continue

        style = styles.get(ds_name, {'color': np.random.rand(3,), 'marker': '.'})
        
        if ds_name == "GSM8K_Direct": continue 

        if "last_token_last_layer" not in all_results[ds_name]:
            continue
            
        pca_data = all_results[ds_name]
        
        ks = pca_data["last_token_last_layer"]["k"]
        comps = pca_data["last_token_last_layer"]["components"]
        
        l1, = ax.plot(
            ks, comps, 
            markersize=6, 
            linestyle='-', 
            linewidth=2.0, 
            color=style['color'], 
            alpha=1.0,
            label=f"{ds_name}"
        )
        lines.append(l1)

    if "GSM8K_Direct" in all_results:
        res = all_results["GSM8K_Direct"]
        val = res["dimension"]
        
        l_direct = ax.axhline(
            y=val, 
            color='#003366', 
            linestyle='--', 
            linewidth=1.5, 
            label=f"GSM8k Direct"
        )
        lines.append(l_direct)

    if not lines:
        plt.close()
        return

    ax.set_xlabel("Text Length (Tokens)") 
    ax.set_ylabel("Dimension")
    ax.grid(True, alpha=0.3)
    #ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig("pca_single_token_state_small.pdf", dpi=300, bbox_inches='tight') 
    print("Saved pca_single_token_state_small.pdf")
    plt.close()

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------

def main():
    tokenizer = load_tokenizer()

    filtered_datasets_map = {}
    dataset_counts = {} 
    
    # --- 1. Process GSM8K First to get statistics ---
    print("\n--- Processing GSM8K ---")
    gsm8k_items = get_gsm8k_data(None)
    gsm8k_filtered = filter_data_by_length(gsm8k_items, tokenizer, min_answer_len=MAX_ANALYSIS_LEN)
    
    if len(gsm8k_filtered) > 10:
        filtered_datasets_map["GSM8K"] = gsm8k_filtered
        dataset_counts["GSM8K"] = len(gsm8k_filtered)
    else:
        print("Warning: GSM8K filtered count too low.")

    # Compute Average Prompt Length from GSM8K
    gsm8k_avg_prompt_len = 0
    if "GSM8K" in filtered_datasets_map:
        print("Computing average GSM8K prompt length...")
        p_lengths = []
        for item in gsm8k_filtered:
            # Re-encode to get prompt length
            p_ids = tokenizer.encode(item['prompt'], add_special_tokens=True)
            p_lengths.append(len(p_ids))
        
        if p_lengths:
            gsm8k_avg_prompt_len = np.mean(p_lengths)
            print(f"Average GSM8K Prompt Length: {gsm8k_avg_prompt_len:.4f}")

    # --- 2. Process WikiText using GSM8K stats ---
    print("\n--- Processing WikiText ---")
    # Wikitext threshold: length must be > 100 + avg_prompt_len
    wikitext_min_token_len = 100 + int(gsm8k_avg_prompt_len)
    
    # We estimate char length (conservative 3 chars per token) for initial efficient filtering
    wikitext_items = get_wikitext_data(NUM_WIKI_SAMPLES, min_char_len=wikitext_min_token_len * 3)
    
    # Filter strictly by token length. For Wikitext, full text is the "answer", prompt is empty.
    # So we pass min_answer_len = wikitext_min_token_len
    wikitext_filtered = filter_data_by_length(wikitext_items, tokenizer, min_answer_len=wikitext_min_token_len)
    
    if len(wikitext_filtered) > 10:
        filtered_datasets_map["WikiText"] = wikitext_filtered
        dataset_counts["WikiText"] = len(wikitext_filtered)
    else:
        print(f"Skipping WikiText - not enough samples > {wikitext_min_token_len} tokens.")

    # --- Load Existing Results ---
    if os.path.exists(RESULTS_FILE):
        print(f"Found existing PCA results file: {RESULTS_FILE}. Loading...")
        with open(RESULTS_FILE, 'r') as f:
            all_results = json.load(f)
    else:
        print("No existing PCA results found. Starting fresh.")
        all_results = {}
    
    # Save the average prompt length metadata
    all_results["gsm8k_avg_prompt_len"] = float(gsm8k_avg_prompt_len)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    recompute_needed = []
    for d in filtered_datasets_map.keys():
        if d not in all_results:
            recompute_needed.append(d)
        elif "last_token_last_layer" not in all_results[d]:
            recompute_needed.append(d)
    
    gsm8k_direct_needed = "GSM8K" in filtered_datasets_map and "GSM8K_Direct" not in all_results

    if recompute_needed or gsm8k_direct_needed:
        # Load model only if needed and do it ONCE
        model = load_model()
        
        # 1. Process Standard Datasets
        for ds_name in recompute_needed:
            print(f"\nProcessing PCA for {ds_name}...")
            data_items = filtered_datasets_map[ds_name]
            
            last_embs, p_lens = compute_embeddings(model, tokenizer, data_items)
            
            # --- SPECIAL HANDLING FOR WIKITEXT ---
            # Start analysis from avg_prompt_length
            if ds_name == "WikiText":
                print(f"WikiText: Overriding prompt lengths to {int(gsm8k_avg_prompt_len)} (Avg GSM8k Prompt Length)")
                p_lens = [int(gsm8k_avg_prompt_len)] * len(last_embs)
            # -------------------------------------
            
            print(f"Analyzing {ds_name} - Last Layer (Single Token State)...")
            ks_last_token, comps_last_token = run_last_token_pca_analysis(last_embs, p_lens)

            if ds_name not in all_results:
                all_results[ds_name] = {}
            
            all_results[ds_name]["last_token_last_layer"] = {"k": ks_last_token, "components": comps_last_token}
            all_results[ds_name]["n_samples"] = dataset_counts[ds_name]
            
            # Clean up keys from previous versions if they exist
            all_results[ds_name].pop("first_layer", None)
            all_results[ds_name].pop("last_layer", None)
            
            print(f"Saving Single Token results for {ds_name}...")
            with open(RESULTS_FILE, 'w') as f:
                json.dump(all_results, f, indent=2)
                
            plot_single_token_results(all_results)
            
            del last_embs, p_lens
            gc.collect()

        # 2. Process GSM8K Direct Answer Control
        if gsm8k_direct_needed:
            print("\nProcessing GSM8K Direct Answer Experiment...")
            direct_texts = [item['prompt'] + " Only output the answer." for item in filtered_datasets_map["GSM8K"]]
            
            direct_embs = compute_last_token_embeddings(model, tokenizer, direct_texts)
            
            dim = run_static_pca_analysis(direct_embs)
            print(f"GSM8K Direct Dimension: {dim}")
            
            all_results["GSM8K_Direct"] = {
                "dimension": dim,
                "n_samples": len(direct_texts)
            }
            
            print(f"Saving GSM8K Direct results...")
            with open(RESULTS_FILE, 'w') as f:
                json.dump(all_results, f, indent=2)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    plot_single_token_results(all_results)

if __name__ == "__main__":
    main()
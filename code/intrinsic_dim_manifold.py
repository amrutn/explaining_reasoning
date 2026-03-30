import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA
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
VARIANCE_THRESHOLD = 0.80       # For eigenvalue-based methods (Isomap, KernelPCA)
MLE_K = 20                      # k neighbors for MLE estimator
ISOMAP_N_NEIGHBORS = 10
ISOMAP_MAX_COMPONENTS = 50
KPCA_MAX_COMPONENTS = 50
MAX_SAMPLES_MANIFOLD = 500      # Subsample for O(n^2) methods (Isomap, KernelPCA)
STEP_SIZE = 1
MAX_ANALYSIS_LEN = 100
RESULTS_FILE = "manifold_dim_results.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running with configuration for: {DEVICE}")

# -----------------------------------------------------------------------------
# DIMENSIONALITY ESTIMATION METHODS
# -----------------------------------------------------------------------------

def estimate_dim_mle(X, k=MLE_K):
    """MLE intrinsic dimension estimator (Levina & Bickel, 2004).

    Uses k-nearest neighbor distances to estimate the intrinsic
    dimensionality at each point, then returns the median.
    """
    k = min(k, X.shape[0] - 2)
    if k < 2:
        return 0

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
    distances, _ = nn.kneighbors(X)
    distances = distances[:, 1:]  # exclude self-distance

    dims = []
    for i in range(len(X)):
        d = distances[i]
        d = d[d > 0]
        if len(d) < 2:
            continue
        T_k = d[-1]
        log_ratios = np.log(T_k / d[:-1])
        total = np.sum(log_ratios)
        if total > 0:
            dims.append(len(log_ratios) / total)

    return float(np.median(dims)) if dims else 0.0


def estimate_dim_twonn(X):
    """Two-NN intrinsic dimension estimator (Facco et al., 2017).

    Estimates intrinsic dimension from the ratio of distances
    to 2nd and 1st nearest neighbors.
    """
    if X.shape[0] < 4:
        return 0.0

    nn = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(X)
    distances, _ = nn.kneighbors(X)

    r1 = distances[:, 1]
    r2 = distances[:, 2]

    mask = r1 > 0
    r1, r2 = r1[mask], r2[mask]
    if len(r1) < 2:
        return 0.0

    mu = r2 / r1
    mu_sorted = np.sort(mu)
    n = len(mu_sorted)

    # Empirical CDF vs theoretical: F(mu) = 1 - mu^(-d)
    # => log(1 - F) = -d * log(mu)
    F = np.arange(1, n + 1) / (n + 1)  # avoid F=1
    log_1_minus_F = np.log(1 - F)
    log_mu = np.log(mu_sorted)

    valid = log_mu > 0
    if not np.any(valid):
        return 0.0

    d = -np.sum(log_1_minus_F[valid]) / np.sum(log_mu[valid])
    return float(d)


def _subsample(X, max_n):
    """Randomly subsample rows if X has more than max_n samples."""
    if X.shape[0] <= max_n:
        return X
    indices = np.random.choice(X.shape[0], max_n, replace=False)
    return X[indices]


def estimate_dim_isomap(X, n_neighbors=ISOMAP_N_NEIGHBORS,
                        max_components=ISOMAP_MAX_COMPONENTS,
                        threshold=VARIANCE_THRESHOLD):
    """Isomap-based dimension estimate using geodesic kernel eigenvalues.

    Fits Isomap once and examines the eigenvalue spectrum of the
    internal kernel PCA to find how many components explain the
    threshold fraction of total variance.
    """
    X = _subsample(X, MAX_SAMPLES_MANIFOLD)

    max_comp = min(max_components, X.shape[0] - 2)
    if max_comp < 1:
        return 0

    try:
        iso = Isomap(n_components=max_comp, n_neighbors=min(n_neighbors, X.shape[0] - 1))
        iso.fit(X)

        eigenvalues = iso.kernel_pca_.eigenvalues_
        eigenvalues = eigenvalues[eigenvalues > 0]
        if len(eigenvalues) == 0:
            return 0

        explained_ratio = eigenvalues / eigenvalues.sum()
        cumulative = np.cumsum(explained_ratio)
        idx = np.searchsorted(cumulative, threshold)
        return int(idx + 1)
    except Exception as e:
        print(f"    Isomap failed: {e}")
        return 0


def estimate_dim_kpca(X, max_components=KPCA_MAX_COMPONENTS,
                      threshold=VARIANCE_THRESHOLD):
    """Kernel PCA (RBF) dimension estimate from eigenvalue spectrum.

    Fits Kernel PCA with an RBF kernel and finds how many components
    explain the threshold fraction of total eigenvalue mass.
    """
    X = _subsample(X, MAX_SAMPLES_MANIFOLD)

    max_comp = min(max_components, X.shape[0] - 1)
    if max_comp < 1:
        return 0

    try:
        kpca = KernelPCA(n_components=max_comp, kernel="rbf", gamma="scale")
        kpca.fit(X)

        eigenvalues = kpca.eigenvalues_
        eigenvalues = eigenvalues[eigenvalues > 0]
        if len(eigenvalues) == 0:
            return 0

        explained_ratio = eigenvalues / eigenvalues.sum()
        cumulative = np.cumsum(explained_ratio)
        idx = np.searchsorted(cumulative, threshold)
        return int(idx + 1)
    except Exception as e:
        print(f"    KernelPCA failed: {e}")
        return 0


# All methods to run
METHODS = {
    "MLE": estimate_dim_mle,
    "TwoNN": estimate_dim_twonn,
    "Isomap": estimate_dim_isomap,
    "KernelPCA": estimate_dim_kpca,
}

# -----------------------------------------------------------------------------
# LOAD MODEL AND DATA (same as intrinsic_dim.py)
# -----------------------------------------------------------------------------

def load_tokenizer():
    print(f"Loading tokenizer: {MODEL_NAME}...")
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def load_model():
    print(f"Loading model: {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
        trust_remote_code=True,
    )
    model.eval()
    return model


def get_gsm8k_data(n=None):
    print("Loading GSM8K...")
    ds = load_dataset("gsm8k", "main", split="train")
    data_items = []

    if n is None:
        print(f"Using full GSM8K dataset ({len(ds)} samples)")
        selection = ds
    else:
        print(f"Sampling {n} from GSM8K")
        selection = ds.select(range(min(n, len(ds))))

    for ex in selection:
        prompt = ex["question"]
        full = f"{ex['question']}\n{ex['answer']}"
        data_items.append({"prompt": prompt, "full": full})

    return data_items


def get_wikitext_data(n=100, min_char_len=200):
    print("Loading WikiText-2 (Normal Text)...")
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        data_items = []
        valid_indices = [
            i for i, x in enumerate(ds["text"])
            if len(x.strip()) > min_char_len and not x.strip().startswith("=")
        ]
        count = min(n, len(valid_indices))
        selected_indices = valid_indices[:count]

        for idx in selected_indices:
            text = ds[idx]["text"].strip()
            data_items.append({"prompt": "", "full": text})

        return data_items
    except Exception as e:
        print(f"Failed to load WikiText: {e}")
        return []


def filter_data_by_length(data_items, tokenizer, min_answer_len):
    filtered_items = []
    print(f"Filtering data (keeping answer len >= {min_answer_len})...")
    for item in tqdm(data_items):
        full_ids = tokenizer.encode(item["full"], add_special_tokens=True)
        prompt_ids = tokenizer.encode(item["prompt"], add_special_tokens=True)
        answer_len = max(0, len(full_ids) - len(prompt_ids))
        if answer_len >= min_answer_len:
            filtered_items.append(item)
    print(f"Kept {len(filtered_items)} / {len(data_items)} samples.")
    return filtered_items


# -----------------------------------------------------------------------------
# COMPUTE EMBEDDINGS (same as intrinsic_dim.py)
# -----------------------------------------------------------------------------

def compute_embeddings(model, tokenizer, data_items):
    last_layer_embeddings_list = []
    prompt_lengths_list = []

    print("Computing embeddings...")
    for item in tqdm(data_items):
        full_text = item["full"]
        prompt_text = item["prompt"]

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
        prompt_len = len(prompt_ids)

        inputs = tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

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
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden = outputs.hidden_states[-1][:, -1, :].squeeze(0).float().cpu().numpy()
        embeddings.append(last_hidden)
        del inputs, outputs
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return embeddings


# -----------------------------------------------------------------------------
# ANALYSIS: APPLY EACH METHOD AT EACH TOKEN POSITION
# -----------------------------------------------------------------------------

def run_manifold_analysis(embeddings_list, prompt_lengths, method_name, method_fn):
    """Run a single dimensionality estimation method across token positions."""
    if not embeddings_list:
        return [], []

    results_k = []
    results_dim = []

    print(f"  Running {method_name} analysis...")
    range_vals = range(STEP_SIZE, MAX_ANALYSIS_LEN + 1, STEP_SIZE)

    for k in tqdm(range_vals, desc=f"    {method_name}"):
        stacked_vectors = []

        for i, emb in enumerate(embeddings_list):
            p_len = prompt_lengths[i]
            target_idx = p_len + k - 1

            if emb.shape[0] > target_idx:
                stacked_vectors.append(emb[target_idx, :])

        if len(stacked_vectors) < 10:
            continue

        X = np.array(stacked_vectors)
        dim = method_fn(X)

        results_k.append(k)
        results_dim.append(float(dim))

    return results_k, results_dim


def run_static_analysis(embeddings, method_name, method_fn):
    """Run a single method on a static set of embeddings (e.g., Direct Answer)."""
    if len(embeddings) < 10:
        return 0.0
    X = np.array(embeddings)
    return float(method_fn(X))


# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------

def plot_results(all_results):
    """Generate one plot per method, same format as intrinsic_dim.py."""
    print("Generating plots...")

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
    })

    styles = {
        "GSM8K": {"color": "#003366", "marker": "o"},
        "WikiText": {"color": "#5dade2", "marker": "x"},
    }

    for method_name in METHODS.keys():
        fig, ax = plt.subplots(figsize=(4, 2.5))
        lines = []
        has_data = False

        for ds_name in ["GSM8K", "WikiText"]:
            if ds_name not in all_results:
                continue
            ds_data = all_results[ds_name]
            if not isinstance(ds_data, dict):
                continue

            key = f"{method_name}_last_token"
            if key not in ds_data:
                continue

            ks = ds_data[key]["k"]
            dims = ds_data[key]["dimension"]

            if not ks:
                continue

            has_data = True
            style = styles.get(ds_name, {"color": "gray", "marker": "."})

            l1, = ax.plot(
                ks, dims,
                markersize=6,
                linestyle="-",
                linewidth=2.0,
                color=style["color"],
                alpha=1.0,
                label=ds_name,
            )
            lines.append(l1)

        # GSM8K Direct baseline
        direct_key = f"GSM8K_Direct_{method_name}"
        if direct_key in all_results:
            val = all_results[direct_key]["dimension"]
            l_direct = ax.axhline(
                y=val,
                color="#003366",
                linestyle="--",
                linewidth=1.5,
                label="GSM8k Direct",
            )
            lines.append(l_direct)

        if not has_data:
            plt.close(fig)
            continue

        ax.set_xlabel("Text Length (Tokens)")
        ax.set_ylabel("Dimension")
        ax.set_title(method_name)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f"manifold_{method_name.lower()}_single_token_state_small.pdf"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        print(f"Saved {fname}")
        plt.close()

    # --- Combined overview plot ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = axes.flatten()

    for idx, method_name in enumerate(METHODS.keys()):
        ax = axes[idx]

        for ds_name in ["GSM8K", "WikiText"]:
            if ds_name not in all_results:
                continue
            ds_data = all_results[ds_name]
            if not isinstance(ds_data, dict):
                continue

            key = f"{method_name}_last_token"
            if key not in ds_data:
                continue

            ks = ds_data[key]["k"]
            dims = ds_data[key]["dimension"]
            if not ks:
                continue

            style = styles.get(ds_name, {"color": "gray", "marker": "."})
            ax.plot(
                ks, dims,
                linestyle="-", linewidth=2.0,
                color=style["color"], alpha=1.0,
                label=ds_name,
            )

        direct_key = f"GSM8K_Direct_{method_name}"
        if direct_key in all_results:
            val = all_results[direct_key]["dimension"]
            ax.axhline(y=val, color="#003366", linestyle="--", linewidth=1.5,
                       label="GSM8k Direct")

        ax.set_title(method_name)
        ax.set_ylabel("Dimension")
        ax.grid(True, alpha=0.3)
        if idx >= 2:
            ax.set_xlabel("Text Length (Tokens)")
        if idx == 0:
            ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig("manifold_all_methods_overview.pdf", dpi=300, bbox_inches="tight")
    print("Saved manifold_all_methods_overview.pdf")
    plt.close()


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    tokenizer = load_tokenizer()

    filtered_datasets_map = {}
    dataset_counts = {}

    # --- 1. Process GSM8K ---
    print("\n--- Processing GSM8K ---")
    gsm8k_items = get_gsm8k_data(None)
    gsm8k_filtered = filter_data_by_length(gsm8k_items, tokenizer, min_answer_len=MAX_ANALYSIS_LEN)

    if len(gsm8k_filtered) > 10:
        filtered_datasets_map["GSM8K"] = gsm8k_filtered
        dataset_counts["GSM8K"] = len(gsm8k_filtered)
    else:
        print("Warning: GSM8K filtered count too low.")

    # Average prompt length
    gsm8k_avg_prompt_len = 0
    if "GSM8K" in filtered_datasets_map:
        print("Computing average GSM8K prompt length...")
        p_lengths = [
            len(tokenizer.encode(item["prompt"], add_special_tokens=True))
            for item in gsm8k_filtered
        ]
        if p_lengths:
            gsm8k_avg_prompt_len = np.mean(p_lengths)
            print(f"Average GSM8K Prompt Length: {gsm8k_avg_prompt_len:.4f}")

    # --- 2. Process WikiText ---
    print("\n--- Processing WikiText ---")
    wikitext_min_token_len = 100 + int(gsm8k_avg_prompt_len)
    wikitext_items = get_wikitext_data(NUM_WIKI_SAMPLES, min_char_len=wikitext_min_token_len * 3)
    wikitext_filtered = filter_data_by_length(wikitext_items, tokenizer, min_answer_len=wikitext_min_token_len)

    if len(wikitext_filtered) > 10:
        filtered_datasets_map["WikiText"] = wikitext_filtered
        dataset_counts["WikiText"] = len(wikitext_filtered)
    else:
        print(f"Skipping WikiText - not enough samples > {wikitext_min_token_len} tokens.")

    # --- Load existing results ---
    if os.path.exists(RESULTS_FILE):
        print(f"Found existing results file: {RESULTS_FILE}. Loading...")
        with open(RESULTS_FILE, "r") as f:
            all_results = json.load(f)
    else:
        print("No existing results found. Starting fresh.")
        all_results = {}

    all_results["gsm8k_avg_prompt_len"] = float(gsm8k_avg_prompt_len)
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    # --- Determine what needs computing ---
    recompute_needed = []
    for ds_name in filtered_datasets_map.keys():
        if ds_name not in all_results:
            recompute_needed.append(ds_name)
        else:
            # Check if all methods have results
            for method_name in METHODS.keys():
                key = f"{method_name}_last_token"
                if key not in all_results[ds_name]:
                    if ds_name not in recompute_needed:
                        recompute_needed.append(ds_name)
                    break

    gsm8k_direct_needed = "GSM8K" in filtered_datasets_map and not all(
        f"GSM8K_Direct_{m}" in all_results for m in METHODS.keys()
    )

    if recompute_needed or gsm8k_direct_needed:
        model = load_model()

        # 1. Process standard datasets
        for ds_name in recompute_needed:
            print(f"\nProcessing manifold analysis for {ds_name}...")
            data_items = filtered_datasets_map[ds_name]

            last_embs, p_lens = compute_embeddings(model, tokenizer, data_items)

            # WikiText: override prompt lengths to match GSM8K avg
            if ds_name == "WikiText":
                print(f"WikiText: Overriding prompt lengths to {int(gsm8k_avg_prompt_len)}")
                p_lens = [int(gsm8k_avg_prompt_len)] * len(last_embs)

            if ds_name not in all_results:
                all_results[ds_name] = {}

            for method_name, method_fn in METHODS.items():
                key = f"{method_name}_last_token"
                if key in all_results[ds_name]:
                    print(f"  Skipping {method_name} (already computed)")
                    continue

                ks, dims = run_manifold_analysis(last_embs, p_lens, method_name, method_fn)
                all_results[ds_name][key] = {"k": ks, "dimension": dims}

                # Save incrementally
                with open(RESULTS_FILE, "w") as f:
                    json.dump(all_results, f, indent=2)
                plot_results(all_results)

            all_results[ds_name]["n_samples"] = dataset_counts[ds_name]

            with open(RESULTS_FILE, "w") as f:
                json.dump(all_results, f, indent=2)

            del last_embs, p_lens
            gc.collect()

        # 2. GSM8K Direct Answer baseline
        if gsm8k_direct_needed:
            print("\nProcessing GSM8K Direct Answer Experiment...")
            direct_texts = [
                item["prompt"] + " Only output the answer."
                for item in filtered_datasets_map["GSM8K"]
            ]
            direct_embs = compute_last_token_embeddings(model, tokenizer, direct_texts)

            for method_name, method_fn in METHODS.items():
                direct_key = f"GSM8K_Direct_{method_name}"
                if direct_key in all_results:
                    print(f"  Skipping {method_name} Direct (already computed)")
                    continue

                dim = run_static_analysis(direct_embs, method_name, method_fn)
                print(f"  GSM8K Direct Dimension ({method_name}): {dim:.2f}")
                all_results[direct_key] = {
                    "dimension": dim,
                    "n_samples": len(direct_texts),
                }

                with open(RESULTS_FILE, "w") as f:
                    json.dump(all_results, f, indent=2)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    plot_results(all_results)


if __name__ == "__main__":
    main()

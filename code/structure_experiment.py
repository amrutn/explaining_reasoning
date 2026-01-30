import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import numpy as np
import json
import os
import queue
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import copy
import uuid
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, get_linear_schedule_with_warmup
from tokenizers import ByteLevelBPETokenizer

# ==========================================
# 0. Global Configuration
# ==========================================

# Dataset & Training
TRAIN_SIZE_PER_DEPTH = 15000
TEST_SIZE = 5000
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 2.5e-3

# Model Architecture & Dimensions
INPUT_DIM = 10      # Dimension of the input vector (the data feature vector)
EMBED_DIM = 64      # Internal embedding dimension for the Transformer
NUM_LAYERS = 2      # Number of Transformer layers
NUM_HEADS = 4       # Number of Attention heads

# Evaluation
MAX_GEN_LEN = 150

# Experiment Structure
DEPTHS = [4, 5, 6, 7]
REPLICATES = 10
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51] # Pool of seeds to draw from

# Hardware / Multiprocessing
# Set GPU_IDS to a list like [0, 1, 2] to restrict usage, or None to use all available.
GPU_IDS = None 
WORKERS_PER_GPU = 5

# ==========================================
# 1. Task Definition
# ==========================================

class VariableTreeParityTask:
    def __init__(self, branching_factors, seed=42, input_dim=64):
        self.branching_factors = branching_factors
        self.n = len(branching_factors)
        self.seed = seed
        self.input_dim = input_dim
        
        self.rng = np.random.RandomState(self.seed)
        
        self.layer_sizes = [1]
        current_size = 1
        for b in branching_factors:
            current_size *= b
            self.layer_sizes.append(current_size)
        
        self.layer_offsets = [0]
        current_offset = 0
        for size in self.layer_sizes[:-1]:
            current_offset += size
            self.layer_offsets.append(current_offset)
            
        self.total_nodes = sum(self.layer_sizes)
        self.num_leaves = self.layer_sizes[-1]
        
        self.hyperplane = self.rng.randn(self.input_dim)
        self.hyperplane /= np.linalg.norm(self.hyperplane)

    def get_layer(self, node_idx):
        for i, offset in enumerate(self.layer_offsets):
            if i == len(self.layer_offsets) - 1:
                return i
            if node_idx < self.layer_offsets[i+1]:
                return i
        return len(self.layer_offsets) - 1

    def get_parent(self, node_idx):
        if node_idx == 0: return None
        layer = self.get_layer(node_idx)
        layer_start = self.layer_offsets[layer]
        parent_layer_start = self.layer_offsets[layer-1]
        local_idx = node_idx - layer_start
        b = self.branching_factors[layer-1]
        parent_local_idx = local_idx // b
        return parent_layer_start + parent_local_idx

    def get_edge_op(self, node_idx):
        if node_idx == 0: return 0
        h = hash((self.seed, node_idx))
        return h % 2

    def get_path(self, target_node):
        path = []
        curr = target_node
        while curr is not None:
            path.append(curr)
            curr = self.get_parent(curr)
        return path[::-1]

    def generate_sample(self, mode='reasoning', return_data=False):
        input_vec = self.rng.uniform(-1, 1, self.input_dim).astype(np.float32)
        
        leaves_start = self.layer_offsets[-1]
        target_node = self.rng.randint(leaves_start, leaves_start + self.num_leaves)
        path = self.get_path(target_node)
        
        values = {}
        values[0] = 0
        
        if len(path) > 1:
            l1_node = path[1]
            proj = np.dot(self.hyperplane, input_vec)
            
            # Determine value based on projection and edge operation
            base_val = 1 if proj > 0 else 0
            op = self.get_edge_op(l1_node)
            
            if op == 0:
                values[l1_node] = base_val
            else:
                values[l1_node] = 1 - base_val
            
            for i in range(2, len(path)):
                node = path[i]
                parent = path[i-1]
                op = self.get_edge_op(node)
                if op == 0:
                    values[node] = values[parent]
                else:
                    values[node] = 1 - values[parent]

        input_str = f"Target: X{target_node}"
        
        if mode == 'direct':
            label_str = f" Answer: X{target_node}={values[target_node]}"
            full_text = input_str + label_str
        elif mode == 'reasoning':
            steps = []
            for node in path:
                if node == 0: continue
                steps.append(f"X{node}={values[node]}")
            label_str = " Answer: " + " ".join(steps)
            full_text = input_str + label_str
            
        if return_data:
            return input_vec, full_text
        return full_text

# ==========================================
# 2. Model & Data Utilities
# ==========================================

def train_tokenizer(task, size=100000):
    # Use UUID to prevent collisions in multiprocessing
    unique_id = uuid.uuid4().hex
    corpus_path = f"temp_corpus_{unique_id}.txt"
    tokenizer_path = f"tokenizer_{unique_id}.json"

    with open(corpus_path, "w") as f:
        for _ in range(size):
            _, text_r = task.generate_sample(mode='reasoning', return_data=True)
            _, text_d = task.generate_sample(mode='direct', return_data=True)
            f.write(text_r + "\n")
            f.write(text_d + "\n")
            
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=[corpus_path], vocab_size=5000, min_frequency=2, special_tokens=["<|endoftext|>", "<pad>"])
    
    # Save to unique path to avoid race conditions
    tokenizer.save(tokenizer_path)
    
    # Load from unique path
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    fast_tokenizer.pad_token = "<pad>"
    fast_tokenizer.eos_token = "<|endoftext|>"
    
    # Clean up unique temporary files
    if os.path.exists(corpus_path):
        os.remove(corpus_path)
    if os.path.exists(tokenizer_path):
        os.remove(tokenizer_path)
        
    return fast_tokenizer

class TreeDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=256):
        self.samples = samples 
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vec, text = self.samples[idx]
        
        try:
            split_idx = text.index(" Answer:")
            prompt_text = text[:split_idx]
        except ValueError:
            prompt_text = text
            
        # Use HF tokenizer with padding and truncation
        # Return tensors='pt' gives us pytorch tensors directly
        encoding = self.tokenizer(
            text, 
            max_length=self.max_len, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        token_ids = encoding["input_ids"].squeeze(0)
        
        # Calculate prompt length for generation masking
        # We encode just the prompt to get its token length
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)
        
        return torch.tensor(vec, dtype=torch.float32), token_ids, prompt_len

class GPT2Small(nn.Module):
    def __init__(self, tokenizer, input_dim=64, embed_dim=64, n_layer=4, n_head=4):
        super().__init__()
        # Matches input vector dimension
        config = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=1024,
            n_embd=embed_dim,    
            n_layer=n_layer,   
            n_head=n_head,     
            bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            # Explicitly set loss type to suppress warning if using older transformers
            loss_type="cross_entropy" 
        )
        self.model = GPT2LMHeadModel(config)
        
        # Projection layer to map input vector to embedding space
        self.projector = nn.Linear(input_dim, embed_dim)
        
    def forward(self, vec, x, src_key_padding_mask=None):
        # vec: [B, input_dim]
        # x: [B, T] input_ids
        
        # 1. Embed Text
        text_embeds = self.model.transformer.wte(x) # [B, T, D]
        
        # 2. Project Vector
        # Apply learnable linear projection before unsqueezing
        vec_projected = self.projector(vec) # [B, D]
        
        # 3. Reshape projected vector
        vec_embeds = vec_projected.unsqueeze(1) # [B, 1, D]
        
        # 4. Concatenate
        inputs_embeds = torch.cat([vec_embeds, text_embeds], dim=1) # [B, T+1, D]
        
        # 5. Handle Attention Mask
        # src_key_padding_mask comes in as [B, T] with True=Pad, False=Keep
        # GPT2 expects [B, T+1] with 1=Keep, 0=Pad
        
        if src_key_padding_mask is not None:
            B = vec.shape[0]
            # Image is always valid (False for padding mask)
            vec_mask = torch.zeros((B, 1), dtype=torch.bool, device=vec.device) 
            combined_pad_mask = torch.cat([vec_mask, src_key_padding_mask], dim=1)
            
            # Invert for GPT2: ~True(Pad) -> False(0), ~False(Keep) -> True(1)
            attention_mask = (~combined_pad_mask).long()
        else:
            attention_mask = None

        # 6. Forward Pass
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        
        # Return logits: [B, T+1, Vocab]
        return out.logits

def calculate_accuracy(model, loader, device, tokenizer):
    model.eval()
    correct = 0
    total = 0
    max_gen_len = MAX_GEN_LEN 
    
    with torch.no_grad():
        for vecs, tokens, prompt_lens in loader:
            vecs = vecs.to(device)
            batch_size = vecs.size(0)
            
            max_p_len = max(prompt_lens).item()
            curr_seqs = torch.zeros((batch_size, max_p_len), dtype=torch.long, device=device)
            
            active_lens = prompt_lens.clone().to(device)
            
            for i in range(batch_size):
                p_len = prompt_lens[i].item()
                curr_seqs[i, :p_len] = tokens[i, :p_len]
            
            # Generation Loop
            for _ in range(max_gen_len):
                src_key_padding_mask = torch.zeros_like(curr_seqs, dtype=torch.bool)
                for i in range(batch_size):
                    src_key_padding_mask[i, active_lens[i]:] = True
                
                logits = model(vecs, curr_seqs, src_key_padding_mask=src_key_padding_mask)
                
                next_tokens_list = []
                for i in range(batch_size):
                    idx = active_lens[i].item()
                    if idx >= logits.size(1):
                        idx = logits.size(1) - 1
                    
                    last_logit = logits[i, idx, :]
                    next_token = torch.argmax(last_logit).item()
                    next_tokens_list.append(next_token)
                
                next_tokens_tensor = torch.tensor(next_tokens_list, device=device).unsqueeze(1)
                
                new_col = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
                curr_seqs = torch.cat([curr_seqs, new_col], dim=1)
                
                for i in range(batch_size):
                    curr_seqs[i, active_lens[i]] = next_tokens_tensor[i]
                    active_lens[i] += 1
                
            generated_strs = tokenizer.batch_decode(curr_seqs, skip_special_tokens=True)
            ground_truth_strs = tokenizer.batch_decode(tokens, skip_special_tokens=True)
            
            for i in range(batch_size):
                gen_text = generated_strs[i]
                gt_text = ground_truth_strs[i]
                
                try:
                    target_part = gt_text.split(" Answer:")[0]
                    target_node = target_part.split("Target: X")[1].strip()
                    search_term = f"X{target_node}="
                    
                    if search_term in gen_text:
                        idx = gen_text.rfind(search_term)
                        val_idx = idx + len(search_term)
                        if val_idx < len(gen_text):
                            pred_val = gen_text[val_idx]
                            
                            if search_term in gt_text:
                                gt_idx = gt_text.rfind(search_term)
                                gt_val_idx = gt_idx + len(search_term)
                                gt_val = gt_text[gt_val_idx]
                                
                                if pred_val == gt_val:
                                    correct += 1
                except:
                    pass
                total += 1
                
    return correct / total

# ==========================================
# 3. Worker Process
# ==========================================

def train_worker(gpu_id, config_queue, result_queue, worker_id):
    device = torch.device(f'cuda:{gpu_id}')
    
    while True:
        try:
            task_config = config_queue.get(timeout=3)
        except queue.Empty:
            break
            
        n = task_config['n']
        structure = task_config['structure']
        k = task_config['k']
        mode = task_config['mode']
        rep_id = task_config['rep']
        seed = task_config['seed']
        result_file = task_config['result_file']
        
        if mode == 'direct':
            branching_factors = [3] * n
            gen_mode = 'direct'
        else:
            branching_factors = [3]*(n-k-1) + [3**(k+1)] + [1]*k
            gen_mode = 'reasoning'

        task = VariableTreeParityTask(branching_factors, seed=seed, input_dim=INPUT_DIM)
        
        # TRAIN TOKENIZER
        tokenizer = train_tokenizer(task)
        
        train_samples = [task.generate_sample(mode=gen_mode, return_data=True) for _ in range(TRAIN_SIZE_PER_DEPTH * n)]
        test_samples = [task.generate_sample(mode=gen_mode, return_data=True) for _ in range(TEST_SIZE)]
        
        os.makedirs("data_logs", exist_ok=True)
        sample_log = {
            "config": task_config,
            "train_example": str(train_samples[0][1]),
            "branching": branching_factors
        }
        with open(f"data_logs/log_n{n}_k{k}_rep{rep_id}.json", "w") as f:
            json.dump(sample_log, f)

        train_ds = TreeDataset(train_samples, tokenizer)
        test_ds = TreeDataset(test_samples, tokenizer)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # Swapped TinyTransformer for GPT2Small
        model = GPT2Small(
            tokenizer, 
            input_dim=INPUT_DIM, 
            embed_dim=EMBED_DIM, 
            n_layer=NUM_LAYERS, 
            n_head=NUM_HEADS
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        
        steps = 0
        model.train()
        
        
        mode_str = "Direct" if mode == 'direct' else "Reason"
        desc = f"GPU {gpu_id} | {mode_str} n={n} k={k} rep={rep_id}"
        
        with tqdm(total=total_steps, desc=desc, position=worker_id, leave=True) as pbar:
            for epoch in range(EPOCHS):
                for vecs, tokens, _ in train_loader:
                    vecs, tokens = vecs.to(device), tokens.to(device)
                    
                    optimizer.zero_grad()
                    logits = model(vecs, tokens)
                    
                    # Training slicing:
                    preds = logits[:, :-1, :].contiguous().view(-1, len(tokenizer))
                    targets = tokens.contiguous().view(-1)
                    
                    loss = criterion(preds, targets)
                    
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    steps += 1
                    
                    pbar.update(1)
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
                
        # Final Evaluation (Only once at the end)
        acc = calculate_accuracy(model, test_loader, device, tokenizer)
        # Use worker_id for write safety if needed, though write is global. 
        # Usually it's better to avoid writing to stdout/err from workers directly with tqdm active, 
        # but tqdm.write handles it reasonably well.
        # We can prepend the worker ID to the message.
        tqdm.write(f"[Worker {worker_id} | GPU {gpu_id}] Final | n={n} k={k} | Gen Acc: {acc:.4f}")
        err = 1.0 - acc
        
        result_dict = {
            'n': n,
            'k': k,
            'structure': structure,
            'rep': rep_id,
            'seed': seed,
            'error': err,
            'mode': mode
        }
        
        with open(result_file, "w") as f:
            json.dump(result_dict, f)
        
        result_queue.put(result_dict)

# ==========================================
# 4. Main Experiment Driver
# ==========================================

def run_experiment():
    if GPU_IDS is None:
        num_gpus = torch.cuda.device_count()
        gpus = list(range(num_gpus))
    else:
        gpus = GPU_IDS

    config_queue = mp.Queue()
    result_queue = mp.Queue()
    
    os.makedirs("results", exist_ok=True)
    
    replicates = REPLICATES
    seeds = SEEDS
    depths = DEPTHS
    
    results = []
    jobs_count = 0
    
    for n in depths:
        # 1. Direct Prediction (Structure = -0.1)
        for r in range(replicates):
            seed = seeds[r]
            fname = f"results/res_n{n}_k0_direct_seed{seed}.json"
            
            if os.path.exists(fname):
                print(f"Skipping {fname}, already exists.")
                with open(fname, 'r') as f:
                    results.append(json.load(f))
            else:
                config_queue.put({
                    'n': n, 'k': 0, 'structure': -0.1, 'mode': 'direct', 
                    'rep': r, 'seed': seed, 'result_file': fname
                })
                jobs_count += 1
            
        # 2. Reasoning Mode (Structure varies)
        for k in range(n):
            structure_val = 1.0 - (k / n)
            for r in range(replicates):
                seed = seeds[r]
                fname = f"results/res_n{n}_k{k}_reasoning_seed{seed}.json"
                
                if os.path.exists(fname):
                    print(f"Skipping {fname}, already exists.")
                    with open(fname, 'r') as f:
                        results.append(json.load(f))
                else:
                    config_queue.put({
                        'n': n, 'k': k, 'structure': structure_val, 'mode': 'reasoning', 
                        'rep': r, 'seed': seed, 'result_file': fname
                    })
                    jobs_count += 1

    if jobs_count > 0:
        processes = []
        worker_id = 0
        for gpu in gpus:
            for _ in range(WORKERS_PER_GPU):
                p = mp.Process(target=train_worker, args=(gpu, config_queue, result_queue, worker_id))
                p.start()
                processes.append(p)
                worker_id += 1

        for _ in range(jobs_count):
            results.append(result_queue.get())
            
        for p in processes:
            p.join()
    
    return results

# ==========================================
# 5. Plotting
# ==========================================

def plot_results(results):
    data_by_n = {}
    direct_by_n = {}
    
    for r in results:
        n = r['n']
        if n not in data_by_n:
            data_by_n[n] = []
            direct_by_n[n] = []
            
        if r['structure'] == 0.0:
            direct_by_n[n].append(r['error'])
        else:
            data_by_n[n].append(r)
            
    exp_data = []
    sanity_results = {}
    
    unique_ns = sorted(data_by_n.keys())
    
    for n in unique_ns:
        rows = data_by_n[n]
        if not rows: continue
        
        struct_map = {}
        for row in rows:
            s = row['structure']
            if s not in struct_map: struct_map[s] = []
            struct_map[s].append(row['error'])
            
        sorted_structs = sorted(struct_map.keys())
        
        means = []
        
        for s in sorted_structs:
            arr = np.array(struct_map[s])
            means.append(np.mean(arr))
            
        exp_data.append({
            'n': n,
            'structure': sorted_structs,
            'error_mean': means
        })
        
        dirs = np.array(direct_by_n[n])
        if len(dirs) > 0:
            sanity_results[n] = {
                'direct_mean': np.mean(dirs)
            }

    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 10,
        'axes.spines.top': False, 'axes.spines.right': False
    })
    
    plt.figure(figsize=(4, 2.5))
    
    # Generate Blue Gradient
    # Normalize n to 0-1 range for colormap
    min_n = min(unique_ns)
    max_n = max(unique_ns)
    
    for idx, data_entry in enumerate(exp_data):
        target_n = data_entry['n']
        
        # Calculate color from Blues colormap
        # We start from 0.4 to avoid too light colors
        if max_n > min_n:
            norm_n = (target_n - min_n) / (max_n - min_n)
        else:
            norm_n = 1.0
            
        color_val = 0.6 + 0.4 * norm_n
        color = cm.Blues(color_val)
        
        x_vals = list(data_entry['structure'])
        y_vals = list(data_entry['error_mean'])
        
        if target_n in sanity_results:
            dir_res = sanity_results[target_n]
            x_vals.append(-0.1)
            y_vals.append(dir_res['direct_mean'])
        
        combined = sorted(zip(x_vals, y_vals))
        x_vals, y_vals = zip(*combined)
        
        plt.plot(
            x_vals, y_vals, '-',
            color=color, 
            linewidth=2,
            markeredgecolor='white', markeredgewidth=1.5,
            label=f'$n={target_n}$'
        )

    plt.xlabel('Structure')
    plt.ylabel('Test Error')
    
    # Custom X-Ticks logic
    ticks = [-0.1, 0.25, 0.5, 0.75, 1.0]
    labels = ["Direct", "0.25", "0.5", "0.75", "1.0"]
    plt.xticks(ticks, labels)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.legend(frameon=False, loc="upper right", handlelength=1.5)
    plt.savefig("experiment_results_structure.pdf", dpi=300, bbox_inches='tight')
    print("Plot saved to experiment_results_structure.pdf")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    results = run_experiment()
    
    with open("experiment_raw_results.json", "w") as f:
        json.dump(results, f)
        
    plot_results(results)
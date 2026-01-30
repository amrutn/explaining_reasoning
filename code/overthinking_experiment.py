import os
import gc

# ==========================================
# Memory Optimization 1: Fragment Allocator
# ==========================================
# Set allocator to handle fragmentation better (Must be set before torch import)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
import random
import json
import uuid
import math
import traceback
from torch.utils.data import Dataset, DataLoader
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

# ==========================================
# 0. Global Configuration
# ==========================================

# Dataset & Training
TRAIN_SIZE = 50000
TEST_SIZE = 5000
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 2.5e-3

# Model Architecture & Dimensions
INPUT_DIM = 10      # Dimension of the input vector
EMBED_DIM = 64      # Internal embedding dimension
NUM_LAYERS = 2      # Number of Transformer layers
NUM_HEADS = 4       # Number of Attention heads

# Evaluation
MAX_GEN_LEN = 300

# Experiment Structure
DEPTH = 3
DEPTH_FACTORS = [2,7/3,8/3, 3]
M_VALUES = list(range(1, 21, 1))
REPLICATES = 20
SEEDS = range(42,42 + REPLICATES)

# Hardware / Multiprocessing
GPU_IDS = None
WORKERS_PER_GPU = 3

# ==========================================
# 1. Logic & Tasks
# ==========================================

class TreeParityTask:
    """
    Standard perfect m-ary tree task.
    """
    def __init__(self, branching_factor, depth, seed=42, input_dim=64):
        self.m = branching_factor
        self.n = depth
        self.seed = seed
        self.input_dim = input_dim
        
        rng = np.random.RandomState(seed) 
        self.hyperplane = rng.uniform(-1, 1, input_dim)
        
        if self.m == 1:
            self.num_leaves = 1
            self.leaf_start_index = self.n
        else:
            self.num_leaves = pow(self.m, self.n)
            self.leaf_start_index = (pow(self.m, self.n) - 1) // (self.m - 1)
            
    def get_parent(self, node_idx):
        if node_idx == 0: return None
        return (node_idx - 1) // self.m

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

    def solve_path(self, target_node, root_val):
        path = self.get_path(target_node)
        values = {}
        values[0] = root_val
        
        for i in range(1, len(path)):
            node = path[i]
            parent = path[i-1]
            op = self.get_edge_op(node)
            
            if op == 0:
                values[node] = values[parent]
            else:
                values[node] = 1 - values[parent]
        return values, path

    def generate_sample(self, mode='reasoning'):
        input_vector = np.random.uniform(-1, 1, self.input_dim).astype(np.float32)
        dot_prod = np.dot(input_vector, self.hyperplane)
        root_val = 1 if dot_prod > 0 else 0
        
        curr = 0
        for _ in range(self.n):
            if self.m > 1:
                branch = np.random.randint(0, self.m)
            else:
                branch = 0
            curr = self.m * curr + 1 + branch
        target_node = curr
        
        values, path = self.solve_path(target_node, root_val)
        
        input_str = f" Target: X{target_node}"
        
        if mode == 'direct':
            label_str = f" Answer: X{target_node}={values[target_node]}"
            full_text = input_str + label_str
            
        elif mode == 'reasoning':
            steps = []
            for i in range(1, len(path)):
                node = path[i]
                val = values[node]
                steps.append(f"X{node}={values[node]}")
            label_str = " Answer: " + " ".join(steps)
            full_text = input_str + label_str
            
        return input_vector, full_text

class RedundantTreeParityTask(TreeParityTask):
    def __init__(self, branching_factor, base_depth, depth_factor, seed=42, input_dim=64):
        self.base_n = base_depth
        self.r = depth_factor
        self.real_depth = int(base_depth * depth_factor)
        super().__init__(branching_factor, self.real_depth, seed, input_dim)
        
        self.logical_task = TreeParityTask(branching_factor, base_depth, seed, input_dim)
        
        self.num_logical_leaves = self.logical_task.num_leaves
        self.num_reasoning_leaves = self.num_leaves
        
        if self.num_logical_leaves > 0:
            self.leaves_per_logical = self.num_reasoning_leaves // self.num_logical_leaves
        else:
            self.leaves_per_logical = 1

    def get_edge_op(self, node_idx):
        if node_idx < self.leaf_start_index:
            return super().get_edge_op(node_idx)
        
        parent = self.get_parent(node_idx)
        parity_reasoning_parent = 0
        curr = parent
        while curr is not None and curr > 0:
            op = super().get_edge_op(curr)
            parity_reasoning_parent ^= op
            curr = self.get_parent(curr)
            
        rel_idx = node_idx - self.leaf_start_index
        logical_rel_idx = rel_idx // self.leaves_per_logical
        logical_leaf_idx = self.logical_task.leaf_start_index + logical_rel_idx
        
        parity_logical = 0
        curr = logical_leaf_idx
        while curr is not None and curr > 0:
            op = self.logical_task.get_edge_op(curr)
            parity_logical ^= op
            curr = self.logical_task.get_parent(curr)
            
        return parity_reasoning_parent ^ parity_logical

    def generate_sample(self, mode='reasoning'):
        input_vector = np.random.uniform(-1, 1, self.input_dim).astype(np.float32)
        dot_prod = np.dot(input_vector, self.hyperplane)
        root_val = 1 if dot_prod > 0 else 0
        
        logical_rel_idx = np.random.randint(0, self.num_logical_leaves)
        logical_leaf_idx = self.logical_task.leaf_start_index + logical_rel_idx
        
        if self.leaves_per_logical > 1:
            offset = np.random.randint(0, self.leaves_per_logical)
        else:
            offset = 0
            
        reasoning_leaf_idx = self.leaf_start_index + (logical_rel_idx * self.leaves_per_logical) + offset
        
        values, path = self.solve_path(reasoning_leaf_idx, root_val)
        
        input_str = f" Target: X{logical_leaf_idx}"
        target_val = values[reasoning_leaf_idx]
        
        if mode == 'direct':
            label_str = f" Answer: X{logical_leaf_idx}={target_val}" 
            full_text = input_str + label_str
            
        elif mode == 'reasoning':
            steps = []
            for i in range(1, len(path)):
                node = path[i]
                val = values[node]
                if i == len(path) - 1:
                    steps.append(f"X{logical_leaf_idx}={val}")
                else:
                    steps.append(f"Y{node}={val}")
                    
            label_str = " Answer: " + " ".join(steps)
            full_text = input_str + label_str
            
        return input_vector, full_text

# ==========================================
# 2. Tokenizers & Data
# ==========================================

def train_tokenizer(task, size=100000):
    unique_id = uuid.uuid4().hex
    corpus_path = f"temp_corpus_{unique_id}.txt"
    tokenizer_path = f"tokenizer_{unique_id}.json"

    # Streaming write to avoid holding big text in RAM
    with open(corpus_path, "w") as f:
        for _ in range(size):
            _, text_r = task.generate_sample(mode='reasoning')
            _, text_d = task.generate_sample(mode='direct')
            f.write(text_r + "\n")
            f.write(text_d + "\n")
            
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=[corpus_path], vocab_size=5000, min_frequency=2, special_tokens=["<|endoftext|>", "<pad>"])
    tokenizer.save(tokenizer_path)
    
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    fast_tokenizer.pad_token = "<pad>"
    fast_tokenizer.eos_token = "<|endoftext|>"
    
    if os.path.exists(corpus_path):
        os.remove(corpus_path)
    if os.path.exists(tokenizer_path):
        os.remove(tokenizer_path)
        
    return fast_tokenizer

class TreeDataset(Dataset):
    """
    Memory Efficient Dataset:
    Pre-tokenizes data into contiguous PyTorch tensors (Int16/Int32) and deletes 
    the original string list. significantly reduces RAM overhead.
    """
    def __init__(self, samples_list, tokenizer, max_len=MAX_GEN_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Pre-allocate tensors
        num_samples = len(samples_list)
        # Use int16 if vocab size < 32767, else int32
        dtype = torch.int16 if tokenizer.vocab_size < 32000 else torch.int32
        
        self.input_ids = torch.full((num_samples, max_len), tokenizer.pad_token_id, dtype=dtype)
        self.input_vecs = torch.zeros((num_samples, len(samples_list[0][0])), dtype=torch.float32)
        
        # We don't need prompt lengths for training usually, but if needed for generation logic
        # we can calculate on fly or store. Evaluating only needs a few samples so we skip storing for all.
        
        print(f"Tokenizing {num_samples} samples into tensors...")
        for i, (vec, text) in enumerate(samples_list):
            self.input_vecs[i] = torch.tensor(vec, dtype=torch.float32)
            
            enc = tokenizer(
                text, 
                max_length=max_len, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            )
            # Store in pre-allocated tensor
            self.input_ids[i] = enc["input_ids"].squeeze(0).to(dtype)
        
        # Clear original list from memory immediately
        del samples_list
        gc.collect()

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        # Return as Long for model compatibility, but storage is small int
        return self.input_vecs[idx], self.input_ids[idx].long(), 0 # 0 is placeholder for prompt_len

# ==========================================
# 3. Model
# ==========================================

class GPT2Small(nn.Module):
    def __init__(self, tokenizer, input_dim=INPUT_DIM, embed_dim=EMBED_DIM, n_layer=NUM_LAYERS, n_head=NUM_HEADS):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        config = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=1024,
            n_embd=embed_dim,    
            n_layer=n_layer,   
            n_head=n_head,     
            bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            loss_type="cross_entropy",
            use_cache=False # Must be False if gradient_checkpointing is True
        )
        self.model = GPT2LMHeadModel(config)
        self.model.gradient_checkpointing_enable()
        
    def forward(self, vec, x, src_key_padding_mask=None):
        # 1. Embed Text
        text_embeds = self.model.transformer.wte(x) 
        
        # 2. Project Vector
        vec_projected = self.input_projection(vec) 
        vec_embeds = vec_projected.unsqueeze(1) 
        
        # 3. Concatenate
        inputs_embeds = torch.cat([vec_embeds, text_embeds], dim=1) 
        
        # 4. Handle Attention Mask
        if src_key_padding_mask is not None:
            B = vec.shape[0]
            vec_mask = torch.zeros((B, 1), dtype=torch.bool, device=vec.device) 
            combined_pad_mask = torch.cat([vec_mask, src_key_padding_mask], dim=1)
            attention_mask = (~combined_pad_mask).long()
        else:
            attention_mask = None

        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return out.logits

def get_model(tokenizer):
    return GPT2Small(tokenizer)

# ==========================================
# 4. Training & Evaluation
# ==========================================

def evaluate_model(model, task, tokenizer, device, mode, batch_size=BATCH_SIZE):
    model.eval()
    
    # We create a small test list. No need for the heavy TreeDataset here, can rely on on-the-fly logic 
    # or just use TreeDataset for consistency but with smaller size.
    test_data = [task.generate_sample(mode=mode) for _ in range(TEST_SIZE)]
    # Use TreeDataset to handle vector/tensor conversion cleanly
    dataset = TreeDataset(test_data, tokenizer, max_len=MAX_GEN_LEN)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    correct = 0
    total = 0
    examples = []
    
    with torch.no_grad():
        # Memory Optimization 3: Mixed Precision Inference
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            for i, (input_vecs, input_ids, _) in enumerate(dataloader):
                input_vecs = input_vecs.to(device)
                input_ids = input_ids.to(device)
                
                src_key_padding_mask = (input_ids == tokenizer.pad_token_id)
                
                logits = model(input_vecs, input_ids, src_key_padding_mask=src_key_padding_mask)
                
                valid_lens = (~src_key_padding_mask).sum(dim=1).long()
                logit_indices = valid_lens - 1
                
                pred_logits = logits[torch.arange(logits.size(0)), logit_indices, :]
                pred_tokens = torch.argmax(pred_logits, dim=-1)
                true_tokens = input_ids[torch.arange(input_ids.size(0)), logit_indices]
                
                correct += (pred_tokens == true_tokens).sum().item()
                total += input_ids.size(0)
                
                # Capture Example Data (First batch only)
                if i == 0:
                    num_ex = min(5, input_ids.size(0))
                    for j in range(num_ex):
                        decoded_text = tokenizer.decode(input_ids[j].cpu().tolist(), skip_special_tokens=True)
                        pred_char = tokenizer.decode([pred_tokens[j].item()])
                        true_char = tokenizer.decode([true_tokens[j].item()])
                        vec_list = input_vecs[j].cpu().tolist()
                        
                        examples.append({
                            "input_vector": vec_list,
                            "full_text_context": decoded_text,
                            "prediction": pred_char,
                            "ground_truth": true_char,
                            "correct": bool(pred_tokens[j].item() == true_tokens[j].item())
                        })
            
    model.train()
    accuracy = correct / total if total > 0 else 0
    
    # Explicit cleanup
    del dataset, dataloader, test_data
    return accuracy, examples

def run_experiment(task, mode, epochs=EPOCHS, train_samples=TRAIN_SIZE, batch_size=BATCH_SIZE, accumulation_steps=1, 
                   checkpoint_dir="results", task_id="default", gpu_id=0, pbar_pos=0):
    
    hist_name = f"acc_history_{task_id}_{mode}.json"
    full_path = os.path.join(checkpoint_dir, hist_name)
    
    if os.path.exists(full_path):
        try:
            with open(full_path, 'r') as f:
                return json.load(f)
        except:
            pass

    if torch.cuda.is_available() and isinstance(gpu_id, int):
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    os.makedirs(checkpoint_dir, exist_ok=True)
    
    tokenizer = train_tokenizer(task, size=min(train_samples, 5000)) # Smaller sample for vocab is usually fine
    model = get_model(tokenizer).to(device)
    
    train_data = [task.generate_sample(mode=mode) for _ in range(train_samples)]
    dataset = TreeDataset(train_data, tokenizer, max_len=MAX_GEN_LEN)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    steps = epochs * train_samples // (batch_size * accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Memory Optimization 3: Mixed Precision Training scaler
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    model.train()
    eval_history = []
    
    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"Dev {gpu_id} | {task_id} | Ep {epoch+1}/{epochs}", 
                            position=pbar_pos, leave=False)
        # Memory Optimization 4: set_to_none=True
        optimizer.zero_grad(set_to_none=True)
        
        for i, (input_vecs, input_ids, _) in enumerate(progress_bar):
            input_vecs = input_vecs.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            
            src_key_padding_mask = (input_ids == tokenizer.pad_token_id)
            
            # Autocast context
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                logits = model(input_vecs, input_ids, src_key_padding_mask=src_key_padding_mask)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids.contiguous()
                loss = criterion(shift_logits.view(-1, tokenizer.vocab_size), shift_labels.view(-1))
                loss = loss / accumulation_steps
            
            # Scaled Backward
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                    
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                progress_bar.set_postfix(loss=f"{loss.item()*accumulation_steps:.4f}")

    final_acc, examples = evaluate_model(model, task, tokenizer, device, mode, batch_size)
    eval_history.append(final_acc)
    
    with open(full_path, 'w') as f:
        json.dump(eval_history, f)
        
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_name = f"log_{task_id}_{mode}.json"
    log_path = os.path.join(log_dir, log_name)
    with open(log_path, 'w') as f:
        json.dump(examples, f, indent=2)
        
    # Memory Optimization 5: Explicit cleanup
    del model, optimizer, dataset, dataloader, scaler
    try:
        del input_ids, input_vecs, logits, loss
    except UnboundLocalError:
        pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
        
    return eval_history

# ==========================================
# 5. Parallel Execution Logic
# ==========================================

def worker_process(rank, world_size, gpu_id, queue, config): 
    my_tasks = config.get('my_tasks', [])
    print(f"[Worker {rank} on Device {gpu_id}] Starting. Tasks: {len(my_tasks)}")
    local_results = [] 
    
    try:
        for task_def in my_tasks:
            try:
                # Force GC between tasks
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                exp_type = task_def['type']
                seed = task_def['seed']
                
                np.random.seed(seed)
                torch.manual_seed(seed)
                random.seed(seed)
                
                epochs = config.get('epochs', EPOCHS)
                train_samples = config.get('train_samples', TRAIN_SIZE)
                batch_size = config.get('batch_size', BATCH_SIZE)
                
                if exp_type == 'exp2':
                    n = task_def['n']
                    r = task_def['r']
                    m = task_def['m']
                    
                    # Baseline
                    task_baseline = TreeParityTask(branching_factor=m, depth=n, seed=seed, input_dim=INPUT_DIM)
                    task_id_base = f"base_m{m}_n{n}_seed{seed}"
                    hist_base = run_experiment(task_baseline, 'reasoning', 
                                               epochs=epochs, train_samples=train_samples, batch_size=batch_size, 
                                               task_id=task_id_base, gpu_id=gpu_id, pbar_pos=rank)
                    acc_baseline = hist_base[-1]

                    # Overthinking
                    task_overthinking = RedundantTreeParityTask(branching_factor=m, base_depth=n, depth_factor=r, seed=seed, input_dim=INPUT_DIM)
                    task_id_over = f"over_m{m}_n{n}_r{r}_seed{seed}"
                    hist_over = run_experiment(task_overthinking, 'reasoning', 
                                               epochs=epochs, train_samples=train_samples, batch_size=batch_size, 
                                               task_id=task_id_over, gpu_id=gpu_id, pbar_pos=rank)
                    acc_overthinking = hist_over[-1]
                    
                    diff = (1.0 - acc_baseline) - (1.0 - acc_overthinking)
                    local_results.append({
                        'type': 'exp2', 
                        'n': n, 'r': r, 'm': m, 'seed': seed, 
                        'diff': diff
                    })
                    tqdm.write(f"[Worker {rank}] Exp2 Finished m={m}, r={r:.2f}, seed={seed}. Benefit={diff:.3f}")
            except Exception as e:
                tqdm.write(f"[Worker {rank}] Task failed: {e}")
                traceback.print_exc()
    except Exception as e:
        tqdm.write(f"[Worker {rank}] Process crashed: {e}")
        traceback.print_exc()
    finally:
        queue.put(local_results)
        tqdm.write(f"[Worker {rank}] Done.")

def run_task_batch(task_list, target_gpus, base_config):
    num_workers = len(target_gpus)
    random.shuffle(task_list)
    worker_tasks = [[] for _ in range(num_workers)]
    for i, task in enumerate(task_list):
        worker_tasks[i % num_workers].append(task)
        
    queue = mp.Queue()
    processes = []
    print(f"\n--- Starting Batch with {len(task_list)} tasks ---")
    
    for rank, gpu_id in enumerate(target_gpus):
        worker_config = base_config.copy()
        worker_config['my_tasks'] = worker_tasks[rank]
        p = mp.Process(target=worker_process, args=(rank, num_workers, gpu_id, queue, worker_config))
        p.start()
        processes.append(p)
    
    all_results = []
    # Collect results gradually to ensure queue doesn't fill up if tasks are huge
    # (Though here results are small dicts, so simple get loop is fine)
    finished_workers = 0
    while finished_workers < num_workers:
        try:
            # We expect each worker to put exactly one list at the end
            res = queue.get()
            all_results.extend(res)
            finished_workers += 1
        except Exception:
            break

    for p in processes:
        p.join()
        
    print("--- Batch Complete ---")
    return all_results

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    if GPU_IDS is None:
        if torch.cuda.is_available():
            available_gpus = list(range(torch.cuda.device_count()))
            target_gpus = available_gpus * WORKERS_PER_GPU
        else:
            target_gpus = ['cpu'] * WORKERS_PER_GPU
    else:
        target_gpus = GPU_IDS * WORKERS_PER_GPU

    print(f"Starting parallel execution on devices: {target_gpus}...")
    
    config_base = {
        'epochs': EPOCHS,
        'train_samples': TRAIN_SIZE,
        'batch_size': BATCH_SIZE
    }
    
    n_values = [DEPTH]
    r_values = DEPTH_FACTORS
    m_values = M_VALUES 
    
    exp2_tasks = []
    for n in n_values:
        for r in r_values:
            for m in m_values:
                for seed in SEEDS:
                    exp2_tasks.append({
                        'type': 'exp2', 
                        'n': n, 'r': r, 'm': m, 'seed': seed,
                    })
    
    results_flat_exp2 = run_task_batch(exp2_tasks, target_gpus, config_base)
    
    results_exp2 = {} 
    
    for item in results_flat_exp2:
        key = (item['n'], item['r'])
        if key not in results_exp2: 
            results_exp2[key] = {}
            
        if item['m'] not in results_exp2[key]: 
            results_exp2[key][item['m']] = []
            
        results_exp2[key][item['m']].append(item['diff'])

    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 10,
        'axes.spines.top': False, 'axes.spines.right': False
    })

    plt.figure(figsize=(4,2.5))
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=1)
    
    n_val = DEPTH
    
    if not results_exp2:
        print("No results collected. Please check logs for errors.")
    else:
        r_keys = sorted([k[1] for k in results_exp2.keys() if k[0] == n_val])
        
        if not r_keys:
            print("No valid r_keys found for plotting.")
        else:
            blues = plt.get_cmap('Blues')
            blue_colors = blues(np.linspace(0.6, 1.0, 256))
            new_cmap = mcolors.LinearSegmentedColormap.from_list('DarkBlues', blue_colors)
            norm = plt.Normalize(min(r_keys), max(r_keys))
            
            for r in r_keys:
                m_dict = results_exp2[(n_val, r)]
                sorted_ms = sorted(m_dict.keys())
                plot_ms = [m for m in sorted_ms] 
                
                means = [np.mean(m_dict[m]) for m in plot_ms]
                
                if len(r_keys) > 1:
                    color = new_cmap(norm(r))
                else:
                    color = 'royalblue'
                
                plt.plot(plot_ms, means, '-', 
                         color=color, 
                         linewidth=2, 
                         label=f"$r={r:.2f}$", 
                         alpha=0.9, 
                         markersize=6, 
                         markeredgecolor='white', 
                         markeredgewidth=1.5)

            plt.xlabel("Degree $m$")
            plt.ylabel("Thinking Gain")
            plt.legend(loc='upper left', frameon=False, bbox_to_anchor=(-.02, 1.05), handlelength=1.5)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.xticks(list(range(2, max(m_values), 4)))
            plt.tight_layout()
            plt.savefig("experiment_results_overthinking.pdf", dpi=300, bbox_inches='tight')
            print("Plot saved to experiment_results_overthinking.pdf")
            
    print("\nExperiment complete.")
import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import uuid
from tqdm import tqdm
from collections import deque
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

# ==========================================
# 1. Task Definition (From Original Code)
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

    def generate_specific_path(self, target_node, input_vec=None):
        """Helper to generate a trace for a specific target node."""
        if input_vec is None:
            input_vec = self.rng.uniform(-1, 1, self.input_dim).astype(np.float32)
            
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
        steps = []
        for node in path:
            if node == 0: continue
            steps.append(f"X{node}={values[node]}")
        label_str = " Answer: " + " ".join(steps)
        full_text = input_str + label_str
        return full_text

    def generate_sample(self):
        """Random sampling wrapper."""
        leaves_start = self.layer_offsets[-1]
        target_node = self.rng.randint(leaves_start, leaves_start + self.num_leaves)
        return self.generate_specific_path(target_node)

# ==========================================
# 2. Trie Implementation
# ==========================================

class TrieNode:
    def __init__(self):
        self.children = {} # Map token_id -> TrieNode
        self.count = 0

def build_trie_and_measure(tokenized_sequences):
    """
    Builds a trie from sequences of token IDs and computes 
    the average branching factor by LAYER.
    
    1. Group nodes by depth.
    2. Calculate average degree for each depth (excluding depths with 0 total degree).
    3. Average the layer-wise averages.
    """
    root = TrieNode()
    
    # Build Trie
    for seq in tokenized_sequences:
        curr = root
        curr.count += 1
        for token in seq:
            if token not in curr.children:
                curr.children[token] = TrieNode()
            curr = curr.children[token]
            curr.count += 1
            
    # Measure by layer using BFS
    queue = deque([(root, 0)]) # (node, depth)
    layer_stats = {} # depth -> {'sum': total_degree, 'count': num_nodes}
    
    while queue:
        node, depth = queue.popleft()
        deg = len(node.children)
        
        if depth not in layer_stats:
            layer_stats[depth] = {'sum': 0, 'count': 0}
        
        layer_stats[depth]['sum'] += deg
        layer_stats[depth]['count'] += 1
        
        for child in node.children.values():
            queue.append((child, depth + 1))
            
    # Compute average per layer, then average of layers
    layer_averages = []
    
    # We sort by depth to be consistent, though order doesn't affect mean
    for d in sorted(layer_stats.keys()):
        stats = layer_stats[d]
        total_deg = stats['sum']
        count = stats['count']
        
        # If the entire layer has 0 children (leaves), skip it
        # as it represents the end of sequences, not a branching decision layer.
        if total_deg == 0:
            continue
            
        layer_averages.append(total_deg / count)
        
    if not layer_averages:
        return 0.0
        
    return sum(layer_averages) / len(layer_averages)

# ==========================================
# 3. Analysis Logic
# ==========================================

def train_temp_tokenizer(texts, vocab_size=5000):
    unique_id = uuid.uuid4().hex
    corpus_path = f"temp_corpus_{unique_id}.txt"
    tokenizer_path = f"tokenizer_{unique_id}.json"

    with open(corpus_path, "w") as f:
        for t in texts:
            f.write(t + "\n")
            
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=[corpus_path], vocab_size=vocab_size, min_frequency=2, special_tokens=["<|endoftext|>", "<pad>"])
    tokenizer.save(tokenizer_path)
    
    # Load into HF wrapper
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    
    # Cleanup
    if os.path.exists(corpus_path): os.remove(corpus_path)
    if os.path.exists(tokenizer_path): os.remove(tokenizer_path)
        
    return fast_tokenizer

def run_analysis():
    depths = [2, 3, 4]
    degrees = list(range(2, 11)) # 2 to 10
    
    results = {n: {'x': [], 'y': []} for n in depths}
    
    total_iterations = len(depths) * len(degrees)
    pbar = tqdm(total=total_iterations, desc="Analyzing Tasks")
    
    for n in depths:
        for m in degrees:
            # 1. Setup Task
            # Uniform branching factor m for depth n
            branching_factors = [m] * n
            task = VariableTreeParityTask(branching_factors, seed=42)
            
            # 2. Generate Traces
            total_leaves = task.num_leaves
            
            texts = []
            
            # Exhaustive generation for all depths
            leaves_start = task.layer_offsets[-1]
            for i in range(total_leaves):
                target_node = leaves_start + i
                texts.append(task.generate_specific_path(target_node))
            
            # 3. Train Tokenizer
            tokenizer = train_temp_tokenizer(texts, vocab_size=5000)
            
            # 4. Tokenize
            tokenized_seqs = []
            for t in texts:
                # encode returns list of ids
                ids = tokenizer.encode(t) 
                tokenized_seqs.append(ids)
                
            # 5. Build Trie & Compute Metric
            avg_degree = build_trie_and_measure(tokenized_seqs)
            
            # Store
            results[n]['x'].append(m)
            results[n]['y'].append(avg_degree)
            
            pbar.update(1)
            pbar.set_postfix(n=n, m=m, deg=f"{avg_degree:.2f}")
            
    pbar.close()
    return results

# ==========================================
# 4. Plotting
# ==========================================

def plot_trie_analysis(results):
    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 10,
        'axes.spines.top': False, 'axes.spines.right': False
    })
    
    plt.figure(figsize=(4, 2.5))
    
    depths = sorted(results.keys())
    min_n = min(depths)
    max_n = max(depths)
    
    for n in depths:
        data = results[n]
        x_vals = data['x']
        y_vals = data['y']
        
        # Color matching previous script logic
        if max_n > min_n:
            norm_n = (n - min_n) / (max_n - min_n)
        else:
            norm_n = 1.0
            
        color_val = 0.6 + 0.4 * norm_n
        color = cm.Blues(color_val)
        
        plt.plot(
            x_vals, y_vals, '-',
            color=color,
            linewidth=2,
            label=f'$n={n}$'
        )
        
    plt.xlabel(r'Degree $m$')
    plt.ylabel('Tokenized Degree')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=False, loc="upper left")
    
    output_filename = "trie_degree_analysis.pdf"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")

if __name__ == "__main__":
    data = run_analysis()
    
    # Save raw data just in case
    with open("trie_analysis_results.json", "w") as f:
        json.dump(data, f)
        
    plot_trie_analysis(data)
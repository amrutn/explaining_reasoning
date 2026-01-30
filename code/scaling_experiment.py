import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit
import json
import os
import queue
import time

# --- Configuration ---

# Hardware Configuration
# "Use the six GPUs" - assuming indices 0-5 are valid
GPU_IDS = [0, 1, 2, 3, 4, 5] 
WORKERS_PER_GPU = 2 # Run 2 jobs per GPU concurrently

# Experiment ranges
# Generate ~100 logarithmically spaced values between 10 and 1500
N_values = np.unique(np.logspace(1, np.log10(1500), 100).astype(int)).tolist()
dimensions = [3, 4, 5]

# Model Hyperparameters
HIDDEN_DIM = 128 
BATCH_SIZE = 512
TRAIN_STEPS = 500 

# --- Helper Functions ---

def generate_prototypes(N, d, device):
    """
    Generates N unit vectors in d dimensions.
    """
    # Generate Gaussian random variables
    vecs = torch.randn(N, d)
    # Normalize to unit sphere
    prototypes = vecs / vecs.norm(dim=1, keepdim=True)
    return prototypes.to(device)

def get_batch(batch_size, d, prototypes, device):
    """
    Generates random inputs and determines their ground truth class.
    """
    # 1. Sample inputs on unit sphere
    inputs = torch.randn(batch_size, d, device=device)
    inputs = inputs / inputs.norm(dim=1, keepdim=True)
    
    # 2. Find closest prototype (argmax of dot product)
    similarities = torch.matmul(inputs, prototypes.T)
    targets = torch.argmax(similarities, dim=1)
    
    return inputs, targets

# --- Model Definition ---

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: (Batch, Input_Dim) -> (Batch, Output_Dim)
        return self.net(x)

# --- Worker Process ---

def train_worker(gpu_id, config_queue, result_queue, worker_id):
    # Set device for this worker
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Worker {worker_id} started on {device}")

    while True:
        try:
            # Timeout allows worker to exit if queue is empty for a while
            task_config = config_queue.get(timeout=3)
        except queue.Empty:
            break
            
        N = task_config['N']
        d = task_config['d']
        result_file = task_config['result_file']
        
        try:
            # 1. Setup Data and Model
            prototypes = generate_prototypes(N, d, device)
            
            # Initialize Simple MLP
            model = SimpleMLP(input_dim=d, hidden_dim=HIDDEN_DIM, output_dim=N).to(device)
            
            optimizer = optim.AdamW(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            # 2. Train
            model.train()
            for step in range(TRAIN_STEPS):
                inputs, targets = get_batch(BATCH_SIZE, d, prototypes, device)
                
                optimizer.zero_grad()
                logits = model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
            
            # 3. Evaluate
            model.eval()
            with torch.no_grad():
                val_inputs, val_targets = get_batch(2000, d, prototypes, device)
                val_logits = model(val_inputs)
                predictions = torch.argmax(val_logits, dim=1)
                accuracy = (predictions == val_targets).float().mean().item()
                error = 1.0 - accuracy
            
            # 4. Save and Report
            result_data = {
                'd': d,
                'N': N,
                'error': error
            }
            
            # Atomic write pattern to avoid corruption
            temp_file = result_file + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(result_data, f)
            os.rename(temp_file, result_file)
            
            print(f"[Worker {worker_id}|GPU {gpu_id}] Finished d={d} N={N} | Err: {error:.4f}")
            result_queue.put(result_data)
            
        except Exception as e:
            print(f"Worker {worker_id} failed on task d={d}, N={N}: {e}")
            result_queue.put({'d': d, 'N': N, 'error': None}) 

# --- Main Driver ---

def run_experiment():
    # Setup directory
    os.makedirs("results", exist_ok=True)
    
    config_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # 1. Populate Queue
    jobs_count = 0
    results = []
    
    for d in dimensions:
        for N in N_values:
            filename = f"results/res_d{d}_N{N}.json"
            
            # Checkpoint check
            if os.path.exists(filename):
                print(f"Skipping existing: {filename}")
                with open(filename, 'r') as f:
                    try:
                        data = json.load(f)
                        results.append(data)
                    except json.JSONDecodeError:
                        print(f"Error reading {filename}, re-queueing.")
                        config_queue.put({'d': d, 'N': N, 'result_file': filename})
                        jobs_count += 1
            else:
                config_queue.put({'d': d, 'N': N, 'result_file': filename})
                jobs_count += 1
    
    # 2. Start Workers
    if jobs_count > 0:
        processes = []
        worker_id = 0
        
        # Determine available GPUs (fallback if hardcoded list is invalid)
        available_gpus = GPU_IDS
        if not torch.cuda.is_available():
            available_gpus = [0] # CPU fallback
        elif torch.cuda.device_count() < len(GPU_IDS):
            available_gpus = list(range(torch.cuda.device_count()))
            
        print(f"Spinning up workers on GPUs: {available_gpus}")
        
        for gpu in available_gpus:
            for _ in range(WORKERS_PER_GPU):
                p = mp.Process(target=train_worker, args=(gpu, config_queue, result_queue, worker_id))
                p.start()
                processes.append(p)
                worker_id += 1
        
        # 3. Collect Results
        print(f"Waiting for {jobs_count} jobs...")
        for _ in range(jobs_count):
            res = result_queue.get()
            if res.get('error') is not None:
                results.append(res)
        
        # 4. Join Processes
        for p in processes:
            p.join()
            
    return results

# --- Plotting ---

def plot_results(flat_results):
    # Reorganize data: {d: [errors corresponding to sorted N_values]}
    
    data_map = {d: {} for d in dimensions}
    for res in flat_results:
        d = res['d']
        N = res['N']
        err = res['error']
        if d in data_map:
            data_map[d][N] = err
            
    # Prepare lists for plotting
    sorted_results = {}
    for d in dimensions:
        errs = []
        # Ensure we pick errors in the order of N_values
        for N in N_values:
            if N in data_map[d]:
                errs.append(data_map[d][N])
            else:
                errs.append(np.nan) # Handle missing data gracefully
        sorted_results[d] = errs

    # Define the scaling function: y = a * N^(2/d) + b
    def scaling_law(N, a, b, dim):
        return a * np.power(N, 2.0/dim) + b

    # Styling
    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
        'axes.spines.top': False, 'axes.spines.right': False
    })

    plt.figure(figsize=(3, 2.5))

    min_d = min(dimensions)
    max_d = max(dimensions)
    
    fit_params = {}
    annotation_coords = [[750, 0.7], [1000,0.53],[700, 0.20]]
    for i,d in enumerate(dimensions):
        y_data = np.array(sorted_results[d])
        x_data = np.array(N_values)
        
        # Filter out NaNs if any jobs failed
        mask = ~np.isnan(y_data)
        x_clean = x_data[mask]
        y_clean = y_data[mask]
        
        if len(x_clean) == 0:
            fit_params[d] = {'a': 0.0, 'b': 0.0}
            continue

        norm_d = (d - min_d) / (max_d - min_d) if max_d > min_d else 0.5
        color_val = 0.6 + 0.4 * norm_d
        color = cm.Blues(color_val)
        
        # Fit the curve
        def func_to_fit(x, a, b):
            return scaling_law(x, a, b, d)
        
        x_fit = x_clean
        y_fit = y_clean

        try:
            if len(x_fit) < 2:
                print(f"Not enough data to fit d={d}")
                a_fit, b_fit = 0, 0
            else:
                popt, _ = curve_fit(func_to_fit, x_fit, y_fit, maxfev=5000)
                a_fit, b_fit = popt
        except:
            print(f"Could not fit curve for d={d}")
            a_fit, b_fit = 0, 0
            
        fit_params[d] = {'a': float(a_fit), 'b': float(b_fit)}

        # Plot Data Points
        plt.plot(
            x_clean, y_clean, 
            marker='o', 
            markersize=3, 
            color=color, 
            linewidth=0,
            markeredgewidth=0, 
            alpha=0.6, 
            label=f'd={d}'
        )
        
        # Plot Fitted Curve
        x_range = np.linspace(min(N_values), max(N_values), 200)
        y_fit = scaling_law(x_range, a_fit, b_fit, d)
        
        plt.plot(
            x_range, y_fit,
            linestyle='-',
            linewidth=1.5,
            color=color,
            alpha=1.0
        )

        plt.text(
            annotation_coords[i][0], annotation_coords[i][1], 
            f" d={d}", 
            color=color, 
            fontsize=12,
            verticalalignment='center'
        )
    
    # Save fit parameters
    with open("fit_parameters.json", "w") as f:
        json.dump(fit_params, f, indent=4)
    print("Fit parameters saved to fit_parameters.json")

    plt.xlabel(r'Number of Classes $m$')
    plt.ylabel('Test Error')
    plt.xscale('linear') 
    #plt.ylim(0, 1.02)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    plt.savefig("scaling_law_results.pdf", dpi=300, bbox_inches='tight')
    print("Plot saved to scaling_law_results.pdf")

if __name__ == "__main__":
    # Required for CUDA multiprocessing
    mp.set_start_method('spawn', force=True)
    
    print("Starting Experiment...")
    final_results = run_experiment()
    
    print("Generating Plot...")
    plot_results(final_results)
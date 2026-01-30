import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import time
import os
import json
import pickle
import random
import torch.multiprocessing as mp
from tqdm import tqdm

# Configuration
BATCH_SIZE = 64
IMG_SIZE = 224
TOTAL_STEPS = 25000 
LR = 3e-4

# ==========================================
# 1. Dataset Wrapper
# ==========================================
class SuperclassCIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, target_classes, mode='semantic', train=True, transform=None, download=False):
        super().__init__(root, train=train, transform=transform, download=download)
        self.target_classes = target_classes
        self.mode = mode
        
        # The CIFAR-100 Coarse mapping (Level 1 hierarchy)
        self.coarse_groups = [
            [4, 30, 55, 72, 95],   # aquatic mammals
            [1, 32, 67, 73, 91],   # fish
            [54, 62, 70, 82, 92],  # flowers
            [9, 10, 16, 28, 61],   # food containers
            [0, 51, 53, 57, 83],   # fruit and vegetables
            [22, 39, 40, 86, 87],  # household electrical devices
            [5, 20, 25, 84, 94],   # household furniture
            [6, 7, 14, 18, 24],    # insects
            [3, 42, 43, 88, 97],   # large carnivores
            [12, 17, 37, 68, 76],  # large man-made outdoor things
            [23, 33, 49, 60, 71],  # large natural outdoor scenes
            [15, 19, 21, 31, 38],  # large omnivores and herbivores
            [34, 63, 64, 66, 75],  # medium-sized mammals
            [26, 45, 77, 79, 99],  # non-insect invertebrates
            [2, 11, 35, 46, 98],   # people
            [27, 29, 44, 78, 93],  # reptiles
            [36, 50, 65, 74, 80],  # small mammals
            [47, 52, 56, 59, 96],  # trees
            [8, 13, 48, 58, 90],   # vehicles 1
            [41, 69, 81, 85, 89],  # vehicles 2
        ]
        
        self.class_map = self._generate_mapping(target_classes)
        self.targets = [self.class_map[t] for t in self.targets]

    def _generate_mapping(self, m):
        mapping = {}
        
        if self.mode == 'semantic':
            # CASE A: Fine-Grained Scaling (20 <= m <= 100)
            if m >= 20:
                new_label_counter = 0
                for i, group in enumerate(self.coarse_groups):
                    n_subclusters = (m // 20) + (1 if i < (m % 20) else 0)
                    fine_classes = sorted(group)
                    for fine_id in fine_classes:
                        local_idx = fine_classes.index(fine_id)
                        subcluster_idx = local_idx % n_subclusters
                        mapping[fine_id] = new_label_counter + subcluster_idx
                    new_label_counter += n_subclusters

            # CASE B: Coarse Scaling (m < 20)
            else:
                coarse_to_new = {}
                for coarse_id in range(20):
                    coarse_to_new[coarse_id] = coarse_id % m
                
                for coarse_id, fine_ids in enumerate(self.coarse_groups):
                    target_label = coarse_to_new[coarse_id]
                    for fine_id in fine_ids:
                        mapping[fine_id] = target_label

        elif self.mode == 'random':
            # Fully random grouping. 
            rng = random.Random(42) 
            all_fine = list(range(100))
            rng.shuffle(all_fine)
            
            current_idx = 0
            for bucket_id in range(m):
                bucket_size = (100 // m) + (1 if bucket_id < (100 % m) else 0)
                bucket_classes = all_fine[current_idx : current_idx + bucket_size]
                current_idx += bucket_size
                
                for fine_id in bucket_classes:
                    mapping[fine_id] = bucket_id
                    
        return mapping

# ==========================================
# 2. Helper Functions
# ==========================================
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    model.train()
    return 100. * correct / total

# ==========================================
# 3. Training Loop (Worker Function)
# ==========================================
def train_experiment(gpu_id, task_list, result_dict, total_steps=TOTAL_STEPS, batch_size=BATCH_SIZE, eval_interval=2000):
    # Set Device for this process
    device = torch.device(f"cuda:{gpu_id}")
    
    tqdm.write(f"[GPU {gpu_id}] Starting process. Assigned tasks: {len(task_list)}")
    
    transform_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for task_idx, task in enumerate(task_list):
        num_classes, mode = task
        
        # Check for existing logs to resume/skip
        log_path = f'training_logs/vit_history_m{num_classes}_{mode}.json'
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    prev_history = json.load(f)
                if 'test_acc' in prev_history and len(prev_history['test_acc']) > 0:
                    best_acc = max(prev_history['test_acc'])
                    tqdm.write(f"[GPU {gpu_id}] Skipping {mode} (m={num_classes}): Found log with Acc {best_acc:.2f}%")
                    result_dict[(mode, num_classes)] = best_acc
                    continue
            except Exception as e:
                tqdm.write(f"[GPU {gpu_id}] Log corrupt for {mode} (m={num_classes}). Re-running. Error: {e}")

        
        # Initialize Progress Bar for this specific task
        pbar = tqdm(total=total_steps, desc=f"G{gpu_id} | {mode[:3]} | m={num_classes}", position=gpu_id, leave=False)
        
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('training_logs', exist_ok=True)

        trainset = SuperclassCIFAR100(root='./data', target_classes=num_classes, mode=mode,
                                   train=True, download=True, transform=transform_train)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

        testset = SuperclassCIFAR100(root='./data', target_classes=num_classes, mode=mode,
                                  train=False, download=True, transform=transform_test)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)

        model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_classes)
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        history = {'steps': [], 'train_loss': [], 'eval_steps': [], 'test_acc': []}
        best_acc = 0.0
        model.train()
        steps = 0
        
        while steps < total_steps:
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()

                steps += 1
                pbar.update(1)
                
                if steps % 100 == 0:
                    history['steps'].append(steps)
                    history['train_loss'].append(loss.item())

                if steps % eval_interval == 0 or steps == total_steps:
                    acc = evaluate(model, testloader, device)
                    history['eval_steps'].append(steps)
                    history['test_acc'].append(acc)
                    
                    pbar.set_postfix({'acc': f"{acc:.1f}%"})
                    
                    if acc > best_acc:
                        best_acc = acc
                        # torch.save(model.state_dict(), f'checkpoints/vit_m{num_classes}_{mode}.pth')

                if steps >= total_steps:
                    break
        
        pbar.close()
        tqdm.write(f"[GPU {gpu_id}] Finished {mode} (m={num_classes}). Best Acc: {best_acc:.2f}%")
        
        # Save logs
        with open(log_path, 'w') as f:
            json.dump(history, f)
        
        # Store result in shared dictionary
        result_dict[(mode, num_classes)] = best_acc

# ==========================================
# 4. Fitting & Plotting
# ==========================================
def power_law_bias(x, a, alpha, b):
    return a * np.power(x, alpha) + b

def main():
    mp.set_start_method('spawn', force=True)
    
    class_counts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    modes = ['semantic', 'random']
    
    all_tasks = []
    for mode in modes:
        for m in class_counts:
            all_tasks.append((m, mode))
            
    # Distribute tasks
    gpu_tasks = {0: [], 1: [], 2: []}
    for i, task in enumerate(all_tasks):
        gpu_id = i % 3
        gpu_tasks[gpu_id].append(task)
        
    print("================================================================")
    print(f"EXPERIMENT: Scaling Laws on ViT-Tiny (Multi-GPU)")
    print(f"Total Steps per Run: {TOTAL_STEPS}")
    print("================================================================\n")

    manager = mp.Manager()
    result_dict = manager.dict()
    
    processes = []
    for gpu_id in [0, 1, 2]:
        p = mp.Process(target=train_experiment, args=(gpu_id, gpu_tasks[gpu_id], result_dict))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    print("\nAll processes completed. Generating plots...")

    # ---------------------------------------------------------
    # Process Results & Plot
    # ---------------------------------------------------------
    
    results = {'semantic': [], 'random': []}
    
    for mode in modes:
        for m in class_counts:
            acc = result_dict.get((mode, m), 0.0)
            results[mode].append(acc / 100.0)

    # Convert to numpy
    x_data_full = np.array(class_counts)
    y_sem_full = 1.0 - np.array(results['semantic']) # Error
    y_rnd_full = 1.0 - np.array(results['random'])   # Error

    # Include all data points (including m=10)
    x_data = x_data_full
    y_sem = y_sem_full
    y_rnd = y_rnd_full

    # Fit Power Law (Semantic Only)
    try:
        popt, pcov = curve_fit(power_law_bias, x_data, y_sem, p0=[0.01, 0.5, 0.1], maxfev=10000)
        fit_a, fit_alpha, fit_b = popt
        fit_str = f"${fit_a:.3f} \cdot m^{{{fit_alpha:.2f}}} + {fit_b:.2f}$"
    except Exception as e:
        print(f"Fitting failed: {e}")
        fit_a, fit_alpha, fit_b = 0, 0, 0
        fit_str = "Fit Failed"

    # ---------------------------------------------------------
    # STYLED PLOT
    # ---------------------------------------------------------
    
    # Update RC params for specific style
    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
        'axes.spines.top': False, 'axes.spines.right': False
    })

    plt.figure(figsize=(3, 2.5))
    
    # Semantic Data (Firebrick, Circles)
    plt.plot(x_data, y_sem, 'o', color='firebrick', markersize=8, label='Semantic')
    
    # Random Data (Royalblue, Squares)
    plt.plot(x_data, y_rnd, 's', color='royalblue', markersize=8, label='Random')

    # Fit Line
    if fit_a != 0:
        x_fit = np.linspace(min(class_counts), max(class_counts), 100)
        y_fit = power_law_bias(x_fit, fit_a, fit_alpha, fit_b)
        plt.plot(x_fit, y_fit, '--', color='firebrick', linewidth=2, label='_nolegend_')
        
        # Annotation
        mid_x = x_fit[50]
        mid_y = y_fit[50]
        #plt.text(mid_x, mid_y - 0.02, fit_str, color='firebrick', fontsize=12, fontweight='bold')

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlabel(r'Number of Classes $m$', fontsize=14)
    plt.ylabel('Test Error')
    plt.legend(frameon=False, loc='lower right', handletextpad=0.1, bbox_to_anchor=(1.05,-0.05))
    
    plt.tight_layout()
    plt.savefig('vit_scaling_comparison.pdf', dpi=300, bbox_inches='tight')
    print("\nSaved comparison plot to 'vit_scaling_comparison.pdf'")

if __name__ == "__main__":
    main()
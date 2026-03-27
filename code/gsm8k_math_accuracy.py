import torch
import torch.multiprocessing as mp
import re
import json
import numpy as np
import os
import math
import time
import shutil
import random
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# Import math_verify
try:
    from math_verify import parse, verify
except ImportError:
    raise ImportError("Please install math_verify: pip install math_verify")

# ==========================================
# Configuration
# ==========================================

# Models to evaluate
MODELS = {
    "Qwen": "Qwen/Qwen2.5-7B-Instruct",
    "Gemma": "google/gemma-3-1b-it"
}

# Hugging Face Token (Required for Gated Models like Gemma)
# Ensure you have accepted the terms at https://huggingface.co/google/gemma-3-4b-it
HF_TOKEN = os.getenv("HF_TOKEN") 

USE_4BIT = False
NUM_SAMPLES = None 
REPS = [0, 1, 2, 3, 4] # 5 Replicates

# Optimization Configs (all replicates are run in a single batch)
QUESTIONS_PER_BATCH = 2
USE_COMPILE = True 

# Dataset Configurations
# Mapped to specific short names for file generation as requested: "gsm8k", "math"
"""
DATASETS = {
    "GSM8K": {
        "hf_path": ("gsm8k", "main"),
        "split": "test",
        "col_q": "question",
        "col_a": "answer",
        "file_short_name": "gsm8k",
        "prompt_set": "GSM"
    }
}
"""
DATASETS = {
    "GSM8K": {
        "hf_path": ("gsm8k", "main"),
        "split": "test",
        "col_q": "question",
        "col_a": "answer",
        "file_short_name": "gsm8k",
        "prompt_set": "GSM"
    },
    "Math-500": {
        "hf_path": ("HuggingFaceH4/MATH-500",),
        "split": "test",
        "col_q": "problem",
        "col_a": "answer", 
        "file_short_name": "math",
        "prompt_set": "MATH"
    }
}

# ==========================================
# Prompts
# ==========================================

PROMPTS_MATH = {
    "Direct" : ("You are a helpful assistant. Solve the math problem. You should not show any work at all. Output the final answer in a box."),
    "L01" : ("You are a helpful assistant. Solve the math problem. Show your work. Only show important steps. Output the final answer in a box."),
    "L02" : ("You are a helpful assistant. Solve the math problem. Show your work step by step. Output the final answer in a box."),
    "L03" : ("You are a helpful assistant. Solve the math problem. Show your work step by step. Check each step to make sure it is correct. Output the final answer in a box."),
    "L04" : ("You are a helpful assistant. Solve the math problem. Show your work step by step. Explain each step. Explicitly check each step to make sure it is correct. Output the final answer in a box."),
    "L05" : ("You are a helpful assistant. Solve the math problem. Show your work step by step. Explain each step. Explicitly double-check each step to make sure it is correct. Output the final answer in a box.")
}

PROMPTS_GSM = {
    "Direct" : ("You are a helpful assistant. Solve the math problem. You should not show any work at all. Prepend #### to your final answer."),
    "L01" : ("You are a helpful assistant. Solve the math problem. Show your work. Only show important steps. Prepend #### to your final answer."),
    "L02" : ("You are a helpful assistant. Solve the math problem. Show your work step by step. Prepend #### to your final answer."),
    "L03" : ("You are a helpful assistant. Solve the math problem. Show your work step by step. Check each step to make sure it is correct. Prepend #### to your final answer."),
    "L04" : ("You are a helpful assistant. Solve the math problem. Show your work step by step. Explain each step. Explicitly check each step to make sure it is correct. Prepend #### to your final answer."),
    "L05" : ("You are a helpful assistant. Solve the math problem. Show your work step by step. Explain each step. Explicitly check each step to make sure it is correct. Then, check each step again to make sure it is correct. Prepend #### to your final answer.")
}

FEW_SHOT_GSM = {
    "Direct" : ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? \n #### 72 \
             \n Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn? \n #### 10 \n"),

    "L01" : ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\
          \n Natalia sold 48/2=24 clips in May. She sold 72 clips altogether in April and May. #### 72 \
             \n Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\
              \n Weng earns 12/60=1/5 dollars per minute. She earns 10 dollars in 50 minutes. #### 10 \n"),

    "L02" : ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\
          \n Natalia sold 48/2=24 clips in May. She sold 48 clips in April. She sold 48+24 = 72 clips altogether in April and May. #### 72 \
             \n Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\
              \n Weng earns 12/60 = 1/5 dollars per minute. In 50 minutes, she earns 1/5 dollars per minute*50 minutes=10 dollars. #### 10 \n"),

    "L03" : ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\
          \n  Natalia sold 48/2=24 clips in May. Checking, half of 48 is 24. She sold 48 clips in April. She sold 48+24 = 72 clips altogether in April and May. Checking, 48+24=72 is correct. #### 72 \
             \n Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\
              \n Weng earns 12/60 = 1/5 dollars per minute. Checking, 12/60=1/5 because 60/12=5. In 50 minutes, she earns 1/5 dollars per minute*50 minutes=10 dollars. Checking, we must multiply time working with the earning rate, 50 * 1/5=10 is correct. #### 10 \n"),

    "L04" : ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\
          \n I should calculate how many clips Natalia sold in May. Natalia sold 48/2=24 clips in May. Checking, half of 48 is 24. She sold 48 clips in April. The total will be the sum of the clips sold in April and May. She sold 48+24 = 72 clips altogether in April and May. Checking, 48+24=72 is correct. #### 72 \
             \n Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\
              \n I should calculate Weng's earning rate per minute. Weng earns 12/60 = 1/5 dollars per minute. Checking, 12/60=1/5 because 60/12=5. To calculate the money earned, I must multiply the earning rate with the time spent working. In 50 minutes, she earns 1/5 dollars per minute*50 minutes=10 dollars. Checking, I must multiply time working with the earning rate, 50 * 1/5=10 is correct. #### 10 \n"),

    "L05" : ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\
          \n I should calculate how many clips Natalia sold in May. Natalia sold 48/2=24 clips in May. Checking, half of 48 is 24. Checking again, 48 divided by 2 is 24. She sold 48 clips in April. The total will be the sum of the clips sold in April and May. She sold 48+24 = 72 clips altogether in April and May. Checking, 48+24=72 is correct. Checking again, the sum of 48 and 24 is 72. #### 72 \
             \n Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\
              \n I should calculate Weng's earning rate per minute. Weng earns 12/60 = 1/5 dollars per minute. Checking, 12/60=1/5 because 60/12=5. Checking again, 12/60=0.2 which is equal to 1/5. To calculate the money earned, I must multiply the earning rate with the time spent working. In 50 minutes, she earns 1/5 dollars per minute*50 minutes=10 dollars. Checking, I must multiply time working with the earning rate, 50*1/5=10 is correct. Checking again, 50*1/5 is 10. #### 10 \n")
}

FEW_SHOT_MATH = {
    "Direct" : (r"Let \[f(x) = \left\{ \begin{array}{cl} ax+3, &\text{ if }x>2, \\ x-5 &\text{ if } -2 \le x \le 2, \\ 2x-b &\text{ if } x <-2. \end{array} \right.\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper). \
        \n $\boxed{0}$\
        \n What is the value of $9^3 + 3(9^2) + 3(9) + 1$?\
        \n $\boxed{1000}$"),
     
    "L01" : (r"Let \[f(x) = \left\{ \begin{array}{cl} ax+3, &\text{ if }x>2, \\ x-5 &\text{ if } -2 \le x \le 2, \\ 2x-b &\text{ if } x <-2. \end{array} \right.\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper). \
        \n For the piecewise function to be continuous, the cases must meet at $2$ and $-2$. This means $ax+3$ must equal $x-5$ when $x=2$ and, therefore, $a=-3$. Similarly, $x-5$ must equal $2x-b$ at $x=-2$, which implies $b=3$. Then, the sum $a+b=0$. \boxed{0}$\
        \n What is the value of $9^3 + 3(9^2) + 3(9) + 1$?\
        \n This expression is a cubic polynomial with coefficients $(1,3,3,1)$ in decreasing order of degree. This means the expression is equal to $(9+1)^3$. Thus, its value is $10^3$. \boxed{1000}$"),
     
    "L02" : (r"Let \[f(x) = \left\{ \begin{array}{cl} ax+3, &\text{ if }x>2, \\ x-5 &\text{ if } -2 \le x \le 2, \\ 2x-b &\text{ if } x <-2. \end{array} \right.\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper). \
        \n For the piecewise function to be continuous, the cases must meet at $2$ and $-2$. This means $ax+3$ must equal $x-5$ when $x=2$. Therefore, $2a+3=2-5$ and $2a+3=-3\rightarrow $2a=-6$ so $a=-3$. Similarly, $x-5$ must equal $2x-b$ at $x=-2$. This means $2(-2)-b=-2-5$ so $-4-b=-7\rightarrow -b=-3$ so $b=3$. Then, the sum $a+b=0$. \boxed{0}$\
        \n What is the value of $9^3 + 3(9^2) + 3(9) + 1$?\
        \n This expression is a cubic polynomial with coefficients $(1,3,3,1)$ in decreasing order of degree. The polynomial $(x+1)^3=x^3+3x^2+3x+1$ has the same coefficients. This means the expression is equal to $(x+1)^3$ for $x=9$ which is $(9+1)^3. Thus, its value is $10^3$. \boxed{1000}$"),

    "L03" : (r"Let \[f(x) = \left\{ \begin{array}{cl} ax+3, &\text{ if }x>2, \\ x-5 &\text{ if } -2 \le x \le 2, \\ 2x-b &\text{ if } x <-2. \end{array} \right.\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper). \
        \n For the piecewise function to be continuous, the cases must meet at $2$ and $-2$. This means $ax+3$ must equal $x-5$ when $x=2$. Therefore, $2a+3=2-5$ and $2a+3=-3\rightarrow $2a=-6$ so $a=-3$. Checking by substitution, $2a+3=2-5$ means $2(-3)+3=-3\rightarrow -6+3=-3$ which is a true expression. Similarly, $x-5$ must equal $2x-b$ at $x=-2$. This means $2(-2)-b=-2-5$ so $-4-b=-7\rightarrow -b=-3$ so $b=3$. Checking by substitution, $2(-2)-3=-2-5\rightarrow -4-3=-7$ which is a true expression. Then, the sum $a+b=0$. \boxed{0}$\
        \n What is the value of $9^3 + 3(9^2) + 3(9) + 1$?\
        \n This expression is a cubic polynomial with coefficients $(1,3,3,1)$ in decreasing order of degree. The polynomial $(x+1)^3=x^3+3x^2+3x+1$ has the same coefficients. Checking, $(x+1)^3=(x+1)(x+1)(x+1)=(x+1)(x^2+2x+1)=x^3+2x^2+x+x^2+2x+1=x^3+3x^2+3x+1$ which yields the correct coefficients. This means the expression is equal to $(x+1)^3$ for $x=9$ which is $(9+1)^3. Thus, its value is $10^3$. \boxed{1000}$"),

    "L04" : (r"Let \[f(x) = \left\{ \begin{array}{cl} ax+3, &\text{ if }x>2, \\ x-5 &\text{ if } -2 \le x \le 2, \\ 2x-b &\text{ if } x <-2. \end{array} \right.\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper). \
        \n For the piecewise function to be continuous, the cases must meet at $2$ and $-2$. This means $ax+3$ must equal $x-5$ when $x=2$. This is because the piecewise function is equal to $ax+3$ above $x=2$ and is equal to $x-5$ below $x=2$. Both lines must meet at exactly $x=2$ for the function to be continuous.  Therefore, $2a+3=2-5$ and $2a+3=-3\rightarrow $2a=-6$ so $a=-3$. Checking by substitution, $2a+3=2-5$ means $2(-3)+3=-3\rightarrow -6+3=-3$ which is a true expression. Similarly, $x-5$ must equal $2x-b$ at $x=-2$. This is because the piecewise function is equal to $x-5$ for $x$ immidiately larger than $-2$, and the function is equal to $2x-b$ when $x$ is smaller than $-2$. Both lines must meet at $x=-2$. This means $2(-2)-b=-2-5$ so $-4-b=-7\rightarrow -b=-3$ so $b=3$. Checking by substitution, $2(-2)-3=-2-5\rightarrow -4-3=-7$ which is a true expression. Then, the sum $a+b=0$. \boxed{0}$\
        \n What is the value of $9^3 + 3(9^2) + 3(9) + 1$?\
        \n This expression is a cubic polynomial with coefficients $(1,3,3,1)$ in decreasing order of degree. Identifying the polynomial nature in the expression will help us solve it. Given that the expression is a polynomial, we can try to factorize it. The polynomial $(x+1)^3=x^3+3x^2+3x+1$ has the same coefficients as the expression. Checking, $(x+1)^3=(x+1)(x+1)(x+1)=(x+1)(x^2+2x+1)=x^3+2x^2+x+x^2+2x+1=x^3+3x^2+3x+1$ which yields the correct coefficients. This means the expression is equal to $(x+1)^3$ for $x=9$ which is $(9+1)^3. By writing the expression in this factorized form, we can evaluate the entire expression by computing the simplified sum $9+1=10$ within the parentheses. Thus, its value is $10^3$. \boxed{1000}$"),

    "L05" : (r"Let \[f(x) = \left\{ \begin{array}{cl} ax+3, &\text{ if }x>2, \\ x-5 &\text{ if } -2 \le x \le 2, \\ 2x-b &\text{ if } x <-2. \end{array} \right.\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper). \
        \n For the piecewise function to be continuous, the cases must meet at $2$ and $-2$. This means $ax+3$ must equal $x-5$ when $x=2$. This is because the piecewise function is equal to $ax+3$ above $x=2$ and is equal to $x-5$ below $x=2$. Both lines must meet at exactly $x=2$ for the function to be continuous.  Therefore, $2a+3=2-5$ and $2a+3=-3\rightarrow $2a=-6$ so $a=-3$. Checking by substitution, $2a+3=2-5$ means $2(-3)+3=-3\rightarrow -6+3=-3$ which is a true expression. Checking again, $2a+3=2-5\rightarrow 2a+3=-3\rightarrow $2a=-6$ so $a=-3$. Similarly, $x-5$ must equal $2x-b$ at $x=-2$. This is because the piecewise function is equal to $x-5$ for $x$ immidiately larger than $-2$, and the function is equal to $2x-b$ when $x$ is smaller than $-2$. Both lines must meet at $x=-2$. This means $2(-2)-b=-2-5$ so $-4-b=-7\rightarrow -b=-3$ so $b=3$. Checking by substitution, $2(-2)-3=-2-5\rightarrow -4-3=-7$ which is a true expression. Checking again, $2(-2)-b=-2-5\rightarrow -4-b=-7\rightarrow -b=-3$ so $b=3$. Then, the sum $a+b=0$. \boxed{0}$\
        \n What is the value of $9^3 + 3(9^2) + 3(9) + 1$?\
        \n This expression is a cubic polynomial with coefficients $(1,3,3,1)$ in decreasing order of degree. Identifying the polynomial nature in the expression will help us solve it. Given that the expression is a polynomial, we can try to factorize it. The polynomial $(x+1)^3=x^3+3x^2+3x+1$ has the same coefficients as the expression. Checking, $(x+1)^3=(x+1)(x+1)(x+1)=(x+1)(x^2+2x+1)=x^3+2x^2+x+x^2+2x+1=x^3+3x^2+3x+1$ which yields the correct coefficients. Checking again, $(x+1)^3$ can be expanded to be (x+1)(x+1)(x+1)=(x^2+2x+1)(x+1)=x^3+2x^2+x+x^2+2x+1=x^3+3x^2+3x+1$, matching the coefficients in the expression. This means the expression is equal to $(x+1)^3$ for $x=9$ which is $(9+1)^3. By writing the expression in this factorized form, we can evaluate the entire expression by computing the simplified sum $9+1=10$ within the parentheses. Thus, its value is $10^3$. \boxed{1000}$")
}

def get_prompts_for_dataset(dataset_name):
    if DATASETS[dataset_name]["prompt_set"] == "GSM":
        return PROMPTS_GSM
    return PROMPTS_MATH

def get_output_path(dataset_name, model_key):
    ds_short = DATASETS[dataset_name]["file_short_name"]
    model_short = model_key.lower()
    return os.path.join("results", f"{ds_short}_{model_short}.json")

def get_summary_path(dataset_name, model_key):
    ds_short = DATASETS[dataset_name]["file_short_name"]
    model_short = model_key.lower()
    return os.path.join("summary", f"{ds_short}_{model_short}_summary_stats.json")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_max_tokens(level_key):
    if level_key == "Direct": return 1024
    if int(level_key[1:3]) <= 3: return 1024
    if int(level_key[1:3]) <= 6: return 2048
    return 8192

def clean_gsm_ground_truth(text):
    if "####" in str(text):
        return text.split("####")[-1].strip()
    return str(text)

def save_atomic(data, filename):
    temp_filename = filename + ".tmp"
    try:
        with open(temp_filename, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(temp_filename, filename)
    except Exception as e:
        print(f"Failed to save: {e}")

def worker_process(rank, gpu_id, dataset_name, model_key, indices, result_queue, incomplete_map):
    try:
        device = torch.device(f"cuda:{gpu_id}")
        dataset_config = DATASETS[dataset_name]
        current_prompts = get_prompts_for_dataset(dataset_name)
        sorted_levels = sorted(current_prompts.keys())
        model_path = MODELS[model_key]

        # Progress Bar
        pbar = tqdm(total=len(indices), 
                    position=rank, 
                    desc=f"GPU {gpu_id} [{dataset_name} | {model_key}]", 
                    leave=True)

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, token=HF_TOKEN)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Model Loading Optimization
        model_kwargs = {
            "trust_remote_code": True, 
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
            "token": HF_TOKEN
        }

        if USE_4BIT:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if not USE_4BIT:
            model = model.to(device)
            
        model.eval()

        if USE_COMPILE:
            try:
                import logging
                torch._logging.set_logs(dynamo=logging.ERROR) 
                model = torch.compile(model, mode="reduce-overhead")
            except Exception as e:
                pass

        dataset = load_dataset(*dataset_config['hf_path'], split=dataset_config['split'])

        # Process in batches
        for i in range(0, len(indices), QUESTIONS_PER_BATCH):
            batch_indices = indices[i : i + QUESTIONS_PER_BATCH]
            batch_data = dataset.select(batch_indices)
            
            questions = batch_data[dataset_config['col_q']]
            answers = batch_data[dataset_config['col_a']]
            
            # Setup Results Storage
            batch_entries = []
            
            # Pre-initialize result structure
            for j, q_idx in enumerate(batch_indices):
                q_idx_int = int(q_idx)
                raw_gt = str(answers[j])
                clean_gt = clean_gsm_ground_truth(raw_gt) if "GSM8K" in dataset_name else raw_gt
                batch_entries.append({
                    "id": q_idx_int,
                    "question": questions[j],
                    "ground_truth_clean": clean_gt,
                    "results": {} # Will populate below
                })

            for condition_name in sorted_levels:
                system_prompt = current_prompts[condition_name]
                max_tokens = get_max_tokens(condition_name)

                # Prepare Prompts
                all_prompts = []
                # Map flattened index back to (batch_local_index, seed)
                map_idx_to_meta = [] 

                # Few shot logic
                few_shot_prefix = ""
                if DATASETS[dataset_name]["prompt_set"] == "GSM" and condition_name in FEW_SHOT_GSM:
                    few_shot_prefix = FEW_SHOT_GSM[condition_name]
                elif DATASETS[dataset_name]["prompt_set"] == "MATH" and condition_name in FEW_SHOT_MATH:
                    few_shot_prefix = FEW_SHOT_MATH[condition_name]


                # Filter which questions in this batch actually NEED this condition
                count_needed = 0

                for j, q_idx in enumerate(batch_indices):
                    q_idx_int = int(q_idx)
                    
                    # CHECK: Do we need this level for this ID?
                    # If ID is in incomplete_map, check if condition is in the missing set.
                    # If ID is NOT in incomplete_map (it's new), we need everything.
                    needs_run = True
                    if q_idx_int in incomplete_map:
                        if condition_name not in incomplete_map[q_idx_int]:
                            needs_run = False
                    
                    if not needs_run:
                        continue

                    count_needed += 1
                    q = questions[j]
                    final_user_content = few_shot_prefix + q
                    prompt_messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": final_user_content}
                    ]
                    text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
                    
                    # Replicates
                    for rep in REPS:
                        all_prompts.append(text)
                        map_idx_to_meta.append((j, rep)) # j is index in batch_entries

                if count_needed == 0:
                    continue

                # Tokenize all needed
                inputs = tokenizer(
                    all_prompts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=8192 
                ).to(device)

                # Generate
                set_seed(42) 
                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=1.0, 
                        top_p=0.9,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id
                    )

                input_len = inputs.input_ids.shape[1]
                output_ids = generated_ids[:, input_len:]
                decoded_responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                # Distribute results
                for k, response_text in enumerate(decoded_responses):
                    batch_local_idx, rep = map_idx_to_meta[k]

                    # Count tokens in answer.
                    # If pad_token_id == eos_token_id, the 'padding' is essentially the sequence of EOS tokens 
                    # that follow the first EOS token. We must count up to and including the first EOS.
                    # If pad_token_id != eos_token_id, we simply count tokens that are not padding.
                    
                    pad_id = tokenizer.pad_token_id
                    eos_id = tokenizer.eos_token_id
                    
                    # Case where ids are None
                    if pad_id is None and eos_id is not None:
                        pad_id = eos_id # Fallback behavior

                    token_count = 0
                    if pad_id is not None and pad_id == eos_id:
                        # Find the first occurrence of the EOS token.
                        # output_ids[k] contains the generated sequence.
                        # We want the length to be index_of_first_eos + 1.
                        # If no EOS found, it means it hit max_length, so use full length.
                        inds = (output_ids[k] == eos_id).nonzero(as_tuple=True)[0]
                        if len(inds) > 0:
                            token_count = inds[0].item() + 1
                        else:
                            token_count = len(output_ids[k])
                    else:
                        # Standard case: pad and eos are different.
                        # Simply count everything that isn't a padding token.
                        if pad_id is not None:
                            token_count = (output_ids[k] != pad_id).sum().item()
                        else:
                            token_count = len(output_ids[k])
                    
                    token_count = int(token_count)

                    truncated = token_count >= max_tokens
                    
                    gt_clean = batch_entries[batch_local_idx]["ground_truth_clean"]
                    
                    try:
                        parsed_answer = parse(response_text)
                        is_correct = verify(gt_clean, parsed_answer)
                    except Exception:
                        is_correct = False

                    result_entry = {
                        "replicate": rep,
                        "response_text": response_text,
                        "token_count": int(token_count),
                        "truncated": truncated,
                        "is_correct": bool(is_correct)
                    }
                    
                    if condition_name not in batch_entries[batch_local_idx]["results"]:
                        batch_entries[batch_local_idx]["results"][condition_name] = []
                    
                    batch_entries[batch_local_idx]["results"][condition_name].append(result_entry)
                
                # Sort replicates
                for entry in batch_entries:
                    if condition_name in entry["results"]:
                        entry["results"][condition_name].sort(key=lambda x: x["replicate"])

            pbar.update(len(batch_indices)) 
            result_queue.put(batch_entries)

        pbar.close()
        result_queue.put("DONE")

    except Exception as e:
        tqdm.write(f"[Worker {rank}] Error: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put("DONE")

def plot_question_performance(dataset_name):
    # Set plot style parameters
    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 10,
        'axes.spines.top': False
    })

    prompt_keys = sorted(get_prompts_for_dataset(dataset_name).keys())
    
    # Increase width slightly to accommodate dual axes
    fig, ax1 = plt.subplots(figsize=(5, 2.5))
    
    # Specific colors for models
    colors = {"Qwen": "darkblue", "Gemma": "darkred"}
    
    # Helper to retrieve processed data for a model
    def get_model_data(m_key):
        output_file = get_output_path(dataset_name, m_key)
        if not os.path.exists(output_file):
            return [], [], []
            
        with open(output_file, 'r') as f: data = json.load(f)
        if not data: return [], [], []

        plot_points = []
        summary_stats = {}

        for prompt_key in prompt_keys:
            # We need to compute statistics across replicates for error bars
            # Gather data per replicate
            rep_stats = {r: {"correct": 0, "total": 0} for r in REPS}
            all_tokens_for_prompt = []
            
            has_data = False

            for entry in data:
                if prompt_key not in entry["results"]: continue
                replicates = entry["results"][prompt_key]
                if not replicates: continue
                
                has_data = True
                
                for r_entry in replicates:
                    r_idx = r_entry["replicate"]
                    if r_idx in rep_stats:
                        rep_stats[r_idx]["total"] += 1
                        if r_entry["is_correct"]:
                            rep_stats[r_idx]["correct"] += 1
                        all_tokens_for_prompt.append(r_entry["token_count"])
            
            if not has_data: continue

            # Calculate Error Rates per Replicate
            replicate_error_rates = []
            for r in REPS:
                total = rep_stats[r]["total"]
                if total > 0:
                    error_rate = 1.0 - (rep_stats[r]["correct"] / total)
                    replicate_error_rates.append(error_rate)
            
            if not replicate_error_rates: continue

            # Compute Mean and Standard Error of the Mean (SEM)
            mean_error_rate = np.mean(replicate_error_rates)
            if len(replicate_error_rates) > 1:
                sem_error_rate = np.std(replicate_error_rates, ddof=1) / np.sqrt(len(replicate_error_rates))
            else:
                sem_error_rate = 0.0

            avg_tokens = np.mean(all_tokens_for_prompt) if all_tokens_for_prompt else 0

            summary_stats[prompt_key] = {
                "error_rate_mean": mean_error_rate, 
                "error_rate_sem": sem_error_rate,
                "average_tokens": avg_tokens
            }

            if prompt_key != "Direct":
                plot_points.append((avg_tokens, mean_error_rate, sem_error_rate))

        # Sort points by token length (x-axis)
        plot_points.sort(key=lambda p: p[0])
        
        if plot_points:
            x_val, y_val, y_err = zip(*plot_points)
        else:
            x_val, y_val, y_err = [], [], []
            
        # Save summary stats side-effect
        summary_path = get_summary_path(dataset_name, m_key)
        with open(summary_path, "w") as f:
            json.dump(summary_stats, f, indent=4)
            
        return x_val, y_val, y_err

    # Plot Qwen on Left Axis (ax1)
    if "Qwen" in MODELS:
        x_q, y_q, err_q = get_model_data("Qwen")
        if x_q:
            color = colors["Qwen"]
            ax1.errorbar(x_q, y_q, yerr=err_q, fmt='-o', color=color, linewidth=2, label="Qwen", zorder=2, capsize=3, markersize=6)
            ax1.set_ylabel("Test Error (Qwen)", color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.spines['left'].set_color(color)

    ax1.set_xlabel("Reasoning Tokens")
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Plot Gemma on Right Axis (ax2)
    if "Gemma" in MODELS:
        ax2 = ax1.twinx()
        x_g, y_g, err_g = get_model_data("Gemma")
        if x_g:
            color = colors["Gemma"]
            ax2.errorbar(x_g, y_g, yerr=err_g, fmt='-o', color=color, linewidth=2, label="Gemma", zorder=2, capsize=3, markersize=6)
            ax2.set_ylabel("Test Error (Gemma)", color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.spines['right'].set_color(color)
            
            # Remove top spine for aesthetic consistency on the secondary axis
            ax2.spines['top'].set_visible(False)

    # Combined Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = (ax2.get_legend_handles_labels()) if 'ax2' in locals() else ([], [])
    
    #if lines1 or lines2:
     #   ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    # Annotate with dataset name
    #plt.text(0.95, 0.95, dataset_name, transform=ax1.transAxes, 
     #        horizontalalignment='center', verticalalignment='top', 
      #       fontsize=12, color='black')

    plt.tight_layout()
    
    short_name = DATASETS[dataset_name]["file_short_name"]
    plt.savefig(f"{short_name}_comparison_error_vs_length.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def process_dataset(dataset_name, model_key):
    config = DATASETS[dataset_name]
    output_file = get_output_path(dataset_name, model_key)
    
    print(f"\n=== Processing Dataset: {dataset_name} | Model: {model_key} ===")
    print(f"Output file: {output_file}")

    existing_results = []
    incomplete_map = {} # Map ID -> Set of missing keys
    
    dataset_prompts = get_prompts_for_dataset(dataset_name)
    required_keys = set(dataset_prompts.keys())

    # MODIFIED: Load existing and check for PARTIAL completeness
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                existing_results = json.load(f)
            
            print(f"Loaded {len(existing_results)} existing entries. Checking for missing keys...")
            
            for entry in existing_results:
                existing_keys = set(entry["results"].keys())
                missing_keys = required_keys - existing_keys
                if missing_keys:
                    # It exists, but is missing L07 (or others)
                    incomplete_map[entry["id"]] = missing_keys
        
        except Exception:
            pass

    # Existing IDs that are totally complete
    fully_complete_ids = {entry["id"] for entry in existing_results if entry["id"] not in incomplete_map}

    dataset_meta = load_dataset(*config['hf_path'], split=config['split'])
    total_count = len(dataset_meta)
    target_count = NUM_SAMPLES if NUM_SAMPLES is not None else total_count
    
    all_indices = list(range(target_count))
    
    # Process if it's new OR if it's in incomplete_map
    indices_to_process = [i for i in all_indices if (i not in fully_complete_ids)]

    if not indices_to_process:
        print(f"All samples fully processed for {dataset_name} ({model_key}).")
        return

    print(f"Processing {len(indices_to_process)} samples (some may be backfills).")

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs found!")
        return 

    chunk_size = math.ceil(len(indices_to_process) / num_gpus)
    chunks = [indices_to_process[i:i + chunk_size] for i in range(0, len(indices_to_process), chunk_size)]
    
    manager = mp.Manager()
    result_queue = manager.Queue()
    # Pass incomplete_map to workers so they know what to skip
    
    processes = []
    active_workers = 0

    for rank, chunk in enumerate(chunks):
        if not chunk: continue
        gpu_id = rank % num_gpus
        p = mp.Process(target=worker_process, args=(rank, gpu_id, dataset_name, model_key, chunk, result_queue, incomplete_map))
        p.start()
        processes.append(p)
        active_workers += 1
        
    finished_workers = 0
    try:
        while finished_workers < active_workers:
            message = result_queue.get()
            if message == "DONE":
                finished_workers += 1
            else:
                new_batch = message
                
                # Convert existing_results to a dict for fast lookup during merge
                existing_map = {e["id"]: e for e in existing_results}
                
                for new_item in new_batch:
                    item_id = new_item["id"]
                    if item_id in existing_map:
                        # Merge the new keys into the old entry
                        existing_map[item_id]["results"].update(new_item["results"])
                    else:
                        # It's a brand new item
                        existing_results.append(new_item)
                        existing_map[item_id] = new_item # Add to map to keep consistent
                
                # Re-sort list by ID
                existing_results.sort(key=lambda x: x["id"])
                save_atomic(existing_results, output_file)
                
    except KeyboardInterrupt:
        print("\nInterrupted! Terminating workers...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
        return

    for p in processes:
        p.join()

def run_experiment():
    torch.set_float32_matmul_precision('high')
    
    # Create directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("summary", exist_ok=True)

    # Run processing
    for ds_name in DATASETS.keys():
        for model_key in MODELS.keys():
            process_dataset(ds_name, model_key)
            
    print("\n=== Experiment Complete ===")
    
    # Run plotting (aggregates both models)
    for ds_name in DATASETS.keys():
        plot_question_performance(ds_name)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    run_experiment()
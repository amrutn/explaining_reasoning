import asyncio
import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Import math_verify
try:
    from math_verify import parse, verify
except ImportError:
    raise ImportError("Please install math_verify: pip install math_verify")

# ==========================================
# Configuration
# ==========================================

MODELS = {
    "gemini-flash": {
        "provider": "google",
        "model_id": "gemini-2.5-flash",
    },
    "deepseek-v3": {
        "provider": "deepseek",
        "model_id": "deepseek-chat",
    },
}

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Dataset Configurations
DATASETS = {
    "GSM8K": {
        "hf_path": ("gsm8k", "main"),
        "split": "test",
        "col_q": "question",
        "col_a": "answer",
        "file_short_name": "gsm8k",
        "prompt_set": "GSM",
    },
    "MATH-500": {
        "hf_path": ("HuggingFaceH4/MATH-500",),
        "split": "test",
        "col_q": "problem",
        "col_a": "answer",
        "file_short_name": "math",
        "prompt_set": "MATH",
    },
}

REPS = [0]

# Per-provider concurrency limits
PROVIDER_SEMAPHORES = {}

# ==========================================
# Prompts
# ==========================================

PROMPTS_GSM = {
    "Direct": "You are a helpful assistant. Solve the math problem. You should not show any work at all. Prepend #### to your final answer.",
    "L01": "You are a helpful assistant. Solve the math problem. Show your work. Only show important steps. Prepend #### to your final answer.",
    "L02": "You are a helpful assistant. Solve the math problem. Show your work step by step. Prepend #### to your final answer.",
    "L03": "You are a helpful assistant. Solve the math problem. Show your work step by step. Check each step to make sure it is correct. Prepend #### to your final answer.",
    "L04": "You are a helpful assistant. Solve the math problem. Show your work step by step. Explain each step. Explicitly check each step to make sure it is correct. Prepend #### to your final answer.",
    "L05": "You are a helpful assistant. Solve the math problem. Show your work step by step. Explain each step. Explicitly check each step to make sure it is correct. Then, check each step again to make sure it is correct. Prepend #### to your final answer.",
}

PROMPTS_MATH = {
    "Direct": "You are a helpful assistant. Solve the math problem. You should not show any work at all. Output the final answer in a box.",
    "L01": "You are a helpful assistant. Solve the math problem. Show your work. Only show important steps. Output the final answer in a box.",
    "L02": "You are a helpful assistant. Solve the math problem. Show your work step by step. Output the final answer in a box.",
    "L03": "You are a helpful assistant. Solve the math problem. Show your work step by step. Check each step to make sure it is correct. Output the final answer in a box.",
    "L04": "You are a helpful assistant. Solve the math problem. Show your work step by step. Explain each step. Explicitly check each step to make sure it is correct. Output the final answer in a box.",
    "L05": "You are a helpful assistant. Solve the math problem. Show your work step by step. Explain each step. Explicitly double-check each step to make sure it is correct. Output the final answer in a box.",
}

# ==========================================
# Few-Shot Examples
# ==========================================

FEW_SHOT_GSM = {
    "Direct": ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? \n #### 72 "
               "\n Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn? \n #### 10 \n"),

    "L01": ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
            "\n Natalia sold 48/2=24 clips in May. She sold 72 clips altogether in April and May. #### 72 "
            "\n Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"
            "\n Weng earns 12/60=1/5 dollars per minute. She earns 10 dollars in 50 minutes. #### 10 \n"),

    "L02": ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
            "\n Natalia sold 48/2=24 clips in May. She sold 48 clips in April. She sold 48+24 = 72 clips altogether in April and May. #### 72 "
            "\n Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"
            "\n Weng earns 12/60 = 1/5 dollars per minute. In 50 minutes, she earns 1/5 dollars per minute*50 minutes=10 dollars. #### 10 \n"),

    "L03": ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
            "\n  Natalia sold 48/2=24 clips in May. Checking, half of 48 is 24. She sold 48 clips in April. She sold 48+24 = 72 clips altogether in April and May. Checking, 48+24=72 is correct. #### 72 "
            "\n Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"
            "\n Weng earns 12/60 = 1/5 dollars per minute. Checking, 12/60=1/5 because 60/12=5. In 50 minutes, she earns 1/5 dollars per minute*50 minutes=10 dollars. Checking, we must multiply time working with the earning rate, 50 * 1/5=10 is correct. #### 10 \n"),

    "L04": ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
            "\n I should calculate how many clips Natalia sold in May. Natalia sold 48/2=24 clips in May. Checking, half of 48 is 24. She sold 48 clips in April. The total will be the sum of the clips sold in April and May. She sold 48+24 = 72 clips altogether in April and May. Checking, 48+24=72 is correct. #### 72 "
            "\n Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"
            "\n I should calculate Weng's earning rate per minute. Weng earns 12/60 = 1/5 dollars per minute. Checking, 12/60=1/5 because 60/12=5. To calculate the money earned, I must multiply the earning rate with the time spent working. In 50 minutes, she earns 1/5 dollars per minute*50 minutes=10 dollars. Checking, I must multiply time working with the earning rate, 50 * 1/5=10 is correct. #### 10 \n"),

    "L05": ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
            "\n I should calculate how many clips Natalia sold in May. Natalia sold 48/2=24 clips in May. Checking, half of 48 is 24. Checking again, 48 divided by 2 is 24. She sold 48 clips in April. The total will be the sum of the clips sold in April and May. She sold 48+24 = 72 clips altogether in April and May. Checking, 48+24=72 is correct. Checking again, the sum of 48 and 24 is 72. #### 72 "
            "\n Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"
            "\n I should calculate Weng's earning rate per minute. Weng earns 12/60 = 1/5 dollars per minute. Checking, 12/60=1/5 because 60/12=5. Checking again, 12/60=0.2 which is equal to 1/5. To calculate the money earned, I must multiply the earning rate with the time spent working. In 50 minutes, she earns 1/5 dollars per minute*50 minutes=10 dollars. Checking, I must multiply time working with the earning rate, 50*1/5=10 is correct. Checking again, 50*1/5 is 10. #### 10 \n"),
}

FEW_SHOT_MATH = {
    "Direct": (r"Let \[f(x) = \left\{ \begin{array}{cl} ax+3, &\text{ if }x>2, \\ x-5 &\text{ if } -2 \le x \le 2, \\ 2x-b &\text{ if } x <-2. \end{array} \right.\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper). "
               r"\n $\boxed{0}$"
               r"\n What is the value of $9^3 + 3(9^2) + 3(9) + 1$?"
               r"\n $\boxed{1000}$"),

    "L01": (r"Let \[f(x) = \left\{ \begin{array}{cl} ax+3, &\text{ if }x>2, \\ x-5 &\text{ if } -2 \le x \le 2, \\ 2x-b &\text{ if } x <-2. \end{array} \right.\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper). "
             r"\n For the piecewise function to be continuous, the cases must meet at $2$ and $-2$. This means $ax+3$ must equal $x-5$ when $x=2$ and, therefore, $a=-3$. Similarly, $x-5$ must equal $2x-b$ at $x=-2$, which implies $b=3$. Then, the sum $a+b=0$. \boxed{0}$"
             r"\n What is the value of $9^3 + 3(9^2) + 3(9) + 1$?"
             r"\n This expression is a cubic polynomial with coefficients $(1,3,3,1)$ in decreasing order of degree. This means the expression is equal to $(9+1)^3$. Thus, its value is $10^3$. \boxed{1000}$"),

    "L02": (r"Let \[f(x) = \left\{ \begin{array}{cl} ax+3, &\text{ if }x>2, \\ x-5 &\text{ if } -2 \le x \le 2, \\ 2x-b &\text{ if } x <-2. \end{array} \right.\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper). "
             r"\n For the piecewise function to be continuous, the cases must meet at $2$ and $-2$. This means $ax+3$ must equal $x-5$ when $x=2$. Therefore, $2a+3=2-5$ and $2a+3=-3\rightarrow $2a=-6$ so $a=-3$. Similarly, $x-5$ must equal $2x-b$ at $x=-2$. This means $2(-2)-b=-2-5$ so $-4-b=-7\rightarrow -b=-3$ so $b=3$. Then, the sum $a+b=0$. \boxed{0}$"
             r"\n What is the value of $9^3 + 3(9^2) + 3(9) + 1$?"
             r"\n This expression is a cubic polynomial with coefficients $(1,3,3,1)$ in decreasing order of degree. The polynomial $(x+1)^3=x^3+3x^2+3x+1$ has the same coefficients. This means the expression is equal to $(x+1)^3$ for $x=9$ which is $(9+1)^3. Thus, its value is $10^3$. \boxed{1000}$"),

    "L03": (r"Let \[f(x) = \left\{ \begin{array}{cl} ax+3, &\text{ if }x>2, \\ x-5 &\text{ if } -2 \le x \le 2, \\ 2x-b &\text{ if } x <-2. \end{array} \right.\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper). "
             r"\n For the piecewise function to be continuous, the cases must meet at $2$ and $-2$. This means $ax+3$ must equal $x-5$ when $x=2$. Therefore, $2a+3=2-5$ and $2a+3=-3\rightarrow $2a=-6$ so $a=-3$. Checking by substitution, $2a+3=2-5$ means $2(-3)+3=-3\rightarrow -6+3=-3$ which is a true expression. Similarly, $x-5$ must equal $2x-b$ at $x=-2$. This means $2(-2)-b=-2-5$ so $-4-b=-7\rightarrow -b=-3$ so $b=3$. Checking by substitution, $2(-2)-3=-2-5\rightarrow -4-3=-7$ which is a true expression. Then, the sum $a+b=0$. \boxed{0}$"
             r"\n What is the value of $9^3 + 3(9^2) + 3(9) + 1$?"
             r"\n This expression is a cubic polynomial with coefficients $(1,3,3,1)$ in decreasing order of degree. The polynomial $(x+1)^3=x^3+3x^2+3x+1$ has the same coefficients. Checking, $(x+1)^3=(x+1)(x+1)(x+1)=(x+1)(x^2+2x+1)=x^3+2x^2+x+x^2+2x+1=x^3+3x^2+3x+1$ which yields the correct coefficients. This means the expression is equal to $(x+1)^3$ for $x=9$ which is $(9+1)^3. Thus, its value is $10^3$. \boxed{1000}$"),

    "L04": (r"Let \[f(x) = \left\{ \begin{array}{cl} ax+3, &\text{ if }x>2, \\ x-5 &\text{ if } -2 \le x \le 2, \\ 2x-b &\text{ if } x <-2. \end{array} \right.\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper). "
             r"\n For the piecewise function to be continuous, the cases must meet at $2$ and $-2$. This means $ax+3$ must equal $x-5$ when $x=2$. This is because the piecewise function is equal to $ax+3$ above $x=2$ and is equal to $x-5$ below $x=2$. Both lines must meet at exactly $x=2$ for the function to be continuous.  Therefore, $2a+3=2-5$ and $2a+3=-3\rightarrow $2a=-6$ so $a=-3$. Checking by substitution, $2a+3=2-5$ means $2(-3)+3=-3\rightarrow -6+3=-3$ which is a true expression. Similarly, $x-5$ must equal $2x-b$ at $x=-2$. This is because the piecewise function is equal to $x-5$ for $x$ immidiately larger than $-2$, and the function is equal to $2x-b$ when $x$ is smaller than $-2$. Both lines must meet at $x=-2$. This means $2(-2)-b=-2-5$ so $-4-b=-7\rightarrow -b=-3$ so $b=3$. Checking by substitution, $2(-2)-3=-2-5\rightarrow -4-3=-7$ which is a true expression. Then, the sum $a+b=0$. \boxed{0}$"
             r"\n What is the value of $9^3 + 3(9^2) + 3(9) + 1$?"
             r"\n This expression is a cubic polynomial with coefficients $(1,3,3,1)$ in decreasing order of degree. Identifying the polynomial nature in the expression will help us solve it. Given that the expression is a polynomial, we can try to factorize it. The polynomial $(x+1)^3=x^3+3x^2+3x+1$ has the same coefficients as the expression. Checking, $(x+1)^3=(x+1)(x+1)(x+1)=(x+1)(x^2+2x+1)=x^3+2x^2+x+x^2+2x+1=x^3+3x^2+3x+1$ which yields the correct coefficients. This means the expression is equal to $(x+1)^3$ for $x=9$ which is $(9+1)^3. By writing the expression in this factorized form, we can evaluate the entire expression by computing the simplified sum $9+1=10$ within the parentheses. Thus, its value is $10^3$. \boxed{1000}$"),

    "L05": (r"Let \[f(x) = \left\{ \begin{array}{cl} ax+3, &\text{ if }x>2, \\ x-5 &\text{ if } -2 \le x \le 2, \\ 2x-b &\text{ if } x <-2. \end{array} \right.\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper). "
             r"\n For the piecewise function to be continuous, the cases must meet at $2$ and $-2$. This means $ax+3$ must equal $x-5$ when $x=2$. This is because the piecewise function is equal to $ax+3$ above $x=2$ and is equal to $x-5$ below $x=2$. Both lines must meet at exactly $x=2$ for the function to be continuous.  Therefore, $2a+3=2-5$ and $2a+3=-3\rightarrow $2a=-6$ so $a=-3$. Checking by substitution, $2a+3=2-5$ means $2(-3)+3=-3\rightarrow -6+3=-3$ which is a true expression. Checking again, $2a+3=2-5\rightarrow 2a+3=-3\rightarrow $2a=-6$ so $a=-3$. Similarly, $x-5$ must equal $2x-b$ at $x=-2$. This is because the piecewise function is equal to $x-5$ for $x$ immidiately larger than $-2$, and the function is equal to $2x-b$ when $x$ is smaller than $-2$. Both lines must meet at $x=-2$. This means $2(-2)-b=-2-5$ so $-4-b=-7\rightarrow -b=-3$ so $b=3$. Checking by substitution, $2(-2)-3=-2-5\rightarrow -4-3=-7$ which is a true expression. Checking again, $2(-2)-b=-2-5\rightarrow -4-b=-7\rightarrow -b=-3$ so $b=3$. Then, the sum $a+b=0$. \boxed{0}$"
             r"\n What is the value of $9^3 + 3(9^2) + 3(9) + 1$?"
             r"\n This expression is a cubic polynomial with coefficients $(1,3,3,1)$ in decreasing order of degree. Identifying the polynomial nature in the expression will help us solve it. Given that the expression is a polynomial, we can try to factorize it. The polynomial $(x+1)^3=x^3+3x^2+3x+1$ has the same coefficients as the expression. Checking, $(x+1)^3=(x+1)(x+1)(x+1)=(x+1)(x^2+2x+1)=x^3+2x^2+x+x^2+2x+1=x^3+3x^2+3x+1$ which yields the correct coefficients. Checking again, $(x+1)^3$ can be expanded to be (x+1)(x+1)(x+1)=(x^2+2x+1)(x+1)=x^3+2x^2+x+x^2+2x+1=x^3+3x^2+3x+1$, matching the coefficients in the expression. This means the expression is equal to $(x+1)^3$ for $x=9$ which is $(9+1)^3. By writing the expression in this factorized form, we can evaluate the entire expression by computing the simplified sum $9+1=10$ within the parentheses. Thus, its value is $10^3$. \boxed{1000}$"),
}


# ==========================================
# Utility Functions
# ==========================================

def get_prompts_for_dataset(dataset_name):
    if DATASETS[dataset_name]["prompt_set"] == "GSM":
        return PROMPTS_GSM
    return PROMPTS_MATH


def get_few_shot_for_dataset(dataset_name):
    if DATASETS[dataset_name]["prompt_set"] == "GSM":
        return FEW_SHOT_GSM
    return FEW_SHOT_MATH


def get_output_path(dataset_name, model_key):
    ds_short = DATASETS[dataset_name]["file_short_name"]
    return os.path.join("results", f"{ds_short}_{model_key}.json")


def get_summary_path(dataset_name, model_key):
    ds_short = DATASETS[dataset_name]["file_short_name"]
    return os.path.join("summary", f"{ds_short}_{model_key}_summary_stats.json")


def get_max_tokens(level_key):
    if level_key == "Direct":
        return 1024
    if int(level_key[1:3]) <= 3:
        return 1024
    if int(level_key[1:3]) <= 6:
        return 2048
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


# ==========================================
# Provider Call Functions
# ==========================================

async def call_google(model_config, system_prompt, user_message, max_tokens):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GOOGLE_API_KEY)

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=1.0,
        max_output_tokens=max_tokens,
    )

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.models.generate_content(
            model=model_config["model_id"],
            contents=user_message,
            config=config,
        ),
    )

    response_text = response.text or ""

    input_tokens = response.usage_metadata.prompt_token_count or 0
    output_tokens = response.usage_metadata.candidates_token_count or 0

    return {
        "response_text": response_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


async def call_deepseek(model_config, system_prompt, user_message, max_tokens):
    import openai

    client = openai.AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    response = await client.chat.completions.create(
        model=model_config["model_id"],
        messages=messages,
        max_tokens=max_tokens,
        temperature=1.0,
    )

    response_text = response.choices[0].message.content or ""

    return {
        "response_text": response_text,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }


# ==========================================
# Unified Dispatcher with Retry
# ==========================================

PROVIDER_FUNCTIONS = {
    "google": call_google,
    "deepseek": call_deepseek,
}


def _get_retry_exceptions():
    """Collect retryable exception types from installed providers."""
    exceptions = [ConnectionError, TimeoutError]
    try:
        import openai
        exceptions.extend([openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError])
    except ImportError:
        pass
    return tuple(exceptions)


RETRY_EXCEPTIONS = _get_retry_exceptions()


@retry(
    wait=wait_exponential(multiplier=2, min=2, max=120),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(RETRY_EXCEPTIONS),
)
async def call_model(model_key, system_prompt, user_message, max_tokens):
    config = MODELS[model_key]
    provider = config["provider"]
    fn = PROVIDER_FUNCTIONS[provider]

    semaphore = PROVIDER_SEMAPHORES[provider]
    async with semaphore:
        return await fn(config, system_prompt, user_message, max_tokens)


# ==========================================
# Core Processing
# ==========================================

async def process_single_call(model_key, system_prompt, user_message, max_tokens, rep, q_idx, gt_clean):
    """Process a single (question, prompt_level, replicate) combination."""
    try:
        result = await call_model(model_key, system_prompt, user_message, max_tokens)

        response_text = result["response_text"]
        output_tokens = result["output_tokens"]

        truncated = output_tokens >= max_tokens

        try:
            parsed_gold = parse(gt_clean)
            parsed_answer = parse(response_text)
            is_correct = verify(parsed_gold, parsed_answer)
        except Exception:
            is_correct = False

        return {
            "replicate": rep,
            "response_text": response_text,
            "token_count": int(output_tokens),
            "truncated": truncated,
            "is_correct": bool(is_correct),
            "q_idx": q_idx,
        }
    except Exception as e:
        print(f"  Error on q={q_idx}, rep={rep}: {e}")
        return {
            "replicate": rep,
            "response_text": "",
            "token_count": 0,
            "truncated": False,
            "is_correct": False,
            "q_idx": q_idx,
        }


async def process_dataset(dataset_name, model_key, num_samples=None):
    config = DATASETS[dataset_name]
    output_file = get_output_path(dataset_name, model_key)

    print(f"\n=== Processing Dataset: {dataset_name} | Model: {model_key} ===")
    print(f"Output file: {output_file}")

    current_prompts = get_prompts_for_dataset(dataset_name)
    few_shot_examples = get_few_shot_for_dataset(dataset_name)
    required_keys = set(current_prompts.keys())

    # Load existing results for resume support
    existing_results = []
    incomplete_map = {}  # Map ID -> set of (condition_name, replicate) pairs still needed

    required_reps = set(REPS)

    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                existing_results = json.load(f)

            print(f"Loaded {len(existing_results)} existing entries. Checking for completeness...")

            for entry in existing_results:
                missing = set()
                for key in required_keys:
                    if key not in entry["results"]:
                        # Entire prompt level missing
                        for r in REPS:
                            missing.add((key, r))
                    else:
                        existing_reps = {r_entry["replicate"] for r_entry in entry["results"][key]}
                        for r in required_reps - existing_reps:
                            missing.add((key, r))
                if missing:
                    incomplete_map[entry["id"]] = missing
        except Exception:
            pass

    fully_complete_ids = {
        entry["id"] for entry in existing_results if entry["id"] not in incomplete_map
    }

    # Load dataset
    dataset = load_dataset(*config["hf_path"], split=config["split"])
    total_count = len(dataset)
    target_count = num_samples if num_samples is not None else total_count

    all_indices = list(range(target_count))
    indices_to_process = [i for i in all_indices if i not in fully_complete_ids]

    if not indices_to_process:
        print(f"All samples fully processed for {dataset_name} ({model_key}).")
        return

    print(f"Processing {len(indices_to_process)} samples.")

    # Process in batches
    batch_size = 20
    for batch_start in range(0, len(indices_to_process), batch_size):
        batch_indices = indices_to_process[batch_start : batch_start + batch_size]

        # Prepare batch entries
        batch_entries = {}
        for q_idx in batch_indices:
            q_idx_int = int(q_idx)
            raw_gt = str(dataset[q_idx][config["col_a"]])
            clean_gt = (
                clean_gsm_ground_truth(raw_gt) if config["prompt_set"] == "GSM" else raw_gt
            )
            batch_entries[q_idx_int] = {
                "id": q_idx_int,
                "question": dataset[q_idx][config["col_q"]],
                "ground_truth_clean": clean_gt,
                "results": {},
            }

        # Create all async tasks
        tasks = []
        task_meta = []  # (q_idx, condition_name) for each task

        for condition_name in sorted(current_prompts.keys()):
            system_prompt = current_prompts[condition_name]
            max_tokens = get_max_tokens(condition_name)
            few_shot_prefix = few_shot_examples.get(condition_name, "")

            for q_idx in batch_indices:
                q_idx_int = int(q_idx)

                question = dataset[q_idx][config["col_q"]]
                user_message = few_shot_prefix + question
                gt_clean = batch_entries[q_idx_int]["ground_truth_clean"]

                for rep in REPS:
                    # Check if this specific (condition, replicate) is needed
                    if q_idx_int in fully_complete_ids:
                        continue
                    if q_idx_int in incomplete_map:
                        if (condition_name, rep) not in incomplete_map[q_idx_int]:
                            continue

                    task = process_single_call(
                        model_key, system_prompt, user_message, max_tokens, rep, q_idx_int, gt_clean
                    )
                    tasks.append(task)
                    task_meta.append((q_idx_int, condition_name))

        if not tasks:
            continue

        # Run all tasks for this batch concurrently
        print(f"  Batch {batch_start // batch_size + 1}: {len(tasks)} API calls for {len(batch_indices)} questions...")
        results = await asyncio.gather(*tasks)

        # Distribute results into batch_entries
        for result, (q_idx_int, condition_name) in zip(results, task_meta):
            if condition_name not in batch_entries[q_idx_int]["results"]:
                batch_entries[q_idx_int]["results"][condition_name] = []
            # Remove internal tracking field
            result_clean = {k: v for k, v in result.items() if k != "q_idx"}
            batch_entries[q_idx_int]["results"][condition_name].append(result_clean)

        # Sort replicates
        for entry in batch_entries.values():
            for condition_name in entry["results"]:
                entry["results"][condition_name].sort(key=lambda x: x["replicate"])

        # Merge into existing results (append replicates, don't overwrite)
        existing_map = {e["id"]: e for e in existing_results}
        for new_item in batch_entries.values():
            item_id = new_item["id"]
            if item_id in existing_map:
                for cond_key, new_reps in new_item["results"].items():
                    if cond_key not in existing_map[item_id]["results"]:
                        existing_map[item_id]["results"][cond_key] = new_reps
                    else:
                        # Append only replicates not already present
                        existing_rep_ids = {
                            r["replicate"] for r in existing_map[item_id]["results"][cond_key]
                        }
                        for r_entry in new_reps:
                            if r_entry["replicate"] not in existing_rep_ids:
                                existing_map[item_id]["results"][cond_key].append(r_entry)
                        existing_map[item_id]["results"][cond_key].sort(
                            key=lambda x: x["replicate"]
                        )
            else:
                existing_results.append(new_item)
                existing_map[item_id] = new_item

        existing_results.sort(key=lambda x: x["id"])
        save_atomic(existing_results, output_file)
        print(f"  Saved {len(existing_results)} entries.")


# ==========================================
# Plotting
# ==========================================

def plot_api_performance(dataset_name):
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    prompt_keys = sorted(get_prompts_for_dataset(dataset_name).keys())

    fig, ax = plt.subplots(figsize=(8, 5))

    # Colors by provider, style by reasoning
    provider_colors = {
        "anthropic": "#2563eb",
        "openai": "#16a34a",
        "google": "#dc2626",
        "deepseek": "#7c3aed",
    }

    model_keys_with_data = []

    for model_key, model_config in MODELS.items():
        output_file = get_output_path(dataset_name, model_key)
        if not os.path.exists(output_file):
            continue

        with open(output_file, "r") as f:
            data = json.load(f)
        if not data:
            continue

        summary_stats = {}
        plot_points = []

        for prompt_key in prompt_keys:
            rep_stats = {r: {"correct": 0, "total": 0} for r in REPS}
            all_tokens = []
            has_data = False

            for entry in data:
                if prompt_key not in entry["results"]:
                    continue
                replicates = entry["results"][prompt_key]
                if not replicates:
                    continue

                has_data = True
                for r_entry in replicates:
                    r_idx = r_entry["replicate"]
                    if r_idx in rep_stats:
                        rep_stats[r_idx]["total"] += 1
                        if r_entry["is_correct"]:
                            rep_stats[r_idx]["correct"] += 1
                        all_tokens.append(r_entry["token_count"])

            if not has_data:
                continue

            replicate_error_rates = []
            for r in REPS:
                total = rep_stats[r]["total"]
                if total > 0:
                    error_rate = 1.0 - (rep_stats[r]["correct"] / total)
                    replicate_error_rates.append(error_rate)

            if not replicate_error_rates:
                continue

            mean_error_rate = np.mean(replicate_error_rates)
            if len(replicate_error_rates) > 1:
                sem_error_rate = np.std(replicate_error_rates, ddof=1) / np.sqrt(
                    len(replicate_error_rates)
                )
            else:
                sem_error_rate = 0.0

            avg_tokens = np.mean(all_tokens) if all_tokens else 0

            summary_stats[prompt_key] = {
                "error_rate_mean": float(mean_error_rate),
                "error_rate_sem": float(sem_error_rate),
                "average_tokens": float(avg_tokens),
            }

            if prompt_key != "Direct":
                plot_points.append((avg_tokens, mean_error_rate, sem_error_rate))

        # Save summary stats
        summary_path = get_summary_path(dataset_name, model_key)
        with open(summary_path, "w") as f:
            json.dump(summary_stats, f, indent=4)

        plot_points.sort(key=lambda p: p[0])

        if not plot_points:
            continue

        model_keys_with_data.append(model_key)

        x_val, y_val, y_err = zip(*plot_points)

        color = provider_colors.get(model_config["provider"], "gray")

        ax.errorbar(
            x_val,
            y_val,
            yerr=y_err,
            fmt="-o",
            color=color,
            linewidth=2,
            label=model_key,
            capsize=3,
            markersize=6,
        )

    ax.set_xlabel("Output Tokens")
    ax.set_ylabel("Test Error")
    ax.set_title(f"{dataset_name}")
    ax.grid(True, linestyle=":", alpha=0.6)

    if model_keys_with_data:
        ax.legend(loc="best")

    plt.tight_layout()
    short_name = DATASETS[dataset_name]["file_short_name"]
    plt.savefig(f"{short_name}_api_comparison_error_vs_length.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {short_name}_api_comparison_error_vs_length.pdf")


# ==========================================
# Cost Estimation
# ==========================================

def estimate_costs(datasets_to_run, models_to_run, num_samples):
    """Print rough cost estimates before running."""
    # Approximate per-1M-token costs (input, output)
    cost_per_1m = {
        "gemini-flash": (0.15, 0.6),
        "deepseek-v3": (0.27, 1.1),
    }
    # Average output tokens per call
    avg_output = {
        "gemini-flash": 200,
        "deepseek-v3": 200,
    }

    print("\n=== Cost Estimate ===")
    print("(Based on approximate average token counts per call)\n")
    total_cost = 0.0
    total_calls = 0
    for ds_name in datasets_to_run:
        config = DATASETS[ds_name]
        dataset = load_dataset(*config["hf_path"], split=config["split"])
        n_questions = num_samples if num_samples else len(dataset)

        for model_key in models_to_run:
            n_calls = n_questions * 6 * len(REPS)
            avg_in = 200  # system prompt + question
            avg_out = avg_output.get(model_key, 300)
            in_cost = (n_calls * avg_in / 1_000_000) * cost_per_1m.get(model_key, (5, 15))[0]
            out_cost = (n_calls * avg_out / 1_000_000) * cost_per_1m.get(model_key, (5, 15))[1]
            model_cost = in_cost + out_cost
            total_cost += model_cost
            total_calls += n_calls
            print(f"  {ds_name:>8} x {model_key:<25} {n_calls:>7} calls  ~${model_cost:>8.2f}")

    print(f"\n  Total: {total_calls} API calls, estimated ~${total_cost:.2f}")
    print()


# ==========================================
# Main
# ==========================================

async def main(args):
    os.makedirs("results", exist_ok=True)
    os.makedirs("summary", exist_ok=True)

    # Initialize per-provider semaphores
    PROVIDER_SEMAPHORES["google"] = asyncio.Semaphore(5)
    PROVIDER_SEMAPHORES["deepseek"] = asyncio.Semaphore(3)

    # Parse model/dataset filters
    models_to_run = list(MODELS.keys())
    if args.models:
        models_to_run = [m.strip() for m in args.models.split(",")]
        for m in models_to_run:
            if m not in MODELS:
                print(f"Unknown model: {m}. Available: {list(MODELS.keys())}")
                return

    datasets_to_run = list(DATASETS.keys())
    if args.datasets:
        datasets_to_run = [d.strip() for d in args.datasets.split(",")]
        for d in datasets_to_run:
            if d not in DATASETS:
                print(f"Unknown dataset: {d}. Available: {list(DATASETS.keys())}")
                return

    if args.dry_run:
        estimate_costs(datasets_to_run, models_to_run, args.num_samples)
        return

    if not args.plot_only:
        for ds_name in datasets_to_run:
            for model_key in models_to_run:
                await process_dataset(ds_name, model_key, num_samples=args.num_samples)

    print("\n=== Generating Plots ===")
    for ds_name in datasets_to_run:
        plot_api_performance(ds_name)

    print("\n=== Experiment Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate frontier LLM APIs on math benchmarks")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model keys to run")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset keys to run")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit number of questions per dataset")
    parser.add_argument("--dry-run", action="store_true", help="Estimate costs without running")
    parser.add_argument("--plot-only", action="store_true", help="Only generate plots from existing results")
    args = parser.parse_args()
    asyncio.run(main(args))

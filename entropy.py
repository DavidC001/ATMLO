import json
import os
import numpy as np
import torch
from torch.nn.functional import softmax, log_softmax
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple
import glob

def load_dataset(dataset_path: str) -> Dict:
    """Load a dataset from JSON file."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_yes_no_tokens(tokenizer) -> Dict[str, List[int]]:
    """Get token IDs for yes/no variations."""
    yes_tokens = []
    no_tokens = []
    
    # Get tokens for different variations
    for yes_word in ["Yes", "yes"]:
        tokens = tokenizer.encode(yes_word, add_special_tokens=False)
        if tokens:
            yes_tokens.extend(tokens)
    
    for no_word in ["No", "no"]:
        tokens = tokenizer.encode(no_word, add_special_tokens=False)
        if tokens:
            no_tokens.extend(tokens)
    
    # Remove duplicates
    yes_tokens = list(set(yes_tokens))
    no_tokens = list(set(no_tokens))
    
    return {
        "yes": yes_tokens,
        "no": no_tokens
    }

def compute_entropy(logits: torch.Tensor, yes_tokens: List[int], no_tokens: List[int]) -> float:
    """
    Compute entropy over yes/no tokens.
    
    Args:
        logits: Model logits [vocab_size]
        yes_tokens: List of token IDs for "yes" variations
        no_tokens: List of token IDs for "no" variations
    
    Returns:
        Entropy value
    """
    # Extract logits for yes and no tokens
    yes_logits = logits[yes_tokens].max()  # Take max across variations
    no_logits = logits[no_tokens].max()    # Take max across variations
    
    # Create probability distribution
    combined_logits = torch.stack([yes_logits, no_logits])
    probs = softmax(combined_logits, dim=0)
    
    # Compute entropy: H = -sum(p * log(p))
    log_probs = log_softmax(combined_logits, dim=0)
    entropy = -(probs * log_probs).sum().item()
    
    return entropy

def process_prompts(model, tokenizer, prompts: List[Dict], yes_no_tokens: Dict, device: str) -> List[float]:
    """Process prompts and compute entropy for each."""
    entropies = []
    
    for prompt_data in tqdm(prompts, desc="Processing prompts"):
        clean_prompt = prompt_data["clean"]
        corrupt_prompt = prompt_data["corrupt"]
        
        # Process clean prompt
        clean_entropy = process_single_prompt(
            model, tokenizer, clean_prompt, yes_no_tokens, device
        )
        
        # Process corrupt prompt  
        corrupt_entropy = process_single_prompt(
            model, tokenizer, corrupt_prompt, yes_no_tokens, device
        )
        
        entropies.append({
            "clean_entropy": clean_entropy,
            "corrupt_entropy": corrupt_entropy,
            "clean_prompt": clean_prompt,
            "corrupt_prompt": corrupt_prompt,
            "answers": prompt_data.get("answers", []),
            "wrong_answers": prompt_data.get("wrong_answers", [])
        })
    
    return entropies

def process_single_prompt(model, tokenizer, prompt: str, yes_no_tokens: Dict, device: str) -> float:
    """Process a single prompt and return entropy."""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last token logits
    
    # Compute entropy
    entropy = compute_entropy(logits, yes_no_tokens["yes"], yes_no_tokens["no"])
    return entropy

def compute_dataset_statistics(entropies: List[Dict]) -> Dict:
    """Compute statistics over the dataset."""
    clean_entropies = [e["clean_entropy"] for e in entropies]
    corrupt_entropies = [e["corrupt_entropy"] for e in entropies]
    
    stats = {
        "clean_entropy": {
            "mean": np.mean(clean_entropies),
            "std": np.std(clean_entropies),
            "min": np.min(clean_entropies),
            "max": np.max(clean_entropies)
        },
        "corrupt_entropy": {
            "mean": np.mean(corrupt_entropies),
            "std": np.std(corrupt_entropies),
            "min": np.min(corrupt_entropies),
            "max": np.max(corrupt_entropies)
        },
        "entropy_difference": {
            "mean": np.mean([e["clean_entropy"] - e["corrupt_entropy"] for e in entropies]),
            "std": np.std([e["clean_entropy"] - e["corrupt_entropy"] for e in entropies])
        },
        "total_samples": len(entropies)
    }
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Compute entropy of yes/no logits for benchmark datasets")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", 
                       help="Model name or path")
    parser.add_argument("--dataset_dir", type=str, default="datasets/circ_disc", 
                       help="Directory containing benchmark datasets")
    parser.add_argument("--device", type=str, default="auto", 
                       help="Device to run on (auto, cuda, cpu)")
    parser.add_argument("--specific_dataset", type=str, default="modus_tollens",
                       help="Specific dataset to process (e.g., 'modus_tollens')")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    
    # Get yes/no tokens
    yes_no_tokens = get_yes_no_tokens(tokenizer)
    print(f"Yes tokens: {yes_no_tokens['yes']}")
    print(f"No tokens: {yes_no_tokens['no']}")
    
    # Find all dataset files
    if args.specific_dataset:
        dataset_pattern = os.path.join(args.dataset_dir, args.model_name, args.specific_dataset, "dataset.json")
        dataset_files = glob.glob(dataset_pattern)
    else:
        dataset_pattern = os.path.join(args.dataset_dir, args.model_name, "*", "dataset.json")
        dataset_files = glob.glob(dataset_pattern)
    
    print(f"Found {len(dataset_files)} dataset files")
    
    # Process each dataset
    for dataset_file in dataset_files:
        print(f"\nProcessing: {dataset_file}")
        
        # Extract model and dataset names from path
        path_parts = dataset_file.replace(args.dataset_dir, "").strip(os.sep).split(os.sep)
        model_dir = path_parts[0]
        dataset_name = path_parts[1]
        
        # Load dataset
        dataset = load_dataset(dataset_file)
        
        if not dataset.get("prompts"):
            print(f"No prompts found in {dataset_file}, skipping...")
            continue
        
        print(f"Processing {len(dataset['prompts'])} prompts...")
        
        # Process prompts
        entropies = process_prompts(
            model, tokenizer, dataset["prompts"], yes_no_tokens, device
        )
        
        # Compute statistics
        stats = compute_dataset_statistics(entropies)
        
        # Prepare output
        output_data = {
            "model_name": args.model_name,
            "model_directory": model_dir,
            "dataset_name": dataset_name,
            "yes_tokens": yes_no_tokens["yes"],
            "no_tokens": yes_no_tokens["no"],
            "statistics": stats,
            "detailed_results": entropies
        }
        
        # Save results
        out_dir = f"results/{args.model_name}/entropy"
        os.makedirs(out_dir, exist_ok=True)
        output_file = os.path.join(out_dir, f"{dataset_name}_entropy.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_file}")
        print(f"Clean entropy mean: {stats['clean_entropy']['mean']:.4f} ± {stats['clean_entropy']['std']:.4f}")
        print(f"Corrupt entropy mean: {stats['corrupt_entropy']['mean']:.4f} ± {stats['corrupt_entropy']['std']:.4f}")
        print(f"Entropy difference mean: {stats['entropy_difference']['mean']:.4f} ± {stats['entropy_difference']['std']:.4f}")

if __name__ == "__main__":
    main()
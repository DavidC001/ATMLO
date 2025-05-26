import json

from tqdm import tqdm
from auto_circuit.data import load_datasets_from_json

alpaca_prompt_template = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
%s

### Input:%s

### Response:
"""

llama_3prompt_template = f"""<|start_header_id|>system<|end_header_id|>%s<|eot_id|><|start_header_id|>user<|end_header_id|>%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

qwen_template = f"""<|im_start|>system
%s<|im_end|>
<|im_start|>user%s<|im_end|>
<|im_start|>assistant
"""

task = {
}

prompt_template = {
    "alpaca": alpaca_prompt_template,
    "llama_3": llama_3prompt_template,
    "qwen": qwen_template,
}

def create_dataset(input_jsons, out_json="datasets/data.json", template = "llama_3", global_padding=False, tokenizer=None):
    """
    Create a dataset from the input json files and save it to the output json file.
    
    Args:
        input_jsons: list of json files to load
        out_json: output json file to save the dataset
        template: template to use for the dataset
        global_padding: if True, pad the input ids to the same length
        tokenizer: tokenizer object, required if global_padding is True
        
    Returns:
        The number of samples in the dataset
    """
    
    assert tokenizer is not None or not global_padding, "Tokenizer is required if global_padding is True"
    
    out_data = {
        "seq_labels": [],
        "word_idxs": {},
        "prompts": []
    }
    
    for json_path in input_jsons:
        # convert the datasets of only phrases with the correct system promt
        data = json.load(open(json_path))
        
        instruction = data.get("instruction", "Please answer the question based on the given context.")
        
        for sample in data["prompts"]:
            if template is not None:
                sample["clean"] = prompt_template[template] % (instruction, sample["clean"])
                sample["corrupt"] = prompt_template[template] % (instruction, sample["corrupt"])
                
            # if tokenizer is not None verify that the clean and corrupt prompts are the same length
            if tokenizer is not None:
                assert len(tokenizer.encode(sample["clean"], add_special_tokens=False)) == len(tokenizer.encode(sample["corrupt"], add_special_tokens=False)), f"Clean and corrupt prompts have different lengths: {len(tokenizer.encode(sample['clean'], add_special_tokens=False))} vs {len(tokenizer.encode(sample['corrupt'], add_special_tokens=False))}"
            
        
        out_data["prompts"].extend(data["prompts"])
        
    
    if global_padding:    
        # compute the max token length
        max_token_length = max([len(tokenizer.encode(sample["clean"], add_special_tokens=False)) for sample in out_data["prompts"]])
        
        print("normalizing the dataset to max token length: ", max_token_length)
        # add the "\n" to the end of the prompt
        for sample in tqdm(out_data["prompts"]):
            while len(tokenizer.encode(sample["clean"], add_special_tokens=False)) < max_token_length:
                sample["clean"] += "\n"
            while len(tokenizer.encode(sample["corrupt"], add_special_tokens=False)) < max_token_length:
                sample["corrupt"] += "\n"
                
                
            assert len(tokenizer.encode(sample["clean"], add_special_tokens=False)) == len(tokenizer.encode(sample["corrupt"], add_special_tokens=False)), f"Clean and corrupt prompts have different lengths: {len(tokenizer.encode(sample['clean'], add_special_tokens=False))} vs {len(tokenizer.encode(sample['corrupt'], add_special_tokens=False))}"
            assert len(tokenizer.encode(sample["clean"], add_special_tokens=False)) == max_token_length, f"Prompt length is not equal to max token length: {len(tokenizer.encode(sample['clean'], add_special_tokens=False))} vs {max_token_length}"
    
    
    json.dump(out_data, open(out_json,"w"), indent = 1)
    # return dataset dimension
    return len(out_data["prompts"])

if __name__ == "__main__":
    create_dataset(
        "datasets/LogicBench/LogicBench(Aug)/propositional_logic/modus_tollens/data.json",
        "datasets/LogicBench/LogicBench(Eval)/BQA/propositional_logic/modus_tollens/data.json",
    )    
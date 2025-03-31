import json
from auto_circuit.data import load_datasets_from_json

alpaca_prompt_template = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
%s

### Input:
%s

### Response:
"""

llama_3prompt_template = f"""<|start_header_id|>system<|end_header_id|>

%s<|eot_id|><|start_header_id|>user<|end_header_id|>

%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

qwen_template = f"""<|im_start|>system
%s<|im_end|>
<|im_start|>user
%s<|im_end|>
<|im_start|>assistant
"""

task = {
}

prompt_template = {
    "alpaca": alpaca_prompt_template,
    "llama_3": llama_3prompt_template,
    "qwen": qwen_template,
}

def create_dataset(input_jsons, out_json="datasets/data.json", template = "llama_3"):
    
    out_data = {
        "seq_labels": [],
        "word_idxs": {},
        "prompts": []
    }
    
    for json_path in input_jsons:
        # convert the datasets of only phrases with the correct system promt
        data = json.load(open(json_path))
        
        instruction = data["instruction"]
        
        for sample in data["prompts"]:
            sample["clean"] = prompt_template[template] % (instruction, sample["clean"])
            sample["corrupt"] = prompt_template[template] % (instruction, sample["corrupt"])
        
        out_data["prompts"].extend(data["prompts"])
    
    json.dump(out_data, open(out_json,"w"), indent = 1)
    # return dataset dimension
    return len(out_data["prompts"])

if __name__ == "__main__":
    create_dataset(
        "datasets/LogicBench/LogicBench(Aug)/propositional_logic/modus_tollens/data.json",
        "datasets/LogicBench/LogicBench(Eval)/BQA/propositional_logic/modus_tollens/data.json",
    )    
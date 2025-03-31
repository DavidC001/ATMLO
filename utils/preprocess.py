import json
import os
import random
from time import sleep
from tqdm import tqdm
import sys
from transformers import AutoTokenizer

sys.path.append(".")

dataset_base_dir = "LogicBench/data"
output_dir = "datasets/LogicBench"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

data_types = ["LogicBench(Aug)","LogicBench(Eval)/BQA"]

bar = tqdm()

def preprocess(model_name, tokenizer=None):
    
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    for data_type in data_types:
        
        data_path = f"{dataset_base_dir}/{data_type}"
            
        for logic_type in os.listdir(data_path):
            
            logic_path = f"{data_path}/{logic_type}"
            if not os.path.isdir(logic_path): continue
                
            for problem_type in os.listdir(logic_path):
                    
                json_path = f"{logic_path}/{problem_type}/data_instances.json"
                data = json.load(open(json_path))
                
                # print(f"=========================\nPROCESSING {json_path}\n=========================")
                
                data_col = "data_samples"
                data_col = data_col if (data_col in data.keys()) else "samples"
                
                # define output file path
                output_json_path = f"{output_dir}/{data_type}/{logic_type}/{problem_type}"
                os.makedirs(output_json_path, exist_ok=True)
                output_json_path+="/data.json"
                output_data = {
                    "instruction": "Based on the given context, you have to respond with yes or no.",
                    "seq_labels": [],
                    "word_idxs": {},
                    "prompts": []
                }
                
                # parse the data
                for sample in data[data_col]:
                    context = sample["context"]
                    
                    clean = sample["qa_pairs"][0]["question"]
                    # corrupt_idx = random.randint(1,len(sample["qa_pairs"])-1)
                    corrupt_idx = 1
                    corrupt = sample["qa_pairs"][corrupt_idx]["question"]
                    
                    # tokenize both and check if they match
                    tokenized_clean = len(tokenizer.tokenize(clean))
                    tokenized_corrupt = len(tokenizer.tokenize(corrupt))
                    
                    while tokenized_corrupt != tokenized_clean:
                        
                        if tokenized_corrupt < tokenized_clean:
                            corrupt += "\n"
                            tokenized_corrupt = len(tokenizer.tokenize(corrupt))
                        else:
                            clean += "\n"
                            tokenized_clean = len(tokenizer.tokenize(clean))
                        
                    clean = context + " " + clean
                    corrupt = context + " " + corrupt
                    
                    assert tokenized_corrupt == tokenized_clean, f"the prompts do not have the same lenght clean:{tokenized_clean} corrupt:{tokenized_corrupt}\nclean: {clean}\ncorrupt: {corrupt}"
                    
                    output_data["prompts"].append({
                        "clean": clean,
                        "corrupt": corrupt,
                        "answers": ["yes"],
                        "wrong_answers": ["no"],
                    })
                    
                # output data to file
                json.dump(output_data, open(output_json_path, "w"), indent=1)
                bar.update()
            
                
if __name__ == "__main__":
    from config import ProjectConfig, load_yaml_config

    random.seed(42)
    
    conf:ProjectConfig = load_yaml_config("conf.yaml")
    preprocess(conf.model.model_name)

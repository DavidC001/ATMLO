from copy import deepcopy
import os
import torch
from torch.utils.data import Dataset
from utils.data.logic_dataset import create_dataset
from utils.preprocess.preprocess import preprocess_LogicBench
from config import ProjectConfig, load_yaml_config, BenchConfig
from auto_circuit.experiment_utils import load_tl_model
import json
from tqdm import tqdm
from utils.dataloader import AC_data


# load configuration
config : ProjectConfig = load_yaml_config("conf.yaml")                

accuracies = {}
retained_dataset_size = {}

temp_base_dir = config.benchmark.dataset_dir + "/bench/"
os.makedirs(temp_base_dir, exist_ok=True)

for template, model_name in zip(config.benchmark.templates, config.benchmark.model_names):
    # load hooked transformers model
    model = load_tl_model(model_name, device="cuda")
    tokenizer = model.tokenizer
    model_dataset = {
        "seq_labels": [],
        "word_idxs": {},
        "prompts": []
    }

    counterfactual_sample = {
        "clean": "",
        "corrupt": "",
        "answers": ["Yes", "yes"],
        "wrong_answers": ["No", "no"],
    }
    
    accuracies[model_name] = {}
    retained_dataset_size[model_name] = {}
    
    for key in config.benchmark.input_jsons:
        
        json_files = []
        for i in range(len(config.benchmark.input_jsons[key])):
            preprocess_LogicBench(model_name, tokenizer=tokenizer, file=config.benchmark.input_jsons[key][i], out=f"datasets/bench/{i}.json")
            json_files.append(f"{temp_base_dir}/{i}.json")
        
        print("loading data:")
        data = AC_data(
            input_jsons=json_files,
            template=template,
            tokenizer=tokenizer,
            global_padding=config.benchmark.global_padding,
        )
        total = len(data)
        correct = 0
        wrong = 0

        print(f"Processing Dataset")
        dataset = tqdm(data)
        turn = 0
        clean_correct = False # wether the clean prompt was correctly classified
        clean_prompt = ""
        for sample in dataset:
            out = model(sample["input_ids"].unsqueeze(0))[0][-1]
            
            correct_value = out[sample["correct"]].sum()
            wrong_value = out[sample["wrong"]].sum()
            
            # check if the answer is the same
            if (correct_value > wrong_value) :
                correct+=1
                
                if turn == 0:
                    clean_correct = True
                    clean_prompt = sample["text"]
                elif clean_correct == True:
                    counterfactual_sample["clean"] = clean_prompt
                    counterfactual_sample["corrupt"] = sample["text"]
                    model_dataset["prompts"].append(deepcopy(counterfactual_sample))
            else:
                wrong+=1
                clean_correct = False
            
            dataset.set_description(desc=f"CORRECT={correct} WRONG={wrong}")    
            
            turn += 1
            turn %= 2
            
        dataset.close()
        print("========================================")
        print(f"Model name: {model_name} on dataset {key}")
        print(f"Accuracy of the model {model_name} on dataset {key}: {correct / total}")
        print(f"Total number of kept samples for the dataset: {len(model_dataset['prompts'])}")
        
        accuracies[model_name][key] = correct / total
        retained_dataset_size[model_name][key] = len(model_dataset["prompts"])
        
        model_dataset["accuracy"] = correct / total
        model_dataset["dataset_size"] = len(model_dataset["prompts"])
        
        # save the dataset
        # create model directory under datasets
        os.makedirs(f"datasets/circ_disc/{model_name}/{key}", exist_ok=True)
        with open(f"datasets/circ_disc/{model_name}/{key}/dataset.json", "w") as f:
            json.dump(model_dataset, f, indent=4)
        print(f"Dataset saved to datasets/{model_name}/{key}/dataset.json")
        print("Done")
        print("========================================")
        
    # delete model to free memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

# delete temp directory
os.rmdir(temp_base_dir)

print("All done")
print("========================================")
print("Accuracies:")
for model_name, dataset in accuracies.items():
    for dataset_name, accuracy in dataset.items():
        print(f"Model: {model_name} on dataset {dataset_name} has accuracy: {accuracy}")
print("========================================")

from copy import deepcopy
import os
from torch.utils.data import Dataset
from utils.dataset import create_dataset
from utils.preprocess import preprocess
from config import ProjectConfig, load_yaml_config, DataConfig
from auto_circuit.experiment_utils import load_tl_model
import json
from tqdm import tqdm


class AC_data(Dataset):
    
    def __init__(self, input_jsons, template, tokenizer):
        """
        Args:
            config (DataConfig): configuration object
            tokenizer: tokenizer object
        """
        super().__init__()
        
        self.text = []
        self.samples = []
        self.GT = []
        self.GT_opposite = []
        
        out_path = "testing.json"
        create_dataset(
            input_jsons=input_jsons,
            out_json=out_path,
            template=template,
        )
        
        # load the data from the json
        with open(out_path) as f:
            data = json.load(f)
            for sample in tqdm(data["prompts"]):
                self.text.append(sample["clean"])
                self.text.append(sample["corrupt"])
                
                clean = tokenizer(sample["clean"], return_tensors="pt")
                corrupt = tokenizer(sample["corrupt"], return_tensors="pt")
                self.samples.append(clean["input_ids"])
                self.samples.append(corrupt["input_ids"])
                
                correct = tokenizer.encode(sample["answers"][0], add_special_tokens=False)[0]
                wrong = tokenizer.encode(sample["wrong_answers"][0], add_special_tokens=False)[0]
                self.GT.append(correct)
                self.GT_opposite.append(wrong)
                
                self.GT.append(wrong)
                self.GT_opposite.append(correct)
                
        # remove the json file
        os.remove(out_path)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        return {
            "text": self.text[index],
            "input_ids": self.samples[index],
            "correct": self.GT[index],
            "wrong": self.GT_opposite[index],
        }

# load configuration
config : ProjectConfig = load_yaml_config("conf.yaml")                

accuracies = {}
retained_dataset_size = {}

for template, model_name in zip(config.data_preprocessing.templates, config.data_preprocessing.model_names):
    # load hooked transformers model
    model = load_tl_model(model_name, device="cuda")
    tokenizer = model.tokenizer
    model_dataset = {
        "seq_labels": [],
        "word_idxs": {},
        "prompts": []
    }
    
    preprocess(model_name, tokenizer=tokenizer)

    counterfactual_sample = {
        "clean": "",
        "corrupt": "",
        "answers": ["yes"],
        "wrong_answers": ["no"],
    }

    print("loading data:")
    data = AC_data(
        input_jsons=config.data_preprocessing.input_jsons,
        template=template,
        tokenizer=tokenizer,
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
        out = model(sample["input_ids"])[0][-1]
        
        correct_value = out[sample["correct"]]
        wrong_value = out[sample["wrong"]]
        
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
    print(f"Model name: {model_name}")
    print(f"Accuracy of the model {model_name} is {correct/total}")
    print(f"Total number of kept samples for the dataset: {len(model_dataset['prompts'])}")
    
    accuracies[model_name] = correct/total
    retained_dataset_size[model_name] = len(model_dataset["prompts"])
    
    model_dataset["accuracy"] = correct / total
    model_dataset["dataset_size"] = len(model_dataset["prompts"])
    
    # save the dataset
    # create model directory under datasets
    os.makedirs(f"datasets/{model_name}", exist_ok=True)
    with open(f"datasets/{model_name}/dataset.json", "w") as f:
        json.dump(model_dataset, f, indent=4)
    print(f"Dataset saved to datasets/{model_name}/dataset.json")
    print("Done")
    print("========================================")

print("All done")
print("========================================")
print("Accuracies:")
for model_name, accuracy in accuracies.items():
    print(f"Model: {model_name} Accuracy: {accuracy} Dataset size: {retained_dataset_size[model_name]}")
print("========================================")

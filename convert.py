import json
import os
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import ProjectConfig, load_yaml_config

device = "cuda" if torch.cuda.is_available() else "cpu"

messages = {
    "default": [
        {"role": "system", "content": "you need to rewrite the following logic problems statement to have them be the same number of tokens. Only write the two new prompts in two lines."},
        {"role": "user", "content": '"If he will receive any salary, does this mean that mason didn\'t leave his job?" and "If he will receive any salary, does this mean that Mason left his job?"'},
        {"role": "assistant", "content": 'If he will receive any salary, does this mean that Mason kept his job?\nIf he will receive any salary, does this mean that Mason lost his job?  '},
    ],
    "modus_tollens": [
        {"role": "system", "content": "you need to rewrite the following logic problems statement to have them be the same number of tokens. The template to use is \"if condition, does this mean that effect\". Only write the two new prompts in two lines."},
        {"role": "user", "content": '"If he will receive any salary, does this mean that mason didn\'t leave his job?" and "If he will receive any salary, does this mean that Mason left his job?"'},
        {"role": "assistant", "content": 'If he will receive any salary, does this mean that Mason kept his job?\nIf he will receive any salary, does this mean that Mason lost his job?  '},
        {"role": "user", "content": '"If he won\'t buy a house, does this imply that jack didn\'t win the lottery?" and "If he won\'t buy a house, does this mean that Jack won the lottery?"'},
        {"role": "assistant", "content": 'If he won\'t buy a house, does this imply that Jack missed the lottery?\nIf he won\'t buy a house, does this mean that Jack hit the lottery?'},
        {"role": "user", "content": '"If he won\'t pass with flying colors, does this imply that levi isn\'t studying for his exam?" and "If he won\'t pass with flying colors, does this imply that Levi is studying for his exam?"'},
        {"role": "assistant", "content": "If he won't pass with flying colors, does this imply that Levi skipped studying for his exam?\nIf he won't pass with flying colors, does this imply that Levi kept studying for his exam?"},
        {"role": "user", "content": '"If he won\'t stay up late to study, does this mean that levi doesn\'t have an exam tomorrow?" and "If he won\'t stay up late to study, does this imply that Levi has an exam tomorrow?"'},
        {"role": "assistant", "content": "If he won't stay up late to study, does this mean that Levi has no exam tomorrow?\nIf he won't stay up late to study, does this imply that Levi has his exam tomorrow?"},
    ],
}

def convert_dataset(model, tokenizer, problem_type, json_path, output_json_path):
    """
    Convert the LogicBench dataset to a format that can be used for circuit discovery more easily by having the same number of tokens in the two prompts.
    """
    
    print(f"=========================\nPROCESSING {json_path}\n=========================")
    data = json.load(open(json_path))
    os.makedirs(output_json_path, exist_ok=True)

    data_col = "data_samples"
    data_col = data_col if (data_col in data.keys()) else "samples"


    # parse the data
    for sample in tqdm(data[data_col]):
        clean = sample["qa_pairs"][0]["question"]
        # corrupt_idx = random.randint(1,len(sample["qa_pairs"])-1)
        corrupt_idx = 1
        corrupt = sample["qa_pairs"][corrupt_idx]["question"]
        
        # send the clean and corrupt samples to the model
        message = f"\"{clean}\" and \"{corrupt}\""
        
        messages[problem_type].append({"role": "user", "content": message})
        text = tokenizer.apply_chat_template(
            messages[problem_type],
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        success = False
        do_sample = False
        
        while not success:
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=do_sample,
                temperature=0.2,
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            response = response.split("Assistant:")[-1].strip()
            print(f"Assistant: {response}")
            
            prompts = response.split("\n")
            if len(prompts) == 2:
                new_clean = prompts[0].strip()
                new_corrupt = prompts[1].strip()
                success = True
            else:
                do_sample = True
                print("Failed to generate two prompts. Retrying...")
            
        sample["qa_pairs"][0]["question"] = new_clean
        sample["qa_pairs"][1]["question"] = new_corrupt

        messages[problem_type].pop()
        
    # save the new dataset
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4)
        

if __name__ == "__main__":
    conf: ProjectConfig = load_yaml_config("conf.yaml")

    model_name = conf.convert_dataset.model
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    for i in tqdm(range(len(conf.convert_dataset.dataset_files))):
        json_path = conf.convert_dataset.dataset_files[i]
        output_json_path = conf.convert_dataset.output_files[i]
        problem_type = conf.convert_dataset.format[i]
        
        convert_dataset(
            model,
            tokenizer,
            problem_type,
            json_path,
            output_json_path
        )
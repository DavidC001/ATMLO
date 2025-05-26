from torch.utils.data import Dataset
import torch
import os
import json
from utils.data.logic_dataset import create_dataset
from tqdm import tqdm

class AC_data(Dataset):
    
    def __init__(self, input_jsons, template, tokenizer, global_padding=False, alignment=True):
        """
        Args:
            input_jsons: list of json files to load
            template: template to use for the dataset
            tokenizer: tokenizer object
            global_padding: if True, pad the input ids to the same length
            alignment: if True, align the dataset with the tokenizer so that all pairs of clean and corrupt prompts have the same length
            
        """
        super().__init__()
        
        self.text = []
        self.samples = []
        self.attention_mask = []
        self.GT = []
        self.GT_opposite = []
        
        out_path = "temp.json"
        create_dataset(
            input_jsons=input_jsons,
            out_json=out_path,
            template=template,
            global_padding=global_padding,
            tokenizer=tokenizer,
            alignment=alignment
        )
        
        # load the data from the json
        with open(out_path) as f:
            data = json.load(f)
            for sample in tqdm(data["prompts"]):
                self.text.append(sample["clean"])
                self.text.append(sample["corrupt"])
                
                clean = tokenizer(sample["clean"], return_tensors="pt")
                corrupt = tokenizer(sample["corrupt"], return_tensors="pt")
                self.samples.append(clean["input_ids"][0])
                self.samples.append(corrupt["input_ids"][0])
                
                self.attention_mask.append(clean["attention_mask"][0])
                self.attention_mask.append(corrupt["attention_mask"][0])
                
                correct = [tokenizer.encode(answer, add_special_tokens=False)[0] for answer in sample["answers"]]
                wrong = [tokenizer.encode(w_answer, add_special_tokens=False)[0] for w_answer in sample["wrong_answers"]]
                
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
            "attention_mask": self.attention_mask[index],
            "correct": self.GT[index],
            "wrong": self.GT_opposite[index],
        }

def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences.
    Pads the input_ids and attention_mask to the maximum length in the batch.
    """
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    
    # Pad the sequences
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_side="left")
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_side="left")
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "text": [item["text"] for item in batch],
        "correct": [item["correct"] for item in batch],
        "wrong": [item["wrong"] for item in batch],
    }
    
def get_dataloader(input_jsons, template, tokenizer, batch_size=32, global_padding=False, alignment=True, split=False):
    """
    Returns a DataLoader for the given dataset.
    
    Args:
        input_jsons: list of json files to load
        template: template to use for the dataset
        tokenizer: tokenizer object
        batch_size: batch size for the DataLoader
        global_padding: if True, pad the input ids to the same length
        alignment: if True, align the dataset with the tokenizer so that all pairs of clean and corrupt prompts have the same length
        split: if True, split the dataset into train and test sets
        
    Returns:
        DataLoader object | tuple of DataLoader objects (train, test) if split is True
    """
    dataset = AC_data(input_jsons, template, tokenizer, global_padding, alignment)
    
    # If split is True, we can split the dataset into train and test sets
    if split:
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
        return train_loader, test_loader
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        return dataloader
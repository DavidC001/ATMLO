import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import json

class Probe_data(Dataset):
    
    def __init__(self, 
                 base_dataset_path="datasets\probing\Qwen\Qwen2.5-1.5B-Instruct\modus_tollens", 
                 layers=["attention_layer0_out_ln", "MLP_layer4_after"],
                 train_test_split=0.8, train=True
                ):
        """
        Args:
            base_dataset_path: path to the dataset folder
        """
        clean_path = os.path.join(base_dataset_path, "clean")
        corrupt_path = os.path.join(base_dataset_path, "corrupt")
        
        clean_data = os.listdir(clean_path)
        corrupt_data = os.listdir(corrupt_path)
        
        train_size = int(len(clean_data) * train_test_split)
        
        if train:
            clean_data = clean_data[:train_size]
            corrupt_data = corrupt_data[:train_size]
        else:
            clean_data = clean_data[train_size:]
            corrupt_data = corrupt_data[train_size:]
        
        self.data_len = len(clean_data)*2
        self.layers = layers
        
        
        self.data = {}
        self.class_ids = []
        for layer in layers:
            self.data[layer] = []
            
        print("Loading Clean Data")
        self.load_data(0, clean_data, clean_path, layers)
        print("Loading Corrupt Data")
        self.load_data(1, corrupt_data, corrupt_path, layers)
    
    def load_data(self, class_id, files, path, layers):
        """
        Load the data from the json files
        """
        for fname in tqdm(files):
            if not fname.endswith(".json"):
                continue
            
            fpath = os.path.join(path, fname)
            with open(fpath, "r") as f:
                data = json.load(f)
                
                self.class_ids.append(class_id)
                for layer in layers:
                    self.data[layer].append(torch.tensor(data[layer]))
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        """
        Get the data at the index idx
        """
        item = {}
        for layer in self.layers:
            item[layer] = self.data[layer][idx]
        
        item["class_id"] = self.class_ids[idx]
        item["class_id"] = torch.tensor(item["class_id"])
        
        return item
    

def get_dataloader(base_dataset_path="datasets\probing\Qwen\Qwen2.5-1.5B-Instruct\modus_tollens",
                   layers=["attention_layer0_out_ln", "MLP_layer4_aften"],
                   train_test_split=0.8,
                   batch_size=32):
    """
    Get the dataloader for the dataset
    """
    train_dataset = Probe_data(base_dataset_path, layers, train_test_split, True)
    test_dataset = Probe_data(base_dataset_path, layers, train_test_split, False)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

if __name__ == "__main__":
    dataset = Probe_data()
    print(dataset[0])
    print(dataset[1])
    print(dataset[2])
    
    # create dataloader
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for batch in dataloader:
        print(batch)
        break
    
    
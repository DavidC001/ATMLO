from torch.utils.data import Dataset
import numpy as np
import torch
from config import DataGenerationConfig
import os

class PriceTask(Dataset):
    def __init__(self, config: DataGenerationConfig):
        self.config = config
        self.data = []
        self.generate_data()
        
    def generate_data(self):
        np.random.seed(self.config.seed)
        system_prompt = "Please say yes only if it costs between [{x}] and [{y}] dollars, otherwise no."
        user_prompt = "[{cost}]"
        
        # gemerate True samples
        for _ in range(self.config.num_samples):
            x = np.random.randint(1, 1000)
            y = np.random.randint(x+1, 1001)
            sys_prompt = system_prompt.format(x=x, y=y)
            cost = np.random.randint(x, y)
            prompt = user_prompt.format(cost=cost)
            
            data = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ]
            
            self.data.append((data, 1))
            
        # generate False samples
        # for _ in range(self.config.num_samples//2):
        #     x = np.random.randint(1, 1000)
        #     y = np.random.randint(x+1, 1001)
        #     sys_prompt = system_prompt.format(x=x, y=y)
            
        #     greater = np.random.choice([True, False])
        #     if greater:
        #         cost = np.random.randint(y+1, 1002)
        #     else:
        #         cost = np.random.randint(0, x)
        #     prompt = user_prompt.format(cost=cost)
            
        #     data = [
        #         {"role": "system", "content": sys_prompt},
        #         {"role": "user", "content": prompt}
        #     ]
            
        #     self.data.append((data, 0))
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            "prompt": self.data[idx][0],
            "label": self.data[idx][1]
        }
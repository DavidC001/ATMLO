import sys
sys.path.append(".")

from utils.probe_dataloader import get_dataloader
from probing.model import ProbeLayer

import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import os
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

def train_probe_model(model, train_loader, test_loader, num_epochs=10, learning_rate=1e-3, layers=[]):
    """
    Train the probe model on the dataset
    """
    # Define the optimizer and loss function
    optimizers = {
        layer: AdamW(model[layer].parameters(), lr=learning_rate) for layer in layers
    }
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for layer in layers: model[layer].train()
    for epoch in range(num_epochs):
        total_loss = {
            layer: 0 for layer in layers
        }
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            labels = batch["class_id"].to(device)
            
            for layer in layers:
                inputs = batch[layer].to(device)
                inputs = inputs.view(inputs.size(0), -1)
                
                optimizers[layer].zero_grad()
                
                outputs = model[layer](inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizers[layer].step()
                
                total_loss[layer] += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss}")
    
    # Evaluation loop
    for layer in layers: model[layer].eval()
    all_preds = {
        layer: [] for layer in layers
    }
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            labels = batch["class_id"].to(device)
            all_labels.extend(labels.cpu().numpy())
            
            for layer in layers:
                inputs = batch[layer].to(device)
                inputs = inputs.view(inputs.size(0), -1)
                
                outputs = model[layer](inputs)
                preds = torch.argmax(outputs, dim=1).detach().cpu()
                
                all_preds[layer].extend(preds)
    
    accuracy = {
        layer: accuracy_score(all_labels, np.round(preds)) for layer, preds in all_preds.items()
    }
    f1 = {
        layer: f1_score(all_labels, np.round(preds)) for layer, preds in all_preds.items()
    }
    precision = {
        layer: precision_score(all_labels, np.round(preds)) for layer, preds in all_preds.items()
    }
    recall = {
        layer: recall_score(all_labels, np.round(preds)) for layer, preds in all_preds.items()
    }
    
    print(f"Accuracy: {accuracy}, F1 Score: {f1}, Precision: {precision}, Recall: {recall}")
    
    return accuracy, f1, precision, recall

def main():
    # Load the dataset
    layers=[
        "MLP_layer1_after_ln", 
        "MLP_layer3_after_ln", 
        "MLP_layer5_after_ln", 
        "MLP_layer8_after_ln",
        "MLP_layer10_after_ln"
    ]
    train_loader, test_loader = get_dataloader(
        base_dataset_path="datasets\probing\meta-llama\Llama-3.2-1B-Instruct\price_game",
        layers=layers,
        train_test_split=0.8,
        batch_size=32
    )
    # layers=[
    #     "MLP_layer1_after_ln", 
    #     "MLP_layer3_after_ln", 
    #     "MLP_layer5_after_ln", 
    #     "MLP_layer8_after_ln",
    #     "MLP_layer10_after_ln",
    #     "MLP_layer16_after_ln"
    # ]
    # train_loader, test_loader = get_dataloader(
    #     base_dataset_path="datasets\probing\Qwen\Qwen2.5-1.5B-Instruct\modus_tollens",
    #     layers=layers,
    #     train_test_split=0.5,
    #     batch_size=32
    # )
    
    models = {}
    for layer in layers:
        # Initialize the probe model
        input_dim = train_loader.dataset[0][layer].shape[0]
        num_classes = 2  # Binary classification
        models[layer] = ProbeLayer(input_dim, num_classes).to(device)
        
    # Train the probe model
    train_probe_model(models, train_loader, test_loader, layers=layers, num_epochs=10)
    
if __name__ == "__main__":
    main()
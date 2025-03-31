import torch
import numpy as np
from tqdm import tqdm, trange
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss

from tutorial_price_tagging_utils import (
    factual_sampler,
    bound_alignment_sampler,
    lower_bound_alignment_example_sampler,
)
import tutorial_price_tagging_utils as price_utils
from pyvene import (
    IntervenableModel,
    BoundlessRotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig,
    set_seed,
)

from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------
# Setup & Data Preparation
# ----------------------------

# Set the random seed
set_seed(42)

# Create the base LLaMA model, tokenizer, and config.
# (The model is assumed to be the instruct-tuned LLaMA-7B.)
# model = "gpt2"
# price_utils.mode = "alpaca"
model = "meta-llama/Llama-3.2-3B-Instruct"
price_utils.mode = "llama_3"
llama = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model)
llama_config = llama.config
llama = llama.to("cuda")
llama.eval()  # evaluation mode for factual performance


# Check the prealign task accuracy.

raw_prealign = factual_sampler(tokenizer, 5000, game="pricing_tag")
prealign_dataset = Dataset.from_dict(
    {"input_ids": raw_prealign[0], "labels": raw_prealign[1]}
)
prealign_dataset.set_format("torch", columns=["input_ids", "labels"])
prealign_dataloader = DataLoader(prealign_dataset, batch_size=8)

total_count = 0
correct_count = 0
with torch.no_grad():
    for step, inputs in enumerate(tqdm(prealign_dataloader)):
        for k, v in inputs.items():
            if v is not None and isinstance(v, torch.Tensor):
                inputs[k] = v.to(llama.device)

        # aligning forward!
        outputs = llama(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        )

        actual_test_labels = inputs["labels"][:, -1]
        pred_test_labels = torch.argmax(outputs.logits[:, -1], dim=-1)

        correct_labels = actual_test_labels == pred_test_labels

        total_count += len(correct_labels)
        correct_count += correct_labels.sum().tolist()
current_acc = round(correct_count / total_count, 2)
print(f"[WARNING: THIS NEEDS TO BE GOOD!] prealign task accuracy: {current_acc}")


# Prepare the dataset using the bound alignment sampler.
raw_data = bound_alignment_sampler(tokenizer, 10000, [lower_bound_alignment_example_sampler])
# Split into train (first 8000), eval (next 1000) and test (last 1000).
raw_train = (raw_data[0][:8000], raw_data[1][:8000], raw_data[2][:8000], raw_data[3][:8000])
raw_eval  = (raw_data[0][8000:9000], raw_data[1][8000:9000], raw_data[2][8000:9000], raw_data[3][8000:9000])
raw_test  = (raw_data[0][9000:],   raw_data[1][9000:],   raw_data[2][9000:],   raw_data[3][9000:])

train_dataset = Dataset.from_dict({
    "input_ids": raw_train[0],
    "source_input_ids": raw_train[1],
    "labels": raw_train[2],
    "intervention_ids": raw_train[3],
}).with_format("torch")
train_dataloader = DataLoader(train_dataset, batch_size=8)

eval_dataset = Dataset.from_dict({
    "input_ids": raw_eval[0],
    "source_input_ids": raw_eval[1],
    "labels": raw_eval[2],
    "intervention_ids": raw_eval[3],
}).with_format("torch")
eval_dataloader = DataLoader(eval_dataset, batch_size=8)

test_dataset = Dataset.from_dict({
    "input_ids": raw_test[0],
    "source_input_ids": raw_test[1],
    "labels": raw_test[2],
    "intervention_ids": raw_test[3],
}).with_format("torch")
test_dataloader = DataLoader(test_dataset, batch_size=8)

# ----------------------------
# Helper Functions
# ----------------------------

def simple_boundless_das_position_config(model_type, intervention_type, layer):
    """Creates a configuration for Boundless DAS given a specific layer."""
    config = IntervenableConfig(
        model_type=model_type,
        representations=[
            RepresentationConfig(layer, intervention_type),
        ],
        intervention_types=BoundlessRotatedSpaceIntervention,
    )
    return config

def compute_metrics(eval_preds, eval_labels):
    """Computes accuracy from logits and labels."""
    total_count = 0
    correct_count = 0
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        actual_test_labels = eval_label[:, -1]
        pred_test_labels = torch.argmax(eval_pred[:, -1], dim=-1)
        correct = (actual_test_labels == pred_test_labels)
        total_count += len(correct)
        correct_count += correct.sum().item()
    accuracy = round(correct_count / total_count, 2)
    return {"accuracy": accuracy}

def calculate_loss(logits, labels, intervenable, model_config):
    """Computes cross-entropy loss with an extra boundary loss term."""
    shift_logits = logits.contiguous().view(-1, model_config.vocab_size)
    shift_labels = labels.contiguous().view(-1)
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(shift_logits, shift_labels)
    # Add the boundary loss from the intervention (assumes one intervention here).
    for _, intervention in intervenable.interventions.items():
        boundary_loss = 1.0 * intervention.intervention_boundaries.sum()
    loss += boundary_loss
    return loss

# ----------------------------
# Experiment Setup
# ----------------------------

# Define the layers and token positions to test.
layers_to_test = [5, 10, 15, 20]
num_tokens = len(raw_train[0][0])  # number of tokens in the input
tokens_to_test = [i*10 for i in range(num_tokens//10)]

# Prepare a results matrix to store test accuracies.
results_matrix = np.zeros((len(layers_to_test), len(tokens_to_test)))

# Training configuration.
epochs = 3
gradient_accumulation_steps = 4

# ----------------------------
# Main Experiment Loop
# ----------------------------

for i, layer in enumerate(layers_to_test):
    for j, token_index in enumerate(tokens_to_test):
        print(f"\nTraining intervention at layer {layer} with token position {token_index}")
        
        # Reinitialize the intervention configuration and model for this combination.
        config_intervention = simple_boundless_das_position_config(type(llama), "block_output", layer)
        intervenable = IntervenableModel(config_intervention, llama)
        intervenable.set_device("cuda")
        intervenable.disable_model_gradients()
        
        # Compute total steps and create a temperature schedule.
        total_steps = len(train_dataloader) * epochs
        
        warm_up_steps = int(0.1 * total_steps)
        temperature_start = 50.0
        temperature_end = 0.1
        temperature_schedule = torch.linspace(temperature_start, temperature_end, total_steps).to(torch.bfloat16).to("cuda")
        intervenable.set_temperature(temperature_schedule[0])
        
        # Setup optimizer: include parameters from rotate_layer and intervention boundaries.
        optimizer_params = []
        for _, intervention in intervenable.interventions.items():
            optimizer_params.append({"params": intervention.rotate_layer.parameters()})
            optimizer_params.append({"params": intervention.intervention_boundaries, "lr": 1e-2})
        optimizer = torch.optim.Adam(optimizer_params, lr=1e-3)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(0.1 * total_steps),
                                                    num_training_steps=total_steps)
        
        # Training loop.
        intervenable.model.train()
        total_step = 0
        for epoch in trange(epochs, desc="Epochs"):
            epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch}")
            for step, inputs in enumerate(epoch_iterator):
                # Move inputs to CUDA.
                for key, value in inputs.items():
                    if value is not None and isinstance(value, torch.Tensor):
                        inputs[key] = value.to("cuda")
                        
                # Apply the intervention at the specified token position.
                # Note: the intervention dict now uses our token_index.
                _, counterfactual_outputs = intervenable(
                    {"input_ids": inputs["input_ids"]},
                    [{"input_ids": inputs["source_input_ids"]}],
                    {"sources->base": token_index},
                )
                
                # Optionally compute metrics on this batch.
                metrics = compute_metrics([counterfactual_outputs.logits], [inputs["labels"]])
                
                # Compute loss.
                loss = calculate_loss(counterfactual_outputs.logits, inputs["labels"], intervenable, llama_config)
                loss_value = loss.item()
                epoch_iterator.set_postfix({"loss": loss_value, "acc": metrics["accuracy"]})
                
                # Backpropagation with gradient accumulation.
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                loss.backward()
                if total_step % gradient_accumulation_steps == 0:
                    if not (gradient_accumulation_steps > 1 and total_step == 0):
                        optimizer.step()
                        scheduler.step()
                        intervenable.set_zero_grad()
                        intervenable.set_temperature(temperature_schedule[total_step])
                total_step += 1
        
        # Evaluation on the test set.
        intervenable.model.eval()
        eval_preds = []
        eval_labels = []
        with torch.no_grad():
            for inputs in tqdm(test_dataloader, desc=f"Evaluating layer {layer} token {token_index}"):
                for key, value in inputs.items():
                    if value is not None and isinstance(value, torch.Tensor):
                        inputs[key] = value.to("cuda")
                _, counterfactual_outputs = intervenable(
                    {"input_ids": inputs["input_ids"]},
                    [{"input_ids": inputs["source_input_ids"]}],
                    {"sources->base": token_index},
                )
                eval_preds.append(counterfactual_outputs.logits.to("cpu"))
                eval_labels.append(inputs["labels"].to("cpu"))
        metrics = compute_metrics(eval_preds, eval_labels)
        accuracy = metrics["accuracy"]
        print(f"Test accuracy for layer {layer} and token {token_index}: {accuracy}")
        results_matrix[i, j] = accuracy

# ----------------------------
# Save Results
# ----------------------------

# Save as CSV. The header is a comma-separated list of token positions.
header = ",".join(map(str, tokens_to_test))
np.savetxt("evaluation_matrix.csv", results_matrix, delimiter=",", header=header, comments="")
# Also save as a NumPy binary file.
np.save("evaluation_matrix.npy", results_matrix)

print("\nFinal Evaluation Matrix (rows: layers, columns: token positions):")
print(results_matrix)

# save as image
import matplotlib.pyplot as plt
plt.imshow(results_matrix, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.savefig('evaluation_matrix.png')

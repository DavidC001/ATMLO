from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.1-70B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [
    {"role": "system", "content": "you need to rewrite the following modus tollens true and false statement to have them be the same number of tokens. The template to use is \"if condition, does this mean that effect\". Only write the two new prompts in two lines."},
    {"role": "user", "content": '"If he will receive any salary, does this mean that mason didn\'t leave his job?" and "If he will receive any salary, does this mean that Mason left his job?"'},
    {"role": "assistant", "content": 'If he will receive any salary, does this mean that Mason kept his job?\nIf he will receive any salary, does this mean that Mason lost his job?  '},
    {"role": "user", "content": '"If he won\'t buy a house, does this imply that jack didn\'t win the lottery?" and "If he won\'t buy a house, does this mean that Jack won the lottery?"'},
    {"role": "assistant", "content": 'If he won\'t buy a house, does this imply that Jack missed the lottery?\nIf he won\'t buy a house, does this mean that Jack hit the lottery?'},
    {"role": "user", "content": '"If he won\'t pass with flying colors, does this imply that levi isn\'t studying for his exam?" and "If he won\'t pass with flying colors, does this imply that Levi is studying for his exam?"'},
    {"role": "assistant", "content": "If he won't pass with flying colors, does this imply that Levi skipped studying for his exam?\nIf he won't pass with flying colors, does this imply that Levi kept studying for his exam?"},
    {"role": "user", "content": '"If he won\'t stay up late to study, does this mean that levi doesn\'t have an exam tomorrow?" and "If he won\'t stay up late to study, does this imply that Levi has an exam tomorrow?"'},
    {"role": "assistant", "content": "If he won't stay up late to study, does this mean that Levi has no exam tomorrow?\nIf he won't stay up late to study, does this imply that Levi has his exam tomorrow?"},
]

while True:
    
    input_text = input("User: ")
    if input_text.lower() == "exit":
        break
    
    messages.append({"role": "user", "content": input_text})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    response = response.split("Assistant:")[-1].strip()
    print(f"Assistant: {response}")
    
    messages.pop()

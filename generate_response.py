import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

# Read the generated prompt
with open("generate_prompt.py", "r") as f:
    exec(f.read())  # This will create the 'prompt' variable

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate a response
print("Generating response...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode the response
response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

print("Model Response:")
print(response)

# Save the response to a file
with open("model_response.txt", "w") as f:
    f.write(f"Prompt:\n{prompt}\n\nResponse:\n{response}")
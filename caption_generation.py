from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Define a function for caption generation
def generate_caption(prompt, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return caption

# Example usage
prompt = "A beautiful scenery of mountains and rivers"
caption = generate_caption(prompt)
print("Generated Caption:", caption)

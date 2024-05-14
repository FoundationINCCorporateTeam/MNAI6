from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# Ensure the model is in evaluation mode
model.eval()

def generate_response(prompt, max_length=100):
    # Encode the prompt into input IDs
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    # Generate response from the model
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    # Decode the output IDs back into text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def chatbot_response(user_input):
    return generate_response(user_input)

def chat():
    print("Chatbot is ready to chat! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = chatbot_response(user_input)
        print("Bot:", response)

if __name__ == "__main__":
    chat()

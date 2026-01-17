import logging
import torch
from collections import deque
from transformers import AutoTokenizer
from tensorlink import DistributedModel, UserNode


model_name = "Qwen/Qwen3-8B-Instruct"

# Parameters for distributed model request.
BATCH_SIZE = 16
PIPELINES = 1
DP_FACTOR = 1

# Chatbot parameters
MAX_HISTORY_TURNS = 6
MAX_TOKENS = 2048
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.4

chat_history = deque(maxlen=MAX_HISTORY_TURNS)
chat_history.append(
    "System: You are a helpful assistant. Respond directly to the user's questions."
)


if __name__ == "__main__":
    # Specifically spawning a user node to change debug print level
    user = UserNode(print_level=logging.DEBUG)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Get distributed model with HF model name
    distributed_model = DistributedModel(
        model_name, training=False, n_pipelines=PIPELINES, node=user
    )

    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "reset":
            chat_history.clear()
            chat_history.append(
                "System: You are a helpful assistant. Respond directly to the user's questions."
            )
            print("Conversation reset.")
            continue

        # Append user input to history
        chat_history.append(f"User: {user_input}")

        # Tokenize input
        full_prompt = "\n".join(chat_history) + "\nBot:"

        inputs = tokenizer.encode(
            full_prompt, return_tensors="pt", truncation=True, max_length=MAX_TOKENS
        )

        # Generate response
        with torch.no_grad():
            outputs = distributed_model.generate(
                inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,  # temperature only has effect if sampling is on
                num_beams=2,
            )

        # Decode and print response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract bot response
        bot_reply = response.split("Bot:")[-1].strip()
        print(f"Bot: {bot_reply}\n")

        # Add bot response to history with clear marker
        chat_history.append(f"Bot: {bot_reply}")

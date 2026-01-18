"""
Distributed Chatbot Example

Shows how to use DistributedModel.generate() with
conversation history.
"""

import torch
import logging
from collections import deque
from transformers import AutoTokenizer
from tensorlink.ml import DistributedModel
from tensorlink.nodes import User, UserConfig

MODEL_NAME = "Qwen/Qwen3-8B-Instruct"

MAX_HISTORY_TURNS = 6
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.4

if __name__ == "__main__":
    user = User(config=UserConfig(print_level=logging.INFO))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = DistributedModel(model=MODEL_NAME, training=False, node=user)

    history = deque(maxlen=MAX_HISTORY_TURNS)
    history.append("System: You are a helpful assistant.")

    print("Chatbot ready. Type 'exit' to quit.\n")

    while True:
        text = input("You: ")
        if text.lower() == "exit":
            break

        history.append(f"User: {text}")
        prompt = "\n".join(history) + "\nAssistant:"

        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reply = response.split("Assistant:")[-1].strip()

        print(f"Assistant: {reply}\n")
        history.append(f"Assistant: {reply}")

    user.cleanup()

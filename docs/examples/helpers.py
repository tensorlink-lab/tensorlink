"""
This file just contains some helper functions for the examples.
"""

from collections import deque

MAX_HISTORY_TURNS = 10


def chat_loop(generate_fn, model=None, tokenizer=None):
    history = deque(maxlen=MAX_HISTORY_TURNS)

    while True:
        text = input("You: ").strip()
        if text.lower() == "exit":
            break

        history.append({"role": "user", "content": text})

        try:
            if model is not None and tokenizer is not None:
                reply = generate_fn(model, tokenizer, list(history))
            else:
                reply = generate_fn(list(history))

            print(f"Assistant: {reply}\n")
            history.append({"role": "assistant", "content": reply})

        except Exception as e:
            print(f"Error: {e}\n")
            history.pop()


system_message = "You are a helpful assistant."

def create_conversation(sample):
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": sample["Question"]},
            {"role": "assistant", "content": sample["Answer"]}
        ]
    }
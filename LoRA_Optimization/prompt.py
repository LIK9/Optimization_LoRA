import torch
import json

def format_prompt_train_single(example, tokenizer):
    system_prompt = """You are an intelligent conversational AI. I’ve set up a persona profile for you. 
Please generate a response based on the persona profile and the dialogue history. 
The response you generate should be consistent with the persona profile I set for you and coherent with the dialogue history. 
Moreover, the richer the content you generate, the better."""

    dialogue_history = example["dialogue"][:-1]  # 마지막 dialogue는 response
    dialogue_text = ""
    for i, turn in enumerate(dialogue_history):
        speaker = "User" if i % 2 == 0 else "You"
        dialogue_text += f"»{speaker}: {turn.strip()}\n" 

    persona_text = "\n".join([f"»{i+1}. {p.strip()}" for i, p in enumerate(example["persona_info"])])

    user_input = f"""The dialogue history between you and a user is as follows:
{dialogue_text.strip()}
The persona profile I set up for you is as follows (in first person):
{persona_text}
Now please generate a response to the user based on the dialogue history and the profile."""

    
    assistant_response = example["dialogue"][-1].strip() # response

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": assistant_response}
    ]


    tokenized = tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=True).squeeze(0).tolist()
    text = tokenizer.apply_chat_template(messages,tokenize=False)

    cut_messages = messages[:-1]
    prompt_tokenized = tokenizer.apply_chat_template(cut_messages, return_tensors="pt", tokenize=True).squeeze(0).tolist()
    prompt_len = len(prompt_tokenized)

    labels = [-100]*prompt_len + tokenized[prompt_len:]
    example['input_ids'] = tokenized
    example['text'] = text
    example['labels'] = labels

    return example

def format_prompt_test_single(example, tokenizer):
    persona_text = "\n".join([f"»{i+1}. {p.strip()}" for i, p in enumerate(example["persona_info"])])


    system_prompt = """You are an intelligent conversational AI. I’ve set up a persona profile for you. 
Please generate a response based on the persona profile and the dialogue history. 
The response you generate should be consistent with the persona profile I set for you and coherent with the dialogue history. 
Moreover, the richer the content you generate, the better."""

    dialogue_history = example["dialogue"][:-1]  # 마지막 dialogue는 response
    dialogue_text = ""
    for i, turn in enumerate(dialogue_history):
        speaker = "User" if i % 2 == 0 else "You"
        dialogue_text += f"»{speaker}: {turn.strip()}\n" 


    user_input = f"""The dialogue history between you and a user is as follows:
{dialogue_text.strip()}
The persona profile I set up for you is as follows (in first person):
{persona_text}
Now please generate a response to the user based on the dialogue history and the profile."""

    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    example['text'] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return example


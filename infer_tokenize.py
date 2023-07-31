

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
simple_audio_conv_multimodal = {
    "system": "You are a helpful language and speech assistant. You are able to understand the speech content that the user provides, and assist the user with a variety of tasks using natural language.",
    "roles": {"human": "USER", "gpt": "ASSISTANT"},
}

def tokenize_baichuan(item, tokenizer):
    roles = simple_audio_conv_multimodal["roles"]
    input_ids = []
    if "instruction" in item and len(item["instruction"]) > 0:
        system = item["instruction"]
    else:
        system = simple_audio_conv_multimodal["system"]
    system_ids = tokenizer.encode(system, add_special_tokens=False)
    input_ids += system_ids
    for i, turn in enumerate(item["conversations"]):
        role = roles.get(turn['from'], 'USER')
        content = turn['value']
        content = content.strip()
        if role == 'ASSISTANT' and content != '':
            content += '</s>'
        role_ids = tokenizer.encode(role + ":", add_special_tokens=False)
        content_ids = tokenizer.encode(content, add_special_tokens=False, truncation=True,
                                       max_length=tokenizer.model_max_length)
        input_ids += role_ids + content_ids

    if tokenizer.add_bos_token:
        input_ids = [tokenizer.bos_token_id] + input_ids

    input_ids = input_ids[-tokenizer.model_max_length:]

    return input_ids

def tokenize_Cllama2(item, tokenizer):
    input_ids = []
    if "instruction" in item and len(item["instruction"]) > 0:
        system = item["instruction"]
    else:
        system = simple_audio_conv_multimodal["system"]
    system = B_SYS + system + E_SYS
    system_ids = tokenizer.encode(system, add_special_tokens=False)
    input_ids += system_ids
    item["conversations"][0]['value'] = system + item["conversations"][0]['value']
    for i, turn in enumerate(item["conversations"]):
        role = turn['from']
        content = turn['value']
        content = content.strip()
        if role == 'human':
            content = f"{B_INST} {content} {E_INST} "
            content_ids = tokenizer.encode(content)
        else:
            # assert role == "gpt"
            if content == "":
                content_ids = []
            else:
                content = f"{content} "
                content_ids = tokenizer.encode(content, add_special_tokens=False) + [tokenizer.eos_token_id]   # add_special_tokens=False remove bos token, and add eos at the end
        input_ids += content_ids

    input_ids = input_ids[-tokenizer.model_max_length:]

    return input_ids


def tokenize(item, tokenizer, llm_type):
    if llm_type == "Chinese_llama2":
        return tokenize_Cllama2(item, tokenizer)
    elif llm_type == "baichuan":
        return tokenize_baichuan(item, tokenizer)
    else:
        raise ValueError (f"Invalid llm type {llm_type}, please choose in ['Chinese_llama2', 'baichuan']")
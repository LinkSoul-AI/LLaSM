import os
import librosa
import argparse

import torch

from transformers import AutoTokenizer
from transformers import (
    WhisperProcessor,
    WhisperModel,
)

from llasm import LlaaaLlamaForCausalLM
from infer_tokenize import tokenize
from logger import print_signature


DEFAULT_AUDIO_PATCH_TOKEN = "<au_patch>"
DEFAULT_AUDIO_START_TOKEN = "<au_start>"
DEFAULT_AUDIO_END_TOKEN = "<au_end>"

class Setting:
    def __init__(self):
        self.device = os.environ.get("LLASM_DEVICE", "cuda")
        self.llasm_context_len = 2048
        self.sampling_rate = 16000
        self.audio_token_len = 64
        self.stop = "</s>"

CONFIG = Setting()


def main(args):
    input_audio_file = args.input_audio_file
    temperature = args.temperature
    max_new_tokens = args.max_new_tokens

    # step0: load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llasm_model)
    # step0-1: add special token <au_patch>/<au_start>/<au_end>
    tokenizer.add_tokens([DEFAULT_AUDIO_PATCH_TOKEN], special_tokens=True)
    tokenizer.add_tokens([DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN], special_tokens=True)

    # step1: load model
    model = LlaaaLlamaForCausalLM.from_pretrained(
        args.llasm_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True).to(CONFIG.device)

    # step2: load audio processor
    audio_processor = WhisperProcessor.from_pretrained(args.llasm_audio_tower, torch_dtype=torch.float16)

    # step3: load audio tower
    audio_tower = WhisperModel.from_pretrained(
        args.llasm_audio_tower,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True).to(CONFIG.device)
    # step3-1: update audio_tower config for setting special tokens
    audio_config = audio_tower.config
    audio_config.audio_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_AUDIO_PATCH_TOKEN])[0]
    audio_config.audio_start_token, audio_config.audio_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN])
    model.get_model().audio_tower[0] = audio_tower

    # step4 preprocessing input audio
    audio, _ = librosa.load(input_audio_file, sr=CONFIG.sampling_rate)
    audio_feat = audio_processor(audio, sampling_rate=CONFIG.sampling_rate, return_tensors="pt").input_features
    audio_feat = audio_feat.unsqueeze(0).unsqueeze(0).to(CONFIG.device, dtype=torch.float16)

    # step5: tokenize
    qs = DEFAULT_AUDIO_START_TOKEN + DEFAULT_AUDIO_PATCH_TOKEN * CONFIG.audio_token_len + DEFAULT_AUDIO_END_TOKEN
    input_qs = {
        "conversations": [{
            "from": "human",
            "value": qs,
        },{
            "from": "gpt",
            "value": ""
        }]
    }
    input_ids = torch.tensor([tokenize(input_qs, tokenizer, args.llm_type)]).to(CONFIG.device)

    # step6: infer run
    stop_str = CONFIG.stop
    output_ids = model.generate(input_ids,audios=audio_feat,do_sample=True,temperature=temperature,max_new_tokens=max_new_tokens)

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    
    label = []
    with open(input_audio_file[:-len('mp3')] + 'txt', 'r') as fh:
        for ln in fh:
            label.append(ln.strip())
    text = ''.join(label)
    
    print_signature()
    print (f"Human: {input_audio_file} ({text})")
    print (f"LLaSM: {outputs}")
    print ("="*80)
    print ("Go to the Demo page, and have a try!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_audio_file', type=str, default='./examples/0.mp3')
    parser.add_argument('--llasm_model', type=str, default='path/to/llasm_model')
    parser.add_argument('--llasm_audio_tower', type=str, default='path/to/whisper_large_v2')
    parser.add_argument('--llm_type', type=str, default='Chinese_llama2')
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    args = parser.parse_args()
    main(args)
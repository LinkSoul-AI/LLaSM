from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers import (
    WhisperProcessor,
    WhisperModel,
)


DEFAULT_AUDIO_PATCH_TOKEN = "<au_patch>"
DEFAULT_AUDIO_START_TOKEN = "<au_start>"
DEFAULT_AUDIO_END_TOKEN = "<au_end>"


class LlaaaConfig(LlamaConfig):
    model_type = "llaaa"


def load_whisper(audio_tower_name):
    model = WhisperModel.from_pretrained(audio_tower_name)
    model.config.forced_decoder_ids = None
    return model


class LlaaaLlamaModel(LlamaModel):
    config_class = LlaaaConfig

    def __init__(self, config: LlamaConfig):
        super(LlaaaLlamaModel, self).__init__(config)

        if hasattr(config, "mm_audio_tower"):
            # HACK: for FSDP
            self.audio_tower = [load_whisper(config.mm_audio_tower)]

        if hasattr(config, "use_mm_proj"):
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)

    def initialize_audio_modules(self, audio_tower, audio_token_len, pretrain_mm_mlp_adapter=None):
        self.config.mm_audio_tower = audio_tower

        processor = WhisperProcessor.from_pretrained(audio_tower)

        if not hasattr(self, 'audio_tower'):
            audio_tower = load_whisper(audio_tower)
        else:
            audio_tower = self.audio_tower[0]
        audio_tower.requires_grad_(False)
        audio_tower = audio_tower.to(torch.float16)
        self.audio_tower = [audio_tower]

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = 1280
        self.config.audio_token_len = audio_token_len

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(self.config.mm_hidden_size, self.config.hidden_size)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            self.mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        return dict(
            processor=processor,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        audios: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for LLaAA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        audio_tower = getattr(self, 'audio_tower', None)
        if audio_tower is not None and (input_ids.shape[1] != 1 or self.training) and audios is not None:
            audio_tower = audio_tower[0]  # HACK: for FSDP
            with torch.no_grad():
                bs_audio_features = []
                for audios_list in audios:
                    if len(audios_list) == 0:
                        dummy_audio_feature = torch.zeros(self.config.audio_token_len, self.config.mm_hidden_size, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                        audio_features = [dummy_audio_feature]
                    else:
                        audio_features = []
                        for audio in audios_list:
                            decoder_input_ids = torch.ones((1, self.config.audio_token_len)) * audio_tower.config.decoder_start_token_id
                            decoder_input_ids = decoder_input_ids.to(audio.device).to(torch.long)
                            audio_feature = audio_tower(audio, decoder_input_ids=decoder_input_ids).last_hidden_state
                            audio_features.append(audio_feature)
                    bs_audio_features.append(audio_features)

            audio_config = audio_tower.config
            new_input_embeds = []
            for cur_input_ids, cur_input_embeds, cur_audio_features in zip(input_ids, inputs_embeds, bs_audio_features):
                if (cur_input_ids == audio_config.audio_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal, for using both language and audio data
                    dummy_audio_features = self.mm_projector(cur_audio_features[0])
                    cur_input_embeds = cur_input_embeds + (0. * dummy_audio_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    continue
                if (cur_input_ids == audio_config.audio_start_token).sum() != (cur_input_ids == audio_config.audio_end_token).sum():
                    raise ValueError("The number of audio start tokens and audio end tokens should be the same.")
                audio_start_tokens = torch.where(cur_input_ids == audio_config.audio_start_token)[0]
                if len(audio_start_tokens) != len(cur_audio_features):
                    raise ValueError(f"The number of audio start tokens ({len(audio_start_tokens)}) and audio features ({len(cur_audio_features)}) should be the same.")
                for audio_start_token_pos, cur_audio_feature in zip(audio_start_tokens, cur_audio_features):
                    cur_audio_feature = self.mm_projector(cur_audio_feature)[0]
                    cur_audio_feature = cur_audio_feature.to(device=cur_input_embeds.device)
                    num_patches = cur_audio_feature.shape[0]
                    if cur_input_ids[audio_start_token_pos + num_patches + 1] != audio_config.audio_end_token:
                        raise ValueError("The audio end token should follow the audio start token.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat(
                            (cur_input_embeds[:audio_start_token_pos].detach(),
                             cur_input_embeds[audio_start_token_pos:audio_start_token_pos+1],
                             cur_audio_feature,
                             cur_input_embeds[audio_start_token_pos + num_patches + 1:audio_start_token_pos + num_patches + 2],
                             cur_input_embeds[audio_start_token_pos + num_patches + 2:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((
                            cur_input_embeds[:audio_start_token_pos+1],
                            cur_audio_feature,
                            cur_input_embeds[audio_start_token_pos + num_patches + 1:]), dim=0)
                new_input_embeds.append(cur_new_input_embeds)

            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(LlaaaLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class LlaaaLlamaForCausalLM(LlamaForCausalLM):
    config_class = LlaaaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlaaaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        audios: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            audios=audios
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "audios": kwargs.get("audios", None),
            }
        )
        return model_inputs

    def initialize_audio_tokenizer(self, tokenizer, device,
                                    tune_mm_mlp_adapter=False, pretrain_mm_mlp_adapter=None):
        num_new_tokens = tokenizer.add_tokens([DEFAULT_AUDIO_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        num_new_tokens += tokenizer.add_tokens([DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

        if tune_mm_mlp_adapter:
            self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
            for p in self.get_input_embeddings().parameters():
                p.requires_grad = True
            for p in self.get_output_embeddings().parameters():
                p.requires_grad = False

        if pretrain_mm_mlp_adapter and num_new_tokens > 0:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
            assert num_new_tokens == 3
            if input_embeddings.shape == embed_tokens_weight.shape:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
            elif embed_tokens_weight.shape[0] == num_new_tokens:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight
            else:
                raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        audio_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_AUDIO_PATCH_TOKEN])[0]
        audio_start_token, audio_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN])
        self.model.audio_tower[0].config.audio_patch_token = audio_patch_token
        self.model.audio_tower[0].config.audio_start_token = audio_start_token
        self.model.audio_tower[0].config.audio_end_token = audio_end_token


AutoConfig.register("llaaa", LlaaaConfig)
AutoModelForCausalLM.register(LlaaaConfig, LlaaaLlamaForCausalLM)

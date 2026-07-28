import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM


def collate_ids(
    sequences: list[torch.Tensor],
    reward_masks: list[torch.Tensor],
    pad_token_id: int,
):
    input_ids = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=pad_token_id,
    )

    attention_mask = (input_ids != pad_token_id).long()

    reward_mask = pad_sequence(
        reward_masks,
        batch_first=True,
        padding_value=0,
    ).bool()

    return input_ids, attention_mask, reward_mask


def get_shift_logits_and_labels(
    hf_model: AutoModelForCausalLM,
    sequences: list[torch.Tensor],
    reward_masks: list[torch.Tensor],
    pad_token_id: int,
):
    device = next(hf_model.parameters()).device

    input_ids, attention_mask, reward_mask = collate_ids(
        sequences=sequences,
        reward_masks=reward_masks,
        pad_token_id=pad_token_id,
    )

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    reward_mask = reward_mask.to(device)

    outputs = hf_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )

    logits = outputs.logits              # [B, T, V]

    shift_logits = logits[:, :-1, :]     # predicts input_ids[:, 1:]
    shift_labels = input_ids[:, 1:]      # actual next token

    shift_attention_mask = attention_mask[:, 1:].bool()
    shift_reward_mask = reward_mask[:, 1:].bool()

    # valid token AND token is in reward region
    shift_mask = shift_attention_mask & shift_reward_mask

    return shift_logits, shift_labels, shift_mask
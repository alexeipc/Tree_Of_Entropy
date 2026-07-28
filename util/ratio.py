import torch
import torch.nn.functional as F

def get_ratio(
    new_token_log_probs: torch.Tensor,   # [B,T,V]
    old_token_log_probs: torch.Tensor,   # [B,T,V]
    labels: torch.Tensor,       # [B,T]
    masks: torch.Tensor,        # [B,T]
):
    '''
    new_token_log_probs = new_log_probs.gather(
        -1,
        labels.unsqueeze(-1)
    ).squeeze(-1)                       # [B,T]

    old_token_log_probs = old_log_probs.gather(
        -1,
        labels.unsqueeze(-1)
    ).squeeze(-1)                       # [B,T]
    '''

    ratio = torch.exp(
        new_token_log_probs - old_token_log_probs
    )                                   # [B,T]

    # Ignore prompts and padding
    ratio = ratio * masks
    new_token_log_probs = new_token_log_probs * masks
    old_token_log_probs = old_token_log_probs * masks

    return ratio, new_token_log_probs, old_token_log_probs
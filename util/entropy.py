import numpy as np 
import torch
import torch.nn.functional as F

def calculate_entropy(logprobs):
    mask = torch.isfinite(logprobs)
    return -(logprobs[mask].exp() * logprobs[mask]).sum()

@torch.no_grad()
def calculate_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Args:
        logits: [B, T, V] tensor

    Returns:
        entropies: [B, T] tensor
    """
    top_k_logits, _ = logits.topk(256, dim=-1) # Change V to 256
    log_probs = F.log_softmax(top_k_logits, dim=-1)   # [B, T, V]
    probs = log_probs.exp()                     # [B, T, V]

    entropies = -(probs * log_probs).sum(dim=-1)  # [B, T]

    return entropies
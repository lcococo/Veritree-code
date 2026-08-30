from typing import Optional

import torch
import torch.nn.functional as functional


def spotlight_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_weights: torch.Tensor,
    ignore_index: int = -100,
    chunk_size: Optional[int] = 1024,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels must align with the first two logits dimensions")
    if token_weights.shape != labels.shape:
        raise ValueError("token_weights must have the same shape as labels")
    if torch.any(token_weights < 0):
        raise ValueError("token_weights must be nonnegative")
    if chunk_size is None:
        chunk_size = logits.shape[0] * max(logits.shape[1] - 1, 1)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    shifted_logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
    shifted_labels = labels[:, 1:].contiguous().view(-1).to(logits.device)
    shifted_weights = token_weights[:, 1:].contiguous().view(-1).to(logits.device, dtype=torch.float32)
    valid = shifted_labels != ignore_index
    effective_weights = shifted_weights * valid.to(torch.float32)
    denominator = effective_weights.sum()
    if denominator.item() == 0:
        return shifted_logits.sum() * 0
    numerator = torch.zeros((), device=logits.device, dtype=torch.float32)
    for start in range(0, shifted_logits.shape[0], chunk_size):
        end = min(start + chunk_size, shifted_logits.shape[0])
        chunk_valid = valid[start:end]
        if not torch.any(chunk_valid):
            continue
        chunk_labels = shifted_labels[start:end].clamp_min(0)
        log_probabilities = functional.log_softmax(shifted_logits[start:end].float(), dim=-1)
        losses = -log_probabilities.gather(-1, chunk_labels.unsqueeze(-1)).squeeze(-1)
        numerator = numerator + (losses * effective_weights[start:end]).sum()
    return numerator / denominator

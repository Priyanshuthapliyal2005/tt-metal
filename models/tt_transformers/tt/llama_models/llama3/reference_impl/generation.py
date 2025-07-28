from dataclasses import dataclass
from typing import Any, List

@dataclass
class ChatPrediction:
    message: str
    tokens: List[int]
    metadata: Any = None

@dataclass
class CompletionPrediction:
    completion: str
    tokens: List[int]
    metadata: Any = None

@dataclass
class TokenResult:
    token: int
    logprob: float
    metadata: Any = None

def sample_top_p(logits, p=0.9):
    # Dummy implementation for compatibility
    import numpy as np
    sorted_indices = np.argsort(logits)[::-1]
    sorted_logits = np.sort(logits)[::-1]
    cumulative_probs = np.cumsum(np.exp(sorted_logits)) / np.sum(np.exp(sorted_logits))
    cutoff = cumulative_probs <= p
    candidates = sorted_indices[cutoff]
    return candidates[0] if len(candidates) > 0 else sorted_indices[0]

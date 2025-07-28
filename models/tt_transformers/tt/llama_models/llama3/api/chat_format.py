from typing import Any, List

class ChatFormat:
    def __init__(self, system_prompt: str = ""):
        self.system_prompt = system_prompt

    def format(self, messages: List[Any]) -> str:
        return self.system_prompt + "\n" + "\n".join([m.content for m in messages])

def create_vision_mask(image_shape, dtype=None):
    # Dummy implementation for compatibility
    import numpy as np
    return np.ones(image_shape, dtype=dtype) if dtype else None

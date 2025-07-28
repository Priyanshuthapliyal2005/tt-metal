from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any

class StopReason(Enum):
    END_OF_SEQUENCE = 'end_of_sequence'
    MAX_TOKENS = 'max_tokens'
    STOP_SEQUENCE = 'stop_sequence'

@dataclass
class InterleavedTextMedia:
    text: str
    metadata: Optional[Any] = None

@dataclass
class ImageMedia:
    image: Any
    metadata: Optional[Any] = None

@dataclass
class UserMessage:
    content: str
    role: str = 'user'
    metadata: Optional[Any] = None

"""Token generation for the patient sequences."""

from dataclasses import dataclass, field
from typing import Dict, List

from dateutil import parser

from odyssey.data.constants import (
    CLASS,
    CLASS_TOKEN,
    LAB,
    MASK_TOKEN,
    MED,
    PAD,
    PAD_TOKEN,
    PROC,
    REGISTER,
    REGISTER_TOKEN,
    TIME_DELTA,
    UNKNOWN_TOKEN,
    VISIT_END,
    VISIT_END_TOKEN,
    VISIT_START,
    VISIT_START_TOKEN,
)


@dataclass
class TokenConfig:
    """Token configuration for the patient sequences.

    Parameters
    ----------
    pad_token : str, optional
        Padding token, by default "[PAD]"
    mask_token : str, optional
        Mask token, by default "[MASK]"
    visit_start_token : str, optional
        Visit start token, by default "[VS]"
    visit_end_token : str, optional
        Visit end token, by default "[VE]"
    class_token : str, optional
        Class token, by default "[CLS]"
    register_token : str, optional
        Register token, by default "[REG]"
    unknown_token : str, optional
        Unknown token, by default "[UNK]"
    time_delta_tokens : List[str], optional
        Time delta tokens, by default field(default_factory=lambda: (
            [f"[W_{i}]" for i in range(0, 4)]
            + [f"[M_{i}]" for i in range(0, 13)]
            + ["[LT]"]
        ))

    """

    pad_token: str = PAD_TOKEN
    mask_token: str = MASK_TOKEN
    visit_start_token: str = VISIT_START_TOKEN
    visit_end_token: str = VISIT_END_TOKEN
    class_token: str = CLASS_TOKEN
    register_token: str = REGISTER_TOKEN
    unknown_token: str = UNKNOWN_TOKEN
    time_delta_tokens: List[str] = field(
        default_factory=lambda: (
            [f"[W_{i}]" for i in range(0, 4)]
            + [f"[M_{i}]" for i in range(0, 13)]
            + ["[LT]"]
        )
    )
    token_type_mapping: Dict[str, int] = field(
        default_factory=lambda: {
            PAD: 0,
            CLASS: 1,
            VISIT_START: 2,
            VISIT_END: 3,
            TIME_DELTA: 4,
            LAB: 5,
            MED: 6,
            PROC: 7,
            REGISTER: 8,
        }
    )

    @property
    def special_tokens(self) -> List[str]:
        """Get the special tokens."""
        return [
            self.pad_token,
            self.mask_token,
            self.visit_start_token,
            self.visit_end_token,
            self.class_token,
            self.register_token,
            self.unknown_token,
        ] + self.time_delta_tokens


class TokenGenerator:
    """Generate tokens for the patient sequences.

    Parameters
    ----------
    max_seq_length : int
        Maximum sequence length
    token_config : TokenConfig
        Token configuration, by default TokenConfig()
    reference_time : str
        Reference time

    """

    def __init__(
        self,
        max_seq_length: int,
        token_config: TokenConfig,
        reference_time: str,
    ):
        """Initialize the token generator."""
        self.max_seq_length = max_seq_length
        self.token_config = token_config
        self.reference_time = parser.parse(reference_time)

    def add_tokens(self, events, encounters):
        """Add tokens to the events."""
        pass

    def truncate_or_pad(self, events, pad_events=False):
        """Truncate or pad the events."""
        pass

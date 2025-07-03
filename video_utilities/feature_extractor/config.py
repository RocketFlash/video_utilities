import torch
from dataclasses import dataclass, field

DEFAULT_PROCESSOR_PARAMS = dict(
    padding="max_length",
    truncation=True,
    max_length=64
)

@dataclass
class FeatureExtractorConfig():
    model_name: str = "google/siglip2-so400m-patch14-384"
    attn_implementation: str = "flash_attention_2"
    dtype: torch.dtype = torch.float16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 64
    normalize_vectors: bool = True
    processor_params: dict = field(default_factory=lambda: DEFAULT_PROCESSOR_PARAMS)
from .gpt import *
from .model import *

try:
    from .vllmModels import vllmModels
except ModuleNotFoundError as exc:
    _VLLM_IMPORT_ERROR = exc

    class vllmModels:  # type: ignore[no-redef]
        """Placeholder used when optional local-model dependencies are absent."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "vllmModels requires the original local-model dependency set "
                "(including torch, transformers, and vllm). Install "
                "requirements.txt for that optional path."
            ) from _VLLM_IMPORT_ERROR

__all__ = [
    'APIModel',
    'LLM',
    'vllmModels',
]

from typing import Type

from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    BaseImageProcessor,
    PretrainedConfig,
    ProcessorMixin,
)
from transformers.utils import is_torchvision_available


def register_image_processor(
    config: Type[PretrainedConfig], image_processor: Type[BaseImageProcessor]
):
    """
    register customized hf image processor while removing hf impl
    """
    if not is_torchvision_available():
        return
    AutoImageProcessor.register(
        config, slow_image_processor_class=image_processor, exist_ok=True
    )


def register_processor(config: Type[PretrainedConfig], processor: Type[ProcessorMixin]):
    """
    register customized hf processor while removing hf impl
    """
    AutoProcessor.register(config, processor, exist_ok=True)

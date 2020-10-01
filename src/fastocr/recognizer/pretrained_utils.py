import os
from .config_recognizer import RecognizerConfig
from .file_utils import *

class PreTrainedModel():

    def __init__(self, config: RecognizerConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, RecognizerConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `RecognizerConfig`. "
                "To create a model from a pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        # Save config in model
        self.config = config

    @classmethod
    def from_pretrained(cls, config, model_file_or_dir=CACHE_PATH):
        r"""
        Load a pretrained model from chosen configuration if it exists

        Examples :
            # >>> from fastocr.recognizer.recognizer import Recognizer
            # >>> from fastocr.recognizer.config_recognizer import RecognizerConfig
            # >>> config = RecognizerConfig()
            # Load pretrained model only from config :
            # >>> model = Recognize.from_pretrained(config)
            # Specify directory where cached models are saved :
            # >>> model = Recognize.from_pretrained(config, "/home/data/models/")
            # Specify model file :
            # >>> model = Recognize.from_pretrained(config, "/home/data/models/TPS-ResNet-BiLSTM-Attn.pth")
        """

        model_file_name = config.transformation \
                          + "-" \
                          + config.feature_extraction \
                          + "-" \
                          + config.sequence_modeling \
                          + "-" \
                          + config.prediction \
                          + ".pth"
        if os.path.isdir(model_file_or_dir):
            if os.path.isfile(os.path.join(model_file_or_dir, model_file_name)):
                archive_file = os.path.join(model_file_or_dir, model_file_name)
                cache_dir = model_file_or_dir
            else:
                raise EnvironmentError(
                    "Error no file named {} found in directory {}".format(
                        model_file_name,
                        model_file_or_dir
                    )
                )
        elif os.path.isfile(model_file_or_dir):
            if model_file_or_dir.split("/")[-1] == model_file_name:
                archive_file = model_file_or_dir
                cache_dir = None
            else:
                raise EnvironmentError(
                    "Specified file path does not fit config requirements. Should be {}".format(
                        model_file_name
                    )
                )
        else:
            if model_file_name in LINKS.keys():
                archive_file = LINKS[model_file_name]
                cache_dir = CACHE_PATH
            else:
                raise EnvironmentError(
                    "no pretrained weights found for model based on current config"
                )
        try:
            resolved_archive_file = cached_path(
                archive_file,
                cache_dir=cache_dir
            )
            if resolved_archive_file is None:
                raise EnvironmentError
        except EnvironmentError:
            msg = (
                f"Can't load weights for '{archive_file}'."
            )
            raise EnvironmentError(msg)

        # Instantiate model.
        model = cls(config, resolved_archive_file)

        return model

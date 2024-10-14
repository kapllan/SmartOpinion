# pylint: disable=missing-module-docstring
from pathlib import Path


def get_module_directory() -> Path:
    """Returns the source directory of the nlp4gpp module.

    :return: A path that points towards the source directory of  the nlp4gpp module.
    """
    return Path(__file__).parent

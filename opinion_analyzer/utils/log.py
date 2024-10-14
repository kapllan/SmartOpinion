"""This module handles the setup of the logging."""

import logging
import sys
from datetime import datetime

from opinion_analyzer import get_module_directory
from opinion_analyzer.utils.helper import get_main_config

config = get_main_config()


def get_logger(level=logging.INFO, log_to_stdout=True) -> logging:
    """Sets up the logging and returns the logging module to be used.

    :return: The logging module
    """
    path_to_logs = get_module_directory().parent / "logs"
    path_to_logs.mkdir(parents=True, exist_ok=True)

    # Configure logging
    handlers = [
        logging.FileHandler(
            filename=f"{config['paths']['logs']}/logs_{datetime.now().isoformat()}.log"
        )
    ]
    if log_to_stdout:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(pathname)s - %(lineno)d') - %(message)s",
        handlers=handlers,
    )

    return logging

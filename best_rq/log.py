import logging
import os
import sys


def get_logger(name: str) -> logging.Logger:
    """Configures a console logger.

    The log level is set depending on the environment variable 'BEST_RQ_LOG_LEVEL'.
    - 1: DEBUG
    - 2: INFO (default)
    - 3: WARNING
    - 4: ERROR
    - 5: CRITICAL

    :param name: The name of the module the logger will be used for.

    :return: The configured logger.
    """
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    log_level = int(os.environ.get("BEST_RQ_LOG_LEVEL", "2"))
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return logger

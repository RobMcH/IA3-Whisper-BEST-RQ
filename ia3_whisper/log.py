"""Implements logging functionalities."""

import logging
import os
import sys

import wandb


def get_logger(name: str) -> logging.Logger:
    """Configure a console logger.

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


def log_metrics(
    batch_idx: int,
    epoch: int,
    metrics: dict,
    use_wandb: bool,
    logger: logging.Logger,
    log_every: int = 1,
    loss_key: str = "loss",
) -> None:
    """Log the given metrics using the logger as well as wandb.

    Only logs using the logger every log_every^th batch to not pollute the logs.

    :param batch_idx: The index of the batch within the epoch.
    :param epoch: The epoch of the training.
    :param metrics: A dictionary holding the metrics to log. Keys 'loss', 'unique_targets' and 'targets' will be
     logged using the logger. Uses -1.0 as a default value when a key is not present.
    :param use_wandb: Whether to log the metrics to wandb.
    :param logger: The logger to use for logging.
    :param log_every: Log only every log_every^th batch.
    :param loss_key: The key in the metrics corresponding to the loss.
    """
    if use_wandb:
        wandb.log(metrics)
    if (batch_idx + 1) % log_every == 0:
        logger.info(
            "Epoch %d - Batch %d - Loss %.5f - #Unique targets %d / %d",
            epoch,
            batch_idx,
            metrics.get(loss_key, -1.0),
            metrics.get("unique_targets", -1.0),
            metrics.get("targets", -1.0),
        )

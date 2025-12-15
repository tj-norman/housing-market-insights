"""Logging utilities for the CBSA data pipeline."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from data_pipeline import pipeline_config as config


def get_logger(name, log_file=None):
    """Return a configured logger that logs to file and console.

    Args:
        name (str): Logger name.
        log_file (Path | None): Optional explicit log file path. Defaults to
            `data_pipeline/logs/pipeline.log`.

    Returns:
        logging.Logger: Logger instance configured with rotating file handler and console handler.
    """
    logger = logging.getLogger(name)

    if getattr(logger, "_cbsa_pipeline_configured", False):
        return logger

    logger.setLevel(logging.INFO)

    log_path = log_file or config.LOG_DIR / config.LOG_FILE_BASENAME
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT,
    )
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger._cbsa_pipeline_configured = True  # type: ignore[attr-defined]
    return logger

import logging

import torch

logger = logging.getLogger("ibgest")


def set_logging_level(level: str | int) -> None:
    """set logging level

    Args:
        level: Can be either a string ('DEBUG', 'INFO', 'WARNING', 'ERROR',
        'CRITICAL') or an integer (10, 20, 30, 40, 50) corresponding to the
        logging levels.
    """

    if isinstance(level, str):
        level = level.upper()
        numeric_level = getattr(logging, level, None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {level}")
    elif isinstance(level, int):
        numeric_level = level
    else:
        raise TypeError("Level must be a string or an integer")

    logger.setLevel(numeric_level)

    # Create a StreamHandler if no handlers exist
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.info(f"Logging level set to {logging.getLevelName(numeric_level)}")


set_logging_level("INFO")


def set_device(device_type: str = "auto") -> torch.device:
    """
    Set the device for PyTorch computations. cuda > mps > cpu

    Args:
        device_type: String specifying the device type.
                     Can be 'cuda', 'mps', 'cpu', or 'auto' (default).

    Returns:
        torch.device: The selected PyTorch device.
    """
    if device_type == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif device_type in ["cuda", "mps", "cpu"]:
        device = torch.device(device_type)
    else:
        raise ValueError(f"Invalid device type: {device_type}")

    logger.debug(f"Using device: {device}")

    return device

import logging
import os

def setup_logging(name: str = "data2dash") -> logging.Logger:
    level = os.getenv("LOG_LEVEL", "INFO").upper()

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # prevent duplicate logs

    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # file handler (VERY important in production)
    fh = logging.FileHandler("data2dash.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

###LOG_LEVEL=DEBUG

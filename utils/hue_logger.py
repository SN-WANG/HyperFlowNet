# Colorful logging utility with ANSI support
# Author: Shengning Wang

import logging
import sys


class HueLogger:
    """Pre-configured logger with ANSI color codes and clean re-initialization."""

    b = "\033[1;34m"
    c = "\033[1;36m"
    m = "\033[1;35m"
    y = "\033[1;33m"
    g = "\033[1;32m"
    r = "\033[1;31m"
    q = "\033[0m"

    def __init__(self, name: str = "hyflow", level: int = logging.INFO) -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            f"\033[90m%(asctime)s{self.q} - {self.b}%(levelname)s{self.q} - %(message)s", "%H:%M:%S"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)


hue = HueLogger()
logger = hue.logger

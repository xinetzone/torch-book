from dataclasses import dataclass, field, asdict
from contextlib import ContextDecorator
import logging

logging.basicConfig(level=logging.INFO)


@dataclass
class trace_context(ContextDecorator):
    """追踪上下文信息"""
    name: str

    def __enter__(self):
        logging.info('Entering: %s', self.name)

    def __exit__(self, *exc_details):
        logging.info('Exiting: %s', self.name)

from base.config import BaseConfigHandler
import yaml
import logging
import re

loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))


class YamlConfigHandler(BaseConfigHandler):
    def __init__(self, callables=None):
        self.config = {}
        self.callables = callables or {}

    def load(self, path: str):
        with open(path, 'r') as f:
            try:
                self.config = yaml.load(f, loader=loader)
            except Exception as e:
                logging.error(f"Error loading config from {path}: {e}")

    def save(self, path: str):
        safe_config = {k: v for k, v in self.config.items() if v is not None and not callable(v)}
        with open(path, 'w') as f:
            try:
                yaml.dump(safe_config, f)
            except Exception as e:
                logging.error(f"Error saving config to {path}: {e}")
    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        if callable(value):
            logging.warning(f"Callable value {value} for key {key} will not be saved.")
        elif value is None:
            logging.warning(f"None value for key {key} will not be saved.")
        self.config[key] = value

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        if callable(value):
            logging.warning(f"Callable value {value} for key {key} will not be saved.")
        elif value is None:
            logging.warning(f"None value for key {key} will not be saved.")
        self.config[key] = value

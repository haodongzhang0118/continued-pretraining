"""Stable-CP - A benchmark and toolkit for continued pretraining methods.

This package provides:
- Core abstractions for implementing custom continued pretraining methods
- Built-in methods (DIET, LeJEPA, MAE, SimCLR)
- Data loading and preprocessing utilities
- Evaluation protocols (k-NN, linear probing, zero-shot)
- Training callbacks and utilities
- Unified CLI for running experiments

Example usage:
    >>> from cp_benchmark.core import BaseCPMethod, register_method
    >>> from cp_benchmark.data import create_data_loaders
    >>> from cp_benchmark.utils import load_backbone

    >>> @register_method("my_method")
    >>> class MyMethod(BaseCPMethod):
    ...     def build_module(self, optim_config):
    ...         # Your implementation
    ...         pass
"""

__version__ = "0.1.0"

from . import core
from . import data
from . import utils
from . import callbacks
from . import evaluation

# Make key APIs easily accessible
from .core import (
    BaseCPMethod,
    MethodConfig,
    register_method,
    get_method,
    list_methods,
)

__all__ = [
    # Submodules
    "core",
    "data",
    "utils",
    "callbacks",
    "evaluation",
    # Key exports
    "BaseCPMethod",
    "MethodConfig",
    "register_method",
    "get_method",
    "list_methods",
    "__version__",
]

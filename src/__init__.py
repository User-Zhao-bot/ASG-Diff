import torch

try:
    from icecream import install
except ModuleNotFoundError:
    def install(*args, **kwargs):
        return None

torch.set_num_threads(1)
install()

from . import env  # noqa
from .data import *  # noqa
from .env import *  # noqa
from .metrics import *  # noqa
from .util import *  # noqa

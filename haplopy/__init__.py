"""HaploPy package

"""

import importlib.metadata

from .datautils import *
from .multinomial import *

try:
    from .plot import *
except ImportError:
    # Allow import plot fail in environments where matplotlib is not installed.
    pass


__version__ = importlib.metadata.version("haplopy")

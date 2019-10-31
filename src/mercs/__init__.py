from pkg_resources import get_distribution, DistributionNotFound

from . import algo, composition, core, graph, tests, utils, visuals
from .core import Mercs

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound

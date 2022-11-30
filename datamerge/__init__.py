from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from . import _version

__version__ = _version.get_versions()["version"]

from .plotting import plotFigure
from .dmdataclasses import (
    scatteringDataObj,
    rangeConfigObj,
    mergeConfigObj,
    outputRangeObj,
)
from .findscaling import findScaling
from .mergecore import mergeCore
from .readersandwriters import (
    scatteringDataObjFromNX,
    outputToNX,
    mergeConfigObjFromYaml,
)

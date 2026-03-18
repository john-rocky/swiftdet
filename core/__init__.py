"""SwiftDet core module — model API, results, and CLI."""

from .model import SwiftDet
from .results import Boxes, Results

__all__ = ["SwiftDet", "Results", "Boxes"]

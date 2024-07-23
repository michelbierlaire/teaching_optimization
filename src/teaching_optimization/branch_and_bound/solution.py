"""Abstract representation of a solution for the branch and bound algorithm

Michel Bierlaire
Sun Jul 14 10:41:12 2024
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import NamedTuple, Any

import numpy as np


class Solution(NamedTuple):
    """A solution and its value"""

    solution: Any
    value: float

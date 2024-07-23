"""Simplex algorithm: tableau version, phase II

Michel Bierlaire
Fri May 3 15:02:23 2024
"""

from __future__ import annotations

from copy import deepcopy
from enum import Enum, auto

import numpy as np

from .tableau import SimplexTableau, RowColumn


class CauseInterruptionIterations(Enum):
    OPTIMAL = auto()
    UNBOUNDED = auto()
    INFEASIBLE = auto()

    def __str__(self) -> str:
        messages = {
            self.OPTIMAL: 'Optimal basis found.',
            self.UNBOUNDED: 'Optimization problem is unbounded.',
            self.INFEASIBLE: 'Optimization problem is infeasible.',
        }
        return messages[self]


class SimplexTableauPhaseTwo:
    """Class implementing the tableau version of the simplex algorithm, when an
    initial tableau is available."""

    def __init__(self, initial_tableau: SimplexTableau) -> None:
        self.current_iteration = 0
        self.tableau = initial_tableau
        self.stopping_cause: CauseInterruptionIterations | None = None

    @property
    def solution(self) -> np.ndarray:
        return self.tableau.feasible_basic_solution

    def __iter__(self) -> SimplexTableauPhaseTwo:
        return self

    def column_entering_basis(self) -> int | None:
        """
        Function that identifies a non-basic index to enter the basis, or detect optimality
        :return: a non-basic index corresponding to a negative reduced cost, or None if optimal.
        """
        # the_tableau.tableau contains the numpy array with the entries of the tableau.
        # This can be done in one line using the following logic in numpy.
        # ####
        # - the_tableau.tableau[-1]: This selects the last row of the array.
        # - the_tableau.tableau[-1] < 0: This creates a boolean array where each element is True if the corresponding
        #   element in the last row of A is negative, and False otherwise.
        # - np.where(the_tableau.tableau[-1] < 0)[0]: np.where returns the indices of the elements that are True.
        #   The [0] extracts the first array of indices since np.where can potentially return a tuple
        #   of arrays if used on a multi-dimensional array.
        # - the second [0]: This selects the first index from the array of indices, which corresponds to
        #   the left-most negative value.
        # - if np.any(the_tableau.tableau[-1] < 0) else None: The if condition checks if there is any negative element
        #   in the last row. If not, it returns None.
        reduced_costs = self.tableau.tableau[-1, :-1]
        index = (
            np.where(reduced_costs < 0)[0][0]  # Excluding the last column
            if np.any(reduced_costs < 0)
            else None
        )
        return index

    def row_leaving_basis(
        self,
        column_entering: int,
    ) -> int | None:
        """function that identifies a row corresponding to the basic variable leaving the basis, or identify an unbounded
            problem.

        :param column_entering: non-basic index entering the basis
        :return: index of the variable leaving the basis, or None if unbounded
        """
        # First, we identify the rows with positive entries.
        positive_row_indices = np.where(self.tableau.tableau[:, column_entering] > 0)[0]

        # If there is no such entry, the problem is unbounded.
        if positive_row_indices.size == 0:
            return None

        # Calculating the vector of alphas
        vector_of_alphas = (
            self.tableau.tableau[positive_row_indices, -1]
            / self.tableau.tableau[positive_row_indices, column_entering]
        )

        # Finding the index of the smallest alpha in vector_of_alphas
        min_alpha_index = np.argmin(vector_of_alphas)

        # Mapping the local index of the smallest alpha to the global index in the_tableau
        row_index = int(positive_row_indices[min_alpha_index])

        return row_index

    def __next__(self) -> SimplexTableau:
        # The first tableau should be yielded in any case
        if self.current_iteration == 0:
            # Yield the initial tableau before any iteration
            self.current_iteration += 1
            return deepcopy(self.tableau)
        pivot_column = self.column_entering_basis()
        if pivot_column is None:
            # Optimal solution found.
            self.stopping_cause = CauseInterruptionIterations.OPTIMAL
            raise StopIteration

        pivot_row = self.row_leaving_basis(column_entering=pivot_column)
        if pivot_row is None:
            # Problem is unbounded.
            self.stopping_cause = CauseInterruptionIterations.UNBOUNDED
            raise StopIteration

        pivot = RowColumn(row=pivot_row, column=pivot_column)
        self.tableau.pivoting(pivot=pivot)
        self.current_iteration += 1
        return deepcopy(self.tableau)

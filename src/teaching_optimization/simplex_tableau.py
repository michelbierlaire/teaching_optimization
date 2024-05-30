"""Simplex algorithm: tableau version, two phases

Michel Bierlaire
Fri May 3 17:51:19 2024
"""

from __future__ import annotations

import numpy as np

from .simplex_tableau_phase_two import (
    SimplexTableauPhaseTwo,
    CauseInterruptionIterations,
)
from .tableau import SimplexTableau, RowColumn


class SimplexAlgorithmTableau:
    """Class implementing the tableau version of the simplex algorithm to solve problems in standard form."""

    def __init__(
        self,
        objective: np.ndarray,
        constraint_matrix: np.ndarray,
        right_hand_side: np.ndarray,
    ) -> None:
        """
        Constructor. The problem is assumed to be written in standard form.

        :param objective: vector of coefficients of the objective function.
        :param constraint_matrix: matrix of coefficients of the constraints.
        :param right_hand_side: right hand side of the constraints.
        """
        if (
            not isinstance(objective, np.ndarray)
            or not isinstance(constraint_matrix, np.ndarray)
            or not isinstance(right_hand_side, np.ndarray)
        ):
            error_msg = (
                'Numpy arrays are requested here. An array of float an be transformed '
                'into a numpy array as follows: np.array([1.0, 2.0, 3.0])'
            )
            raise AttributeError(error_msg)
        # We need to make sure that it is a 2d array, even if there is only one row.
        self.constraint_matrix = np.atleast_2d(constraint_matrix)
        self.objective = objective
        self.right_hand_side = right_hand_side

        # Verify the dimensions
        if len(self.objective) != self.constraint_matrix.shape[1]:
            error_msg = f'Incompatible sizes: {len(self.objective)} and {self.constraint_matrix.shape[1]}'
            raise ValueError(error_msg)
        if len(self.right_hand_side) != self.constraint_matrix.shape[0]:
            error_msg = f'Incompatible sizes: {len(self.right_hand_side)} and {self.constraint_matrix.shape[0]}'
            raise ValueError(error_msg)

        self.stopping_cause: CauseInterruptionIterations | None = None

        # We need to make sure that all entries of the right hand side are non negative.
        negative_mask = self.right_hand_side < 0
        self.constraint_matrix[negative_mask] *= -1
        self.right_hand_side[negative_mask] *= -1

    @property
    def n_variables(self) -> int:
        return len(self.objective)

    @property
    def n_constraints(self) -> int:
        return len(self.right_hand_side)

    def initial_tableau_phase_one(self) -> SimplexTableau:
        """Prepare the initial_tableau for Phase One."""
        identity_matrix = np.eye(self.n_constraints)  # Identity matrix of size m
        first_row = np.hstack(
            (
                self.constraint_matrix,
                identity_matrix,
                self.right_hand_side[:, np.newaxis],
            )
        )
        # Second row components
        e = np.ones(self.n_constraints)  # Vector of ones of size m
        neg_e_transposed_matrix = -np.dot(e, self.constraint_matrix)  # -e^T A
        neg_e_transposed_right_hand_side = -np.dot(e, self.right_hand_side)  # -e^T b
        zeros = np.zeros(self.n_constraints)  # m zeros
        second_row = np.hstack(
            (
                neg_e_transposed_matrix,
                zeros,
                np.array([neg_e_transposed_right_hand_side]),
            )
        )

        # Combine both rows
        complete_array = np.vstack((first_row, second_row))
        the_tableau = SimplexTableau(tableau=complete_array)
        return the_tableau

    def solve_phase_one(self, initial_tableau: SimplexTableau) -> SimplexTableau | None:
        """Solve the auxiliary problem, and clean the optimal tableau. If the original problem is not feasible,
        return None"""

        algorithm = SimplexTableauPhaseTwo(initial_tableau=initial_tableau)
        optimal_tableau_phase_one = initial_tableau
        for tableau in algorithm:
            optimal_tableau_phase_one = tableau

        if not np.isclose(optimal_tableau_phase_one.value_objective_function, 0.0):
            self.stopping_cause = CauseInterruptionIterations.INFEASIBLE
            return None
        return optimal_tableau_phase_one

    def remove_auxiliary_variables_from_basis(
        self, optimal_tableau_phase_one
    ) -> SimplexTableau:
        """Remove auxiliary variables and possibly redundant constraints from the optimal tableau"""

        clean_tableau = False
        while not clean_tableau:
            basic_variables = optimal_tableau_phase_one.identify_basic_variables()
            for basic_var in basic_variables:
                if (basic_var.column >= self.n_variables) and (
                    basic_var.column < (self.n_variables + self.n_constraints)
                ):
                    # Find a non zero element in the corresponding part of the row corresponding to original variables.
                    row_slice = optimal_tableau_phase_one.tableau[
                        basic_var.row, : self.n_variables
                    ]
                    nonzero_indices = np.nonzero(row_slice)[0]

                    if len(nonzero_indices) == 0:
                        """No way to pivot. The constraint is redundant and must be removed"""
                        optimal_tableau_phase_one = (
                            optimal_tableau_phase_one.remove_row(basic_var.row)
                        )
                        break

                    # We pivot to remove the auxiliary variable out of the basis
                    pivot = RowColumn(row=basic_var.row, column=int(nonzero_indices[0]))
                    optimal_tableau_phase_one.pivoting(pivot=pivot)
                    break

            else:
                clean_tableau = True

        return optimal_tableau_phase_one

    def prepare_tableau_for_phase_two(
        self, cleaned_tableau: SimplexTableau
    ) -> SimplexTableau:
        """Remove columns corresponding to auxiliary variables and calculate the reduced costs.

        :param cleaned_tableau: tableau where all auxiliary variables have been removed from the basis
        :return: initial tableau for phase II
        """
        basic_variables = cleaned_tableau.identify_basic_variables()
        for basic_var in basic_variables:
            if (basic_var.column >= self.n_variables) and (
                basic_var.column < (self.n_variables + self.n_constraints)
            ):
                error_msg = f'Auxiliary variable {basic_var.column} is in the basis. Tableau cannot be cleaned'
                raise ValueError(error_msg)

        # Delete the columns corresponding to the auxiliary variables.
        slice_to_delete = np.s_[
            self.n_variables : self.n_variables + self.n_constraints
        ]
        new_array = np.delete(cleaned_tableau.tableau, slice_to_delete, axis=1)
        phase_two_tableau = SimplexTableau(tableau=new_array)

        # Calculate the reduced costs
        phase_two_tableau.recalculate_reduced_costs(costs=self.objective)

        return phase_two_tableau

    def solve_phase_two(self, initial_tableau: SimplexTableau) -> SimplexTableau | None:
        """Run the phase II algorithm"""

        # Phase II
        phase_two = SimplexTableauPhaseTwo(initial_tableau=initial_tableau)

        optimal_tableau = None
        for tableau in phase_two:
            optimal_tableau = tableau

        self.stopping_cause = phase_two.stopping_cause

        return optimal_tableau

    def solve(self) -> SimplexTableau | None:
        """Solve the problem using the complete simplex tableau algorithm"""
        first_tableau = self.initial_tableau_phase_one()
        optimal_tableau_phase_one = self.solve_phase_one(initial_tableau=first_tableau)
        if optimal_tableau_phase_one is None:
            return None
        clean_tableau = self.remove_auxiliary_variables_from_basis(
            optimal_tableau_phase_one=optimal_tableau_phase_one
        )
        first_tableau_phase_two = self.prepare_tableau_for_phase_two(
            cleaned_tableau=clean_tableau
        )
        optimal_tableau = self.solve_phase_two(initial_tableau=first_tableau_phase_two)
        return optimal_tableau

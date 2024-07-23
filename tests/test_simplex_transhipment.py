"""Test simplex tableau, complete.

Michel Bierlaire
Sun Jul 7 14:39:11 2024

"""

import unittest

import numpy as np

from teaching_optimization.simplex_tableau import (
    SimplexAlgorithmTableau,
    SimplexTableau,
)


class TestSimplexAlgorithmTableau(unittest.TestCase):

    def setUp(self):
        objective = np.array([36, 28, 50, 71, 83, 50])
        matrix = np.array(
            [
                [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, -1.0, -1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0, -1.0, -1.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            ]
        )
        rhs = np.array([-1.0, 0.0, 0.0, 1.0])
        self.algorithm = SimplexAlgorithmTableau(
            objective=objective, constraint_matrix=matrix, right_hand_side=rhs
        )

    def test_initial_tableau(self):
        first_tableau: SimplexTableau = self.algorithm.initial_tableau_phase_one()
        self.assertEqual(first_tableau.n_variables, 10)
        self.assertEqual(first_tableau.n_constraints, 4)

    def test_phase_one(self):
        first_tableau: SimplexTableau = self.algorithm.initial_tableau_phase_one()
        optimal_tableau_phase_one = self.algorithm.solve_phase_one(
            initial_tableau=first_tableau
        )
        self.assertEqual(optimal_tableau_phase_one.value_objective_function, 0.0)

    def test_clean(self):
        first_tableau: SimplexTableau = self.algorithm.initial_tableau_phase_one()
        optimal_tableau_phase_one = self.algorithm.solve_phase_one(
            initial_tableau=first_tableau
        )
        clean_tableau = self.algorithm.remove_auxiliary_variables_from_basis(
            optimal_tableau_phase_one=optimal_tableau_phase_one
        )
        for row_col in clean_tableau.identify_basic_variables():
            self.assertLessEqual(row_col.column, 6)

    def test_phase_two(self):
        first_tableau: SimplexTableau = self.algorithm.initial_tableau_phase_one()
        optimal_tableau_phase_one = self.algorithm.solve_phase_one(
            initial_tableau=first_tableau
        )
        clean_tableau = self.algorithm.remove_auxiliary_variables_from_basis(
            optimal_tableau_phase_one=optimal_tableau_phase_one
        )
        first_tableau_phase_two = self.algorithm.prepare_tableau_for_phase_two(
            cleaned_tableau=clean_tableau
        )
        optimal_tableau = self.algorithm.solve_phase_two(
            initial_tableau=first_tableau_phase_two
        )
        self.assertEqual(optimal_tableau.value_objective_function, 99)

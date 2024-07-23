"""Test simplex tableau, phase two

Michel Bierlaire
Fri May 3 17:30:16 2024

"""

import numpy as np
import unittest
from copy import deepcopy

from numpy.testing import assert_array_equal

from teaching_optimization.simplex_tableau_phase_two import (
    SimplexTableauPhaseTwo,
    CauseInterruptionIterations,
)
from teaching_optimization.tableau import SimplexTableau


# Assuming the SimplexTableau and RowColumn classes are defined elsewhere, and the class definitions from your question.


class TestSimplexTableauPhaseTwo(unittest.TestCase):
    def setUp(self):
        # Data for a basic tableau that can be solved
        standard_a = np.array([[2, 1, 1, 0], [1, -1, 0, 1]])
        standard_b = np.array([6, 2])
        standard_c = np.array([-4, 3, 0, 0])
        Ab = np.column_stack((standard_a, standard_b))
        self.initial_tableau = np.row_stack((Ab, np.append(standard_c, 0)))

    def test_optimal_solution_detection(self):
        # Manually set the tableau to an optimal state if necessary
        # For example, make last row all non-negative

        tableau = SimplexTableau(deepcopy(self.initial_tableau))
        algorithm = SimplexTableauPhaseTwo(initial_tableau=tableau)
        algorithm.tableau.tableau[-1] = np.array([0, 0, 0, 0, 1])
        # As the first tableau is always generated, we need to skip it first
        next(algorithm)
        with self.assertRaises(StopIteration):
            next(algorithm)
        self.assertEqual(algorithm.stopping_cause, CauseInterruptionIterations.OPTIMAL)

    def test_unbounded_problem_detection(self):
        # Manually set the tableau to reflect an unbounded problem
        # Assume the entering basis column leads to no positive ratios in the test
        tableau = SimplexTableau(deepcopy(self.initial_tableau))
        algorithm = SimplexTableauPhaseTwo(initial_tableau=tableau)
        algorithm.tableau.tableau[:, 0] = np.array([-1, -1, -1])
        # As the first tableau is always generated, we need to get it first
        next(algorithm)
        with self.assertRaises(StopIteration):
            next(algorithm)
        self.assertEqual(
            algorithm.stopping_cause, CauseInterruptionIterations.UNBOUNDED
        )

    def test_algorithm_iteration(self):
        # Verify that the algorithm correctly performs a pivot
        # This test needs to be designed based on specific data that will lead to a pivot
        tableau = SimplexTableau(deepcopy(self.initial_tableau))
        algorithm = SimplexTableauPhaseTwo(initial_tableau=tableau)
        # The first tableau is always generated. We skipit first.
        next(algorithm)
        try:
            next(algorithm)  # Perform one iteration
        except StopIteration:
            pass
        # Check that tableau has changed indicating a pivot
        self.assertNotEqual(
            self.initial_tableau.tolist(), algorithm.tableau.tableau.tolist()
        )

    def test_complete_algorithm(self):
        # Verify a full run of the algorithm
        tableau = SimplexTableau(deepcopy(self.initial_tableau))
        algorithm = SimplexTableauPhaseTwo(initial_tableau=tableau)
        for _ in algorithm:
            pass
        the_solution = algorithm.solution
        expected_solution = np.array([2.66666667, 0.66666667, 0, 0])
        self.assertTrue(
            np.allclose(the_solution, expected_solution, atol=1e-3),
            f"Arrays are not almost equal: {the_solution} != {expected_solution}",
        )

    def test_start_with_optimal_tableau(self):
        initial_tableau = np.array([[-1.0, -1.0, 1.0, 1.0], [1.0, 1.0, 0.0, -1.0]])
        tableau = SimplexTableau(tableau=initial_tableau)
        algorithm = SimplexTableauPhaseTwo(initial_tableau=tableau)
        for _ in algorithm:
            pass
        expected_solution = np.array([0, 0, 1])
        self.assertTrue(
            np.allclose(algorithm.solution, expected_solution, atol=1e-3),
            "Arrays are not almost equal",
        )
        expected_tableau = np.array([[-1.0, -1.0, 1.0, 1.0], [1.0, 1.0, 0.0, -1.0]])
        assert_array_equal(
            algorithm.tableau.tableau, expected_tableau, "Arrays should be equal"
        )


# Run the tests
if __name__ == '__main__':
    unittest.main()

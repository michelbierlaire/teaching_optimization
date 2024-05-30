"""Test simplex tableau, complete.

Michel Bierlaire
Sun May 5 15:09:47 2024

"""

import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from teaching_optimization.simplex_tableau import (
    SimplexAlgorithmTableau,
    SimplexTableau,
    CauseInterruptionIterations,
)


def print_array_for_initialization(array):
    # Convert the NumPy array to a string that looks like a Python list
    formatted_string = np.array2string(array, separator=', ')
    # Clean up the string to make it suitable for initialization
    # formatted_string = formatted_string.replace('[', '').replace(']', '')
    formatted_string = 'np.array([' + formatted_string + '])'
    print('\n', formatted_string)


class TestSimplexAlgorithmTableau(unittest.TestCase):

    def setUp(self):
        # Problem 1 (Example 16.16)
        self.objective_1 = np.array([2, 3, 3, 1, -2])
        self.constraints_1 = np.array(
            [[-1, -3, 0, -4, -1], [1, 2, 0, -3, 1], [-1, -4, 3, 0, 0]]
        )
        self.rhs_1 = np.array([-2, 2, 1])

        # Problem 2 (Example 16.15)
        self.objective_2 = np.array([1, 1, 1, 0])
        self.constraints_2 = np.array(
            [[1, 2, 3, 0], [-1, 2, 6, 0], [0, 4, 9, 0], [0, 0, 3, 1]]
        )
        self.rhs_2 = np.array([3, 2, 5, 1])

        # Problem 3 (Infeasible)
        self.objective_3 = np.array([1, 0])
        self.constraints_3 = np.array([1, 1])
        self.rhs_3 = np.array([-1])

        # Problem 4 (wrong instance)
        self.objective_4 = np.array([1, 0])
        self.constraints_4 = np.array([1, 1])
        self.rhs_4 = np.array([-1, 1])

    def test_initialization(self):
        # Test initialization and attribute assignments
        algorithm_1 = SimplexAlgorithmTableau(
            objective=self.objective_1,
            constraint_matrix=self.constraints_1,
            right_hand_side=self.rhs_1,
        )

        self.assertIsInstance(algorithm_1.objective, np.ndarray)
        self.assertIsInstance(algorithm_1.constraint_matrix, np.ndarray)
        # Check if the change of sign has been performed
        expected_rhs_1 = np.array([2, 2, 1])
        assert_array_equal(
            algorithm_1.right_hand_side, expected_rhs_1, "Arrays should be equal"
        )
        algorithm_2 = SimplexAlgorithmTableau(
            objective=self.objective_2,
            constraint_matrix=self.constraints_2,
            right_hand_side=self.rhs_2,
        )
        expected_objective_2 = np.array([1, 1, 1, 0])
        assert_array_equal(
            algorithm_2.objective, expected_objective_2, "Arrays should be equal"
        )
        algorithm_3 = SimplexAlgorithmTableau(
            objective=self.objective_3,
            constraint_matrix=self.constraints_3,
            right_hand_side=self.rhs_3,
        )
        expected_constraints_3 = np.array([[-1, -1]])
        assert_array_equal(
            algorithm_3.constraint_matrix,
            expected_constraints_3,
            "Arrays should be equal",
        )
        with self.assertRaises(ValueError):
            _ = SimplexAlgorithmTableau(
                objective=self.objective_4,
                constraint_matrix=self.constraints_4,
                right_hand_side=self.rhs_4,
            )

    def test_initial_tableau_phase_one(self):
        # Test initial tableau creation for phase one
        algorithm_1 = SimplexAlgorithmTableau(
            objective=self.objective_1,
            constraint_matrix=self.constraints_1,
            right_hand_side=self.rhs_1,
        )
        tableau_1 = algorithm_1.initial_tableau_phase_one()
        expected_tableau_1 = np.array(
            [
                [1, 3, 0, 4, 1, 1, 0, 0, 2],
                [1, 2, 0, -3, 1, 0, 1, 0, 2],
                [-1, -4, 3, 0, 0, 0, 0, 1, 1],
                [-1, -1, -3, -1, -2, 0, 0, 0, -5],
            ]
        )
        self.assertIsInstance(tableau_1, SimplexTableau)
        self.assertEqual(
            tableau_1.tableau.shape, (4, 9)
        )  # Assumes specific tableau structure
        assert_array_equal(
            tableau_1.tableau, expected_tableau_1, "Arrays should be equal"
        )
        algorithm_2 = SimplexAlgorithmTableau(
            objective=self.objective_2,
            constraint_matrix=self.constraints_2,
            right_hand_side=self.rhs_2,
        )
        expected_tableau_2 = np.array(
            [
                [1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 0.0, 0.0, 3.0],
                [-1.0, 2.0, 6.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0],
                [0.0, 4.0, 9.0, 0.0, 0.0, 0.0, 1.0, 0.0, 5.0],
                [0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [-0.0, -8.0, -21.0, -1.0, 0.0, 0.0, 0.0, 0.0, -11.0],
            ]
        )
        tableau_2 = algorithm_2.initial_tableau_phase_one()
        self.assertIsInstance(tableau_2, SimplexTableau)
        self.assertEqual(
            tableau_2.tableau.shape, (5, 9)
        )  # Assumes specific tableau structure
        assert_array_equal(
            tableau_2.tableau, expected_tableau_2, "Arrays should be equal"
        )
        algorithm_3 = SimplexAlgorithmTableau(
            objective=self.objective_3,
            constraint_matrix=self.constraints_3,
            right_hand_side=self.rhs_3,
        )
        expected_tableau_3 = np.array([[-1.0, -1.0, 1.0, 1.0], [1.0, 1.0, 0.0, -1.0]])
        tableau_3 = algorithm_3.initial_tableau_phase_one()
        self.assertIsInstance(tableau_3, SimplexTableau)
        self.assertEqual(
            tableau_3.tableau.shape, (2, 4)
        )  # Assumes specific tableau structure
        assert_array_equal(
            tableau_3.tableau, expected_tableau_3, "Arrays should be equal"
        )

    def test_solve_phase_one(self):
        # Test phase one with a feasible setup
        algorithm_1 = SimplexAlgorithmTableau(
            objective=self.objective_1,
            constraint_matrix=self.constraints_1,
            right_hand_side=self.rhs_1,
        )
        tableau_1 = algorithm_1.initial_tableau_phase_one()
        result_1 = algorithm_1.solve_phase_one(tableau_1)
        expected_result_1 = np.array(
            [
                [1.0, 3.0, 0.0, 4.0, 1.0, 1.0, 0.0, 0.0, 2.0],
                [0.0, -1.0, 0.0, -7.0, 0.0, -1.0, 1.0, 0.0, 0.0],
                [
                    0.0,
                    -1 / 3,
                    1.0,
                    4 / 3,
                    1 / 3,
                    1 / 3,
                    0.0,
                    1 / 3,
                    1.0,
                ],
                [0.0, 1.0, 0.0, 7.0, 0.0, 2.0, 0.0, 1.0, 0.0],
            ]
        )
        self.assertTrue(np.isclose(result_1.value_objective_function, 0))
        assert_array_equal(
            result_1.tableau, expected_result_1, "Arrays should be equal"
        )
        algorithm_2 = SimplexAlgorithmTableau(
            objective=self.objective_2,
            constraint_matrix=self.constraints_2,
            right_hand_side=self.rhs_2,
        )
        tableau_2 = algorithm_2.initial_tableau_phase_one()
        result_2 = algorithm_2.solve_phase_one(tableau_2)
        expected_result_2 = np.array(
            [
                [1.0, 0.0, 0.0, 0.5, 0.5, -0.5, 0.0, 0.5, 1.0],
                [0.0, 1.0, 0.0, -0.75, 0.25, 0.25, 0.0, -0.75, 0.5],
                [0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1 / 3, 0.0, 0.0, 0.0, 1 / 3, 1 / 3],
                [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 1.0, 0.0],
            ]
        )
        self.assertTrue(np.isclose(result_2.value_objective_function, 0))
        assert_array_equal(
            result_2.tableau, expected_result_2, "Arrays should be equal"
        )
        algorithm_3 = SimplexAlgorithmTableau(
            objective=self.objective_3,
            constraint_matrix=self.constraints_3,
            right_hand_side=self.rhs_3,
        )
        tableau_3 = algorithm_3.initial_tableau_phase_one()
        _ = algorithm_3.solve_phase_one(tableau_3)

        self.assertEqual(
            algorithm_3.stopping_cause, CauseInterruptionIterations.INFEASIBLE
        )

    def test_remove_auxiliary_variables_from_basis_1(self):
        # Test removing auxiliary variables from basis
        algorithm = SimplexAlgorithmTableau(
            objective=self.objective_1,
            constraint_matrix=self.constraints_1,
            right_hand_side=self.rhs_1,
        )
        tableau = algorithm.initial_tableau_phase_one()
        phase_one_tableau = algorithm.solve_phase_one(tableau)
        cleaned_tableau = algorithm.remove_auxiliary_variables_from_basis(
            phase_one_tableau
        )
        # print_array_for_initialization(cleaned_tableau_1.tableau)
        # Check that no auxiliary variables remain in the basis
        self.assertTrue(
            all(
                [
                    var.column < algorithm.n_variables
                    for var in cleaned_tableau.identify_basic_variables()
                ]
            )
        )
        expected_result = np.array(
            [
                [1.0, 0.0, 0.0, -17.0, 1.0, -2.0, 3.0, 0.0, 2.0],
                [-0.0, 1.0, -0.0, 7.0, -0.0, 1.0, -1.0, -0.0, -0.0],
                [
                    0.0,
                    0.0,
                    1.0,
                    11 / 3,
                    1 / 3,
                    2 / 3,
                    -1 / 3,
                    1 / 3,
                    1.0,
                ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            ]
        )
        assert_allclose(
            cleaned_tableau.tableau,
            expected_result,
            err_msg="Arrays should be equal",
        )

    def test_remove_auxiliary_variables_from_basis_2(self):
        # Test removing auxiliary variables from basis
        algorithm = SimplexAlgorithmTableau(
            objective=self.objective_2,
            constraint_matrix=self.constraints_2,
            right_hand_side=self.rhs_2,
        )
        tableau = algorithm.initial_tableau_phase_one()
        phase_one_tableau = algorithm.solve_phase_one(tableau)
        cleaned_tableau = algorithm.remove_auxiliary_variables_from_basis(
            phase_one_tableau
        )
        # Check that no auxiliary variables remain in the basis
        self.assertTrue(
            all(
                [
                    var.column < algorithm.n_variables
                    for var in cleaned_tableau.identify_basic_variables()
                ]
            )
        )
        expected_result = np.array(
            [
                [1.0, 0.0, 0.0, 0.5, 0.5, -0.5, 0.0, 0.5, 1.0],
                [0.0, 1.0, 0.0, -0.75, 0.25, 0.25, 0.0, -0.75, 0.5],
                [0.0, 0.0, 1.0, 0.33333333, 0.0, 0.0, 0.0, 0.33333333, 0.33333333],
                [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 1.0, 0.0],
            ]
        )
        assert_allclose(
            cleaned_tableau.tableau,
            expected_result,
            err_msg="Arrays should be equal",
        )

    def test_prepare_tableau_for_phase_two_1(self):
        # Test preparation of tableau for phase two
        algorithm = SimplexAlgorithmTableau(
            objective=self.objective_1,
            constraint_matrix=self.constraints_1,
            right_hand_side=self.rhs_1,
        )
        tableau = algorithm.initial_tableau_phase_one()
        phase_one_tableau = algorithm.solve_phase_one(tableau)
        cleaned_tableau = algorithm.remove_auxiliary_variables_from_basis(
            phase_one_tableau
        )
        phase_two_tableau = algorithm.prepare_tableau_for_phase_two(cleaned_tableau)
        expected_result = np.array(
            [
                [1.0, 0.0, 0.0, -17.0, 1.0, 2.0],
                [-0.0, 1.0, -0.0, 7.0, -0.0, -0.0],
                [0.0, 0.0, 1.0, 3.66666667, 0.33333333, 1.0],
                [0.0, 0.0, 0.0, 3.0, -5.0, -7.0],
            ]
        )
        assert_allclose(
            phase_two_tableau.tableau,
            expected_result,
            err_msg="Arrays should be equal",
        )

    def test_prepare_tableau_for_phase_two_2(self):
        # Test preparation of tableau for phase two
        algorithm = SimplexAlgorithmTableau(
            objective=self.objective_2,
            constraint_matrix=self.constraints_2,
            right_hand_side=self.rhs_2,
        )
        tableau = algorithm.initial_tableau_phase_one()
        phase_one_tableau = algorithm.solve_phase_one(tableau)
        cleaned_tableau = algorithm.remove_auxiliary_variables_from_basis(
            phase_one_tableau
        )
        phase_two_tableau = algorithm.prepare_tableau_for_phase_two(cleaned_tableau)
        expected_result = np.array(
            [
                [1.0, 0.0, 0.0, 0.5, 1.0],
                [0.0, 1.0, 0.0, -0.75, 0.5],
                [0.0, 0.0, 1.0, 0.33333333, 0.33333333],
                [0.0, 0.0, 0.0, -0.08333333, -1.83333333],
            ]
        )
        assert_allclose(
            phase_two_tableau.tableau,
            expected_result,
            err_msg="Arrays should be equal",
        )

    def test_solve_complete_1(self):
        # Test the entire solve process from phase one to phase two
        algorithm = SimplexAlgorithmTableau(
            objective=self.objective_1,
            constraint_matrix=self.constraints_1,
            right_hand_side=self.rhs_1,
        )
        result = algorithm.solve()
        self.assertEqual(algorithm.stopping_cause, CauseInterruptionIterations.OPTIMAL)
        optimal_solution = result.feasible_basic_solution
        expected_solution = np.array([0, 0, 1 / 3, 0, 2])
        assert_allclose(optimal_solution, expected_solution)
        self.assertTrue(np.isclose(result.value_objective_function, -3))

    def test_solve_complete_2(self):
        # Test the entire solve process from phase one to phase two
        algorithm = SimplexAlgorithmTableau(
            objective=self.objective_2,
            constraint_matrix=self.constraints_2,
            right_hand_side=self.rhs_2,
        )
        result = algorithm.solve()
        self.assertEqual(algorithm.stopping_cause, CauseInterruptionIterations.OPTIMAL)
        optimal_solution = result.feasible_basic_solution
        expected_solution = np.array([0.5, 1.25, 0.0, 1.0])
        assert_allclose(optimal_solution, expected_solution)
        self.assertTrue(np.isclose(result.value_objective_function, 1.75))


if __name__ == '__main__':
    unittest.main()

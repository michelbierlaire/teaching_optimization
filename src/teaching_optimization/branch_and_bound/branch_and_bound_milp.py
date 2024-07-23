"""Example of branch and bound

Michel Bierlaire
Sun Jul 14 13:50:36 2024

Example from the slides

"""

import logging

import numpy as np

from teaching_optimization.branch_and_bound.solution import Solution
from teaching_optimization.branch_and_bound.subproblem import Subproblem
from teaching_optimization.linear_constraints import (
    Constraint,
    AllConstraints,
    StandardCanonicalForms,
    Term,
    SignConstraint,
)
from teaching_optimization.simplex_tableau import SimplexAlgorithmTableau

logger = logging.getLogger(__name__)


def is_solution_integer(solution: np.array) -> bool:
    """Checks if an array contains only integer values"""
    return all(element.is_integer() for element in solution)


def first_non_integer_index(arr: np.ndarray) -> int | None:
    # Check if elements are integers using np.mod
    non_integer_mask = np.mod(arr, 1) != 0
    # Find the index of the first non-integer element
    non_integer_indices = np.where(non_integer_mask)[0]
    if non_integer_indices.size > 0:
        return int(non_integer_indices[0])
    else:
        return None  # Return -1 if all elements are integers


class LinearProblem(Subproblem):

    def __init__(
        self,
        objective_coefficients: np.array,
        constraints: AllConstraints,
        the_name: str,
    ):
        self.objective_coefficients = objective_coefficients
        self.constraints = constraints
        super().__init__(the_name)

    def branch(self) -> list[Subproblem]:
        """Branch on a non integer solution"""

        if self.solution is None:
            raise ValueError(f'No lower bound has been calculated yet.')

        size = len(self.solution.solution)
        index_for_branching = first_non_integer_index(self.solution.solution)
        non_integer_value = self.solution.solution[index_for_branching]

        new_b_left = np.floor(non_integer_value)
        the_left_term = Term(
            coefficient=1.0,
            variable=self.constraints.index_to_variable[index_for_branching],
        )
        left_constraint = Constraint(
            name=f'rounding down {self.name}',
            left_hand_side=[the_left_term],
            sign=SignConstraint.LESSER_OR_EQUAL,
            right_hand_side=new_b_left,
        )
        constraints_left_problem = AllConstraints(
            self.constraints.constraints + [left_constraint]
        )
        left_subproblem = LinearProblem(
            objective_coefficients=self.objective_coefficients,
            constraints=constraints_left_problem,
            the_name=f'{self.name}_1',
        )

        new_b_right = np.ceil(non_integer_value)
        the_right_term = Term(
            coefficient=1.0,
            variable=self.constraints.index_to_variable[index_for_branching],
        )
        right_constraint = Constraint(
            name=f'rounding up {self.name}',
            left_hand_side=[the_right_term],
            sign=SignConstraint.GREATER_OR_EQUAL,
            right_hand_side=new_b_right,
        )
        constraints_right_problem = AllConstraints(
            self.constraints.constraints + [right_constraint]
        )
        right_subproblem = LinearProblem(
            objective_coefficients=self.objective_coefficients,
            constraints=constraints_right_problem,
            the_name=f'{self.name}_2',
        )
        return [left_subproblem, right_subproblem]

    def bound(self) -> tuple[Solution, bool] | None:
        """
        This method tries to solve the problem or to calculate a lower bound. The bool is True is an optimal
        solution has been found, and False otherwise.
        """
        logger.info(f'Calculating lower bound for {self.name} with constraints')
        logger.info(f'\n{str(self.constraints)}')
        standard_canonical = StandardCanonicalForms(self.constraints)

        # Add the coefficients (zeros) of the slack variables
        num_columns = standard_canonical.standard_matrix.shape[1]

        # Get the current length of the vector
        original_length = len(self.objective_coefficients)

        # how many zeros to add
        num_zeros_to_add = num_columns - original_length

        # If the vector is shorter, append zeros
        standard_objective_coefficients = np.append(
            self.objective_coefficients, np.zeros(num_zeros_to_add)
        )

        the_simplex_algorithm = SimplexAlgorithmTableau(
            objective=standard_objective_coefficients,
            constraint_matrix=standard_canonical.standard_matrix,
            right_hand_side=standard_canonical.standard_vector,
        )
        optimal_tableau = the_simplex_algorithm.solve()
        if optimal_tableau is None:
            return None

        optimal_solution = optimal_tableau.feasible_basic_solution[:original_length]
        the_solution = Solution(
            solution=optimal_solution,
            value=optimal_tableau.value_objective_function,
        )

        if is_solution_integer(the_solution.solution):
            return the_solution, True

        return the_solution, False

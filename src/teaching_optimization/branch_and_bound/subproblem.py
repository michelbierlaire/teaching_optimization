"""Branch and bound algorithm

Michel Bierlaire
Sun Jul 14 10:41:12 2024
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from teaching_optimization.branch_and_bound.solution import Solution

logger = logging.getLogger(__name__)


class Subproblem(ABC):
    """The abstract solver is used by the branch and bound algorithm to solve (relaxations of) sub problems, and to
    branch a problem into sub-problems"""

    def __init__(self, the_name: str) -> None:
        self._name = the_name
        self.solution = None
        self.optimal = None
        calculated_bound = self.bound()
        if calculated_bound is not None:
            self.solution, self.optimal = calculated_bound

    @property
    def name(self) -> str:
        """Name of the subproblem. Must be unique as it is used as an identifier."""
        return self._name

    @abstractmethod
    def branch(self) -> list[Subproblem]:
        """
        Abstract method to be implemented by subclasses.
        This method should handle branching logic.
        """
        pass

    @abstractmethod
    def bound(self) -> tuple[Solution, bool] | None:
        """
        Abstract method to be implemented by subclasses.
        This method tries to solve the problem or to calculate a lower bound.
        If the problem is infeasible, None is returned instead of a solution.

        :return: a solution, and a boolean that is True if the solution is optimal for the subproblem. Or None
            if the problem is infeasible.
        """
        pass

    def __str__(self) -> str:
        return f'{self.name}'

    def solve(self, upper_bound: float) -> Solution | None:
        """Solve the subproblem

        :param upper_bound: upper bound on the optimal value.
        :return: the optimal solution of the subproblem, or None if the problem is infeasible or suboptimal.
        """
        logger.info(f'**** Solving problem {self.name} ****')
        if self.solution is None:
            # The subproblem is infeasible
            logger.info(f'{self.name}: infeasible')
            return None
        if self.optimal:
            logger.info(f'{self.name}: optimal solution {self.solution}')
            return self.solution

        logger.info(f'{self.name}: solution of relaxation {self.solution}')

        if self.solution.value >= upper_bound:
            """This subproblem is suboptimal. No need to solve it."""
            logger.info(
                f'{self.name}: lower bound {self.solution.value} larger than upper bound {upper_bound}'
            )
            return None

        subproblems = self.branch()

        best_solution = None

        for subproblem in subproblems:
            solution: Solution | None = subproblem.solve(upper_bound=upper_bound)
            if solution is None:
                continue
            if best_solution is None:
                best_solution = solution
            elif solution.value < best_solution.value:
                best_solution = solution
            if best_solution.value < upper_bound:
                upper_bound = best_solution.value

        logger.info(f'{self.name}: optimal solution {best_solution}')
        return best_solution

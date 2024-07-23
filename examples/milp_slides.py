"""Branch and bound for the MILP in the slides

Michel Bierlaire
Wed Jul 17 17:37:28 2024
"""

import logging

import numpy as np

from teaching_optimization.branch_and_bound.branch_and_bound_milp import LinearProblem
from teaching_optimization.linear_constraints import (
    Term,
    Variable,
    SignVariable,
    Constraint,
    SignConstraint,
    AllConstraints,
)

logging.basicConfig(level=logging.INFO)


coefficients = np.array([1, -2])

x1 = Variable('x1', sign=SignVariable.NON_NEGATIVE)
x2 = Variable('x2', sign=SignVariable.NON_NEGATIVE)
term_1_1 = Term(
    variable=x1,
    coefficient=-4,
)
term_1_2 = Term(variable=x2, coefficient=6)
constraint_1 = Constraint(
    name='first constraint',
    left_hand_side=[term_1_1, term_1_2],
    sign=SignConstraint.LESSER_OR_EQUAL,
    right_hand_side=9,
)

term_2_1 = Term(
    variable=x1,
    coefficient=1,
)
term_2_2 = Term(variable=x2, coefficient=1)
constraint_2 = Constraint(
    name='second constraint',
    left_hand_side=[term_2_1, term_2_2],
    sign=SignConstraint.LESSER_OR_EQUAL,
    right_hand_side=4,
)

all_constraints = AllConstraints(constraints=[constraint_1, constraint_2])


the_problem = LinearProblem(
    objective_coefficients=coefficients, constraints=all_constraints, the_name='P'
)

solution = the_problem.solve(upper_bound=np.inf)

print(solution)

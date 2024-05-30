"""Illustration of the simplex tableau algorithm, phase II

Michel Bierlaire
Fri May 3 15:16:15 2024
"""

import numpy as np

from biogeme_optimization.teaching.simplex_tableau_phase_two import (
    SimplexTableauPhaseTwo,
)
from biogeme_optimization.teaching.tableau import SimplexTableau

standard_a = np.array([[2, 1, 1, 0], [1, -1, 0, 1]])
standard_b = np.array([6, 2])
standard_c = np.array([-4, 3, 0, 0])

# As the right hand side is non-negative, we choose the slack variables as basic variables, so that the basic
# matrix is the identity matrix. The first tableau contains the data of the problem:
#
# $$ \begin{array}{c|c} A & b \\ \hline c^T & 0 \end{array}$$

# We merge A and b horizontally
Ab = np.column_stack((standard_a, standard_b))

# Then we add the last row ####
initial_tableau = np.row_stack((Ab, np.append(standard_c, 0)))

the_tableau = SimplexTableau(initial_tableau)

print('Initial tableau:')
print(the_tableau.report())

# Initialize the algorithm
algorithm = SimplexTableauPhaseTwo(initial_tableau=the_tableau)

# Run the algorithm and store the reports of each iteration.
iterations = [iterate_tableau for iterate_tableau in algorithm]

# Print the result.
print(f'Optimal solution: {algorithm.solution}')
print(f'Exit status: {algorithm.stopping_cause}')

# Analyze the iterations
for tableau in iterations:
    print(f'=========================')
    print(tableau.report())

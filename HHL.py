#%%
# This document completes the examples and problems from section 5.1 of the
# Qiskit textbook on the HHL algorithm for solving systems of linear
# equations

# This cell completes example 4.A: Running HHL on a simulator: general method

import numpy as np
from linear_solvers import NumPyLinearSolver, HHL

# Create the matrix A and the vector b in |x> = A^-1|b>
matrix = np.array([ [1, -1/3], [-1/3, 1] ])
vector = np.array([1, 0])

# Obtain the exact solution using HHL. Note: this will NOT as log(N) since
# the log(N) scaling only occurs when using HHL to obtain an approximation
# to the solution.
naive_hhl_solution = HHL().solve(matrix, vector)

# Obtain the exact solution using a classical algorithm
classical_solution = NumPyLinearSolver().solve(matrix, vector/np.linalg.norm(vector))

# Obtain the solution using a matrix that HHL can approximately solve with
# log(N) scaling. The tridiagonal Toeplitx matrix works for this example
# (and happens to be exact for 2x2 matrices but won't be exact for larger
# matrices).
from linear_solvers.matrices.tridiagonal_toeplitz import TridiagonalToeplitz
tridi_matrix = TridiagonalToeplitz(1, 1, -1/3)
tridi_solution = HHL().solve(tridi_matrix, vector)

# Print the solution state |x> from the classical method and the quantum
# circuits used to create the solution using HHL. Note: the quantum circuit
# print statements are currently commented out due to an encoding error
# preventing them from printing in my current setup.
print('classical state:', classical_solution.state)
print('naive state:')
# print(naive_hhl_solution.state)
print('tridiagonal state:')
# print(tridi_solution.state)

# Print the Euclidean norm of |x> from each method. HHL can always calculate
# the Euclidean norm of |x>, without using any additional gates, since it is
# simply the probability of measuring |1> in the ancillary qubit. Notice how
# the methods agree up to 13 decimal places.
print('classical Euclidean norm:', classical_solution.euclidean_norm)
print('naive Euclidean norm:', naive_hhl_solution.euclidean_norm)
print('tridiagonal Euclidean norm:', tridi_solution.euclidean_norm)

# While the elements of |x> cannot be read out while maintaining log(N)
# scaling for HHL, they will be read out for instructional purposes of this
# small example.
from qiskit.quantum_info import Statevector

naive_sv = Statevector(naive_hhl_solution.state).data
tridi_sv = Statevector(tridi_solution.state).data

# We only want the quantum states from the circuit that have the ancillary
# qubit in the state |1> and the working qubits that encoded the eigenvalues
# of A in the state |00>. The solution vector's first (second) element will
# be in the state 10000 (10001), corresponding to index 16 (17) of the
# solution state obtained above.
naive_full_vector = np.array([naive_sv[16], naive_sv[17] ])
tridi_full_vector = np.array([tridi_sv[16], tridi_sv[17] ])

print('naive raw solution vector:', naive_full_vector)
print('tridi raw solution vector:', tridi_full_vector)

# The raw solution vector above may contain a constant accumulated from
# gates in the circuit. To obtain the solution vector |x>, divide the full
# vector from above by its norm to remove those gate constants and multiply
# by the Euclidean norm. Note: the real part is taken in the first line in
# the function to remove the imaginary parts, which are extremely small and
# only non-zero because of imperfect computer accuracy.
def get_solution_vector(solution):
    solution_vector = Statevector(solution.state).data[16:18].real
    norm = solution.euclidean_norm
    return norm * solution_vector / np.linalg.norm(solution_vector)

# Print the solution vector from all 3 methods. Notice how they're the same.
print('full naive solution vector:', get_solution_vector(naive_hhl_solution))
print('full tridi solution vector:', get_solution_vector(tridi_solution))
print('classical state:', classical_solution.state)

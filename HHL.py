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

#%%
# This cell runs a larger example of the HHL algorithm that can't be
# completed exactly.

from scipy.sparse import diags

# Use 2 qubits for the |b> state, requiring a 4x4 matrix
num_qubits = 2
matrix_size = 2**num_qubits

a = 1
b = -1/3

matrix = diags([b, a, b],
               [-1, 0, 1],
               shape = (matrix_size, matrix_size)).toarray()

# |b> = |1000>
vector = np.array([1] + [0]*(matrix_size - 1))

classical_solution = NumPyLinearSolver().solve(matrix, vector / np.linalg.norm(vector))
naive_hhl_solution = HHL().solve(matrix, vector)
tridi_matrix = TridiagonalToeplitz(num_qubits, a, b)
tridi_solution = HHL().solve(tridi_matrix, vector)

# Notice how the quantum solutions don't match the classical solution exactly
print('classical euclidean norm:', classical_solution.euclidean_norm)
print('naive euclidean norm:', naive_hhl_solution.euclidean_norm)
print('tridiagonal euclidean norm:', tridi_solution.euclidean_norm)

from qiskit import transpile

max_qubits = 4
i = 1
naive_depths = []
tridi_depths = []
for nqubits in range(1, max_qubits+1):
    matrix = diags([b, a, b],
                   [-1, 0, 1],
                   shape = (2**nqubits, 2**nqubits)).toarray()
    vector = np.array([1] + [0]*(2**nqubits-1))

    naive_hhl_solution = HHL().solve(matrix, vector)
    tridi_matrix = TridiagonalToeplitz(nqubits, a, b)
    tridi_solution = HHL().solve(tridi_matrix, vector)

    naive_qc = transpile(naive_hhl_solution.state, basis_gates = ['id', 'rz', 'sx', 'x', 'cx'])
    tridi_qc = transpile(tridi_solution.state, basis_gates = ['id', 'rz', 'sx', 'x', 'cx'])

    naive_depths.append(naive_qc.depth())
    tridi_depths.append(tridi_qc.depth())

    i += 1

sizes = [f"{2**nqubits}x{2**nqubits}" for nqubits in range(1, max_qubits+1)]
columns = ['size of the system', 'quantum_solution depth', 'tridi_solution depth']
data = np.array([sizes, naive_depths, tridi_depths])
row_format = "{:>23}" * (len(columns) + 2)
for team, row in zip(columns, data):
    print(row_format.format(team, *row))

print('excess:', [naive_depths[i] - tridi_depths[i] for i in range(0, len(naive_depths))])
# %%
# This cell

from linear_solvers.observables import AbsoluteAverage, MatrixFunctional

num_qubits = 1
matrix_size = 2**num_qubits

a = 1
b = -1/3

matrix = diags([b, a, b],
               [-1, 0, 1],
               shape = (matrix_size, matrix_size)).toarray()
vector = np.array([1] + [0]*(matrix_size - 1))
tridi_matrix = TridiagonalToeplitz(1, a, b)

average_solution = HHL().solve(tridi_matrix, vector, AbsoluteAverage())
classical_average = NumPyLinearSolver().solve(matrix, vector / np.linalg.norm(vector), AbsoluteAverage())

print('quantum average:', average_solution.observable)
print('classical average:', classical_average.observable)
print('quantum circuit results:', average_solution.circuit_results)

observable = MatrixFunctional(1, 1/2)

functional_solution = HHL().solve(tridi_matrix, vector, observable)
classical_functional = NumPyLinearSolver().solve(matrix, vector / np.linalg.norm(vector), observable)

print('quantum functional:', functional_solution.observable)
print('classical functional:', classical_functional.observable)
print('quantum circuit results:', functional_solution.circuit_results)

from qiskit import Aer

backend = Aer.get_backend('aer_simulator')
hhl = HHL(1e-3, quantum_instance=backend)

accurate_solution = hhl.solve(matrix, vector)
classical_solution = NumPyLinearSolver(
                    ).solve(matrix,
                            vector / np.linalg.norm(vector))

print(accurate_solution.euclidean_norm)
print(classical_solution.euclidean_norm)

# %%

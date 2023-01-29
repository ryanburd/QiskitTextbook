#%%
# This file completes the examples and problems from section 3.6 of the Qiskit
# textbook on quantum phase estimation (QPE).

#initialization
import matplotlib.pyplot as plt
import numpy as np

# importing Qiskit
from qiskit import IBMQ, Aer, transpile, assemble
from qiskit import QuantumCircuit

# import basic plot tools and circuits
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT

#%%
# This cell completes Example: T-gate, where we want to show that QPE returns
# theta = 1/8 since a T-gate applies a phase of exp[i*pi/4].

# Create a circuit with 4 qubits and 3 classical bits for the readout of theta.
# The first 3 counting qubits will encode theta and the fourth qubit will be
# the eigenvector of the T-gate operation, |1>.
qpe = QuantumCircuit(4, 3)
qpe.x(3)
qpe.draw(output='mpl')

# Put all the counting qubits in the |+> state so they can receive phase
# kickback when controlling the U operations to the eigenvector qubit below.
for qubit in range(3):
    qpe.h(qubit)
qpe.draw(output='mpl')

# Apply the controlled-U operations to the eigenvector qubit. Use pi/4 for the
# Qiskit phase gate to recreate the T-gate. The first counting qubit applies 1
# CU, and each successive counting qubit applies twice the number of CUs as the
# previous counting qubit.
repetitions = 1
for counting_qubit in range(3):
    for i in range(repetitions):
        qpe.cp(np.pi/4, counting_qubit, 3)
    repetitions *= 2
qpe.draw()

# (2^n)theta is now encoded in the counting qubits, but they are in the Fourier
# basis. Apply the inverse QFT to put the qubits into the computational basis.
# Then measure the qubits, storing the results in the classical bits.
qpe.barrier()
qpe = qpe.compose(QFT(3, inverse=True), [0, 1, 2])
qpe.barrier()
for n in range(3):
    qpe.measure(n,n)
qpe.draw()

# Run the code on the simulator. The state closest to (2^n)theta will have the
# highest probability. Divide the binary number represented by the measured
# state by 2^n to obtain theta. Since (2^n)*1/8 can be calculated exactly with
# 3 qubits, the result below should be the state |001> with certainty.
aer_sim = Aer.get_backend('aer_simulator')
shots = 2048
t_qpe = transpile(qpe, aer_sim)
results = aer_sim.run(t_qpe, shots=shots).result()
answer = results.get_counts()

plot_histogram(answer)

#%%
# This cell completes Example: Getting More Precision, where (2^n)theta cannot
# be calculated exactly. Theta = 1/3

nqubits = 6
qpe2 = QuantumCircuit(nqubits, nqubits-1)

for qubit in range(nqubits-1):
    qpe2.h(qubit)

qpe2.x(nqubits-1)

angle = 2*np.pi/3
repetitions = 1
for counting_qubit in range(nqubits-1):
    for i in range(repetitions):
        qpe2.cp(angle, counting_qubit, nqubits-1)
    repetitions = 2

qpe2 = qpe2.compose(QFT(nqubits-1, inverse=True), [i for i in range(nqubits-1)])

for n in range(nqubits-1):
    qpe2.measure(n,n)

qpe2.draw()

aer_sim = Aer.get_backend('aer_simulator')
shots = 4096
t_qpe2 = transpile(qpe2, aer_sim)
results = aer_sim.run(t_qpe2, shots=shots).result()
answer = results.get_counts()

plot_histogram(answer)
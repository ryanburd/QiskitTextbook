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
# be calculated exactly with n <= 6 and theta = 1/3. The code is virtually
# identical to the above cell. It has been compiled into a function to reuse in
# later problems.

def QPE_sim(circuit, angle, nqubits):

    for qubit in range(nqubits-1):
        circuit.h(qubit)

    repetitions = 1
    for counting_qubit in range(nqubits-1):
        for i in range(repetitions):
            circuit.cp(angle, counting_qubit, nqubits-1)
        repetitions *= 2

    circuit = circuit.compose(QFT(nqubits-1, inverse=True), range(nqubits-1))

    for n in range(nqubits-1):
        circuit.measure(n,n)

    circuit.draw()

    aer_sim = Aer.get_backend('aer_simulator')
    shots = 4096
    t_circuit = transpile(circuit, aer_sim)
    results = aer_sim.run(t_circuit, shots=shots).result()
    answer = results.get_counts()

    return answer


# Use 6 qubits to estimate theta = 1/3.
nqubits = 6
qpe2 = QuantumCircuit(nqubits, nqubits-1)
qpe2.x(nqubits-1)
answer = QPE_sim(qpe2, 2*np.pi/3, nqubits)
plot_histogram(answer)

# %%
# This cell completes Experiment with Real Devices. The T-gate example from
# above will be repeated, but this time performed on a real quantum computer to
# show the influence of noise and gate errors. The |001> state should still be
# the most probable, but it will no longer be measured with certainty.

nqubits = 4
qpe3 = QuantumCircuit(nqubits, nqubits-1)
qpe3.x(nqubits-1)

for qubit in range(nqubits-1):
    qpe3.h(qubit)

repetitions = 1
for counting_qubit in range(nqubits-1):
    for i in range(repetitions):
        qpe3.cp(np.pi/4, counting_qubit, nqubits-1)
    repetitions *= 2

qpe3.barrier()
qpe3 = qpe3.compose(QFT(nqubits-1, inverse=True), range(nqubits-1))
qpe3.barrier()
for n in range(nqubits-1):
    qpe3.measure(n,n)
qpe3.draw(output='mpl')

# Find the least busy computer.
from qiskit.providers.ibmq import least_busy
IBMQ.load_account()
from qiskit.tools.monitor import job_monitor
provider = IBMQ.get_provider(hub='ibm-q')
backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= (n+1) and
                                   not x.configuration().simulator and x.status().operational==True))
print("least busy backend: ", backend)

# Run with 2048 shots.
shots = 2048
t_qpe3 = transpile(qpe3, backend, optimization_level=3)
job = backend.run(t_qpe3, shots=shots)
job_monitor(job)

# Get the results from the computation.
results = job.result()
answer = results.get_counts(qpe3)

plot_histogram(answer)

# %%
# The remaining cells completes the problems in section 3.6.

# Problem 1: Try the experiments above with different gates (CNOT, Controlled-S,
# Controlled-T+); what results do you expect? What results do you get?

# CNOT: theta = 0 since phase = 1 with |psi> = |+>
# Regardless of n, (2^n)theta should equal 0.

# Can also use |psi> = |->, where theta = 1/2.
# With n = 3, (2^n)theta should equal 4.
nqubits = 4
qpe4 = QuantumCircuit(nqubits, nqubits-1)
# qpe4.x(nqubits-1)
qpe4.h(nqubits-1)

for qubit in range(nqubits-1):
        qpe4.h(qubit)

repetitions = 1
for counting_qubit in range(nqubits-1):
    for i in range(repetitions):
        qpe4.cx(counting_qubit, nqubits-1)
    repetitions *= 2

qpe4 = qpe4.compose(QFT(nqubits-1, inverse=True), range(nqubits-1))

for n in range(nqubits-1):
    qpe4.measure(n,n)

qpe4.draw()

aer_sim = Aer.get_backend('aer_simulator')
shots = 4096
t_qpe4 = transpile(qpe4, aer_sim)
results = aer_sim.run(t_qpe4, shots=shots).result()
answer = results.get_counts()

plot_histogram(answer)

#%%
# Problem 1: Controlled-S: theta = 1/4, |psi> = |1>
# With n = 3, (2^n)theta should equal 2, |010>
nqubits = 4
qpe2 = QuantumCircuit(nqubits, nqubits-1)
qpe2.x(nqubits-1)
answer = QPE_sim(qpe2, np.pi/2, nqubits)
plot_histogram(answer)

#%%
# Problem 1: Controlled-T+: theta = 5/8, |psi> = |1>
# Theta comes from exp[i*pi/4] for T gate + exp[i*pi] to negate the phase.
# With n = 3, (2^n)theta should equal 5, |101>
nqubits = 4
qpe2 = QuantumCircuit(nqubits, nqubits-1)
qpe2.x(nqubits-1)
answer = QPE_sim(qpe2, np.pi + np.pi/4, nqubits)
plot_histogram(answer)

# %%
# This cell completes Problem 2: Try the experiment with a Controlled-Y-gate,
# do you get the result you expected? (Hint: Remember to make sure |psi> is an
# eigenstate of Y!)

# Controlled-Y: theta = 0 since phase = 1 with |psi> = |0> + i|1>
# Regardless of n, (2^n)theta should equal 0.
nqubits = 4
qpe4 = QuantumCircuit(nqubits, nqubits-1)
# qpe4.x(nqubits-1)
qpe4.h(nqubits-1)
qpe4.s(nqubits-1)

for qubit in range(nqubits-1):
        qpe4.h(qubit)

repetitions = 1
for counting_qubit in range(nqubits-1):
    for i in range(repetitions):
        qpe4.cy(counting_qubit, nqubits-1)
    repetitions *= 2

qpe4 = qpe4.compose(QFT(nqubits-1, inverse=True), range(nqubits-1))

for n in range(nqubits-1):
    qpe4.measure(n,n)

qpe4.draw()

aer_sim = Aer.get_backend('aer_simulator')
shots = 4096
t_qpe4 = transpile(qpe4, aer_sim)
results = aer_sim.run(t_qpe4, shots=shots).result()
answer = results.get_counts()

plot_histogram(answer)
# %%

#%%
import numpy as np
from numpy import pi
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram, plot_bloch_multivector

#%%
# This cell completes a 3 qubit QFT

# Create a 3 qubit circuit
qc = QuantumCircuit(3)

# Hadamard gate applied to qubit 2
qc.h(2)

# CROT quarter turn, qubit 1 controlling qubit 2
qc.cp(pi/2, 1, 2)

# CROT eighth turn, qubit 0 controlling qubit 2
qc.cp(pi/4, 0, 2)

# Following same pattern on qubits 1 and 0
qc.h(1)
qc.cp(pi/2, 0, 1)
qc.h(0)

# Swap qubits 0 and 2 to put Fourier transformed qubits in correct order
qc.swap(0, 2)

qc.draw(output='mpl')

# %%
# This cell completes an n-qubit QFT

# Peforms the rotations starting from the specified most significant qubit.
# Recursion completes the rotations for each successive less signifcant qubit until reaching qubit 0.
def qft_rotations(circuit,n):
    # End the recursion once all qubits in the QFT have been rotated
    if n == 0:
        return circuit
    # 'n' is the number of qubits in the QFT, so -1 to get the correct index
    n -= 1
    circuit.h(n)
    for qubit in range(n):
        circuit.cp(pi/2**(n-qubit), qubit, n)
    qft_rotations(circuit, n)

# Swaps the order of the qubits in the QFT
def swap_registers(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

# Apply rotations and then swap the qubit order to obtain the QFT
def qft(circuit, n):
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit

# Use the number 8 in the computational basis to test the QFT code.
# Rather than testing a number needing 3 qubits like in the text and problems,
# I am using a number that requires 4 qubits.
# Running the line below provides 8 in binary: 1000
bin(8)

# Create a circuit of 4 qubits (to represent 8 in binary) and obtain its QFT
qc = QuantumCircuit(4)
# Intialize qubit 0 in the 1 state to create the 1000 state.
qc.x(3)

# Use the simulator to check that the qubits are in the state 1000.
sim = Aer.get_backend("aer_simulator")
qc_init = qc.copy()
qc_init.save_statevector()
statevector = sim.run(qc_init).result().get_statevector()
plot_bloch_multivector(statevector)

# Obtain the QFT and show the qubit states.
qft(qc,4)
qc.draw(output='mpl')
qc.save_statevector()
statevector = sim.run(qc).result().get_statevector()
plot_bloch_multivector(statevector)
#%%
# Below, we will test the n-qubit QFT circuit on a real quantum computer. We
# will first create the Fourier transformed state 8, apply the inverse of the
# circuit, and measure the inversed state on the real quantum computer. We
# should obtain the 100 state with the highest probability.

# Define the inverse of the n-qubit QFT circuit above.
def inverse_qft(circuit, n):
    qft_circ = qft(QuantumCircuit(n), n)
    invqft_circ = qft_circ.inverse()
    circuit.append(invqft_circ, circuit.qubits[:n])
    return circuit.decompose()

# Create the qubits in their Fourier transformed states.
number = 8
nqubits = 4
qc = QuantumCircuit(nqubits)
for qubit in range(nqubits):
    qc.h(qubit)
qc.p(number*pi/8,0)
qc.p(number*pi/4,1)
qc.p(number*pi/2,2)
qc.p(number*pi,3)
qc.draw(output='mpl')

# Verify the prepared state matches the QFT state from above.
sim = Aer.get_backend("aer_simulator")
qc_init = qc.copy()
qc_init.save_statevector()
statevector = sim.run(qc_init).result().get_statevector()
plot_bloch_multivector(statevector)

# Create the inversed QFT circuit
qc = inverse_qft(qc, nqubits)
qc.measure_all()
qc.draw(output='mpl')

# Get the least busy backend with at least the same number of qubits as nqubits
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= nqubits
                                        and not x.configuration().simulator
                                        and x.status().operational==True))
print("least busy backend: ", backend)

# Run the job
shots = 2048
transpiled_qc = transpile(qc, backend, optimization_level=3)
job = backend.run(transpiled_qc, shots=shots)
job_monitor(job)
counts = job.result().get_counts()
plot_histogram(counts)
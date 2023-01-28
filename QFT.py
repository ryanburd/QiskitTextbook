#%%
import numpy as np
from numpy import pi
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram, plot_bloch_multivector

#%%
# This cell completes a 3 qubit QFT

# def main():
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

# if __name__ == "main":
#     main()
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
# Running the line below provides 8 in binary: 1000
bin(8)

# Create a circuit of 4 qubits (to represent 8 in binary) and obtain its QFT
qc = QuantumCircuit(4)
# Intialize qubit 0 in the 1 state to create the 1000 state.
qc.x(0)

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
# %%

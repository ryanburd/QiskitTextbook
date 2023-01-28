#%%
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram, plot_bloch_multivector

# def main():
# Create a 3 qubit circuit
qc = QuantumCircuit(3)

# Hadamard gate applied to qubit 2
qc.h(2)

qc.draw()

# if __name__ == "main":
#     main()
# %%

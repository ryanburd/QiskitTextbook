from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver

# Use PySCF, a classical computational chemistry software
# package, to compute the one-body and two-body integrals in
# electronic-orbital basis, necessary to form the Fermionic operator
driver = PySCFDriver(
    atom='H .0 .0 .0; H .0 .0 0.735',
    unit=DistanceUnit.ANGSTROM,
    basis='sto3g',
)
problem = driver.run()

# setup the mapper and qubit converter
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.mappers import QubitConverter

mapper = ParityMapper()
converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)

# setup the classical optimizer for the VQE
from qiskit.algorithms.optimizers import L_BFGS_B

optimizer = L_BFGS_B()

# setup the estimator primitive for the VQE
from qiskit.primitives import Estimator

estimator = Estimator()

# setup the ansatz for VQE
from qiskit_nature.second_q.circuit.library import UCCSD

ansatz = UCCSD()

# use a factory to complement the VQE and its components at runtime
from qiskit_nature.second_q.algorithms import VQEUCCFactory

vqe_factory = VQEUCCFactory(estimator, ansatz, optimizer)

# prepare the ground-state solver and run it
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

algorithm = GroundStateEigensolver(converter, vqe_factory)

electronic_structure_result = algorithm.solve(problem)
print(electronic_structure_result)
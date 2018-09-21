# Qubit Dynamics and Quantum Error Correction

## Qubit dynamics
The qubit is the fundamental building block of a quantum computer, it is a two-level quantum system, realistically implemented in many different ways (spin of an electron, optical lattices, polarised light). An isolated qubit is described by a wave function of the form |𝜑⟩ = a|0⟩ + b|1⟩ , whose dynamics obey the Schrödinger equation, resulting in Rabi oscillations between the ground and excited state. We cannot, however, completely isolate a qubit from the environment, and its interactions with it lead to decoherence phenomena such as spontaneous emission, where a qubit decays from its higher state by emitting a photon. Studying the behaviour of many individual two-level systems, we will obtain a quantum trajectory describing the behaviour of the ensemble. This is done theoretically by solving the optical Bloch equations (read more in [this paper](https://www.researchgate.net/publication/243218822_Analytic_solutions_of_the_optical_Bloch_equations)), but their analytical solution introduces some approximations, therefore we will simulate this system using a Monte Carlo wavefunction (MCWF) method.

We start from the behaviour of an isolated qubit, oscillating in a sinusoidal fashion between its excited and ground. Random decays are introduced probabilistically for every timestep, yielding the quantum trajectory of one qubit. We then average many of these trajectories to obtain the behaviour of the system and compare it to the analytical solution of the optical Bloch equations.

The results of the simulation are obtained in the QubitDynamics.py script, and yeald results with residuals of less than 1%.


## Quantum Error Correction

Classical error correction is based on the repetition code, but in the quantum world the no cloning theorem doesn’t allow us to replicate qubits. Quantum error correction is thus achieved through syndrome extraction, projecting the state of an encoded qubit onto orthogonal error subspaces measured using ancilla qubits without affecting the original qubit. Different codes can be used for error correction, in this project three have been simulated, that use 3, 4 and 9 quibits respectively. More detailed information is given in the pdf file named QECreport. The results all prove that applying error correction (or error detection in the case of the 4-qubit code) improves the fidelity of the encoded qubit if the error probability is lower than a certain threshold. This threshold depends on the code used and is the interest of this investigation.

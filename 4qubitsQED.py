import numpy as np
import matplotlib.pyplot as pyplot
import random

X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
I = np.array([[1,0],[0,1]])
H = (1/(np.sqrt(2)))*np.array([[1, 1],[1, -1]])
Zero = np.array([[1],[0]])
One = np.array([[0],[1]])

def NKron(*args):
	'''Calculate a Kronecker product over a variable number of inputs'''
	result = np.array([[1]])
	for op in args:
		result = np.kron(result, op)
	return result


ZeroZero = (1/(np.sqrt(2)))*(NKron(Zero,Zero,Zero,Zero)+NKron(One,One,One,One))
ZeroOne = (1/(np.sqrt(2)))*(NKron(One,One,Zero,Zero)+NKron(Zero,Zero,One,One))
OneZero = (1/(np.sqrt(2)))*(NKron(One,Zero,One,Zero)+NKron(Zero,One,Zero,One))
OneOne = (1/(np.sqrt(2)))*(NKron(Zero,One,One,Zero)+NKron(One,Zero,Zero,One))

states = [ZeroZero, ZeroOne, OneZero, OneOne]
EncState = random.choice(states)


#define bit flip err matrices
Bit1 = NKron(X,I,I,I)
Bit2 = NKron(I,X,I,I)
Bit3 = NKron(I,I,X,I)
Bit4 = NKron(I,I,I,X)

BitErrs = [Bit1, Bit2, Bit3, Bit4]

#define phase flip err matrices
Ph1 = NKron(Z,I,I,I)
Ph2 = NKron(I,Z,I,I)
Ph3 = NKron(I,I,Z,I)
Ph4 = NKron(I,I,I,Z)

PhErrs = [Ph1, Ph2, Ph3, Ph4]


#define bit and phase flip detection matrices
BitDet = NKron(Z,Z,Z,Z)
PhDet = NKron(X,X,X,X)


#define projection matrices for error prob measurement
PiBit = ((np.identity(2**4) - BitDet)/2)
PiPh = ((np.identity(2**4) - PhDet)/2)



def BitErr(qubit, p): # bit flip
	NewErrBit = np.array(random.sample(BitErrs, 4))
	for i in range(4):
		if random.random() < p:
			qubit = np.dot(NewErrBit[i], qubit)
		else:
			qubit = qubit
	return qubit



def PhErr(qubit, p): #phase flip
	NewErrPh = np.array(random.sample(PhErrs, 4))
	for i in range(4):
		if random.random() < p:
			qubit = np.dot(NewErrPh[i], qubit)
		else:
			qubit = qubit
	return qubit


#only if 4 flips occur code fails (fidelity 0), if error is detected discard qubit, if no error is detected return fidelity 1
def AvgFid(qubit, n, p):
	TotFid = 0
	NQb = 0
	for i in range(n):
		x = np.copy(qubit)


		#detectd errs
		qubitBit = BitErr(qubit,p)
		qubitPh = PhErr(qubitBit,p)

		#return one or zero
		ErrPh = np.dot(qubitPh.T,(np.dot(PiPh, qubitPh)))
		ErrBit = np.dot(qubitPh.T,(np.dot(PiBit, qubitPh)))

		if ErrBit == 0 and ErrPh == 0:
			NQb += 1
			if np.array_equal(x, qubitPh):
				TotFid += 1
			else:
				TotFid += 0
		else:
			NQb += 0
			TotFid += 0

	if NQb == 0:
		AvgFid = 0
	else:
		AvgFid = float(TotFid)/NQb
	AvgFid = np.sqrt(AvgFid)

	return AvgFid


def AvgFidUncorr1(qubit, n, p): # average fidelity of qubit subject to bit flip only - no error detection
	TotFid = 0
	for i in range(n):
		x = np.copy(qubit)
		if random.random() < p: #bit flip err
			qubit = np.dot(X, qubit)
		else:
			qubit = qubit

		if np.array_equal(x, qubit):
			fidelity = 1
		else:
			fidelity = 0

		TotFid += fidelity

	AvgFid = float(TotFid)/n
	AvgFid = np.sqrt(AvgFid)

	return AvgFid

def AvgFidUncorr1Both(qubit, n, p): # average fidelity of qubit subject to bit flip and phase flip - no error detection
	TotFid = 0
	for i in range(n):
		x = np.copy(qubit)
		if random.random() < p: #bit flip err
			qubit = np.dot(X, qubit)
		else:
			qubit = qubit

		if random.random() < p: #phase flip err
			qubit = np.dot(Z, qubit)
		else:
			qubit = qubit

		if np.array_equal(x, qubit):
			fidelity = 1
		else:
			fidelity = 0

		TotFid += fidelity

	AvgFid = float(TotFid)/n
	AvgFid = np.sqrt(AvgFid)

	return AvgFid

x_vals = np.array([]) #x axis values (probabilities)
y_vals_4_qb = np.array([]) #x axis values for 3 qubit code (avg fidelity)
y_vals_uncorr1 = np.array([]) #uncorrected 1 qubit (bit flip)
y_vals_uncorr1_both = np.array([]) #uncorrected 1 qubit (bit and phase flip)

n_qubits = 1000
x_steps = 51 #range of probabilities


		
for p in np.linspace(0.0, 0.5, num = x_steps): #y vals for 4 qubit code

	x_vals = np.append(x_vals, np.array([p]))

	Fid9 = AvgFid(EncState, n_qubits, p)
	y_vals_4_qb = np.append(y_vals_4_qb, np.array([Fid9]))
	print('first', p) #check progress


for p in np.linspace(0.0, 0.5, num = x_steps): #y vals for uncerrected 1 qubit in either 0 or 1 (bit and phase)
	state1 = [Zero, One]
	StateIn = random.choice(state1)
	FidUncorr1 = AvgFidUncorr1Both(StateIn, n_qubits, p)
	y_vals_uncorr1 = np.append(y_vals_uncorr1, np.array([FidUncorr1]))
	print('second', p) # check progress


'''plot all of the things'''

pyplot.figure(figsize = (5,3), dpi = 100)

pyplot.plot(x_vals, y_vals_4_qb, linewidth = 1, color = 'r', label='4 qubit code')
pyplot.plot(x_vals, y_vals_uncorr1, linewidth = 1, color = 'k', label='bit flip error')

pyplot.xlabel('Error Probability')
pyplot.ylabel('Average Fidelity')
pyplot.minorticks_on()
pyplot.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on')

pyplot.tight_layout()
pyplot.savefig('4qubit_plot_vec.jpg', dpi = 400)

pyplot.show()
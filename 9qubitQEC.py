import numpy as np
import matplotlib.pyplot as pyplot
import random


'''9-qubit code: 9 possible bit flip errors (only 3 max correctable), 9 possible phase flip errors (1 correctable), can
correct for both phase and bit flip error on a single qubit
three copies of 3 qubit code, phase flip detected comparing two copies
each 3qubit block is encoded to 1/sqrt(8) (|000>+or-|111>), phase flip detected by comparing signs
phase flip in hadamard basis is bit flip (|+> to |->)'''

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


Zero3 = NKron(Zero,Zero,Zero)
One3 = NKron(One,One,One)

Zero9 =(1/np.sqrt(8))*NKron((Zero3+One3),(Zero3+One3),(Zero3+One3))
One9 = (1/np.sqrt(8))*NKron((Zero3-One3),(Zero3-One3),(Zero3-One3))

states = [Zero9, One9]
EncState = random.choice(states)

#define bit flip err matrices
Bit1 = NKron(X,I,I,I,I,I,I,I,I)
Bit2 = NKron(I,X,I,I,I,I,I,I,I)
Bit3 = NKron(I,I,X,I,I,I,I,I,I)
Bit4 = NKron(I,I,I,X,I,I,I,I,I)
Bit5 = NKron(I,I,I,I,X,I,I,I,I)
Bit6 = NKron(I,I,I,I,I,X,I,I,I)
Bit7 = NKron(I,I,I,I,I,I,X,I,I)
Bit8 = NKron(I,I,I,I,I,I,I,X,I)
Bit9 = NKron(I,I,I,I,I,I,I,I,X)

BitErrs = [Bit1, Bit2, Bit3, Bit4, Bit5, Bit6, Bit7, Bit8, Bit9]

#define bit flip detection matrices
Det1 = NKron(Z,Z,I,I,I,I,I,I,I)
Det2 = NKron(I,Z,Z,I,I,I,I,I,I)
Det3 = NKron(I,I,I,Z,Z,I,I,I,I)
Det4 = NKron(I,I,I,I,Z,Z,I,I,I)
Det5 = NKron(I,I,I,I,I,I,Z,Z,I)
Det6 = NKron(I,I,I,I,I,I,I,Z,Z)

BitDets = [Det1,Det2,Det3,Det4,Det5,Det6]

#define projection matrices for error prob measurement
Pi1 = ((np.identity(2**9) - Det1)/2)
Pi2 = ((np.identity(2**9) - Det2)/2)
Pi3 = ((np.identity(2**9) - Det3)/2)
Pi4 = ((np.identity(2**9) - Det4)/2)
Pi5 = ((np.identity(2**9) - Det5)/2)
Pi6 = ((np.identity(2**9) - Det6)/2)

Projs = [Pi1,Pi2,Pi3,Pi4,Pi5,Pi6]


def BitErr(qubit, p): # bit flip error
	for i in range(9):
		if random.random() < p:
			qubit = np.dot(BitErrs[i], qubit)
		else:
			qubit = qubit
	return qubit


def BitDet(qubit): # bit flip error detection
	P = np.array([])
	for i in range (6):
		ProbErr = np.dot(qubit.T,(np.dot(Projs[i], qubit)))
		P = np.append(P, ProbErr)
	return P

def BitCorr(qubit): # bit flip error correction
	P = BitDet(qubit)
	
	#correct first block
	if round(P[0]) == 1 and P[1] == 0: #error on qb1
		qubit = np.dot(BitErrs[0], qubit)
	elif round(P[0]) == 1 and round(P[1]) == 1: #err on qb2, probably floating point prec error on P[i] == 1
		qubit = np.dot(BitErrs[1], qubit)
	elif P[0] == 0 and round(P[1]) == 1: #err on qb3
		qubit = np.dot(BitErrs[2], qubit)
	elif P[0] == 0 and P[1] == 0: #no err
		qubit = qubit

	#correct second block
	if round(P[2]) == 1 and P[3] == 0: #error on qb4
		qubit = np.dot(BitErrs[3], qubit)
	elif round(P[2]) == 1 and round(P[3]) == 1: #err on qb5
		qubit = np.dot(BitErrs[4], qubit)
	elif P[2] == 0 and round(P[3]) == 1: #err on qb6
		qubit = np.dot(BitErrs[5], qubit)
	elif P[2] == 0 and P[3] == 0: #no err
		qubit = qubit

	#correct third block
	if round(P[4]) == 1 and P[5] == 0: #error on qb7
		qubit = np.dot(BitErrs[6], qubit)
	elif round(P[4]) == 1 and round(P[5]) == 1: #err on qb8
		qubit = np.dot(BitErrs[7], qubit)
	elif P[4] == 0 and round(P[5]) == 1: #err on qb9
		qubit = np.dot(BitErrs[8], qubit)
	elif P[4] == 0 and P[5] == 0: #no err
		qubit = qubit

	return qubit



#define phase flip err matrices
Ph1 = NKron(Z,I,I,I,I,I,I,I,I)
Ph2 = NKron(I,Z,I,I,I,I,I,I,I)
Ph3 = NKron(I,I,Z,I,I,I,I,I,I)
Ph4 = NKron(I,I,I,Z,I,I,I,I,I)
Ph5 = NKron(I,I,I,I,Z,I,I,I,I)
Ph6 = NKron(I,I,I,I,I,Z,I,I,I)
Ph7 = NKron(I,I,I,I,I,I,Z,I,I)
Ph8 = NKron(I,I,I,I,I,I,I,Z,I)
Ph9 = NKron(I,I,I,I,I,I,I,I,Z)

PhErrs = [Ph1, Ph2, Ph3, Ph4, Ph5, Ph6, Ph7, Ph8, Ph9]

#define phase flip detection matrices
DetPh1 = NKron(X,X,X,X,X,X,I,I,I)
DetPh2 = NKron(I,I,I,X,X,X,X,X,X)

PhDets = [Det1,Det2]

#define projection matrices for error prob measurement
PiPh1 = ((np.identity(2**9) - DetPh1)/2)
PiPh2 = ((np.identity(2**9) - DetPh2)/2)

PhProjs = [PiPh1,PiPh2]

def PhErr(qubit, p): #phase flip error
	for i in range(9):
		if random.random() < p:
			qubit = np.dot(PhErrs[i], qubit)
		else:
			qubit = qubit
	return qubit


def PhDet(qubit): #phase flip error detection
	P = np.array([])
	for i in range (2):
		ProbErr = np.dot(qubit.T,(np.dot(PhProjs[i], qubit)))
		P = np.append(P, ProbErr)
	return P

def PhCorr(qubit): #phase flip error correction
	#qubit = BitErr(qubit, p)
	P = PhDet(qubit)
	
	#correct first block
	if round(P[0]) == 1 and P[1] == 0: #error on block1
		qubit = np.dot(PhErrs[0], qubit)
	elif round(P[0]) == 1 and round(P[1]) == 1: #err on block2, probably floating point prec error on P[i] == 1
		qubit = np.dot(PhErrs[3], qubit)
	elif P[0] == 0 and round(P[1]) == 1: #err on block3
		qubit = np.dot(PhErrs[6], qubit)
	elif P[0] == 0 and P[1] == 0: #no err
		qubit = qubit

	return qubit




def AvgFid(qubit, n, p): # average fidelity of corrected qubit
	TotFid = 0
	for i in range(n):
		x = np.copy(qubit)
		QbErr = BitErr(qubit,p)
		QbCorr = BitCorr(QbErr)
		QbPhErr = PhErr(QbCorr,p)
		QbPhCorr = PhCorr(QbPhErr)
		if np.array_equal(x, QbPhCorr):
			fidelity = 1
		else:
			fidelity = 0

		TotFid += fidelity

	AvgFid = float(TotFid)/n
	AvgFid = np.sqrt(AvgFid)

	return AvgFid


def AvgFidUncorr9(qubit, n, p): # average fidelity of 9 uncorrected qubits
	TotFid = 0
	for i in range(n):
		x = np.copy(qubit)
		QbErr = BitErr(qubit,p)
		QbErr = PhErr(QbErr,p)
		if np.array_equal(x, QbErr):
			fidelity = 1
		else:
			fidelity = 0

		TotFid += fidelity

	AvgFid = float(TotFid)/n
	AvgFid = np.sqrt(AvgFid)

	return AvgFid



def AvgFidUncorr1Both(qubit, n, p): # average fidelity of uncorrected single qubit
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



'''initialise x and y arrays'''

x_vals = np.array([]) #x axis values (probabilities)
y_vals_9_qb = np.array([]) #x axis values for 9 qubit code (avg fidelity)
y_vals_uncorr9 = np.array([]) #uncorrected 9 qubits
y_vals_uncorr1 = np.array([]) #uncorrected 1 qubit (bit and phase flip)
y_vals_uncorr1_both = np.array([]) #uncorrected 1 qubit (bit and phase flip)

n_qubits = 10000
x_steps = 51 #number of probabilities evaluated


'''populate x and y arrays'''
		
for p in np.linspace(0.0, 0.2, num = x_steps): #y vals for corrected 9 qubit

	x_vals = np.append(x_vals, np.array([p]))

	Fid9 = AvgFid(EncState, n_qubits, p)
	y_vals_9_qb = np.append(y_vals_9_qb, np.array([Fid9]))
	print( 'first', p)


for p in np.linspace(0.0, 0.2, num = x_steps): #y vals for uncorrected 9 qubit

	FidUncorr9 = AvgFidUncorr9(EncState, n_qubits, p)
	y_vals_uncorr9 = np.append(y_vals_uncorr9, np.array([FidUncorr9]))
	print( 'second', p)


for p in np.linspace(0.0, 0.2, num = x_steps): #y vals for uncerrected 1 qubit in either 0 or 1 (bit and phase)
	state1 = [Zero, One]
	StateIn = random.choice(state1)
	FidUncorr1 = AvgFidUncorr1Both(StateIn, n_qubits, p)
	y_vals_uncorr1 = np.append(y_vals_uncorr1, np.array([FidUncorr1]))
	print( 'third', p)



'''plot all of the things'''

pyplot.figure(figsize = (5,3), dpi = 100)

pyplot.plot(x_vals, y_vals_9_qb, linewidth = 0.5, color = 'C1', label='9 qubit code')
pyplot.plot(x_vals, y_vals_uncorr9, linewidth = 0.5, color = 'r', label='no error correction on 9 qubits')
pyplot.plot(x_vals, y_vals_uncorr1, ls = '--', linewidth = 0.5, color = 'k', label='error 1qb')

pyplot.xlabel('Error Probability')
pyplot.ylabel('Average Fidelity')
pyplot.minorticks_on()
pyplot.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on')

pyplot.tight_layout()
pyplot.savefig('9qubit_plot_10000.jpg', dpi = 400)


for i in range(len(y_vals_9_qb)): #print intersect between encoded and unencoded
	if abs(y_vals_9_qb[i] -  y_vals_uncorr1[i]) < 0.005:
		print (x_vals[i])





pyplot.show()
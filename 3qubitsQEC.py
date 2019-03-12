import numpy as np
import matplotlib.pyplot as pyplot
import random

'''simulate a 3-qubit code to detect and correct a bit flip that might occur in a qubit'''


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


states = [Zero, One]
qb1 = random.choice(states)

EncState = NKron(qb1, qb1, qb1)

#define error matrices
Err1 = NKron(X,I,I)
Err2 = NKron(I,X,I)
Err3 = NKron(I,I,X)


#define error detecting matrices
Det1 = NKron(Z,Z,I)
Det2 = NKron(I,Z,Z)



def Error3(qubit, p):

	'''introduce a random error (qubit flip)'''

	if random.random() < p:
		qubit = np.dot(Err1, qubit) #error on qubit 1
	else:
		qubit = qubit
	if random.random() < p:
		qubit = np.dot(Err2, qubit) #error on qubit 2
	else:
		qubit = qubit
	if random.random() < p:
		qubit = np.dot(Err3, qubit) #error on qubit 3
	else:
		qubit = qubit
	return qubit



def ErrDet(ErrState):

	'''detect error'''

	PiMinus = ((np.identity(8) - Det1)/2)
	PiPlus = ((np.identity(8) - Det2)/2)
	ProbErr1 = np.dot(ErrState.T,(np.dot(PiMinus, ErrState)))
	ProbErr2 = np.dot(ErrState.T,(np.dot(PiPlus, ErrState)))
	PErr = np.array([ProbErr1, ProbErr2])
	return PErr



def ErrCorr(ErrState):

	'''correct error'''

	P = ErrDet(ErrState)
	if P[0] == 1 and P[1] == 1: #error on qb2
		ErrState = np.dot(Err2, ErrState)
	elif P[0] == 1 and P[1] == 0: #err on qb1
		ErrState = np.dot(Err1, ErrState)
	elif P[0] == 0 and P[1] == 1: #err on qb3
		ErrState = np.dot(Err3, ErrState)
	elif P[0] == 0 and P[1] == 0: #no err
		ErrState = ErrState

	return ErrState

def AvgFid(QbIn, n, p):

	'''calculate average fidelity, i.e. how effective the error correcting code is, in relation to the
	probability of an error occurring'''

	tot_fidelity = 0
	for i in range(n):
		x = np.copy(QbIn)
		QbErr = Error3(QbIn, p)
		QbCorr = ErrCorr(QbErr)
		if np.array_equal(x, QbCorr):
			fidelity = 1
		else:
			fidelity = 0

		tot_fidelity += fidelity

	avg_fidelity = float(tot_fidelity)/n
	avg_fidelity = np.sqrt(avg_fidelity)

	return avg_fidelity


x_vals = np.array([]) #x axis values (probabilities)
y_vals_3_qb = np.array([]) #x axis values for 3 qubit code (avg fidelity)

n_qubits = 5000
x_steps = 101 #range of probabilities


		
for p in np.linspace(0.0, 0.9, num = x_steps):
	x_vals = np.append(x_vals, np.array([p]))

	fidelity_3 = AvgFid(EncState, n_qubits, p)
	y_vals_3_qb = np.append(y_vals_3_qb, np.array([fidelity_3]))
	print(p)


y_th_3 = np.sqrt(1 - 3*(x_vals)**2 + 2*(x_vals)**3) # teoretical fidelity for 3 qubits
y_th_1 = np.sqrt(1 - x_vals) # teoretical fidelity for 1 qubit

'''plot all of the things'''

pyplot.figure(figsize = (10,6), dpi = 100)

pyplot.plot(x_vals, y_th_1, linewidth = 0.5, ls = ':', color = 'k', label='theory 1 qubit')
pyplot.plot(x_vals, y_th_3, linewidth = 0.5, ls = '--', color = 'k', label='theory 3 qubit')
pyplot.plot(x_vals, y_vals_3_qb, linewidth = 0.5, color = 'C1', label='3 qubit code')
pyplot.xlabel('Error Probability')
pyplot.ylabel('Average Fidelity')
pyplot.legend()
pyplot.minorticks_on()
pyplot.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on')

pyplot.tight_layout()
pyplot.savefig('qec_plot.jpg', dpi = 400)

pyplot.show()

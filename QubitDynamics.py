import numpy as np
import matplotlib.pyplot as pyplot
import random


'''define initial parameters'''
gamma = 0.1 #rate at which probablity transfers
omega = 1.0 #Rabi frequency
gamma_wfmc = gamma*(3.0/4.0) #frequency for monte carlo wavefn method
T = 60.0  #time in s
N_slice = 1000 #number of slices in which T is divided
trials = 1000 #10000 gives residuals  of 0.006 for method 2




def prob_1(t):
    '''fn to calculate probablity of being in state 1 without detuning'''
    return -0.5*np.cos(omega*t)+0.5




ts = np.linspace(0.0, T, N_slice) #values on horizontal axis

dt = T/N_slice #timestep

p_nodec_inslice = 1-(gamma_wfmc)*dt #probability of not decayng to state 0 in slice dt (use gamma_wfmc)

y_ax = np.zeros(int(N_slice)) #array to hold y axis values for graph of one particle

total_ys = np.zeros(int(N_slice)) #initialise array for avg value of ys



    



def prob_1_dec(y_array): #can use any 1D array as argument since all of its values are replaced
    '''fn to calculate probability of being in state 1 with random decays, input array outputs array'''
    
    i = 0.0 #initialise variable i

    for j in range(0, int(N_slice)):
        i += dt #next timestep
        r = random.uniform(0, 1) #random number between 0.0 and 1.0
        if p_nodec_inslice < r:
            y_array[j] = 0.0 #drop to ground state
            i=0.0 #this resets prob_1 to 0 in next step
        
        elif p_nodec_inslice >= r:
            y_array[j] = prob_1(i) #prob fn continues normally (no decay)

    return y_array






def prob_1_dec_prime(y_array):
    '''fn to calculate prob of decay in time T, instead of at each step'''
    j = 0
    t_passed = 0.0
    while t_passed < T - dt: #why need dt? floatnig point precision (t_passed gets to 159.999... and gets stuck in this while loop)
        
        p_nodec = random.uniform(0, 1) #random probability of decaying
        t_nodec = (-np.log(p_nodec))/(gamma_wfmc) #particle decays after time t_nodec, gamma_wfmc
        
        i = 0.0
        while i < t_nodec and j < N_slice:
            '''plot function without decay up to time t_nodec, then reset it'''
            y_array[j] = prob_1(i)
            i += dt
            t_passed += dt
            j += 1           
    
    return y_array






def avg_prob(avg_y_vals, any_array):
    for k in range(trials):
        '''run multiple trials of probablity_1 with decay, method 1'''
        ys = prob_1_dec(any_array)
        for i in range(int (N_slice)):
            '''add y values for each dt and store them in an 1D array'''
            avg_y_vals[i] += ys[i]

    for i in range(int(N_slice)):
        '''get avg value of y for each dt and store in same 1D array'''
        avg_y_vals[i] /= trials

    return avg_y_vals





def avg_prob_prime(avg_y_vals, any_array):
    for k in range(trials):
        '''run multiple trials of probablity_1 with decay, method 1'''
        ys = prob_1_dec_prime(any_array)
        for i in range(int (N_slice)):
            '''add y values for each dt and store them in an 1D array'''
            avg_y_vals[i] += ys[i]

    for i in range(int(N_slice)):
        '''get avg value of y for each dt and store in same 1D array'''
        avg_y_vals[i] /= trials

    return avg_y_vals





def theory_avg(t):
    '''calculate theoretical prediction for avg probability of many particles'''
    A = 1.0 / (1.0 + (gamma**2)/(2.0*(omega**2)))
    C = ((omega**2)-((gamma**2)/16.0))**0.5
    B = (3.0*A*gamma) / (4.0*C)
    rho_diff = np.exp((-3.0/4.0)*gamma*t)*((A*np.cos(C*t))+(B*np.sin(C*t))) + (1.0 / (1 + 2.0*((omega**2)/(gamma**2))))
    return (1 - rho_diff)/2.0 #probability




def residuals(fn1, fn2):
    '''calculate residuals between two functions'''
    res = np.zeros(int(N_slice)) #array to hold y values
    for i in range(int(N_slice)):
        res[i] = (fn1[i] - fn2[i])
    return res





'''plot all of the things'''

pyplot.figure(figsize=(10,6), dpi=100)


# plot with method 1

pyplot.subplot(411)
pyplot.title('WFMC simulation of decoherent Rabi oscillations')
pyplot.plot(ts, prob_1_dec(y_ax)*0.5-0.6, color = 'black', label = 'single particle probability') #plot of one particle method 1
pyplot.plot(ts, avg_prob(total_ys, y_ax), color = 'red', label = 'WFMC simulation method 1') #plot of avg of many particles method 1
pyplot.plot(ts, theory_avg(ts), color = 'black', ls = '--', label = 'teoretical prediction') #plot of theoretical avg
pyplot.ylabel('P1')

pyplot.legend()

pyplot.subplot(412)
pyplot.plot(ts, residuals(avg_prob(total_ys, y_ax), theory_avg(ts))) #plot of residuals method 1
pyplot.ylabel('residuals')

pyplot.subplot2grid((4, 4), (0, 0), colspan=4, rowspan=3)
pyplot.plot(ts, prob_1_dec_prime(total_ys)*0.2-0.3, color = 'black', linewidth = 1, label = 'single particle probability') #plot of one particle with method 2
pyplot.plot(ts, avg_prob_prime(total_ys, y_ax), color = 'red', linewidth = 1, label = 'WFMC simulation') #plot of avg of many particles method 2
pyplot.plot(ts, theory_avg(ts), ls = '--', color = 'black',  linewidth = 1,label = 'theoretical prediction') #plot of theoretical avg
pyplot.ylabel('P1')
pyplot.minorticks_on()
pyplot.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on')



# plot with method 2

pyplot.subplot2grid((5, 5), (0, 0), colspan=5, rowspan=1)
pyplot.plot(ts, prob_1_dec_prime(total_ys), color = 'black', linewidth = 1, label = 'single particle probability') #plot of one particle with method 2
pyplot.ylabel('Prob(1)')
pyplot.minorticks_on()
pyplot.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='off')

pyplot.subplot2grid((5, 5), (1, 0), colspan=5, rowspan=3)
pyplot.plot(ts, avg_prob_prime(total_ys, y_ax), color = 'red', linewidth = 1, label = 'WFMC simulation') #plot of avg of many particles method 2
pyplot.plot(ts, theory_avg(ts), ls = '--', color = 'black',  linewidth = 1,label = 'theoretical prediction') #plot of theoretical avg
pyplot.ylabel('Excited State Population')
pyplot.minorticks_on()
pyplot.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='off')

pyplot.legend()

pyplot.subplot2grid((5, 5), (4, 0), colspan=5, rowspan=1)
pyplot.plot(ts, residuals(avg_prob_prime(total_ys, y_ax), theory_avg(ts)), color = 'black', linewidth = 1) #plot of residuals method 2
pyplot.xlabel('Time (Rabi Period)')
pyplot.ylabel('Residuals')
pyplot.minorticks_on()
pyplot.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on')

pyplot.tight_layout()
pyplot.subplots_adjust(hspace=0.0)
pyplot.savefig('qubit_plot.jpg', dpi=400)


pyplot.show()


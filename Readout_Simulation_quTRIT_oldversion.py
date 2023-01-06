import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

from Readout_Simulation_qubit import *
#QUBIT READOUT

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.rc('text', usetex=True)

# QUBIT READOUT

ROTATING_FRAME = True
HAMILTONIAN = 'normal'  # 'normal'

wr = 6.17 * 2 * np.pi  # resonator

wq = 3.56 * 2 * np.pi  # qubit frequency

alpha = -240e-3 * 2 * np.pi
g = 69.3e-3 * 2 * np.pi
delta = (wq - wr)
# coupling strength, Grad/s
chi = g ** 2 * alpha / (delta * (alpha + delta)) #double check

NUMBER_TRAJECTORIES = 50

print(chi)
phi = 0  # np.pi / 2  # phase for resonator drive
T = 1  # readout duration in s

kappa = -1 / 3 * chi / (2 * np.pi)  # 5e-7 * 2 * np.pi
gamma = 1 / (46e3)  # Hz  # rate of energy relaxation, Grad/s

print(chi / kappa)
# Resonator operators
Nr = 30  # Hilbert-space cutoff -> 30
Nq = 3
a = qt.tensor(qt.destroy(Nr), qt.qeye(Nq))  # annihilation
n = a.dag() * a  # photo n number
x = qt.tensor(1/np.sqrt(2) * qt.position(Nr), qt.qeye(Nq))  # x = 1/sqrt(2) * (a + a.dag())
p = qt.tensor(qt.momentum(Nr), qt.qeye(Nq))  # p = -1j/sqrt(2) * (a - a.dag())

# Qubit operators
a_qubit = qt.tensor(qt.qeye(Nr), qt.desroy(Nq))
n_qubit = a_qubit.dag() * a

# identity matrix
eye = qt.tensor(qt.qeye(Nr), qt.qeye(Nq))

# inital Hamiltonian for resonator and qubit are the same
H0R = 0 * wr * (a.dag() * a)
H0q = 0 * wq * (a_qubit.dag() * a_qubit)

# drive term of Hamiltonian is independet of choice of Ham -> wd is adapted in the if clause and access in that
# shouldn't make any difference
wd = wr - wr  # - chi #+ 1e9  # drive frequency


def H_drive1_coeff(t, args):
    return np.exp(1j * (wd * t) + phi)


def H_drive2_coeff(t, args):
    return np.exp(1j * (wd * t) + phi).conj()


u = 0.06  # 10 photons

H_drive1 = 1 / 2 * u * a
H_drive2 = 1 / 2 * u * a.dag()

if HAMILTONIAN == 'normal':
    # apply U = exp(i wr a.dag()a - i wq /2 sigma_z )
    # resonator and qubit term of Ham removed

    '''w_rotating = wr

    wr = 0  # wr - w_rotating
    wq = 0  # wq - w_rotating
    wd = 0  # wd - w_rotating  # resonator is driven at resonator freq -> wd = 0'''

    # timed dependet coupling hamilotonian after transformation in rotating frame
    H_coupling1 = g * a * a_qubit.dag()  # -> rotate H_coupling as well # RWA not applied
    H_coupling2 = g * a * a_qubit
    H_coupling3 = g * a.dag() * a_qubit.dag()
    H_coupling4 = g * a.dag() * a_qubit


    def H_coup1_coeff(t, args):
        return np.exp(-1j * (wr * t - wq * t))


    def H_coup2_coeff(t, args):
        return np.exp(-1j * (wr * t + wq * t))


    def H_coup3_coeff(t, args):
        return np.exp(1j * (wr * t + wq * t))


    def H_coup4_coeff(t, args):
        return np.exp(1j * (wr * t - wq * t))


    # total Hamiltonian
    H = [H0R, H0q,
         [H_coupling1, H_coup1_coeff], [H_coupling2, H_coup2_coeff], [H_coupling3, H_coup3_coeff],
         [H_coupling3, H_coup4_coeff],
         [H_drive1, H_drive1_coeff], [H_drive2, H_drive2_coeff]]

    sampling_freq = 1

elif HAMILTONIAN == 'dispersive':

        #wrong chi -> use g^2 / delta instead?
    H_coupling = chi * n * sz / 2  # maybe put 1/2

    # H = [H_coupling, [H_drive1, H_drive1_coeff], [H_drive2, H_drive2_coeff]]
    H = [H_coupling, H_drive1 + H_drive2]

    sampling_freq = 5e-1  # 5e8 #chi in order of <10Mhz maybe use 1e7

print(f'{HAMILTONIAN} Hamiltonian')

# step resolution for time

ns = int(round(sampling_freq * T))  # number of samples in the pulse
print(f'Number of timesteps: {ns}')
ts = np.linspace(0, T, ns, endpoint=False)

# Initial state
psi0_g = qt.tensor(qt.coherent(Nr, 0), qt.basis(Nq, 0))  # cavity in coherent state, qubit in |g>
psi0_e = qt.tensor(qt.coherent(Nr, 0), qt.basis(Nq, 1))  # is psi0_e still prepared in resonator g?
psi0_f = qt.tensor(qt.coherent(Nr, 0), qt.basis(Nq, 2))

c_ops = [np.sqrt(kappa) * a]  # np.sqrt(gamma) * sm,



temp_g, result_master_g = get_template(H, psi0_g, ts, c_ops=c_ops, e_ops=[x, p])
temp_e, result_master_e = get_template(H, psi0_e, ts, c_ops=c_ops, e_ops=[x, p])
temp_f, result_master_f = get_template(H, psi0_f, ts, c_ops=c_ops, e_ops=[x, p])

theta_eg = 0.5 * (norm_funk(temp_e) ** 2 - norm_funk(temp_g) ** 2)
theta_fe = 0.5 * (norm_funk(temp_f) ** 2 - norm_funk(temp_e) ** 2)
theta_gf = 0.5 * (norm_funk(temp_g) ** 2 - norm_funk(temp_f) ** 2)

R_ge = []
R_fe = []
R_gf = []
c = 0

for n in range(NUMBER_TRAJECTORIES):

    m_g = calc_measurment_results(H, psi0_g, ts, sc_ops=[np.sqrt(kappa) * a], ntraj=1, nsubsteps=10,
                                  method='heterodyne')
    m_e = calc_measurment_results(H, psi0_e, ts, sc_ops=[np.sqrt(kappa) * a], ntraj=1, nsubsteps=10,
                                  method='heterodyne')
    m_f = calc_measurment_results(H, psi0_f, ts, sc_ops=[np.sqrt(kappa) * a], ntraj=1, nsubsteps=10,
                                  method='heterodyne')
    R_ge.append(calc_R(m_g, temp_e, temp_g))
    R_fe.append(calc_R(m_e, temp_f, temp_e))
    R_gf.append(calc_R(m_f, temp_g, temp_f))

    c += 1
    if c % 10 == 0:
        print(f'Number of success full runs: {c} / {NUMBER_TRAJECTORIES}')








# plt.plot(R_g_filled, counts_g, color='b', label=r'g')
# plt.plot(R_e_filled, counts_e, color='r', label=r'e')
n, bins, patches = plt.hist(x=R_ge, bins='auto', color='b',
                            alpha=0.7, rwidth=0.85, label='g')
n, bins, patches = plt.hist(x=R_e, bins='auto', color='r',
                            alpha=0.7, rwidth=0.85, label='e')
# plt.xlim(-0.05, 0.05)

'''mu, std = norm.fit(R_g)
p_g = norm.pdf(ts, mu, std)
plt.plot(ts, p_g, 'k', linewidth=2)

mu, std = norm.fit(R_e)
p_e = norm.pdf(ts, mu, std)
plt.plot(ts, p_e, 'k', linewidth=2)'''

plt.xlabel(r'$R_{eg}$')
plt.ylabel(r'Counts')
plt.legend()
plt.show()

plt.plot(ts, temp_e.real)
plt.plot(ts, temp_e.imag)
plt.plot(ts, temp_g.real)
plt.plot(ts, temp_g.imag)
plt.show()




















'''USE_DAMPING = False
ROTATING_FRAME = True

wr = 6.17e9  # resonator
Nr = 5  # Hilbert-space cutoff

wq = 3.56e9  # qubit frequency
Nq = 3
alpha = -0.240e9 # anharmonicity

w_rotating = wr

g = 69.3e6 # coupling strength, Grad/s


if USE_DAMPING:
    kappa = 615e6 #Hz # according to ruggeros thesis/ paper
    gamma = 1/(46e-6) #Hz  # rate of energy relaxation, Grad/s
else:
    kappa = 0.0
    gamma = 0.0

# Resonator operators
a = qt.tensor(qt.destroy(Nr), qt.qeye(Nq))  # annihilation
n = a.dag() * a  # photo n number
x = qt.tensor(qt.position(Nr), qt.qeye(Nq))  # x = 1/sqrt(2) * (a + a.dag())
p = qt.tensor(qt.momentum(Nr), qt.qeye(Nq))  # p = -1j/sqrt(2) * (a - a.dag())
# with these definitions: <x>^2 / 2 + <p>^2 / 2 = <n>

#Qubit operators
a_qubit = qt.tensor(qt.qeye(Nr), qt.desroy(Nq))
n_qubit = a_qubit.dag() * a

# identity matrix
eye = qt.tensor(qt.qeye(Nr), qt.qeye(Nq))

# Hamiltonian
if ROTATING_FRAME == True:
    wr = wr - w_rotating
    wq = wq - w_rotating
else:
    pass

H0R = wr * (a.dag() * a )
H0q = - wq * (a_qubit.dag()*a) + alpha/2 * a_qubit.dag() * a_qubit.dag() * a * a

H_coupling = g * (a.dag() * a_qubit + a * a_qubit.dag()) #RWA applied

wd = 10e9  # drive frequency
phi = np.pi / 2  # phase for resonator drive
T = 200e-6  # readout duration in s

fs = 1e9 # sampling rate

ns = int(round(fs * T)) # number of samples in the pulse
print(ns)
ts = np.linspace(0, T, ns)


def H_drive1_coeff(t, args):
    return np.exp(1j * (wd * t) + phi)

def H_drive2_coeff(t, args):
    return np.exp(1j * (wd * t) + phi).conj()

H_drive1 = 1/2 * gamma * a
H_drive2 = 1/2 * gamma * a.dag()


#total Hamiltonian
H = [H0R, H0q, H_coupling, [H_drive1, H_drive1_coeff], [H_drive2, H_drive2_coeff]]


# Initial state
psi0_g = qt.tensor(qt.coherent(Nr, 5), qt.basis(Nq, 1))  # cavity in coherent state, qubit in |g>
psi0_e = qt.tensor(qt.coherent(Nr, 5), qt.basis(Nq, 1))

if USE_DAMPING:
    c_ops = [
        np.sqrt(gamma) * a_qubit, #qubit decays with T_1
        np.sqrt(kappa) * a, #Resonator decays with kappa
    ]
else:
    c_ops = []


# Actually run the simulation, (can be) time consuming!
if USE_DAMPING:
    output = qt.mesolve(H, psi0_g, ts, c_ops=c_ops)  # solve Lindblad master equation
else:
    output = qt.sesolve(H, psi0_g, ts)  # solve Schrödinger equation


#print(output.states)

x_expect = qt.expect(x, output.states)
p_expect = qt.expect(p, output.states)


print(x_expect)
print(p_expect)

fig, axs = plt.subplots(1, 1)
axs.plot(ts,  ) #T -> transpose

axs.grid(linestyle="--")
axs.tick_params(
    direction="in", left=True, right=True, top=True, bottom=True)
axs.set_xlabel("Time [ns]")
axs.set_ylabel("Population")
#plt.legend(, loc='right')
plt.show()'''

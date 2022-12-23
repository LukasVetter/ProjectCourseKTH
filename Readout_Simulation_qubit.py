from Helper_functions import *
import time
plt.rc('text', usetex=True)
from scipy.optimize import curve_fit

# QUBIT READOUT
#PARAMETER SPEC

#plot histogram R_eg vs counts
PLOT_HIST = False
#plot templates of e and g vs time
PLOT_TEMP = False
#plot amp_phase_sep vs driving frequency
PLOT_AMP_PHASE_Sep = True
#sweep driving frequency
FREQ_SWEEP = True
NUMBER_TRAJECTORIES = 0

t_start = time.time()
HAMILTONIAN = 'dispersive'# 'normal'

wr = 6.17 * 2 * np.pi  # resonator
N = 30  # Hilbert-space cutoff -> 30
wq = 3.56 * 2 * np.pi  # qubit frequency

alpha = -240e-3 * 2 * np.pi

delta = (wq - wr)
# coupling strength, Grad/s
chi = -155e-6 * 2 * np.pi  #2 * g ** 2 * alpha / (delta * (alpha + delta))
#chi = g**2 / delta
kappa = 615e-6 * 2 * np.pi  #-4*chi #/2 #/ (2*np.pi) #
gamma = 1 / (46e3)  # Hz  # rate of energy relaxation, Grad/s
g = np.sqrt(np.abs(chi * delta * (delta+alpha) / alpha))  #69.3e-3 * 2 * np.pi

#Rotation for resonator
w_rot_r = wr

#Rotating for qubit
w_rot_q = wq

T = 10 * 2 / kappa  #1e3  # readout duration in s


sweep = 3e-3 * 2 * np.pi

if FREQ_SWEEP == True:

    wds = np.linspace(wr-sweep, wr+sweep, 50, endpoint=False)
else:
    #only one frequency
    wd = wr #+ 3e-3 * 2 * np.pi / 2
    wds = [wd]  #- chi #+ 1e9  # drive frequency


if HAMILTONIAN == 'dispersive':
    sampling_freq = 5e-1 # 5e-1  # 5e8 #chi in order of <10Mhz maybe use 1e(7-9)

if HAMILTONIAN == 'normal':
    sampling_freq = 5

#option to drive at several driving frequencies




'----------------------------------------------------------------------------'
'Operator + Hamiltonian'
#Template + determining R
'----------------------------------------------------------------------------'

#phase of drive
phi = 0  #np.pi / 2  # phase for resonator drive

print(f' $\chi / \kappa$ {chi / kappa}')

# Resonator operators
a = qt.tensor(qt.destroy(N), qt.qeye(2))  # annihilation
n = a.dag() * a  # photo n number
x = qt.tensor(1/np.sqrt(2) * qt.position(N), qt.qeye(2))  # x = 1/sqrt(2) * (a + a.dag())
p = qt.tensor(1/np.sqrt(2) * qt.momentum(N), qt.qeye(2))  # p = -1j/sqrt(2) * (a - a.dag())


# Qubit operators (Pauli matrices)
sx = qt.tensor(qt.qeye(N), qt.sigmax())  # sp + sm
sy = qt.tensor(qt.qeye(N), qt.sigmay())  # 1j * (sm - sp)
sz = qt.tensor(qt.qeye(N), qt.sigmaz())
sp = qt.tensor(qt.qeye(N), qt.sigmap())  # (sx + 1j * sy) / 2 == sm.dag()
sm = qt.tensor(qt.qeye(N), qt.sigmam())  # (sx - 1j * sy) / 2 == sp.dag()

# identity matrix
eye = qt.tensor(qt.qeye(N), qt.qeye(2))

print(f'{HAMILTONIAN} Hamiltonian')

# step resolution for time

ns = int(round(sampling_freq * T))  # number of samples in the pulse
print(f'Number of timesteps: {ns}')
ts = np.linspace(0, T, ns, endpoint=False)

# Initial state
psi0_ground = qt.tensor(qt.coherent(N, 0), qt.basis(2, 0))  # cavity in coherent state, qubit in |g>
psi0_excited = qt.tensor(qt.coherent(N, 0), qt.basis(2, 1))  # is psi0_excited still prepared in resonator ground?

c_ops = [np.sqrt(kappa) * a]  # np.sqrt(gamma) * sm,


Amp_gs = []
Amp_es = []
phi_gs = []
phi_es = []
Ss = []

x_steady_gs = []
p_steady_gs = []
x_steady_es = []
p_steady_es = []


for wd in wds:
    print(f'Finished one loop: {list(wds).index(wd)}/{len(wds)}')
    #inital Hamiltonian for resonator and qubit are the same
    H0R = (wr - w_rot_r) * (a.dag() * a)
    H0q = - (wq - w_rot_q) * sz / 2

    #drive term of Hamiltonian is independet of choice of Ham -> wd is adapted in the if clause and access in that
    #shouldn't make any difference

    #for wd in wds:

    def H_drive1_coeff(t, args):
        return np.exp(1j * ((wd-w_rot_r) * t) + phi)

    def H_drive2_coeff(t, args):
        return np.exp(1j * ((wd-w_rot_r) * t) + phi).conj()

    photon_number = 10
    u_g = np.sqrt(photon_number) / np.abs(tf(w0=wr-chi, wd=wr, kappa=kappa))
    u_e = np.sqrt(photon_number) / np.abs(tf(w0=wr+chi, wd=wr, kappa=kappa))

    H_drive1_g = u_g * a / 2
    H_drive2_g = u_g * a.dag() / 2
    H_drive1_e = u_e * a / 2
    H_drive2_e = u_e * a.dag() / 2

    if HAMILTONIAN == 'normal':
        # apply U = exp(i wr a.dag()a - i wq /2 sigma_z )
        # resonator and qubit term of Ham removed

        #timed dependet coupling hamilotonian after transformation in rotating frame
        H_coupling1 = g * a * sp  # -> rotate H_coupling as well # RWA not applied
        H_coupling2 = g * a * sm
        H_coupling3 = g * a.dag() * sp
        #print('RWA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        H_coupling4 = g * a.dag() * sm

        def H_coup1_coeff(t, args):
            return np.exp(-1j*(w_rot_r*t + w_rot_q*t))

        def H_coup2_coeff(t, args):
            return np.exp(-1j*(w_rot_r*t - w_rot_q*t))

        def H_coup3_coeff(t, args):
            return np.exp(1j*(w_rot_r*t - w_rot_q*t))

        def H_coup4_coeff(t, args):
            return np.exp(1j*(w_rot_r*t + w_rot_q*t))


        # total Hamiltonian
        H_g = [H0R, H0q,
             [H_coupling1, H_coup1_coeff], [H_coupling2, H_coup2_coeff], [H_coupling3, H_coup3_coeff], [H_coupling3, H_coup4_coeff],
             [H_drive1_g, H_drive1_coeff], [H_drive2_g, H_drive2_coeff]]

        H_e = [H0R, H0q,
             [H_coupling1, H_coup1_coeff], [H_coupling2, H_coup2_coeff], [H_coupling3, H_coup3_coeff], [H_coupling3, H_coup4_coeff],
             [H_drive1_e, H_drive1_coeff], [H_drive2_e, H_drive2_coeff]]


    elif HAMILTONIAN == 'dispersive':

        H_coupling = chi * n * sz  #* 2 #maybe put 1/2

        #H = [H_coupling, [H_drive1, H_drive1_coeff], [H_drive2, H_drive2_coeff]]
        H_g = [H_coupling, [H_drive1_g, H_drive1_coeff], [H_drive2_g, H_drive2_coeff]]
        H_e = [H_coupling, [H_drive1_e, H_drive1_coeff], [H_drive2_e, H_drive2_coeff]]




    temp_ground, result_master_ground = get_template(H_g, psi0_ground, ts, kappa, c_ops=c_ops, e_ops=[x, p])
    temp_excited, result_master_excited = get_template(H_e, psi0_excited, ts, kappa, c_ops=c_ops, e_ops=[x, p])

    if wd-wr < 0.01:
        temp_ground_safe = temp_ground
        temp_excited_safe = temp_excited

    theta_eg = 0.5 * (norm_funk(temp_excited)**2 - norm_funk(temp_ground)**2)

    R_g = []
    R_e = []
    c = 0

    if NUMBER_TRAJECTORIES == 0:
        pass
    else:
        for n in range(NUMBER_TRAJECTORIES):
            m_g = calc_measurment_results(H_g, psi0_ground, ts, sc_ops=[np.sqrt(kappa) * a], ntraj=1, nsubsteps=10, method='heterodyne')
            m_e = calc_measurment_results(H_e, psi0_excited, ts, sc_ops=[np.sqrt(kappa) * a], ntraj=1, nsubsteps=10, method='heterodyne')

            R_g.append(calc_R(m_g, temp_excited, temp_ground))
            R_e.append(calc_R(m_e, temp_excited, temp_ground))

            c += 1
            if c % 10 == 0:
                print(f'Number of success full runs: {c} / {NUMBER_TRAJECTORIES}')





    if PLOT_AMP_PHASE_Sep:
        x_steady_g = result_master_ground.expect[0][-1]
        p_steady_g = result_master_ground.expect[1][-1]

        x_steady_e = result_master_excited.expect[0][-1]
        p_steady_e = result_master_excited.expect[1][-1]

        x_steady_gs.append(x_steady_g)
        p_steady_gs.append(p_steady_g)
        x_steady_es.append(x_steady_e)
        p_steady_es.append(p_steady_e)

        S = np.abs(temp_ground[-1]-temp_excited[-1])
        Ss.append(S)

        #A_g = np.sqrt(np.abs(x_steady_g)**2 + np.abs(p_steady_g)**2)
        #phi_g = np.arctan2(x_steady_g, p_steady_g)
        A_g = np.abs(temp_ground[-1])
        phi_g = np.angle(temp_ground[-1])
        Amp_gs.append(A_g)
        phi_gs.append(phi_g)

        #
        #A_e = np.sqrt(x_steady_e ** 2 + p_steady_e ** 2)
        A_e = np.abs(temp_excited[-1])
        #phi_e = np.arctan2(x_steady_e, p_steady_e)
        phi_e = np.angle(temp_excited[-1])
        Amp_es.append(A_e)
        phi_es.append(phi_e)

if FREQ_SWEEP:
    phi_es_unwrap = np.unwrap(phi_es, period=2*np.pi)+T*wds-(wds[0])*T
    phi_gs_unwrap = np.unwrap(phi_gs, period=2*np.pi)+T*wds-(wds[0])*T
'-----------------------------------------------------------------------------------'
#Evaluation
'-----------------------------------------------------------------------------------'

#which plots are created is determined by the bool variables at the beginning of the script


'''Amp_gs = np.sqrt(np.abs(x_steady_gs) ** 2 + np.abs(p_steady_gs) ** 2)
phi_gs = np.arctan2(x_steady_gs, p_steady_gs)
Amp_es = np.sqrt(np.abs(x_steady_es) ** 2 + np.abs(p_steady_es) ** 2)
phi_es = np.arctan2(x_steady_es, p_steady_es)'''

if PLOT_AMP_PHASE_Sep:


    #print(time.time()-t_start)
    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot((wds-wr)*1e3 / (2*np.pi), Amp_gs, label='ground')
    ax[0].plot((wds-wr)*1e3 / (2*np.pi), Amp_es, label='excited')
    ax[1].plot((wds-wr)*1e3 / (2*np.pi), phi_gs_unwrap, label='ground')
    ax[1].plot((wds-wr)*1e3 / (2*np.pi), phi_es_unwrap, label='excited')
    ax[2].plot((wds-wr)*1e3 / (2*np.pi), Ss)
    ax[2].set_xlabel('Driving frequency $w_d$ [MHz]')
    ax[0].set_ylabel('Amplitude')
    ax[1].set_ylabel('Phase')
    ax[2].set_ylabel('Separation')
    plt.show()


if PLOT_HIST:
    fig, ax = plt.subplots(1, 1)

    n_g, bins_g, patches_g = plt.hist(x=R_g, bins=30, color='orange', density=True,
                                alpha=0.3, label='ground', histtype='stepfilled')
    popt_g, pcov = curve_fit(gaussian, bins_g, np.append(n_g, 0), p0=[-8, 5], bounds=[(-10, 0), (-5, 15)])
    plt.plot(bins_g, gaussian(bins_g, *popt_g), color='orange')


    n_e, bins_e, patches_e = plt.hist(x=R_e, bins=30, color='lightskyblue', density=True,
                                alpha=0.6, label='excited', histtype='stepfilled')
    popt_e, pcov = curve_fit(gaussian, bins_e, np.append(n_e, 0))#, p0=[8, 5], bounds=[(5, 0), (10, 15)])
    plt.plot(bins_e, gaussian(bins_e, *popt_e), color='lightskyblue')

    ax.set_xlabel(r'$R_{eg}$')
    ax.set_ylabel(r'Counts normalized')
    ax.text(0.05, 1,
            f'Driving Strength $A_g=${round(u_g,4)}, $A_e=${round(u_e,4)} \n $\chi / \kappa=$ {round(chi/kappa,2)} \n '
            f'$\chi$={round(chi * 1e3,2)}[MHz] \n'
            f'Gr:$\mu_g=${round(popt_g[0], 2)},$\sigma_g=${round(popt_g[1], 2)}'
            f'Ex:$\mu_e=${round(popt_e[0], 2)},$\sigma_e=${round(popt_e[1],2)}',
            transform=ax.transAxes)

    ax.text(0.8, 1.05, f'Ham: {HAMILTONIAN} \n T = {round(T / 1e3,2)}[$\mu s$]', transform=ax.transAxes)
    plt.legend()
    plt.show()

    print(f'Readout fidelity: {fidelity(popt_g[0], popt_g[1], popt_e[0], popt_e[1])}')

#plot templates
if PLOT_TEMP:
    plt.plot(ts, temp_excited_safe.real, label='Excited Real')
    plt.plot(ts, temp_excited_safe.imag, label='Excited Imag')
    plt.plot(ts, temp_ground_safe.real, label='Ground Real')
    plt.plot(ts, temp_ground_safe.imag, label='Ground Imag')
    plt.xlabel(f'Time t [ns]')
    plt.ylabel('template')
    plt.show()





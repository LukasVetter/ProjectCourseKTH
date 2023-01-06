import matplotlib.pyplot as plt

from Helper_functions import *
import time

plt.rc('text', usetex=True)
from scipy.optimize import curve_fit

# QUBIT READOUT
# PARAMETER SPEC

# plot histogram R_eg vs counts
PLOT_HIST = False
# Colormap to compare R_eg and R_ef
COLORMAP = False
# plot templates of e and g vs time
PLOT_TEMP = False
# plot amp_phase_sep vs driving frequency
PLOT_AMP_PHASE_Sep = False
# sweep driving frequency
FREQ_SWEEP = False
NUMBER_TRAJECTORIES = 500

USE_Orthogonal_temps = False
d3_plot = True
d2_plot_parameter_variation = False
#d2_plot = False

t_start = time.time()
HAMILTONIAN = 'dispersive'  # 'normal'

wr = 6.17 * 2 * np.pi  # resonator
Nr = 30  # Hilbert-space cutoff -> 30
Nq = 3
wq = 3.56 * 2 * np.pi  # qubit frequency

alpha = -240e-3 * 2 * np.pi

delta = (wq - wr)
# coupling strength, Grad/s
chi = -155e-6 * 2 * np.pi  # 2 * g ** 2 * alpha / (delta * (alpha + delta))
# chi = g**2 / delta
chi_0 = -chi
chi_1 = chi
kappa = 615e-6 * 2 * np.pi  # -4*chi #/2 #/ (2*np.pi) #
gamma = 1 / (46e3)  # Hz  # rate of energy relaxation, Grad/s
g = np.sqrt(np.abs(chi * delta * (delta + alpha) / (alpha)))  # 69.3e-3 * 2 * np.pi
# -> is this formula really true??? -> I should clarify this
# chi_2 = g ** 2 * (2 / (delta+alpha) - 3/(delta+2*alpha))
chi_2 = g ** 2 * 3 * alpha / ((alpha + delta) * (2 * alpha + delta))

Lambda_1 = g ** 2 / delta
Lambda_2 = g ** 2 * 2 / (delta + alpha)

print(f'$\chi_2=${chi_2 / (2 * np.pi)}')
print(f'$\chi_1=${chi_1 / (2 * np.pi)}')
# Rotation for resonator


# Rotating for qubit
w_rot_q = wq + Lambda_1

T = 5 * 2 / kappa  # 1e3  # readout duration in s

sweep = 3e-3 * 2 * np.pi

if FREQ_SWEEP == True:

    wds = np.linspace(wr - 2e-3 * 2 * np.pi, wr + 2e-3 * 2 * np.pi, 50, endpoint=False)
else:
    # only one frequency
    wd = wr #- 8e-5 * 2 * np.pi
    wds = [wd]  # - chi #+ 1e9  # drive frequency

if HAMILTONIAN == 'dispersive':
    sampling_freq = 5e-1 #* 10 # 5e-1  # 5e8 #chi in order of <10Mhz maybe use 1e(7-9)

if HAMILTONIAN == 'normal':
    sampling_freq = 5

# option to drive at several driving frequencies


'----------------------------------------------------------------------------'
'Operator + Hamiltonian'
# Template + determining R
'----------------------------------------------------------------------------'

# phase of drive
phi = 0  # np.pi / 2  # phase for resonator drive

print(f' $\chi / \kappa$ {chi / kappa}')

# Resonator operators

a = qt.tensor(qt.destroy(Nr), qt.qeye(Nq))  # annihilation
n = a.dag() * a  # photo n number
x = qt.tensor(1 / np.sqrt(2) * qt.position(Nr), qt.qeye(Nq))  # x = 1/sqrt(2) * (a + a.dag())
p = qt.tensor(1 / np.sqrt(2) * qt.momentum(Nr), qt.qeye(Nq))  # p = -1j/sqrt(2) * (a - a.dag())

# Qubit operators
a_qubit = qt.tensor(qt.qeye(Nr), qt.destroy(Nq))
n_qubit = a_qubit.dag() * a
g_op = qt.tensor(qt.qeye(Nr), qt.basis(Nq, 0) * qt.basis(Nq, 0).trans())
e_op = qt.tensor(qt.qeye(Nr), qt.basis(Nq, 1) * qt.basis(Nq, 1).trans())
f_op = qt.tensor(qt.qeye(Nr), qt.basis(Nq, 2) * qt.basis(Nq, 2).trans())

# identity matrix
eye = qt.tensor(qt.qeye(Nr), qt.qeye(Nq))

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

Amp_gs = []
Amp_es = []
Amp_fs = []
phi_gs = []
phi_es = []
phi_fs = []
S_egs = []
S_efs = []
S_gfs = []

x_steady_gs = []
p_steady_gs = []
x_steady_es = []
p_steady_es = []

if d2_plot_parameter_variation:
    dist_ge_s = []
    dist_ef_s = []
    dist_gf_s = []

#forloop to sweep over different driving frequencies if FREQ_SWEEP = True, otherwise simply one element in wds
for wd in wds:
    w_rot_r = wr
    print(f'Finished driving frequency loop: {list(wds).index(wd)}/{len(wds)}')
    # inital Hamiltonian for resonator and qubit are the same
    H0R = (wr - w_rot_r) * (a.dag() * a)


    # H0q = (wq - w_rot_q) * a_qubit.dag() * a_qubit

    # drive term of Hamiltonian is independet of choice of Ham -> wd is adapted in the if clause and access in that
    # shouldn't make any difference



    def H_drive1_coeff(t, args):
        return np.exp(1j * ((wd - w_rot_r) * t) + phi)


    def H_drive2_coeff(t, args):
        return np.exp(1j * ((wd - w_rot_r) * t) + phi).conj()


    photon_number = 10

    u_g = np.sqrt(photon_number) / np.abs(tf(w0=wr + chi_0, wd=wr, kappa=kappa))
    u_e = np.sqrt(photon_number) / np.abs(tf(w0=wr + chi_1, wd=wr, kappa=kappa))
    u_f = np.sqrt(photon_number) / np.abs(tf(w0=wr + chi_2, wd=wr, kappa=kappa))  # what is resonance of f=?
    u = u_g
    H_drive1_g = u * a / 2
    H_drive2_g = u * a.dag() / 2
    H_drive1_e = u * a / 2
    H_drive2_e = u * a.dag() / 2
    H_drive1_f = u * a / 2
    H_drive2_f = u * a.dag() / 2

    if HAMILTONIAN == 'normal':
        # apply U = exp(i wr a.dag()a - i wq /2 sigma_z )
        # resonator and qubit term of Ham removed

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
        H_g = [H0R, H0q,
               [H_coupling1, H_coup1_coeff], [H_coupling2, H_coup2_coeff], [H_coupling3, H_coup3_coeff],
               [H_coupling3, H_coup4_coeff],
               [H_drive1_g, H_drive1_coeff], [H_drive2_g, H_drive2_coeff]]

        H_e = [H0R, H0q,
               [H_coupling1, H_coup1_coeff], [H_coupling2, H_coup2_coeff], [H_coupling3, H_coup3_coeff],
               [H_coupling3, H_coup4_coeff],
               [H_drive1_e, H_drive1_coeff], [H_drive2_e, H_drive2_coeff]]

        H_f = [H0R, H0q,
               [H_coupling1, H_coup1_coeff], [H_coupling2, H_coup2_coeff], [H_coupling3, H_coup3_coeff],
               [H_coupling3, H_coup4_coeff],
               [H_drive1_f, H_drive1_coeff], [H_drive2_f, H_drive2_coeff]]


    elif HAMILTONIAN == 'dispersive':

        #qubit Hamiltonian is set to 0 because I can use different frequencies in the transformation for the qubit
        # corresponds to U = exp(-i \sum wq_n n|n><n|), where wq_n is the qubit frequency of each level
        #even if I use the same driving freq for every level I get the same plot with the weird oscillation

        H0q = (wq + Lambda_1 - w_rot_q) * e_op * 0 + (2 * (wq - w_rot_q) + alpha + Lambda_2) * f_op * 0

        H_coupling = a.dag() * a * (chi_0 * g_op + chi_1 * e_op + chi_2 * f_op)

        # H = [H_coupling, [H_drive1, H_drive1_coeff], [H_drive2, H_drive2_coeff]]
        H_g = [H0q, H0R, H_coupling, [H_drive1_g, H_drive1_coeff], [H_drive2_g, H_drive2_coeff]]
        H_e = [H0q, H0R, H_coupling, [H_drive1_e, H_drive1_coeff], [H_drive2_e, H_drive2_coeff]]
        H_f = [H0q, H0R, H_coupling, [H_drive1_f, H_drive1_coeff], [H_drive2_f, H_drive2_coeff]]

    temp_g, result_master_g = get_template(H_g, psi0_g, ts, kappa, c_ops=c_ops, e_ops=[x, p])
    temp_e, result_master_e = get_template(H_e, psi0_e, ts, kappa, c_ops=c_ops, e_ops=[x, p])
    temp_f, result_master_f = get_template(H_f, psi0_f, ts, kappa, c_ops=c_ops, e_ops=[x, p])

    if USE_Orthogonal_temps:
        temp_g = temp_g
        temp_e = temp_e - scalar_product(temp_g, temp_e) / scalar_product(temp_g, temp_g) * temp_g
        temp_f = temp_f - scalar_product(temp_g, temp_f) / scalar_product(temp_g, temp_g) * temp_g \
                 - scalar_product(temp_e, temp_f) / scalar_product(temp_e, temp_e) * temp_e

    if wd - wr < 0.01:
        temp_g_safe = temp_g
        temp_e_safe = temp_e
        temp_f_safe = temp_f

    theta_eg = 0.5 * (norm_funk(temp_e) ** 2 - norm_funk(temp_g) ** 2)
    theta_fe = 0.5 * (norm_funk(temp_f) ** 2 - norm_funk(temp_e) ** 2)
    theta_gf = 0.5 * (norm_funk(temp_g) ** 2 - norm_funk(temp_f) ** 2)

    R_eg_g = []
    R_fe_g = []
    R_gf_g = []

    R_eg_e = []
    R_fe_e = []
    R_gf_e = []

    R_eg_f = []
    R_fe_f = []
    R_gf_f = []

    c = 0

    if NUMBER_TRAJECTORIES == 0:
        pass
    else:
        for n in range(NUMBER_TRAJECTORIES):
            # TODO: figure out if in all three measurements the driving strength of the Hamiltionian is the same
            m_g = calc_measurment_results(H_g, psi0_g, ts, sc_ops=[np.sqrt(kappa) * a], ntraj=1, nsubsteps=10,
                                          method='heterodyne')
            m_e = calc_measurment_results(H_e, psi0_e, ts, sc_ops=[np.sqrt(kappa) * a], ntraj=1, nsubsteps=10,
                                          method='heterodyne')
            m_f = calc_measurment_results(H_f, psi0_f, ts, sc_ops=[np.sqrt(kappa) * a], ntraj=1, nsubsteps=10,
                                          method='heterodyne')


            R_eg_g.append(calc_R(m_g, temp_e, temp_g))
            R_eg_e.append(calc_R(m_e, temp_e, temp_g))
            R_eg_f.append(calc_R(m_f, temp_e, temp_g))

            R_fe_g.append(calc_R(m_g, temp_f, temp_e))
            R_fe_e.append(calc_R(m_e, temp_f, temp_e))
            R_fe_f.append(calc_R(m_f, temp_f, temp_e))

            R_gf_g.append(calc_R(m_g, temp_g, temp_f))
            R_gf_e.append(calc_R(m_e, temp_g, temp_f))
            R_gf_f.append(calc_R(m_f, temp_g, temp_f))

            c += 1
            if c % 10 == 0:
                print(f'Number of success full runs: {c} / {NUMBER_TRAJECTORIES}')

    if PLOT_AMP_PHASE_Sep:
        x_steady_g = result_master_g.expect[0][-1]
        p_steady_g = result_master_g.expect[1][-1]

        x_steady_e = result_master_e.expect[0][-1]
        p_steady_e = result_master_e.expect[1][-1]

        x_steady_gs.append(x_steady_g)
        p_steady_gs.append(p_steady_g)
        x_steady_es.append(x_steady_e)
        p_steady_es.append(p_steady_e)

        S_eg = np.abs(temp_g[-1] - temp_e[-1])
        S_egs.append(S_eg)
        S_ef = np.abs(temp_e[-1] - temp_f[-1])
        S_efs.append(S_ef)
        S_gf = np.abs(temp_g[-1] - temp_f[-1])
        S_gfs.append(S_gf)

        # A_g = np.sqrt(np.abs(x_steady_g)**2 + np.abs(p_steady_g)**2)
        # phi_g = np.arctan2(x_steady_g, p_steady_g)
        A_g = np.abs(temp_g[-1])
        phi_g = np.angle(temp_g[-1])
        Amp_gs.append(A_g)
        phi_gs.append(phi_g)

        #
        # A_e = np.sqrt(x_steady_e ** 2 + p_steady_e ** 2)
        A_e = np.abs(temp_e[-1])
        # phi_e = np.arctan2(x_steady_e, p_steady_e)
        phi_e = np.angle(temp_e[-1])
        Amp_es.append(A_e)
        phi_es.append(phi_e)

        A_f = np.abs(temp_f[-1])
        phi_f = np.angle(temp_f[-1])
        Amp_fs.append(A_f)
        phi_fs.append(phi_f)

    if d2_plot_parameter_variation:

        m_g_ideal = temp_g
        m_e_ideal = temp_e
        m_f_ideal = temp_f

        # add ideal R calculated from analytic solution of template
        R_eg_g.append(calc_R(m_g_ideal, temp_e, temp_g))
        R_eg_e.append(calc_R(m_e_ideal, temp_e, temp_g))
        R_eg_f.append(calc_R(m_f_ideal, temp_e, temp_g))

        R_fe_g.append(calc_R(m_g_ideal, temp_f, temp_e))
        R_fe_e.append(calc_R(m_e_ideal, temp_f, temp_e))
        R_fe_f.append(calc_R(m_f_ideal, temp_f, temp_e))

        R_gf_g.append(calc_R(m_g_ideal, temp_g, temp_f))
        R_gf_e.append(calc_R(m_e_ideal, temp_g, temp_f))
        R_gf_f.append(calc_R(m_f_ideal, temp_g, temp_f))

        ax = plt.figure().add_subplot(projection='3d')

        R_eg_ges = R_eg_g + R_eg_e + R_eg_f
        R_fe_ges = R_fe_g + R_fe_e + R_fe_f
        R_gf_ges = R_gf_g + R_gf_e + R_gf_f

        data = np.array([R_eg_ges, R_fe_ges, R_gf_ges]).transpose()
        A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  # coefficients

        # evaluate it on grid
        X, Y = np.meshgrid(np.arange(min(R_eg_ges), max(R_eg_ges)), np.arange(min(R_fe_ges), max(R_fe_ges)))
        # evaluate it on grid
        Z = C[0] * X + C[1] * Y + C[2]

        n = [C[0], C[1], -1]  # vector perpendicular to plane
        p = [0, 0, C[2]]  # stützvector

        # new basis vectors in 2d plane
        b_1 = [1, 1, C[0] + C[1]]
        b_2 = [-1 - C[1] * (C[0] + C[1]), (C[0] + C[1]) * C[0] + 1, C[1] + C[0]]

        # matrix that changes basis of linear map from 3d to 2d
        # pinv is quasi inver of matrix
        basis_change_matrix = scipy.linalg.pinv(np.array([b_1, b_2]).transpose())
        '''R_g_1s = []
        R_g_2s = []
        R_e_1s = []
        R_e_2s = []
        R_f_1s = []
        R_f_2s = []'''

        for i in range(len(R_eg_g)):
            R_g = np.matmul(basis_change_matrix, np.array([R_eg_g[i], R_fe_g[i], R_gf_g[i]]).transpose())
            R_e = np.matmul(basis_change_matrix, np.array([R_eg_e[i], R_fe_e[i], R_gf_e[i]]).transpose())
            R_f = np.matmul(basis_change_matrix, np.array([R_eg_f[i], R_fe_f[i], R_gf_f[i]]).transpose())

            dist_ge_s.append(abs(R_g[1] - R_e[1]))
            dist_gf_s.append(abs(R_g[1] - R_f[1]))
            dist_ef_s.append(abs(R_e[1] - R_f[1]))


if FREQ_SWEEP:
    try:
        phi_es_unwrap = np.unwrap(phi_es, period=2 * np.pi) + T * wds - (wds[0]) * T
        phi_gs_unwrap = np.unwrap(phi_gs, period=2 * np.pi) + T * wds - (wds[0]) * T
        phi_fs_unwrap = np.unwrap(phi_fs, period=2 * np.pi) + T * wds - (wds[0]) * T

    except:
        pass
'-----------------------------------------------------------------------------------'
# Evaluation
'-----------------------------------------------------------------------------------'

# which plots are created is determined by the bool variables at the beginning of the script


'''Amp_gs = np.sqrt(np.abs(x_steady_gs) ** 2 + np.abs(p_steady_gs) ** 2)
phi_gs = np.arctan2(x_steady_gs, p_steady_gs)
Amp_es = np.sqrt(np.abs(x_steady_es) ** 2 + np.abs(p_steady_es) ** 2)
phi_es = np.arctan2(x_steady_es, p_steady_es)'''

if PLOT_AMP_PHASE_Sep:
    # print(time.time()-t_start)
    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot((wds - wr) * 1e3 / (2 * np.pi), Amp_gs, label='$|g>$')
    ax[0].plot((wds - wr) * 1e3 / (2 * np.pi), Amp_es, label='$|e>$')
    ax[0].plot((wds - wr) * 1e3 / (2 * np.pi), Amp_fs, label='$|f>$')
    ax[1].plot((wds - wr) * 1e3 / (2 * np.pi), phi_gs_unwrap, label='$|g>$')
    ax[1].plot((wds - wr) * 1e3 / (2 * np.pi), phi_es_unwrap, label='$|e>$')
    ax[1].plot((wds - wr) * 1e3 / (2 * np.pi), phi_fs_unwrap, label='$|f>$')
    ax[2].plot((wds - wr) * 1e3 / (2 * np.pi), S_egs, color='red', label='$g-e$')
    ax[2].plot((wds - wr) * 1e3 / (2 * np.pi), S_efs, color='black', label='$e-f$')
    ax[2].plot((wds - wr) * 1e3 / (2 * np.pi), S_gfs, color='pink', label='$g-f$')
    ax[2].set_xlabel('Driving frequency $w_d$ [MHz]')
    ax[0].legend()
    # ax[1].legend()
    ax[2].legend()
    ax[0].set_ylabel('Amplitude')
    ax[1].set_ylabel('Phase')
    ax[2].set_ylabel('Separation')
    plt.show()

if PLOT_HIST:
    # plot histogram for R_eg and R_fe:
    # R_eg -> templeate references g and e where two gaussian distributions are plotted
    # (in this case prepare the system in ground and excited state)

    # plot histogram for R_eg
    fig, ax = plt.subplots(1, 1)

    n_g, bins_g, patches_g = plt.hist(x=R_eg_g, bins=30, color='orange', density=True,
                                      alpha=0.3, label='g', histtype='stepfilled')
    popt_g, pcov = curve_fit(gaussian, bins_g, np.append(n_g, 0))  # , p0=[-8, 5], bounds=[(-10, 0), (-5, 15)])
    plt.plot(bins_g, gaussian(bins_g, *popt_g), color='orange')

    n_e, bins_e, patches_e = plt.hist(x=R_eg_e, bins=30, color='lightskyblue', density=True,
                                      alpha=0.6, label='e', histtype='stepfilled')
    popt_e, pcov = curve_fit(gaussian, bins_e, np.append(n_e, 0))  # , p0=[8, 5], bounds=[(5, 0), (10, 15)])
    plt.plot(bins_e, gaussian(bins_e, *popt_e), color='lightskyblue')

    ax.set_xlabel(r'$R_{eg}$')
    ax.set_ylabel(r'Counts normalized')
    ax.text(0.05, 1,
            f'Driving Strength $A_g=${round(u_g, 4)}, $A_e=${round(u_e, 4)} \n $\chi_1 / \kappa=$ {round(chi_1 / kappa, 2)} \n '
            f'$\chi_1$={round(chi_1 * 1e3, 2)}[MHz] \n'
            f'$\chi_2$={round(chi_2 * 1e3, 2)}[MHz] \n',
            # f'Gr:$\mu_g=${round(popt_g[0], 2)},$\sigma_g=${round(popt_g[1], 2)}'
            # f'Ex:$\mu_e=${round(popt_e[0], 2)},$\sigma_e=${round(popt_e[1], 2)}',
            transform=ax.transAxes)

    ax.text(0.8, 1.05, f'Ham: {HAMILTONIAN} \n T = {round(T / 1e3, 2)}[$\mu s$]', transform=ax.transAxes)
    plt.legend()
    plt.show()
    print(f'Readout fidelity ge: {fidelity(popt_g[0], popt_g[1], popt_e[0], popt_e[1])}')
    # histogram for R_fe
    fig_2, ax_2 = plt.subplots(1, 1)

    n_f, bins_f, patches_f = plt.hist(x=R_fe_f, bins=30, color='green', density=True,
                                      alpha=0.3, label='f', histtype='stepfilled')
    popt_f, pcov = curve_fit(gaussian, bins_f, np.append(n_f, 0))  # , p0=[-8, 5], bounds=[(-10, 0), (-5, 15)])
    plt.plot(bins_f, gaussian(bins_f, *popt_f), color='green')

    n_e, bins_e, patches_e = plt.hist(x=R_fe_e, bins=30, color='lightskyblue', density=True,
                                      alpha=0.6, label='e', histtype='stepfilled')
    popt_e, pcov = curve_fit(gaussian, bins_e, np.append(n_e, 0))  # , p0=[8, 5], bounds=[(5, 0), (10, 15)])
    plt.plot(bins_e, gaussian(bins_e, *popt_e), color='lightskyblue')

    ax_2.set_xlabel(r'$R_{fe}$')
    ax_2.set_ylabel(r'Counts normalized')
    ax_2.text(0.05, 1,
              f'Driving Strength $A_f=${round(u_f, 4)}, $A_e=${round(u_e, 4)} \n $\chi_1 / \kappa=$ {round(chi / kappa, 2)} \n '
              f'$\chi_1$={round(chi_1 * 1e3, 2)}[MHz] \n'
              f'$\chi_2$={round(chi_2 * 1e3, 2)}[MHz] \n',
              # f'Gr:$\mu_g=${round(popt_g[0], 2)},$\sigma_g=${round(popt_g[1], 2)}'
              # f'Ex:$\mu_e=${round(popt_e[0], 2)},$\sigma_e=${round(popt_e[1], 2)}',
              transform=ax.transAxes)

    ax_2.text(0.8, 1.05, f'Ham: {HAMILTONIAN} \n T = {round(T / 1e3, 2)}[$\mu s$]', transform=ax.transAxes)
    plt.legend()
    plt.show()
    print(f'Readout fidelity fe: {fidelity(popt_f[0], popt_f[1], popt_e[0], popt_e[1])}')

    fig_3, ax_3 = plt.subplots(1, 1)

    n_f, bins_f, patches_f = plt.hist(x=R_gf_f, bins=30, color='green', density=True,
                                      alpha=0.3, label='f', histtype='stepfilled')
    popt_f, pcov = curve_fit(gaussian, bins_f, np.append(n_f, 0))  # , p0=[-8, 5], bounds=[(-10, 0), (-5, 15)])
    plt.plot(bins_f, gaussian(bins_f, *popt_f), color='green')

    n_g, bins_g, patches_g = plt.hist(x=R_gf_g, bins=30, color='orange', density=True,
                                      alpha=0.6, label='g', histtype='stepfilled')
    popt_g, pcov = curve_fit(gaussian, bins_g, np.append(n_g, 0))  # , p0=[8, 5], bounds=[(5, 0), (10, 15)])
    plt.plot(bins_g, gaussian(bins_g, *popt_g), color='lightskyblue')

    ax_3.set_xlabel(r'$R_{gf}$')
    ax_3.set_ylabel(r'Counts normalized')
    ax_3.text(0.05, 1,
              f'Driving Strength $A_f=${round(u_f, 4)}, $A_g=${round(u_g, 4)} \n $\chi_1 / \kappa=$ {round(chi / kappa, 2)} \n '
              f'$\chi_1$={round(chi_1 * 1e3, 2)}[MHz] \n'
              f'$\chi_2$={round(chi_2 * 1e3, 2)}[MHz] \n',
              # f'Gr:$\mu_g=${round(popt_g[0], 2)},$\sigma_g=${round(popt_g[1], 2)}'
              # f'Ex:$\mu_e=${round(popt_e[0], 2)},$\sigma_e=${round(popt_e[1], 2)}',
              transform=ax.transAxes)

    ax_3.text(0.8, 1.05, f'Ham: {HAMILTONIAN} \n T = {round(T / 1e3, 2)}[$\mu s$]', transform=ax.transAxes)
    plt.legend()
    plt.show()
    print(f'Readout fidelity fe: {fidelity(popt_g[0], popt_g[1], popt_f[0], popt_f[1])}')

if COLORMAP:
    # calculate counts and corresponding bins for ground state case

    # histogram data for R_eg temp

    fix, ax = plt.subplots(1)
    plt.hist2d(R_eg_g, R_fe_g, bins=30)
    ax.set_xlabel(r'$R_{ge}$')
    ax.set_ylabel(r'$R_{fe}$')
    plt.title('ground')
    plt.show()

    fig, ax = plt.subplots(1)
    plt.hist2d(R_eg_e, R_fe_e, bins=30)
    ax.set_xlabel(r'$R_{ge}$')
    ax.set_ylabel(r'$R_{fe}$')
    plt.title(r'excited')
    plt.show()

    fix, ax = plt.subplots(1)
    plt.hist2d(R_eg_f, R_fe_f, bins=30)
    ax.set_xlabel(r'$R_{ge}$')
    ax.set_ylabel(r'$R_{fe}$')
    plt.title('2nd excited')
    plt.show()

# plot templates
if PLOT_TEMP:

    #analytical description of templates
    #popt, pcov = curve_fit(p_analytic, ts, temp_g_safe.imag, p0=[38.7, 38.7, 3.86e-3, 9.738e-4, 1.368e-2])  #, bounds=[(0, 0, 0, 0, 0), (50, 50, 1, 1e-3, 1e-1)])
    #plt.plot(ts, p_analytic(ts, *popt), color='lightskyblue')
                                                                #wr, w_rot_r, kappa, chi, u
    #plt.plot(ts, np.sqrt(kappa) * x_analytic(ts, wr, w_rot_r, kappa, chi_0, u_g), 'k--', label='g Real analytic')
    #plt.plot(ts, np.sqrt(kappa) * p_analytic(ts, wr, w_rot_r, kappa, chi_0, u_g), 'k--', label='g Imag analytic')
    #plt.plot(ts, np.sqrt(kappa) * x_analytic(ts, wr, w_rot_r, kappa, chi_1, u_e), 'k--', label='e Real analytic')
    #plt.plot(ts, np.sqrt(kappa) * p_analytic(ts, wr, w_rot_r, kappa, chi_1, u_e), 'k--', label='e Imag analytic')
    #plt.plot(ts, np.sqrt(kappa) * x_analytic(ts, wr, w_rot_r, kappa, chi_2, u_f), 'k--', label='f Real analytic')
    #plt.plot(ts, np.sqrt(kappa) * p_analytic(ts, wr, w_rot_r, kappa, chi_2, u_f), 'k--', label='f Imag analytic')

    plt.plot(ts, temp_g_safe.real, 'b', linewidth=0.5, label='g Real')
    plt.plot(ts, temp_g_safe.imag, 'g', linewidth=0.5, label='g Imag')
    plt.plot(ts, temp_e_safe.real, 'r', linewidth=0.5, label='e Real')
    plt.plot(ts, temp_e_safe.imag, 'c', linewidth=0.5, label='e Imag')
    plt.plot(ts, temp_f_safe.real, 'm', linewidth=0.5, label='f Real')
    plt.plot(ts, temp_f_safe.imag, 'y', linewidth=0.5, label='f Imag')
    plt.xlabel(f'Time t [ns]')
    plt.ylabel('template')
    plt.legend()
    plt.show()

if d3_plot:
    #u_g = np.sqrt(photon_number) / np.abs(tf(w0=wr + chi_0, wd=wd, kappa=kappa))
    #u_e = np.sqrt(photon_number) / np.abs(tf(w0=wr + chi_1, wd=wd, kappa=kappa))
    #u_f = np.sqrt(photon_number) / np.abs(tf(w0=wr + chi_2, wd=wd, kappa=kappa))
    #template = ideal measurement
    #m_g_ideal = template_analytic(ts, wr, wd, kappa, chi_0, u_g)
    #m_e_ideal = template_analytic(ts, wr, wd, kappa, chi_1, u_e)
    #m_f_ideal = template_analytic(ts, wr, wd, kappa, chi_2, u_f)
    m_g_ideal = temp_g
    m_e_ideal = temp_e
    m_f_ideal = temp_f

    #add ideal R calculated from analytic solution of template
    R_eg_g.append(calc_R(m_g_ideal, temp_e, temp_g))
    R_eg_e.append(calc_R(m_e_ideal, temp_e, temp_g))
    R_eg_f.append(calc_R(m_f_ideal, temp_e, temp_g))

    R_fe_g.append(calc_R(m_g_ideal, temp_f, temp_e))
    R_fe_e.append(calc_R(m_e_ideal, temp_f, temp_e))
    R_fe_f.append(calc_R(m_f_ideal, temp_f, temp_e))

    R_gf_g.append(calc_R(m_g_ideal, temp_g, temp_f))
    R_gf_e.append(calc_R(m_e_ideal, temp_g, temp_f))
    R_gf_f.append(calc_R(m_f_ideal, temp_g, temp_f))

    ax = plt.figure().add_subplot(projection='3d')

    #merging the three lists together
    R_eg_ges = R_eg_g + R_eg_e + R_eg_f
    R_fe_ges = R_fe_g + R_fe_e + R_fe_f
    R_gf_ges = R_gf_g + R_gf_e + R_gf_f

    data = np.array([R_eg_ges, R_fe_ges, R_gf_ges]).transpose()
    A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
    C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  # coefficients

    # evaluate it on grid
    X, Y = np.meshgrid(np.arange(min(R_eg_ges), max(R_eg_ges)), np.arange(min(R_fe_ges), max(R_fe_ges)))
    # evaluate it on grid
    Z = C[0] * X + C[1] * Y + C[2]

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(R_eg_g, R_fe_g, R_gf_g, label='g')
    ax.scatter(R_eg_e, R_fe_e, R_gf_e, label='e')
    ax.scatter(R_eg_f, R_fe_f, R_gf_f, label='f')
    ax.set_xlabel(r'$R_{ge}$')
    ax.set_ylabel(r'$R_{fe}$')
    ax.set_zlabel(r'$R_{gf}$')
    ax.view_init(20, 45)
    plt.legend()
    plt.show()

    n = [C[0], C[1], -1]  #vector perpendicular to plane
    p = [0, 0, C[2]]  #stützvector

    # new basis vectors in 2d plane
    b_1 = [1, 1, C[0]+C[1]]
    b_2 = [-1-C[1] * (C[0] + C[1]), (C[0] + C[1]) * C[0] + 1, C[1] + C[0]]

    #matrix that changes basis of linear map from 3d to 2d
    # pinv is quasi inver of matrix
    basis_change_matrix = scipy.linalg.pinv(np.array([b_1, b_2]).transpose())
    R_g_1s = []
    R_g_2s = []
    R_e_1s = []
    R_e_2s = []
    R_f_1s = []
    R_f_2s = []


    for i in range(len(R_eg_g)):
        R_g = np.matmul(basis_change_matrix, np.array([R_eg_g[i], R_fe_g[i], R_gf_g[i]]).transpose())
        R_g_1s.append(R_g[0])
        R_g_2s.append(R_g[1])

        R_e = np.matmul(basis_change_matrix, np.array([R_eg_e[i], R_fe_e[i], R_gf_e[i]]).transpose())
        R_e_1s.append(R_e[0])
        R_e_2s.append(R_e[1])

        R_f = np.matmul(basis_change_matrix, np.array([R_eg_f[i], R_fe_f[i], R_gf_f[i]]).transpose())
        R_f_1s.append(R_f[0])
        R_f_2s.append(R_f[1])

    #iterate over all different combinations of F E G

    rotate_R_12_plane = True
    if rotate_R_12_plane:
        # find new basis vectors of 2 dim R_1/R_2 plane that sperates all three states the best
        # template with R_1 / R_2 coordinates
        G = np.array([R_g_1s[-1], R_g_2s[-1]])
        E = np.array([R_e_1s[-1], R_e_2s[-1]])
        F = np.array([R_f_1s[-1], R_f_2s[-1]])

        sigma_g = sigma_calc(G, R_g_1s, R_g_2s)
        sigma_e = sigma_calc(E, R_e_1s, R_e_2s)
        sigma_f = sigma_calc(F, R_f_1s, R_f_2s)
        factor = 2.5
        option_1 = rotated_basis(G, E, F, sigma_g, sigma_e, sigma_f, factor)
        option_2 = rotated_basis(G, F, E, sigma_g, sigma_f, sigma_e, factor)
        option_3 = rotated_basis(E, F, G, sigma_g, sigma_e, sigma_f, factor)
        print(option_1)
        print(option_2)
        print(option_3)

        basis_rotation_matrix = option_1[0]

        #2nd iteration with adapted basis_change_matrix with additional rotation
        R_g_1s = []
        R_g_2s = []
        R_e_1s = []
        R_e_2s = []
        R_f_1s = []
        R_f_2s = []
        angle = 0 * np.pi / 180
        basis_rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        basis_change_matrix_new = np.matmul(basis_rotation_matrix, basis_change_matrix)

        for i in range(len(R_eg_g)):
            R_g = np.matmul(basis_change_matrix_new, np.array([R_eg_g[i], R_fe_g[i], R_gf_g[i]]).transpose())
            R_g_1s.append(R_g[0])
            R_g_2s.append(R_g[1])

            R_e = np.matmul(basis_change_matrix_new, np.array([R_eg_e[i], R_fe_e[i], R_gf_e[i]]).transpose())
            R_e_1s.append(R_e[0])
            R_e_2s.append(R_e[1])

            R_f = np.matmul(basis_change_matrix_new, np.array([R_eg_f[i], R_fe_f[i], R_gf_f[i]]).transpose())
            R_f_1s.append(R_f[0])
            R_f_2s.append(R_f[1])



    plt.scatter(R_g_1s, R_g_2s, label='g')
    plt.scatter(R_e_1s, R_e_2s, label='e')
    plt.scatter(R_f_1s, R_f_2s, label='f')


    #plot last point separately as ideal case
    plt.plot(R_g_1s[-1], R_g_2s[-1], 'o', color='tab:red', label='g analy')
    plt.plot(R_e_1s[-1], R_e_2s[-1], 'o', color='tab:brown', label='e analy')
    plt.plot(R_f_1s[-1], R_f_2s[-1], 'o', color='tab:purple', label='f analy')
    plt.xlabel(r'$R_1$')
    plt.ylabel(r'$R_2$')
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(1, 1)

    n_g, bins_g, patches_g = plt.hist(x=R_g_2s, bins=30, color='orange', #density=True,
                                      alpha=0.3, label='g', histtype='stepfilled')
    popt_g, pcov = curve_fit(gaussian_unnorm, bins_g, np.append(n_g, 0), p0=[-6, 1, 1])#, bounds=[(-10, 0), (-5, 15)])

    n_e, bins_e, patches_e = plt.hist(x=R_e_2s, bins=30, color='lightskyblue',# density=True,
                                      alpha=0.3, label='e', histtype='stepfilled')
    popt_e, pcov = curve_fit(gaussian_unnorm, bins_e, np.append(n_e, 0), p0=[-2, 1, 1])#, bounds=[(5, 0), (10, 15)])

    n_f, bins_f, patches_f = plt.hist(x=R_f_2s, bins=30, color='green',#, density=True,
                                      alpha=0.3, label='f', histtype='stepfilled')
    popt_f, pcov = curve_fit(gaussian_unnorm, bins_f, np.append(n_f, 0), p0=[2, 1, 1])#, bounds=[(-10, 0), (-5, 15)])

    R_minmax = np.linspace(min(bins_g.tolist()+bins_e.tolist()+bins_f.tolist()), max(bins_g.tolist()+bins_e.tolist()+bins_f.tolist()), 100)

    plt.plot(R_minmax, gaussian_unnorm(R_minmax, *popt_e),
             color='lightskyblue')
    plt.plot(R_minmax, gaussian_unnorm(R_minmax, *popt_g),
             color='orange')
    plt.plot(R_minmax, gaussian_unnorm(R_minmax, *popt_f), color='green')

    ax.set_xlabel(r'$R_2$')
    ax.set_ylabel(r'Counts')
    ax.text(0.05, 1,
            f'Readout fidelity ge: {round(fidelity(*popt_g, *popt_e),4)} \n'
            f'Readout fidelity gf: {round(fidelity(*popt_g, *popt_f),4)} \n'
            f'Readout fidelity ef: {round(fidelity(*popt_e, *popt_f),4)} \n',
            transform=ax.transAxes)

    #ax.text(0.8, 1.05, f'Ham: {HAMILTONIAN} \n T = {round(T / 1e3, 2)}[$\mu s$]', transform=ax.transAxes)
    plt.legend()
    plt.show()


if d2_plot_parameter_variation:

    '''#ax = plt.figure().add_subplot(projection='3d')

    dist_ge_s = []
    dist_ef_s = []
    dist_gf_s = []

    #for kappa in np.linspace(1 / 10 * chi_0, 10 * chi_0, 5):
    wds = np.linspace(wr - 1e-3 * 2 * np.pi, wr + 1e-3 * 2 * np.pi, 500, endpoint=False)
    for wd in wds:
        R_eg_g = []
        R_fe_g = []
        R_gf_g = []

        R_eg_e = []
        R_fe_e = []
        R_gf_e = []

        R_eg_f = []
        R_fe_f = []
        R_gf_f = []

        u_g = np.sqrt(photon_number) / np.abs(tf(w0=wr + chi_0, wd=wd, kappa=kappa))
        u_e = np.sqrt(photon_number) / np.abs(tf(w0=wr + chi_1, wd=wd, kappa=kappa))
        u_f = np.sqrt(photon_number) / np.abs(tf(w0=wr + chi_2, wd=wd, kappa=kappa))
        # assume N=0
        temp_g = template_analytic(ts, wr, wd, kappa, chi_0, u_g)
        temp_e = template_analytic(ts, wr, wd, kappa, chi_1, u_e)
        temp_f = template_analytic(ts, wr, wd, kappa, chi_2, u_f)

        R_eg_g.append(calc_R(temp_g, temp_e, temp_g))
        R_eg_e.append(calc_R(temp_e, temp_e, temp_g))
        R_eg_f.append(calc_R(temp_f, temp_e, temp_g))

        R_fe_g.append(calc_R(temp_g, temp_f, temp_e))
        R_fe_e.append(calc_R(temp_e, temp_f, temp_e))
        R_fe_f.append(calc_R(temp_f, temp_f, temp_e))

        R_gf_g.append(calc_R(temp_g, temp_g, temp_f))
        R_gf_e.append(calc_R(temp_e, temp_g, temp_f))
        R_gf_f.append(calc_R(temp_f, temp_g, temp_f))

        # merging the three lists together
        R_eg_ges = R_eg_g + R_eg_e + R_eg_f
        R_fe_ges = R_fe_g + R_fe_e + R_fe_f
        R_gf_ges = R_gf_g + R_gf_e + R_gf_f

        data = np.array([R_eg_ges, R_fe_ges, R_gf_ges]).transpose()
        A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  # coefficients

        # evaluate it on grid
        X, Y = np.meshgrid(np.arange(min(R_eg_ges), max(R_eg_ges)), np.arange(min(R_fe_ges), max(R_fe_ges)))
        # evaluate it on grid
        Z = C[0] * X + C[1] * Y + C[2]

        n = [C[0], C[1], -1]  # vector perpendicular to plane
        p = [0, 0, C[2]]  # stützvector

        # new basis vectors in 2d plane
        b_1 = [1, 1, C[0] + C[1]]
        b_2 = [-1 - C[1] * (C[0] + C[1]), (C[0] + C[1]) * C[0] + 1, C[1] + C[0]]

        # matrix that changes basis of linear map from 3d to 2d
        # pinv is quasi inver of matrix
        basis_change_matrix = scipy.linalg.pinv(np.array([b_1, b_2]).transpose())
        R_g_1s = []
        R_g_2s = []
        R_e_1s = []
        R_e_2s = []
        R_f_1s = []
        R_f_2s = []

        for i in range(len(R_eg_g)):
            R_g = np.matmul(basis_change_matrix, np.array([R_eg_g[i], R_fe_g[i], R_gf_g[i]]).transpose())
            R_e = np.matmul(basis_change_matrix, np.array([R_eg_e[i], R_fe_e[i], R_gf_e[i]]).transpose())
            R_f = np.matmul(basis_change_matrix, np.array([R_eg_f[i], R_fe_f[i], R_gf_f[i]]).transpose())

            dist_ge_s.append(abs(R_g[1] - R_e[1]))
            dist_gf_s.append(abs(R_g[1] - R_f[1]))
            dist_ef_s.append(abs(R_e[1] - R_f[1]))'''

    plt.plot((wds - wr) * 1e3 / (2 * np.pi), dist_ge_s, label='g-e')
    plt.plot((wds - wr) * 1e3 / (2 * np.pi), dist_gf_s, label='g-f')
    plt.plot((wds - wr) * 1e3 / (2 * np.pi), dist_ef_s, label='e-f')
    plt.xlabel(r'driving frequency $\omega_d - \omega_r$ [MHz]')
    plt.ylabel(r'Separation between $R_2$')
    plt.legend()
    plt.grid()
    plt.show()


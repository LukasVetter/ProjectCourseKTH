import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import scipy
from scipy import integrate
from scipy.stats import norm



plt.rc('text', usetex=True)

def tf(w0, wd, kappa):
    return w0 / (w0**2 - wd**2 + 1j*wd*kappa)

def scalar_product(x, y):
    return sum(x[:] * y[:].conj()).real


def norm_funk(x):
    return np.sqrt(scalar_product(x, x).real)


'''result_master_ground = qt.mesolve(H, qt.ket2dm(psi0_ground), ts, c_ops=c_ops, e_ops=[x, p])
result_master_excited = qt.mesolve(H, qt.ket2dm(psi0_excited), ts, c_ops=c_ops, e_ops=[x, p])

temp_ground = np.sqrt(kappa) * (result_master_ground.expect[0] + 1j * result_master_ground.expect[1])
temp_excited = np.sqrt(kappa) * (result_master_excited.expect[0] + 1j * result_master_excited.expect[1])
'''


def get_template(H, psi0, ts, kappa, c_ops, e_ops):
    result_master = qt.mesolve(H, qt.ket2dm(psi0), ts, c_ops=c_ops, e_ops=e_ops)  # do oly psi0 qt.ket2dm(psi0)-> psi0
    temp = np.sqrt(kappa) * (result_master.expect[0] + 1j * result_master.expect[1])
    return temp, result_master

def calc_measurment_results(H, psi0, ts, sc_ops, ntraj, nsubsteps, method):
    exp = qt.ssesolve(H,
                      psi0,
                      ts,
                      sc_ops=sc_ops,
                      # e_ops=[x, p],
                      ntraj=ntraj,
                      nsubsteps=nsubsteps,
                      store_measurement=True,
                      # dW_factors=[1],
                      method=method,
                      #progress_bar='',
                      )  # 'heterodyne')


    I = np.array(exp.measurement).mean(axis=0)[:, 0, 0].real
    Q = np.array(exp.measurement).mean(axis=0)[:, 0, 1].real
    m = 0.5 * (I + 1j * Q)
    return m

def calc_R(m, temp_i, temp_j, NEW=False):
    if NEW:
        theta_ij = 0.5 * (norm_funk(temp_i)**2 - norm_funk(temp_j)**2) # -> has to be a number use norm  sum (temp_excited * signal)
        p_i = scalar_product(m, temp_i)
        p_j = scalar_product(m, temp_j)
        return p_i - p_j - theta_ij
    else:
        return scalar_product(m, temp_i)

def calc_R_new(m, temp, temp2):
    return scalar_product(m, temp)

def determine_count(R, resolution):
    '''
    R: list of R = p_e-p_g - theta_eg values len(R) = NUMBER_TRAJECTORIES
    resolution: additional factor to change resolution (not necessary)

    in the function first every R value is converted to a real integer and the number of occurences in R is counted
    e.g. R = [2,3,3,7]
        R_filled = [2,3,7], counts = [1,2,1]
    :return: R_filled,counts
    '''

    R = (np.array(R) / resolution).real.astype(np.int)

    R_positive = R + np.abs(np.amin(R))

    counts = np.bincount(R_positive)
    R_filled = np.arange(np.amax(R_positive) + 1) - np.abs(np.amin(R))

    R_filled_new = []
    counts_new = []
    for i in range(len(counts)):
        counts_e_new = []
        R_e_filled_new = []
        if counts[i] == 0:
            pass
        else:
            counts_new.append(counts[i])
            R_filled_new.append(R_filled[i]*resolution)

    return R_filled_new, counts_new

def gaussian(x, mu, std):
    return 1/(std * np.sqrt(2*np.pi)) * np.exp(-1/2*(x-mu)**2/std**2)

def gaussian_unnorm(x, mu, std, a):
    return a * np.exp(-1 / 2 * (x - mu) ** 2 / std ** 2)


def fidelity(mu_i, sigma_i, a_i, mu_j, sigma_j, a_j):
    '''
    fidelity of readout between state |i> and |j>
    :params: mu_k expectation value, sigma_k standard deviation
    '''
    #eps_i = error_wrong_state(mu_i, sigma_i, a_i, mu_j, sigma_j, a_j)
    #eps_j = error_wrong_state(mu_j, sigma_j, a_j, mu_i, sigma_j, a_j)
    normalization_i = 1 / (sigma_i * np.sqrt(2 * np.pi) * a_i)
    normalization_j = 1 / (sigma_j * np.sqrt(2 * np.pi) * a_j)
    if mu_i < mu_j:
        f_i = lambda x, mu_i, sigma_i, a_i: normalization_i * gaussian_unnorm(x, mu_i, sigma_i, a_i)
        f_j = lambda x, mu_j, sigma_j, a_j: normalization_j * gaussian_unnorm(x, mu_j, sigma_j, a_j)
        A_i = integrate.quad(f_i, -np.inf, (mu_i + mu_j) / 2, args=(mu_i, sigma_i, a_i))
        A_j = integrate.quad(f_j, (mu_i + mu_j) / 2, np.inf, args=(mu_j, sigma_j, a_j))
        eps_i = (1 - A_i[0]) / 2
        eps_j = (1 - A_j[0]) / 2
    else:
        print('error in fidelity function')
        eps_i = 1
        eps_j = 1
    return 1 - (eps_i + eps_j) / 2

def error_wrong_state(mu_i, sigma_i, mu_j, sigma_j):
    x = abs(mu_i) / (sigma_i * np.sqrt(2))
    return (1 - scipy.special.erf(x))/2

def x_analytic(ts, wr, wd, kappa, chi, u):
    delta = wr - wd
    #return 2 * u / 2 * 1 / (kappa**2 + 4*(delta+chi)**2) * ((2*delta + kappa + 2*chi)*(1 - np.exp(-ts*kappa/2) * np.cos(ts*(delta + chi)))
    #    + (2*delta - kappa + 2*chi) * np.exp(-ts*kappa/2) * np.sin(ts*(delta + chi)))
    return 2 * u / 2 * np.exp(-ts*kappa/2) * (-2 * np.exp(ts*kappa/2) * (delta + chi) + 2 * (delta+chi) * np.cos(ts*(delta + chi))
        + kappa * np.sin(ts*(delta + chi))) / (kappa**2 + 4*(delta+chi)**2)

def p_analytic(ts, wr, wd, kappa, chi, u):
    delta = wr - wd

    #delta = wr - wd
    #return 2 * u / 2 * 1 / (kappa ** 2 + 4 * (delta + chi) ** 2) * (
    #            (-2 * delta + kappa - 2 * chi) * (1 - np.exp(-ts * kappa / 2) * np.cos(ts * (delta + chi)))
    #            + (2 * delta + kappa + 2 * chi) * np.exp(-ts * kappa / 2) * np.sin(ts * (delta + chi)))
    return 2 * u / 2 * np.exp(-ts*kappa/2) * (-kappa * np.exp(ts*kappa/2) + kappa * np.cos(ts*(delta + chi))
        - 2 * (delta+chi) * np.sin(ts*(delta + chi))) / (kappa**2 + 4*(delta+chi)**2)


def template_analytic(ts, wr, wd, kappa, chi, u):
    #chi and u are depending on the state that the system was prepared in
    x = x_analytic(ts, wr, wd, kappa, chi, u)
    p = p_analytic(ts, wr, wd, kappa, chi, u)
    return np.sqrt(kappa)*(x + 1j * p)

def x_anal(t, a, b, c, d, e):
    return a + np.exp(b*t)*(c*np.cos(d*t) + e*np.sin(d*t))


def sigma_calc(P, R_1s, R_2s):
    return np.sqrt((np.sum(np.array([(R_1 - P[0]) ** 2 for R_1 in R_1s])) + np.sum(
        np.array([(R_2 - P[1]) ** 2 for R_2 in R_2s]))) / (len(R_1s)-1))

def norm_vec(v):
    norm = np.sqrt(v[0]**2 + v[1]**2)
    return v / norm


def rotated_basis(A, B, C, sigma_A, sigma_B, sigma_C, x):
    basis_1 = B - C + (A - B) / 2
    basis_2 = np.array([-basis_1[1], basis_1[0]])

    basis_change_matrix = np.array([basis_1.tolist(), basis_2.tolist()]).T
    A_prime = A + x * sigma_A * norm_vec(B - A)
    B_prime = B - x * sigma_B * norm_vec(B - A)
    C_prime_plus = C + x * sigma_C * norm_vec(B - A) + basis_1
    C_prime_minus = C - x * sigma_C * norm_vec(B - A) + basis_1
    print(C_prime_minus)
    heuristic_A = -np.sqrt((A_prime[0] - C_prime_plus[0]) ** 2 + (A_prime[1] - C_prime_plus[1]) ** 2) + np.sqrt(
        (A[0] - C_prime_plus[0]) ** 2 + (A[1] - C_prime_plus[1]) ** 2)
    heuristic_B = - np.sqrt((B_prime[0] - C_prime_minus[0]) ** 2 + (B_prime[1] - C_prime_minus[1]) ** 2) + np.sqrt(
        (B[0] - C_prime_minus[0]) ** 2 + (B[1] - C_prime_minus[1]) ** 2)

    return basis_change_matrix, heuristic_A, heuristic_B
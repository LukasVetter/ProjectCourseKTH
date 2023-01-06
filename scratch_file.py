'''
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

wr = 6.17e9  # resonator
N = 2  # Hilbert-space cutoff -> 30
wq = 3.56e9  # qubit frequency

g = 69.3e6  # coupling strength, Grad/s

a = qt.tensor(qt.destroy(N), qt.qeye(2))  # annihilation
n = a.dag() * a  # photo n number
x = qt.tensor(1/np.sqrt(2) * qt.position(N), qt.qeye(2))  # x = 1/sqrt(2) * (a + a.dag())
p = qt.tensor(1/np.sqrt(2) * qt.momentum(N), qt.qeye(2))  # p = -1j/sqrt(2) * (a - a.dag())
# with these definitions: <x>^2 / 2 + <p>^2 / 2 = <n>

# Qubit operators (Pauli matrices)
sx = qt.tensor(qt.qeye(N), qt.sigmax())  # sp + sm
sy = qt.tensor(qt.qeye(N), qt.sigmay())  # 1j * (sm - sp)
sz = qt.tensor(qt.qeye(N), qt.sigmaz())
sp = qt.tensor(qt.qeye(N), qt.sigmap())  # (sx + 1j * sy) / 2 == sm.dag()
sm = qt.tensor(qt.qeye(N), qt.sigmam())  # (sx - 1j * sy) / 2 == sp.dag()

# identity matrix
eye = qt.tensor(qt.qeye(N), qt.qeye(2))

#inital Hamiltonian for resonator and qubit are the same
H0R = 0 * wr * (a.dag() * a + 0 * 1 / 2 * eye)
H0q = - 0 * wq * sz / 2

chi = g ** 2 / (wr - wq)

H_coupling = chi * (a+a.dag()) * sx

#H = [H_coupling, [H_drive1, H_drive1_coeff], [H_drive2, H_drive2_coeff]]
H = [H_coupling]#, H_drive1+H_drive2]
sampling_freq = 1e10
T = 1e-6

ns = int(round(sampling_freq * T))  # number of samples in the pulse
print(f'Number of timesteps: {ns}')
ts = np.linspace(0, T, ns)

psi0 = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))

result_master = qt.mesolve(H, qt.ket2dm(psi0), ts, c_ops=[], e_ops=[sx])

r = [5]'''

class test:

    def __init__(self):
        self.hook_pos = {}

    '''def save_hook_pos(self, v, hook_pos):
        self.hook_pos[v] = hook_pos

    def test_func(self, a, b):
        for i in range(3):
            c, d = a+i, b+i
            self.save_hook_pos(c, d)'''

    def test_func(self, ):
        self.hook_pos[1] = 2
        return self.hook_pos


class test2:

    def __init__(self):
        self.attri = 1

    def mu(self):
        test.test_func()

test = test()
test2 = test2()

test2.mu()

print(test.hook_pos)
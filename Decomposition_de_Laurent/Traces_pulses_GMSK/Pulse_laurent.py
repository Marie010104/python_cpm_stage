import numpy as np
from scipy.special import erfc
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

# Fonction Q (probabilité de queue de la loi normale)
def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2))


# Génère l'impulsion gaussienne GMSK gt et son intégrale qt
def create_gmsk_pulse(span, Ns):
    Ts    = 1.0 / Ns
    t     = np.arange(-span / 2, span / 2 + Ts, Ts)
    BT    = 0.3
    alpha = 2 * np.pi * BT / np.sqrt(np.log(2))
    gauss = qfunc(alpha * (t - 0.5)) - qfunc(alpha * (t + 0.5))
    Cst   = 0.5 / (np.sum(gauss) * Ts)  # normalisation
    gt    = Cst * gauss
    qt    = cumulative_trapezoid(gt) * Ts  # intégration numérique
    return gt, qt


# Convertit un entier k en tableau de bits
def dec2bin(k, nbits):
    bits = np.zeros(nbits, dtype=int)
    for i in range(nbits):
        bits[i] = k % 2
        k //= 2
    return bits

# Calcule la fonction s(t) utilisée dans la décomposition de Laurent
def s_func(idx, q_interp, L, h, t):
    x = t + idx
    s = np.zeros_like(x)
    m1 = (x >= 0) & (x <= L)  # première partie du support
    s[m1] = np.sin(2*np.pi*h*q_interp(x[m1])) / np.sin(np.pi*h)
    m2 = (x > L) & (x <= 2*L)  # deuxième partie du support
    s[m2] = np.sin(2 * np.pi * h * (0.5 - q_interp(x[m2] - L))) / np.sin(np.pi * h)
    return s

# Construit les impulsions de Laurent g_k(t) pour chaque combinaison de bits
def Laurent_PAM(M, L, h, q_interp, Ns, t):
    K = 2**(L-1)  # nombre d'impulsions
    s_0 = s_func(0, q_interp, L, h, t)
    if L > 1:
        g_k = np.tile(s_0, (K, 1))
        for k in range(K):
            b_k = dec2bin(k, L-1)  # combinaison binaire associée à k
            for i in range(1, L):
                g_k[k, :] *= s_func(i + L*b_k[i-1], q_interp, L, h, t)
    else:
        g_k = s_0.reshape(1, -1)
    return g_k


# Paramètres de la modulation GMSK
L  = 3
h  = 0.5
Ns = 8
M  = 2


Ts = 1.0 / Ns

# Génération de l'impulsion gaussienne et de son intégrale
gt, qt = create_gmsk_pulse(L, Ns)

t_g = np.arange(-L/2, L/2 + Ts, Ts)
t_q = t_g[1:]


# Interpolation de q(t) pour évaluation en des points arbitraires
def q_interp(x):
    xc = x - L/2
    return np.interp(xc, t_q, qt, left=0.0, right=0.5)

# Construction des pulses de Laurent
t = np.arange(0, 2*L, Ts)
g_k = Laurent_PAM(M, L, h, q_interp, Ns, t)


# Affichage des impulsions obtenues
plt.figure(figsize=(8,5))
for k in range(g_k.shape[0]):
    plt.plot(t, g_k[k, :], label=f"$C_{{{k}}}(t)$")
plt.xlabel("t/T")
plt.ylabel("Amplitude")
plt.title(f"Impulsions de Laurent — GMSK BT=0.3, L={L}")
plt.legend()
plt.grid(True)
plt.show()
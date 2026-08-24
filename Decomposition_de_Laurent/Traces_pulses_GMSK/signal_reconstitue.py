import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from matplotlib.ticker import MultipleLocator


# Fonction Q (probabilité de queue de la loi normale)
def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2))

# Génère N bits aléatoires (0 ou 1)
def genBin(N):
    return np.random.randint(2, size=N)

# Génère la séquence de symboles pilotes
def genPilotes(M, L):
    return np.hstack((
        -(M - 1) * np.ones(16),
        (M - 1) * np.ones(32),
        -(M - 1) * np.ones(16),
        -(M - 1) * np.ones(int((L - 1) / 2))
    ))

# Mapping bits en symboles bipolaires (0->-1, 1->+1)
def mapp(bits):
    return 2 * bits - 1

# Génère la séquence d'apprentissage
def genTr(M, L0):
    return [-1, +1, -1, +1, -1, +1, -1, +1]

# Génère l'impulsion gaussienne GMSK gt et son intégrale qt
def create_gmsk_pulse(span, Ns):
    Ts = 1.0 / Ns
    t = np.arange(-span / 2, span / 2 + Ts, Ts)
    BT = 0.3
    alpha = 2 * np.pi * BT / np.sqrt(np.log(2))
    gauss = qfunc(alpha * (t - 0.5)) - qfunc(alpha * (t + 0.5))
    Cst = 0.5 / (np.sum(gauss) * Ts)  # normalisation
    gt = Cst * gauss
    qt = np.cumsum(gt) * Ts  # intégration numérique (cumulée)
    return gt, qt, t

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
    s[m1] = np.sin(2 * np.pi * h * q_interp(x[m1])) / np.sin(np.pi * h)
    m2 = (x > L) & (x <= 2 * L)  # deuxième partie du support
    s[m2] = np.sin(2 * np.pi * h * (0.5 - q_interp(x[m2] - L))) / np.sin(np.pi * h)
    return s

# Construit les impulsions de Laurent g_k(t) et les combinaisons de bits associées
def Laurent_PAM(L, h, q_interp, t):
    K = 2 ** (L - 1)  # nombre d'impulsions
    s_0 = s_func(0, q_interp, L, h, t)
    b_all = np.zeros((K, max(L - 1, 0)), dtype=int)
    if L > 1:
        g_k = np.tile(s_0, (K, 1))
        for k in range(K):
            b_k = dec2bin(k, L - 1)  # combinaison binaire associée à k
            b_all[k] = b_k
            for i in range(1, L):
                g_k[k, :] *= s_func(i + L * b_k[i - 1], q_interp, L, h, t)
    else:
        g_k = s_0.reshape(1, -1)
    return g_k, b_all


# Paramètres généraux de la simulation
Npilotes = 64
Nbits    = 512
M        = 2
os_      = 8
L        = 3
h        = 0.5

np.random.seed(0)

# Construction de la trame : pilotes + séquence d'apprentissage + données
bits         = genBin(Nbits)
data_symb    = mapp(bits)
pilotes_symb = genPilotes(M, L)
Tr           = genTr(M, Npilotes)

frame_symb  = np.concatenate((pilotes_symb, Tr, data_symb))

# Ajout de zéros de retard pour aligner la trame avec le filtre
zeros_retard  = np.zeros(L // 2, dtype=float)
symb_n_concat = np.concatenate((zeros_retard, frame_symb, zeros_retard))

alpha    = symb_n_concat
N        = len(alpha)
n_offset = L // 2

# Génération de l'impulsion gaussienne et de son intégrale
gt, qt, t_g = create_gmsk_pulse(L, os_)
Ts  = 1.0 / os_
t_q = t_g

# Interpolation de q(t) pour évaluation en des points arbitraires
def q_interp(x):
    xc = x - L / 2
    return np.interp(xc, t_q, qt, left=0.0, right=0.5)

# Construction des pulses de Laurent et des combinaisons de bits
t_pulse    = np.linspace(0, 2 * L, int(round(2 * L * os_)) + 1)
g_k, b_all = Laurent_PAM(L, h, q_interp, t_pulse)
K          = g_k.shape[0]     # K = 2^(L-1) composantes


# Phase cumulée du signal GMSK et composante principale a0
psi = np.pi * h * np.cumsum(alpha)
a0  = np.exp(1j * psi)

# Calcul des pseudo-symboles a_k pour chaque composante de Laurent
a = np.zeros((K, N), dtype=complex)
a[0, :] = a0
for k in range(1, K):
    prod = np.ones(N, dtype=complex)
    for i in range(1, L):
        b = b_all[k, i - 1]
        if b == 1:
            shifted = np.concatenate((np.ones(i), alpha[:-i]))
            prod *= shifted
    phase_fix = np.exp(1j * np.pi * h * L * np.sum(b_all[k]))
    a[k, :] = a0 * prod * phase_fix

# Sur-échantillonne une séquence en insérant des zéros entre les symboles
def upsample(seq, os):
    s = np.zeros(len(seq) * os, dtype=complex)
    s[::os] = seq
    return s

# Reconstruction cumulative du signal en sommant les contributions de chaque composante
conv_len   = N * os_ + len(t_pulse) - 1
s_cumul    = np.zeros(conv_len, dtype=complex)
partial_signals = []

for k in range(K):
    ak_up   = upsample(a[k, :], os_)
    contrib = np.convolve(ak_up, g_k[k, :])
    s_cumul = s_cumul + contrib
    partial_signals.append(s_cumul.copy())  # somme partielle jusqu'à la composante k


Nbits_tot = len(symb_n_concat)

# Génération du signal GMSK directement (méthode de référence, sans décomposition)
bits_s = np.zeros(Nbits_tot * os_)
bits_s[::os_] = symb_n_concat

SN_direct    = np.convolve(bits_s, gt)
Phi_N_direct = np.cumsum(SN_direct) * Ts
tx_direct    = np.exp(1j * 2 * np.pi * h * Phi_N_direct)


# Comparaison de l'erreur relative entre signal direct et reconstructions partielles
lo, hi = 400, min(len(tx_direct), len(partial_signals[-1])) - 400
for name, sig in [("C0 seul", partial_signals[0]),
                   ("C0 + C1", partial_signals[1]),
                   ("Toutes les composantes", partial_signals[-1])]:
    err = np.linalg.norm(tx_direct[lo:hi] - sig[lo:hi]) / np.linalg.norm(tx_direct[lo:hi])
    print(f"{name:24s} erreur relative = {err:.6f}")

# Préparation des données pour l'affichage
n_plot_start = n_offset * os_
n_plot_len   = min(500 * os_,
                    len(tx_direct) - n_plot_start,
                    len(partial_signals[1]) - n_plot_start)
t_axis = np.arange(n_plot_len) / os_

signaux = [
    (tx_direct,          "Signal direct (sans decomposition)", "black",     "-"),
    (partial_signals[0], r"Laurent : $C_0$ seul ", "tab:blue", "--"),
    (partial_signals[1], r"Laurent : $C_0 + C_1$",                "tab:red",  "--"),
]

# Tracé des composantes I et Q : signal direct vs reconstructions de Laurent
for comp_idx, (comp_label, ax_label) in enumerate([("I (partie reelle)", "I(t)"),
                                                     ("Q (partie imaginaire)", "Q(t)")]):
    fig, ax = plt.subplots(figsize=(13, 5))
    for sig, lbl, color, ls in signaux:
        seg = sig[n_plot_start:n_plot_start + n_plot_len]
        val = seg.real if comp_idx == 0 else seg.imag
        ax.plot(t_axis[:len(val)], val, ls, label=lbl, color=color, linewidth=1.8)

    ax.set_xlabel(r"Temps")
    ax.set_ylabel(ax_label)
    ax.set_title(f"Composante {comp_label} : signal direct vs reconstructions de Laurent")
    ax.set_xlim(0, min(500, t_axis[-1]))
    ax.xaxis.set_major_locator(MultipleLocator(16))
    ax.grid(True, which="major")
    ax.legend()
    plt.tight_layout()
    plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.integrate import cumulative_trapezoid


# Fonction Q (probabilité de queue de la loi normale)
def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2))


# Génère N bits aléatoires (0 ou 1)
def genBin(N):
    return np.random.randint(2, size=N)


# Génère la séquence de symboles pilotes (préambule connu)
def genPilotes(M,L):
    pilotes_symb = np.hstack((
        -(M - 1) * np.ones(16),
        (M - 1) * np.ones(32),
        -(M - 1) * np.ones(16),
        -(M-1)* np.ones(int((L-1)/2))
    ))
    return pilotes_symb


# Mapping bits -> symboles bipolaires (0->-1, 1->+1)
def mapp(bits):
    return 2 * bits - 1


# Génère la séquence d'apprentissage (training sequence)
def genTr(M, L0):
    Tr=[-1,+1,-1,+1,-1,+1,-1,+1]
    return Tr


# Génère l'impulsion en cosinus surélevé (RC) gt et son intégrale qt
def create_rc_pulse(pulse_length, os):
    Ts    = 1.0 / os
    t     = np.arange(0, pulse_length + Ts, Ts)
    g_t   = (1 / (2 * pulse_length)) * (1 - np.cos(2 * np.pi * t / pulse_length))
    K     = 0.5 / (np.sum(g_t) * Ts)  # normalisation
    g_t   = K * g_t
    q_t   = cumulative_trapezoid(g_t, dx=Ts, initial=0)  # intégration numérique
    return g_t, q_t


# Ajoute un bruit blanc gaussien complexe au signal, selon le SNR souhaité (en dB)
def awgn(signal, snr_db, Ns):
    signal_power = np.mean(np.abs(signal) ** 2) * Ns
    snr_lin      = 10 ** (snr_db / 10)
    noise_power  = signal_power / snr_lin
    noise        = np.sqrt(noise_power / 2) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )
    return signal + noise



# Paramètres

Npilotes         = 64
Nbits            = 16384
M                = 2
os               = 8
L                = 3
Ts               = 1.0 / os
modulation_index = 0.5
trames           = 1

Signal_Fichier_RC    = "SignalFichier_Test_RC"
Symboles_Fichier_RC  = "SymbolesFichier_Test_RC"
Bits_Fichier_RC      = "BitsFichier_Test_RC"

# Impulsion de mise en forme RC
gt, qt = create_rc_pulse(L, os)

Tr           = genTr(M, Npilotes)
pilotes_symb = genPilotes(M,L)

bits_n_trames     = np.zeros(trames, dtype=object)
symboles_n_trames = np.zeros(trames, dtype=object)

bits_n_exp  = []
symb_n_exp  = []

# Construction de trames : bits -> symboles -> ajout pilotes + séquence d'apprentissage
for i in range(trames):
    bits = genBin(Nbits)
    bits_n_trames[i] = bits
    bits_n_exp = np.concatenate((bits_n_exp, bits))


    data_symb = mapp(bits)


    frame_symb = np.concatenate((pilotes_symb, Tr, data_symb))

    symboles_n_trames[i] = frame_symb
    symb_n_exp = np.concatenate((symb_n_exp, frame_symb))

Nbits = len(symb_n_exp)


# Ajout de zéros de retard pour aligner la trame avec le filtre
zeros_retard = np.zeros(L // 2, dtype=float)
symb_n_concat = np.concatenate((zeros_retard, symb_n_exp, zeros_retard))
Nbits_tot = len(symb_n_concat)

# Suréchantillonnage
bits_s = np.zeros(Nbits_tot * os)
bits_s[::os] = symb_n_concat

t_seq  = np.arange(0, Nbits - Ts, Ts)
SN    = np.convolve(bits_s, gt)[:len(t_seq)]
Phi_N = np.cumsum(SN) * Ts

# Signal transmis (modulation de phase CPM)
tx_signal = np.exp(1j * 2 * np.pi * modulation_index * Phi_N)

print(f"Nombre de trames        : {trames}")
print(f"Bits/trame (pilotes)    : {Npilotes}")
print(f"Nombre d'échantillons   : {len(tx_signal)}")

#génération fichier txt

# Entrelacement I/Q du signal pour export texte
Valeur_intercale = []
for i in range(len(tx_signal)):
    Valeur_intercale.append(f"{tx_signal[i].real:.8f}")
    Valeur_intercale.append(f"{tx_signal[i].imag:.8f}")

try:
    with open(Signal_Fichier_RC, "w") as fichier:
        fichier.write("\t".join(Valeur_intercale))
    print(f"Fichier '{Signal_Fichier_RC}' généré avec succès!")
    print("Structure: [I_Echan1][Tab][Q_Echan1][Tab][I_Echan2]...")
except Exception as e:
    print(f"Une erreur est survenue lors de la génération du fichier : {e}")


# Export des symboles émis
symb_str = [f"{s:.8f}" for s in symb_n_exp]
try:
    with open(Symboles_Fichier_RC, "w") as fichier:
        fichier.write("\t".join(symb_str))
    print(f"Fichier '{Symboles_Fichier_RC}' généré avec succès!")
except Exception as e:
    print(f"Une erreur est survenue lors de la génération du fichier : {e}")


# Export des bits émis
bits_str = [f"{int(b)}" for b in bits_n_exp]
try:
    with open(Bits_Fichier_RC, "w") as fichier:
        fichier.write("\t".join(bits_str))
    print(f"Fichier '{Bits_Fichier_RC}' généré avec succès!")
except Exception as e:
    print(f"Une erreur est survenue lors de la génération du fichier : {e}")



from matplotlib.ticker import MultipleLocator

# Phase instantanée du signal (dépliée pour éviter les sauts de 2*pi)
phi_t = np.unwrap(np.angle(tx_signal))

# Axe temporel exprimé en nombre de périodes symbole Ts

t_symb = np.arange(len(tx_signal)) / os


# Tracé de la phase du signal transmis
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(t_symb, phi_t)
ax.set_xlabel(r"Temps (en nombre de $T_s$)")
ax.set_ylabel(r"$\phi(t)$ [rad]")
ax.set_title(r"Vue d'ensemble : $\phi(t)$")
ax.set_xlim(0, 500)
ax.xaxis.set_major_locator(MultipleLocator(16))
ax.grid(True, which="major")
plt.tight_layout()
plt.tight_layout()
plt.show()
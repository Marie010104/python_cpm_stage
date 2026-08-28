import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt

# Plage de Eb/N0
EbN0_dB = np.arange(0, 14.5, 0.5)
EbN0_lin = 10 ** (EbN0_dB / 10)

# Fonction Q
def Q(x):
    return 0.5 * erfc(x / np.sqrt(2))

# Formules théoriques du BER pour les modulations classiques

# BPSK
ber_bpsk = Q(np.sqrt(2 * EbN0_lin))

# QPSK
ber_qpsk = Q(np.sqrt(2 * EbN0_lin))

# 8-PSK
def ber_mpsk(EbN0_lin, M):
    k = np.log2(M)
    return (2 / k) * Q(
        np.sqrt(2 * k * EbN0_lin) * np.sin(np.pi / M)
    )

ber_8psk = ber_mpsk(EbN0_lin, 8)

# 16-QAM
def ber_mqam(EbN0_lin, M):
    k = np.log2(M)
    return (4 / k) * (1 - 1 / np.sqrt(M)) * Q(
        np.sqrt(3 * k / (M - 1) * EbN0_lin)
    )

ber_16qam = ber_mqam(EbN0_lin, 16)

# Fonction théorique pour GMSK et RC
def ber_theoretical(ebn0_db, dmin2):
    ebn0_lin = 10 ** (np.asarray(ebn0_db, dtype=float) / 10)
    return erfc(np.sqrt(dmin2 * ebn0_lin / 2)) / 2

# GMSK L=3 avec dmin^2 = 1.78
ber_gmsk_L3 = ber_theoretical(EbN0_dB, dmin2=1.78)

# RC L=3 avec dmin^2 = 1.9
ber_rc_L3 = ber_theoretical(EbN0_dB, dmin2=1.9)

# Tracé superposé des courbes
plt.figure(figsize=(9, 6))

plt.semilogy(
    EbN0_dB, ber_bpsk,
    'o-', label='BPSK',
    linewidth=1.8, markersize=4
)

plt.semilogy(
    EbN0_dB, ber_qpsk,
    's--', label='QPSK',
    linewidth=1.8, markersize=4
)

plt.semilogy(
    EbN0_dB, ber_8psk,
    '^-.', label='8-PSK',
    linewidth=1.8, markersize=4
)

plt.semilogy(
    EbN0_dB, ber_16qam,
    'd:', label='16-QAM',
    linewidth=1.8, markersize=4
)

plt.semilogy(
    EbN0_dB, ber_gmsk_L3,
    'x-', label='GMSK L=3',
    linewidth=2.0, markersize=5
)

plt.semilogy(
    EbN0_dB, ber_rc_L3,
    '+--', label='RC L=3',
    linewidth=2.0, markersize=6
)

# Mise en forme du graphique
plt.xlabel(r'$E_b/N_0$ (dB)', fontsize=12)
plt.ylabel("Taux d'erreur binaire (BER)", fontsize=12)

plt.title(
    'Performances théoriques des modulations numériques classiques\n'
    'sur canal AWGN',
    fontsize=12
)

plt.grid(True, which='both', linestyle=':', linewidth=0.6)
plt.legend(fontsize=10)

plt.ylim(1e-6, 1)
plt.xlim(0, 14)

plt.tight_layout()
plt.show()

# Affichage des points de chaque courbe
courbes = {
    "BPSK": ber_bpsk,
    "QPSK": ber_qpsk,
    "8-PSK": ber_8psk,
    "16-QAM": ber_16qam,
    "GMSK": ber_gmsk_L3,
    "RC ": ber_rc_L3,
}

for nom, ber in courbes.items():
    print(f"\n--- {nom} : Eb/N0 (dB), BER ---")

    for x, y in zip(EbN0_dB, ber):
        print(f"{x:.1f},{y:.6e}")
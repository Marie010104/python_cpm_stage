import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.integrate import cumulative_trapezoid

os = 64
pulse_length = 2

# Fonction Q
def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2))

# Pulse GMSK
def create_gmsk_pulse(pulse_length, os):
    Ts = 1.0 / os

    t = np.arange(-pulse_length,
                  pulse_length + Ts,
                  Ts)

    BT = 0.3
    alpha = 2 * np.pi * BT / np.sqrt(np.log(2))

    g_t = qfunc(alpha * (t - 0.5)) - qfunc(alpha * (t + 0.5))

    K = 0.5 / (np.sum(g_t) * Ts)
    g_t = K * g_t

    q_t = cumulative_trapezoid(g_t, dx=Ts, initial=0)

    return t, g_t, q_t

# Pulse RC-CPM
def create_rc_pulse(pulse_length, os):
    Ts = 1.0 / os

    t = np.arange(0, pulse_length + Ts, Ts)

    g_t = (1 / (2 * pulse_length)) * (
        1 - np.cos(2 * np.pi * t / pulse_length)
    )

    K = 0.5 / (np.sum(g_t) * Ts)
    g_t = K * g_t

    q_t = cumulative_trapezoid(g_t, dx=Ts, initial=0)

    return t, g_t, q_t

# Création des pulses
t_gmsk, g_gmsk, q_gmsk = create_gmsk_pulse(pulse_length, os)
t_rc, g_rc, q_rc = create_rc_pulse(pulse_length, os)

# Affichage
plt.figure(figsize=(10, 8))

# Réponses impulsionnelles
plt.subplot(2, 1, 1)
plt.plot(t_gmsk, g_gmsk, label='GMSK')
plt.plot(t_rc, g_rc, label='RC-CPM')
plt.grid(True)
plt.xlabel('t / Ts')
plt.ylabel('g(t)')
plt.title('Réponse impulsionnelle GMSK et RC')
plt.legend()

# Réponses de phase
plt.subplot(2, 1, 2)
plt.plot(t_gmsk, q_gmsk, label='GMSK')
plt.plot(t_rc, q_rc, label='RC')
plt.grid(True)
plt.xlabel('t / Ts')
plt.ylabel('q(t)')
plt.title('Réponse de phase GMSK et RC')
plt.legend()

plt.tight_layout()
plt.show()




#RC (ou gmsk seule)
plt.figure(figsize=(12, 5))



# Réponse fréquentielle à gauche
plt.subplot(1, 2, 1)
plt.plot(t_rc, g_rc)
plt.grid(True)
plt.xlabel('Fréquence')
plt.ylabel('g(t)')
plt.title('Réponse fréquentielle RC')

# Réponse de phase à droite
plt.subplot(1, 2, 2)
plt.plot(t_rc, q_rc)
plt.grid(True)
plt.xlabel('t / Ts')
plt.ylabel('q(t)')
plt.title('Réponse de phase RC')

plt.tight_layout()
plt.show()



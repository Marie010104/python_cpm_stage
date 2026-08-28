import numpy as np
import pandas as pd
from scipy.special import erfc
from scipy.integrate import cumulative_trapezoid
from itertools import product as iterproduct
from fractions import Fraction
import matplotlib.pyplot as plt


# Fonction Q (probabilité de queue de la loi normale)
def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2))


# Génère l'impulsion en cosinus surélevé (RC) gt et son intégrale qt
def create_rc_pulse(pulse_length, os):
    Ts    = 1.0 / os
    t     = np.arange(0, pulse_length + Ts, Ts)
    g_t   = (1 / (2 * pulse_length)) * (1 - np.cos(2 * np.pi * t / pulse_length))
    K     = 0.5 / (np.sum(g_t) * Ts)  # normalisation
    g_t   = K * g_t
    q_t   = cumulative_trapezoid(g_t, dx=Ts, initial=0)  # intégration numérique cumulée
    return g_t, q_t

# Ajoute un bruit blanc gaussien complexe au signal, selon le SNR souhaité (en dB)
def awgn(signal, snr_db, os):
    signal_power = np.mean(np.abs(signal) ** 2) * os
    snr_lin = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_lin
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )
    return signal + noise


# Génère la séquence de symboles pilotes (préambule connu)
def genPilotes(M):
    return np.hstack((
        -(M - 1) * np.ones(16),
        (M - 1) * np.ones(32),
        -(M - 1) * np.ones(16)
    ))


# Génère la séquence d'apprentissage (training sequence)
def genTr(M, L0):
    return np.array([-1, 1, -1, 1, -1, 1, -1, 1], dtype=float)


# Mapping bits -> symboles bipolaires (0->-1, 1->+1)
def mapp(bits, M=2):
    return 2 * bits - 1


# Détecte le début de trame par corrélation avec une forme d'onde de référence
def detect_frame_start(r, ref_wave):
    correlation = np.correlate(r, ref_wave, mode="valid")
    return int(np.argmax(np.abs(correlation)))


# Estime l'offset de fréquence, le retard et la phase à partir des symboles pilotes
def sync_freq_phase(r_pilot, L0, OS, h, M):
    Ts = 1.0 / OS
    N = L0 * OS

    r1 = np.zeros(N, dtype=complex)
    r2 = np.zeros(N, dtype=complex)

    # Séparation du pilote en deux fenêtres avec compensation de phase
    r1[0: int(N / 4)] = r_pilot[0: int(N / 4)]
    r1[3 * int(N / 4): N] = r_pilot[3 * int(N / 4): N] * np.exp(-1j * (M - 1) * np.pi * h * L0)
    r2[int(N / 4): 3 * int(N / 4)] = r_pilot[int(N / 4): 3 * int(N / 4)] * np.exp(1j * (M - 1) * np.pi * h * L0 / 2)

    t_seq = np.arange(0, L0, Ts)
    n = min(len(t_seq), N)
    t_seq = t_seq[:n]
    r1 = r1[:n]
    r2 = r2[:n]

    r1p = r1 * np.exp(1j * (M - 1) * np.pi * h * t_seq)
    r2p = r2 * np.exp(-1j * (M - 1) * np.pi * h * t_seq)

    # Estimation grossière de la fréquence par FFT (recherche du pic spectral)
    kf = 4
    Nf = kf * OS * L0
    r1p_fft = np.fft.fftshift(np.fft.fft(r1p, int(Nf)))
    r2p_fft = np.fft.fftshift(np.fft.fft(r2p, int(Nf)))
    f = np.fft.fftshift(np.fft.fftfreq(int(Nf), Ts))
    X = np.abs(r1p_fft) + np.abs(r2p_fft)

    idx_max = int(np.argmax(X))
    fd_est = f[idx_max]

    # Affinement de l'estimation de fréquence par interpolation logarithmique autour du pic
    v_1, v0, v1 = idx_max - 1, idx_max, idx_max + 1
    if v_1 < 0 or v1 >= len(f):
        fd_est_int = fd_est
    else:
        fd_est_int = fd_est + (1 / (2 * L0 * kf)) * ((np.log(X[v_1]) - np.log(X[v1])) / \
                     (np.log(X[v_1]) + np.log(X[v1]) - 2 * np.log(X[v0])))

    # Estimation du retard et de la phase à partir des deux corrélations compensées en fréquence
    lambda1 = np.sum(r1p * np.exp(-1j * 2 * np.pi * fd_est_int * t_seq))
    lambda2 = np.sum(r2p * np.exp(-1j * 2 * np.pi * fd_est_int * t_seq))

    eps_est = np.angle(lambda1 * np.conj(lambda2)) / (2 * (M - 1) * np.pi * h)
    theta_est = np.angle(
        np.exp(-1j * (M - 1) * np.pi * h * eps_est) * lambda1 +
        np.exp(1j * (M - 1) * np.pi * h * eps_est) * lambda2
    )
    theta_est = (theta_est - np.pi) % (2 * np.pi)

    return fd_est_int, eps_est, theta_est

# Génère toutes les combinaisons possibles de "length" éléments pris dans alphabet
def _permu_repet(alphabet, length):
    combos = list(iterproduct(alphabet, repeat=length))
    return np.array(combos)


# Calcule le nombre d'états, de branches et les phases possibles du treillis CPM
def states_branches(modulation_index, pulse_length, M_ary):
    frac = Fraction(modulation_index).limit_denominator(100)
    m_num, p_den = frac.numerator, frac.denominator

    if m_num % 2 == 0:
        states_number = p_den * M_ary ** (pulse_length - 1)
        branche_number = p_den * M_ary ** pulse_length
        phase_states = np.mod(np.arange(0, p_den) * np.pi * m_num / p_den, 2 * np.pi)
    else:
        states_number = 2 * p_den * M_ary ** (pulse_length - 1)
        branche_number = 2 * p_den * M_ary ** pulse_length
        phase_states = np.mod(np.arange(0, 2 * p_den) * np.pi * m_num / p_den, 2 * np.pi)
    return states_number, branche_number, phase_states


# Construit la table des transitions d'états du treillis CPM (branches = arcs entre états)
def states_transition(states_number, branche_number, phase_states,
                       pulse_length, M_ary, Am, modulation_index):

    state_vector_number = M_ary ** (pulse_length - 1)
    state_vector = _permu_repet(Am, pulse_length - 1)
    branche_vector = _permu_repet(Am, pulse_length)

    # Construction de la liste des états (phase + symboles mémorisés)
    states = np.zeros((states_number, pulse_length), dtype=float)
    states[:, 0] = np.repeat(phase_states, state_vector_number)
    states[:, 1:] = np.tile(state_vector, (len(phase_states), 1))

    # Construction de la liste des branches (phase + symboles + nouveau symbole)
    branche_vector_number = M_ary ** pulse_length
    branches = np.zeros((branche_number, pulse_length + 1), dtype=float)
    branches[:, 0] = np.repeat(phase_states, branche_vector_number)
    branches[:, 1:] = np.tile(branche_vector, (len(phase_states), 1))

    states_list = [tuple(states[i]) for i in range(states_number)]

    transitions = np.zeros((branche_number, 3), dtype=int)

    # Pour chaque branche, on retrouve l'état de départ et l'état d'arrivée correspondants
    for i in range(branche_number):
        from_key = tuple(branches[i, :pulse_length])
        from_idx = None
        for si, sk in enumerate(states_list):
            if all(abs(np.array(from_key) - np.array(sk)) < 1e-5):
                from_idx = si
                break
        transitions[i, 0] = from_idx

        # Nouvelle phase après transition (mise à jour de phase CPM)
        next_phase = np.mod(branches[i, 1] * np.pi * modulation_index + branches[i, 0], 2 * np.pi)
        for ps in phase_states:
            if abs(next_phase - ps) <= 1e-5 or abs(next_phase - 2 * np.pi) <= 1e-5:
                next_phase = ps if abs(next_phase - ps) <= 1e-5 else 0.0
                break

        to_key = tuple([next_phase] + list(branches[i, 2:]))
        to_idx = None
        for si, sk in enumerate(states_list):
            if all(abs(np.array(to_key) - np.array(sk)) < 1e-5):
                to_idx = si
                break
        transitions[i, 1] = to_idx
        transitions[i, 2] = i

    # Tri des transitions par état d'arrivée (utile pour le parcours du Viterbi)
    sort_idx = np.argsort(transitions[:, 1], kind='stable')
    states_sort = transitions[sort_idx]
    return branches, states_sort


# Calcule la forme d'onde de référence (métrique de branche) associée à une séquence de symboles
def compute_branch_metric(os, modulation_index, pulse_length, alpha, gt):
    Ts = 1.0 / os
    Nbits = len(alpha)
    bits_s = np.zeros(Nbits * os)
    bits_s[::os] = alpha

    t_seq = np.arange(0, Nbits - Ts, Ts)

    SN = np.convolve(bits_s, gt)[:len(t_seq)]
    PhiN = np.cumsum(SN) * Ts

    start = (pulse_length - 1) * os - 2
    stop = pulse_length * os
    psi = np.exp(1j * 2 * np.pi * modulation_index * PhiN[start:stop])

    return psi


# Décode une trame CPM par algorithme de Viterbi (MLSE) à partir du signal reçu
def decode_frame_gmsk(received, states_number, branche_number, branches,
                       states_sort, branch_metrics, os, pulse_length,
                       decision_delay, Nbits_frame):

    Ts = 1.0 / os

    detected_bits = np.zeros(Nbits_frame - decision_delay, dtype=float)
    dbits_idx = 0
    pathmetric = np.zeros(states_number)
    survivor_path = np.zeros((states_number, decision_delay), dtype=int)

    for n in range(1, Nbits_frame - pulse_length + 1):
        pathmetric_n = np.zeros(states_number)
        bm_cum = np.zeros(branche_number)

        # Métrique de branche par corrélation avec chaque forme d'onde candidate
        for i in range(branche_number):
            if n != 1:
                seg = received[(n - 1) * os: n * os + 1]
            else:
                seg = received[0: os + 1]
            z = np.real(
                np.exp(-1j * branches[i, 0])
                * np.trapezoid(seg * np.conj(branch_metrics[i])) * Ts
            )
            bm_cum[i] = z

        # Sélection du meilleur chemin (survivor) pour chaque état d'arrivée
        for i in range(0, branche_number, M):
            candidates = np.array([
                bm_cum[states_sort[i + k, 2]] + pathmetric[states_sort[i + k, 0]]
                for k in range(M)
            ])
            best_k = np.argmax(candidates)

            new_state = states_sort[i + best_k, 1]
            prev_state = states_sort[i + best_k, 0]
            pathmetric_n[new_state] = candidates[best_k]

            if n <= decision_delay:
                survivor_path[new_state, n - 1] = prev_state
            else:
                if i == 0:
                    survivor_path[:, :-1] = survivor_path[:, 1:]  # décalage de la fenêtre glissante
                survivor_path[new_state, -1] = prev_state

        pathmetric = pathmetric_n.copy()

        # Décision retardée (traceback) une fois la profondeur de décision atteinte
        if n > decision_delay:
            idx_path = np.argmax(pathmetric)
            curr_state = idx_path

            for jj in range(decision_delay, 0, -1):
                prev_state = survivor_path[curr_state, jj - 1]
                if jj > 1:
                    curr_state = prev_state

            mask = np.all(states_sort[:, :2] == [prev_state, curr_state], axis=1)
            matched_idx = int(np.where(mask)[0][0])

            detected_bits[dbits_idx] = branches[states_sort[matched_idx, 2], 1]
            dbits_idx += 1

    return detected_bits[:dbits_idx]


# Paramètres généraux
os = 8
L = 3
M = 2
modulation_index = 0.5
decision_delay = 50
Am = np.array([2 * m - 1 - M for m in range(1, M + 1)])

Npilotes = 64
Nbits_payload = 16384
SNR_dB = 21

Signal_Fichier = "Test_CPM_RC_1.lvm"
Bits_Fichier   = "BitsFichier_Test_RC"

# Impulsion de mise en forme et construction du treillis CPM
gt, qt = create_rc_pulse(L, os)

states_number, branche_number, phase_states = states_branches(modulation_index, L, M)
branches, states_sort = states_transition(states_number, branche_number, phase_states,
                                           L, M, Am, modulation_index)

# Précalcul des formes d'onde de référence (métriques de branche) pour chaque branche du treillis
_test_psi = compute_branch_metric(os, modulation_index, L, branches[0, 1:], gt)
metric_len = len(_test_psi)
branch_metrics = np.zeros((branche_number, metric_len), dtype=complex)
branch_metrics[0] = _test_psi
for i in range(1, branche_number):
    alpha = branches[i, 1:]
    branch_metrics[i] = compute_branch_metric(os, modulation_index, L, alpha, gt)

# Lit un fichier LabVIEW (.lvm) et reconstruit le signal complexe I+jQ
def read_lvm_iq(path):
    with open(path, "r", encoding="latin-1") as f:
        lines = f.readlines()

    # Trouve dynamiquement où commencent les données (après la ligne d'en-tête de colonnes)
    data_start = None
    for i, l in enumerate(lines):
        if l.startswith("X_Value"):
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError("Ligne d'en-tête 'X_Value' introuvable dans le fichier LVM")

    df = pd.read_csv(path, sep="\t", skiprows=data_start,
                      header=None, decimal=",", engine="python")

    I = df.iloc[:, 1].to_numpy()
    Q = df.iloc[:, 2].to_numpy()
    return I + 1j * Q

# Chargement du signal reçu et des bits de référence émis
tx_signal = read_lvm_iq(Signal_Fichier)

df_bits = pd.read_csv(Bits_Fichier, sep="\t", header=None)
bits_ref = df_bits.to_numpy(dtype=int).flatten()


snr_range = np.arange(0, 11)

# Indices de début de trame et de début des données utiles (après pilotes + Tr)
frame_start = int(np.round((L-1)/2)*os+1)
data_start = frame_start+(64+8+L-1)*os


ber_sim = np.zeros(len(snr_range))

# Boucle principale : pour chaque SNR, on bruite le signal, on synchronise, on décode et on calcule le BER
for snr, ebn0_db in enumerate(snr_range):
    total_errors = 0
    total_bits = 0

    snr_db_channel = ebn0_db
    r = awgn(tx_signal, snr_db_channel, os)
    np.random.seed(42)

    r_pilot = r[frame_start: frame_start + Npilotes * os]
    fd_est, eps_est, theta_est = sync_freq_phase(r_pilot, Npilotes, os, modulation_index, M)

    # Correction de la phase estimée
    r_sync = r * np.exp(-1j * theta_est)

    received_data = r_sync[data_start-34:]

    detected_bits = decode_frame_gmsk(
            received_data, states_number, branche_number, branches,
            states_sort, branch_metrics, os, L, decision_delay,
            Nbits_frame=Nbits_payload
        )

    bits_m = mapp(bits_ref)

    # Fenêtres d'alignement pour comparer bits émis et détectés (compense les retards)
    sl = 2 + L + 2
    el = Nbits_payload - decision_delay - 2 - L - 7
    sl2 = 1 + L + 7
    el2 = len(detected_bits) - L - 2

    sent = bits_m[sl:el]
    recv = detected_bits[sl2:el2]

    errors = int(np.count_nonzero(sent - recv))
    total_errors += errors
    total_bits += len(recv)

    ber = total_errors / total_bits if total_bits > 0 else np.nan
    ber_sim[snr] = ber
    print(f"SNR = {snr:2d} dB | BER = {ber_sim[snr]:.3e}")

# --- Tracé ---
# Création de la figure
fig, ax = plt.subplots(figsize=(8, 5))

# Courbe BER
ax.semilogy(
    snr_range,
    ber_sim,
    '-o',
    color='tab:blue',
    linewidth=2,
    markersize=6,
    markerfacecolor='white',
    markeredgewidth=1.5,
    label='Viterbi MLSE'
)

# Axes
ax.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=12)
ax.set_ylabel("Bit Error Rate (BER)", fontsize=12)
ax.set_title("BER Performance of GMSK", fontsize=14, fontweight='bold')

# Limites
ax.set_xlim(min(snr_range), max(snr_range))
ax.set_ylim(1e-6, 1)

# Grille
ax.grid(True, which='major', linestyle='-', alpha=0.4)
ax.grid(True, which='minor', linestyle=':', alpha=0.3)

# Légende
ax.legend(loc='best', fontsize=11)

# Mise en page
plt.tight_layout()

# Affichage
plt.show()


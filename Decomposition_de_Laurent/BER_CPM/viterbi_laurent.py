import numpy as np
from scipy.special import erfc
from itertools import product as iterproduct
from fractions import Fraction
from matplotlib import pyplot as plt




import time

debut = time.time()

# Impulsion GMSK + décomposition de Laurent


# Fonction Q (probabilité de queue de la loi normale)
def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2))


# Génère l'impulsion gaussienne GMSK gt et son intégrale qt
def create_gmsk_pulse(span, os):
    Ts = 1.0 / os
    t = np.arange(-span / 2, span / 2 + Ts, Ts)
    BT = 0.3
    alpha_ = 2 * np.pi * BT / np.sqrt(np.log(2))
    gauss = qfunc(alpha_ * (t - 0.5)) - qfunc(alpha_ * (t + 0.5))
    Cst = 0.5 / (np.sum(gauss) * Ts)  # normalisation
    gt = Cst * gauss
    qt = np.cumsum(gt) * Ts  # intégration numérique cumulée
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
            b_k = dec2bin(k, L - 1)
            b_all[k] = b_k
            for i in range(1, L):
                g_k[k, :] *= s_func(i + L * b_k[i - 1], q_interp, L, h, t)
    else:
        g_k = s_0.reshape(1, -1)
    return g_k, b_all


#  Treillis CPM


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


# Calcule les pseudo-symboles a0 (fenêtre glissante) et a1 pour la décomposition de Laurent
def laurent_pseudo_symbols(alpha_local, h, L, b1_pattern):
    psi = np.pi * h * np.cumsum(alpha_local)
    a0_window = np.exp(1j * psi)
    prod = 1.0 + 0j
    for i in range(1, L):
        b = b1_pattern[i - 1]
        if b == 1:
            idx = len(alpha_local) - 1 - i
            val = alpha_local[idx] if idx >= 0 else 1.0
            prod *= val
    phase_fix = np.exp(1j * np.pi * h * L * np.sum(b1_pattern))
    a1 = a0_window[-1] * prod * phase_fix
    return a0_window, a1


# Autocorrélation discrète d'une paire d'impulsions de Laurent, avec un retard l (en symboles)
def pulse_autocorr(gk, gj, l, os, Ts):
    if l == 0:
        return Ts * np.sum(gk * gj)
    return Ts * np.sum(gk[l * os:] * gj[:len(gj) - l * os])


# Applique le filtrage adapté (corrélation) avec g0 et g1 pour chaque instant symbole
def matched_filter_outputs(received, g0, g1, os, delta_symbols):
    Ts = 1.0 / os
    Lg = len(g0)
    shift = delta_symbols * os
    padded = np.concatenate([received, np.zeros(Lg + shift + os, dtype=complex)])
    Nsym = len(received) // os + 1
    y0 = np.zeros(Nsym, dtype=complex)
    y1 = np.zeros(Nsym, dtype=complex)
    for k in range(Nsym):
        start = k * os - shift
        if start < 0:
            continue
        seg = padded[start:start + Lg]
        y0[k] = np.sum(seg * g0) * Ts
        y1[k] = np.sum(seg * g1) * Ts
    return y0, y1


DELTA_SYMBOLS = 3


# Ajoute un bruit blanc gaussien complexe au signal, selon le SNR souhaité (en dB)
def awgn(signal, snr_db, os):
    signal_power = np.mean(np.abs(signal) ** 2) * os
    snr_lin = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_lin
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise


# Paramètres généraux de la modulation CPM/GMSK
L = 3
h = 0.5
os = 8
M_ary = 2
Am = np.array([-1, 1])
Ts = 1.0 / os

# Génération de l'impulsion gaussienne et de son intégrale
gt, qt, t_g = create_gmsk_pulse(L, os)


# Interpolation de q(t) pour évaluation en des points arbitraires
def q_interp(x):
    xc = x - L / 2
    return np.interp(xc, t_g, qt, left=0.0, right=0.5)


# Construction des impulsions de Laurent (composantes g_k) et des combinaisons de bits
t_pulse = np.linspace(0, 2 * L, int(round(2 * L * os)) + 1)
g_k, b_all = Laurent_PAM(L, h, q_interp, t_pulse)
K = g_k.shape[0]  # 2^(L-1) = 8

# Énergie de chaque impulsion de Laurent
energies = np.array([np.sum(np.abs(g_k[k]) ** 2) * Ts for k in range(K)])

# On ne garde que les deux composantes principales g0, g1
g0, g1 = g_k[0], g_k[1]
b1_pattern = b_all[1]

# Corrélations des impulsions de Laurent (retards l = 0..L-1)
R00 = np.array([pulse_autocorr(g0, g0, l, os, Ts) for l in range(L)])
R01 = np.array([pulse_autocorr(g0, g1, l, os, Ts) for l in range(L)])
R11 = np.array([pulse_autocorr(g1, g1, l, os, Ts) for l in range(L)])


# Treillis


pulse_length = L
modulation_index = h

# Construction du treillis CPM (états, branches, phases)
states_number, branche_number, phase_states = states_branches(modulation_index, pulse_length, M_ary)
branches, states_sort = states_transition(states_number, branche_number, phase_states,
                                           pulse_length, M_ary, Am, modulation_index)


# Pseudo-symboles a0 (fenêtre complète), a1, et biais Ungerboeck par branche
a0_window_branch = np.zeros((branche_number, L), dtype=complex)
a1_branch = np.zeros(branche_number, dtype=complex)
for i in range(branche_number):
    a0_window_branch[i], a1_branch[i] = laurent_pseudo_symbols(branches[i, 1:], h, L, b1_pattern)
a0_branch = a0_window_branch[:, -1]

# Calcul du biais de métrique (algorithme d'Ungerboeck) pour chaque branche du treillis
bias_branch = np.zeros(branche_number)
for i in range(branche_number):
    b = 2 * np.real(np.conj(a0_branch[i]) * a1_branch[i] * R01[0])
    for l in range(1, L):
        a0_prev = a0_window_branch[i, L - 1 - l]
        b += 2 * np.real(np.conj(a0_branch[i]) * a0_prev * R00[l])
    bias_branch[i] = b


# Mapping bits en symboles bipolaires (0->-1, 1->+1)
def mapp(bits):
    return 2 * bits - 1


# Génère le signal GMSK directement à partir des symboles (référence, sans décomposition)
def modulate_gmsk(alpha, gt, h, os):
    Ts = 1.0 / os
    Nbits = len(alpha)
    bits_s = np.zeros(Nbits * os)
    bits_s[::os] = alpha
    SN = np.convolve(bits_s, gt)
    PhiN = np.cumsum(SN) * Ts
    return np.exp(1j * 2 * np.pi * h * PhiN)


# Algorithme de Viterbi : décode les bits à partir des sorties du filtre adapté
def run_viterbi(y0, y1, idx_offset, branches, states_sort, a0_branch, a1_branch, bias_branch,
                 states_number, branche_number, M_ary, pulse_length, decision_delay, Nbits):
    detected_bits = np.zeros(Nbits - decision_delay, dtype=float)
    dbits_idx = 0
    pathmetric = np.zeros(states_number)
    survivor_path = np.zeros((states_number, decision_delay), dtype=int)

    np.random.seed(33)
    for n in range(1, Nbits - pulse_length + 1):

        pathmetric_n = np.zeros(states_number)
        bm_cum = np.zeros(branche_number)
        k_sym = n + idx_offset
        yk0 = y0[k_sym] if 0 <= k_sym < len(y0) else 0.0
        yk1 = y1[k_sym] if 0 <= k_sym < len(y1) else 0.0
        # Métrique de branche (corrélation - biais Ungerboeck) pour chaque branche
        for i in range(branche_number):
            corr = np.real(
                np.exp(-1j * branches[i, 0]) *
                (np.conj(a0_branch[i]) * yk0 + np.conj(a1_branch[i]) * yk1)
            )
            bm_cum[i] = corr - 0.5 * bias_branch[i]

        # Sélection du meilleur chemin (survivor) pour chaque état d'arrivée
        for i in range(0, branche_number, M_ary):
            candidates = np.array([
                bm_cum[states_sort[i + k, 2]] + pathmetric[states_sort[i + k, 0]]
                for k in range(M_ary)
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


# Évalue le nombre d'erreurs entre symboles émis et détectés sur un signal reçu
def evaluate(received, alpha_true, Nbits, idx_offset, decision_delay):
    y0, y1 = matched_filter_outputs(received, g0, g1, os, DELTA_SYMBOLS)
    detected = run_viterbi(y0, y1, idx_offset, branches, states_sort, a0_branch, a1_branch,
                            bias_branch, states_number, branche_number, M_ary, pulse_length,
                            decision_delay, Nbits)
    # Fenêtres d'alignement pour comparer symboles émis et détectés (compense les retards)
    sl = 2 + pulse_length
    el = Nbits - decision_delay - 2 - pulse_length
    sl2 = 1 + pulse_length
    el2 = len(detected) - 3 - pulse_length
    sent = alpha_true[sl:el]
    recv = detected[sl2:el2]
    n_compare = min(len(sent), len(recv))
    errs = int(np.count_nonzero(sent[:n_compare] - recv[:n_compare]))
    return errs, n_compare


# BER théorique en fonction de Eb/N0
def ber_theoretical(ebn0_db, dmin2=1.78):
    ebn0_lin = 10 ** (np.asarray(ebn0_db, dtype=float) / 10)
    return erfc(np.sqrt(dmin2 * ebn0_lin / 2)) / 2


IDX_OFFSET = 2


# Simulation Monte Carlo du BER pour un Eb/N0 donné (accumulation d'erreurs par blocs)
def simulate_ber(ebn0_db, min_errors=30, max_bits=400_000, block_bits=25_000,
                  decision_delay=50, seed=None):
    rng = np.random.default_rng(seed)
    total_errors = 0
    total_bits = 0

    while total_errors < min_errors and total_bits < max_bits:
        bits = rng.integers(0, 2, block_bits)
        alpha = mapp(bits)
        tx = modulate_gmsk(alpha, gt, h, os)
        received = awgn(tx[(pulse_length - 1) * os - 1:], ebn0_db, os)

        y0, y1 = matched_filter_outputs(received, g0, g1, os, DELTA_SYMBOLS)
        detected = run_viterbi(y0, y1, IDX_OFFSET, branches, states_sort,
                                a0_branch, a1_branch, bias_branch, states_number,
                                branche_number, M_ary, pulse_length, decision_delay, block_bits)

        sl = 2 + pulse_length
        el = block_bits - decision_delay - 2 - pulse_length
        sl2 = 1 + pulse_length
        el2 = len(detected) - 3 - pulse_length
        sent = alpha[sl:el]
        recv = detected[sl2:el2]
        n = min(len(sent), len(recv))

        total_errors += int(np.count_nonzero(sent[:n] - recv[:n]))
        total_bits += n

    ber = total_errors / total_bits if total_bits > 0 else np.nan
    return ber, total_errors, total_bits


if __name__ == "__main__":
    decision_delay = 50

    # Calibration de idx_offset (inchangée) : recherche du décalage donnant le moins d'erreurs
    Nbits_test = 10000
    bits_test = np.random.randint(0, 2, Nbits_test)
    alpha_test = mapp(bits_test)
    tx_test = modulate_gmsk(alpha_test, gt, h, os)
    received_test = tx_test[(pulse_length - 1) * os - 1:]

    best_offset, best_errs = None, None
    for cand in range(-6, 7):
        errs, n_compare = evaluate(received_test, alpha_test, Nbits_test, cand, decision_delay)
        if best_errs is None or errs < best_errs:
            best_errs, best_offset = errs, cand

    # Boucle principale : simulation du BER sur une plage de Eb/N0
    ebn0_range = np.arange(0, 11, 1)
    ber_sim = np.zeros(len(ebn0_range))
    ber_theory = ber_theoretical(ebn0_range, dmin2=1.78)

    print("=" * 78)
    print(" BER Monte Carlo -- Récepteur de Laurent (C0+C1), GMSK BT=0.3, L=4, h=0.5")
    print("=" * 78)

    for idx, ebn0_db in enumerate(ebn0_range):
        ber, errs, nbits = simulate_ber(int(ebn0_db), min_errors=30,
                                         max_bits=400_000, block_bits=25_000,
                                         decision_delay=30, seed=1000 + idx)
        ber_sim[idx] = ber
        ecart = ber - ber_theory[idx]
        print(f"Eb/N0 = {ebn0_db:2d} dB | BER sim = {ber:.4e} | BER théo = {ber_theory[idx]:.4e} ")

    # Évite les valeurs nulles pour l'échelle logarithmique du graphique
    ber_sim_plot = np.maximum(ber_sim, 1e-7)
    ber_theory_plot = np.maximum(ber_theory, 1e-7)

    # Tracé de la courbe BER simulée vs théorique
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(ebn0_range, ber_sim_plot, marker='o', linewidth=2,
                label='Récepteur de Laurent')
    ax.semilogy(ebn0_range, ber_theory_plot, linestyle='--', linewidth=2,
                label='Courbe théorique')
    ax.set_xlabel("Eb/N0 (dB)", fontsize=12)
    ax.set_ylabel("Taux d'erreur binaire (BER)", fontsize=12)
    ax.set_title("Récepteur de Laurent", fontsize=13)
    ax.grid(True, which='both', linestyle='--', linewidth=0.6)
    ax.set_ylim(1e-6, 1)
    ax.set_xlim(ebn0_range[0], ebn0_range[-1])
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.show()


fin = time.time()

duree = fin - debut
print(f"Durée de la simulation : {duree:.3f} secondes")
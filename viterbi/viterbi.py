import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.integrate import cumulative_trapezoid
from itertools import product as iterproduct

import time

debut = time.time()

def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2))


def awgn(signal, snr_db, os):
    signal_power = np.mean(np.abs(signal) ** 2)*os
    snr_lin      = 10 ** (snr_db / 10)
    noise_power  = signal_power / snr_lin
    noise        = np.sqrt(noise_power / 2) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )
    return signal + noise



#  Impulsion GMSK


def create_gmsk_pulse(pulse_length, os):
    Ts    = 1.0 / os
    t     = np.arange(-pulse_length / 2, pulse_length / 2 + Ts, Ts)
    BT    = 0.3
    alpha = 2 * np.pi * BT / np.sqrt(np.log(2))
    gauss = qfunc(alpha * (t - 0.5)) - qfunc(alpha * (t + 0.5))
    Cst   = 0.5 / (np.sum(gauss) * Ts)
    gt    = Cst * gauss
    qt    = cumulative_trapezoid(gt)* Ts
    return gt, qt


#  Treillis


def _permu_repet(alphabet, length):
    combos = list(iterproduct(alphabet, repeat=length))
    return np.array(combos)


def states_branches(modulation_index, pulse_length, M_ary):
    from fractions import Fraction
    frac = Fraction(modulation_index).limit_denominator(100)
    m_num, p_den = frac.numerator, frac.denominator

    if m_num % 2 == 0:
        states_number  = p_den * M_ary ** (pulse_length - 1)
        branche_number = p_den * M_ary ** pulse_length
        phase_states   = np.mod(
            np.arange(0, p_den) * np.pi * m_num / p_den, 2 * np.pi
        )
    else:
        states_number  = 2 * p_den * M_ary ** (pulse_length - 1)
        branche_number = 2 * p_den * M_ary ** pulse_length
        phase_states   = np.mod(
            np.arange(0, 2 * p_den) * np.pi * m_num / p_den, 2 * np.pi
        )
    return states_number, branche_number, phase_states


def states_transition(states_number, branche_number, phase_states,
                      pulse_length, M_ary, Am, modulation_index):

    state_vector_number  = M_ary ** (pulse_length - 1)
    state_vector         = _permu_repet(Am, pulse_length - 1)   # (N, L-1)
    branche_vector       = _permu_repet(Am, pulse_length)        # (N, L)

    # Build states table : [phase | bit_combo…]
    states = np.zeros((states_number, pulse_length), dtype=float)
    states[:, 0] = np.repeat(phase_states, state_vector_number)
    states[:, 1:] = np.tile(state_vector, (len(phase_states), 1))

    # Build branches table : [phase | bit_combo…]
    branche_vector_number = M_ary ** pulse_length
    branches = np.zeros((branche_number, pulse_length + 1), dtype=float)
    branches[:, 0]  = np.repeat(phase_states, branche_vector_number)
    branches[:, 1:] = np.tile(branche_vector, (len(phase_states), 1))

    # State transitions
    # Convert states to list of tuples for fast lookup
    states_list = [tuple(states[i]) for i in range(states_number)]
    states_dict = {s: idx for idx, s in enumerate(states_list)}

    transitions = np.zeros((branche_number, 3), dtype=int)  # [from, to, branch_idx]

    for i in range(branche_number):
        # Current state key : first pulse_length columns of branch
        from_key = tuple(branches[i, :pulse_length])
        # Find matching state (with tolerance)
        from_idx = None
        for si, sk in enumerate(states_list):
            if all(abs(np.array(from_key) - np.array(sk)) < 1e-5):
                from_idx = si
                break
        transitions[i, 0] = from_idx

        # Next phase
        next_phase = np.mod(
            branches[i, 1] * np.pi * modulation_index + branches[i, 0],
            2 * np.pi
        )
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
        transitions[i, 2] = i  # branch index

    # Sort by destination state (column 1) — "states_sort" du MATLAB
    sort_idx    = np.argsort(transitions[:, 1], kind='stable')
    states_sort = transitions[sort_idx]
    return branches, states_sort

#  Métrique de branche
def compute_branch_metric(os, modulation_index, pulse_length, alpha, gt):
    Ts     = 1.0 / os
    Nbits  = len(alpha)
    bits_s = np.zeros(Nbits * os)
    bits_s[::os] = alpha                           # upsample

    t_seq = np.arange(0, Nbits - Ts, Ts)   # 0:Ts:Nbits-Ts

    SN    = np.convolve(bits_s, gt)[:len(t_seq)]
    PhiN  = np.cumsum(SN) * Ts

    start = (pulse_length - 1) * os -2
    stop  = pulse_length * os
    psi   = np.exp(1j * 2 * np.pi * modulation_index * PhiN[start:stop])
    psi_t   = np.exp(1j * 2 * np.pi * modulation_index * PhiN)


    return psi

#  Mapping / démapping (M=2, pulse≠1)

def mapp(bits, M_ary):
    return 2 * bits - 1

#  BER théorique


def ber_theoretical(ebn0_db, dmin2=1.78):
    ebn0_lin = 10 ** (np.asarray(ebn0_db, dtype=float) / 10)
    #return qfunc(np.sqrt(dmin2 * ebn0_lin))
    return erfc(np.sqrt(dmin2 * ebn0_lin/2))/2


#  Simulation Monte Carlo + Viterbi


def run_simulation():
    # Paramètres
    pulse_length     = 4
    os               = 8
    Ts               = 1.0 / os
    M_ary            = 2
    modulation_index = 0.5
    dmin2            = 1.78
    decision_delay   = 50

    snr_range  = np.arange(0, 10)   # 0 … 10 dB
    #snr_range = 8*np.ones(1).astype(int)
    Am         = np.array([2 * m - 1 - M_ary for m in range(1, M_ary + 1)])  # [-1, +1]

    #NBits_limit = (10**4, 10**6)
    NBits_limit = (10**4, 10**6)

    # Impulsion
    gt, qt = create_gmsk_pulse(pulse_length, os)

    # Treillis
    states_number, branche_number, phase_states = states_branches(
        modulation_index, pulse_length, M_ary
    )
    branches, states_sort = states_transition(
        states_number, branche_number, phase_states,
        pulse_length, M_ary, Am, modulation_index
    )

    # Métriques de branche
    # Calcul de la taille réelle via un appel test sur la première branche
    _test_psi  = compute_branch_metric(os, modulation_index, pulse_length,
                                       branches[0, 1:], gt)
    metric_len = len(_test_psi)
    branch_metrics = np.zeros((branche_number, metric_len), dtype=complex)
    branch_metrics[0] = _test_psi
    for i in range(1, branche_number):
        alpha = branches[i, 1:]
        branch_metrics[i] = compute_branch_metric(
            os, modulation_index, pulse_length, alpha, gt
        )

    # BER théorique
    pe_vec = ber_theoretical(snr_range, dmin2)

    # Calcul de la longueur de trame adaptative
    frame_max_length = 300.0 / pe_vec * np.log2(M_ary)

    ber_sim    = np.zeros(len(snr_range))
    ber_theory = pe_vec.copy()

    print()
    print("=" * 100)
    print(f"  GMSK BT=0.3 | pulse_length={pulse_length} | os={os} | "
          f"M={M_ary} | h={modulation_index} | d²min={dmin2}")
    print("=" * 100)

    for snr_idx, ebn0_db in enumerate(snr_range):

        pe = pe_vec[snr_idx]

        # Longueur de trame
        NBits_total = int(np.clip(frame_max_length[snr_idx],
                                  NBits_limit[0], NBits_limit[1]))
        pack        = 10**5
        Nbits_div   = max(1, round(NBits_total / pack))

        total_errors = 0
        total_bits   = 0

        for I in range(1, Nbits_div + 1):
            Nbits = int(np.log2(M_ary)) * int(
                round(NBits_total / Nbits_div) // int(np.log2(M_ary))
            )

            bits   = np.random.randint(0, 2, Nbits)
            bits_m = mapp(bits, M_ary)
            Nbits  = len(bits_m)
            bits_s = np.hstack((

                bits_m[:, None],  # colonne des symboles (N,1)
                np.zeros((len(bits_m), os - 1))  # (N, OS-1) zéros
            )).flatten()

            # Modulation CPM
            #bits_s_ = np.zeros(Nbits * os)
            #bits_s_[::os] = bits_m
            t_seq  = np.arange(0, Nbits - Ts, Ts)  # 0:Ts:Nbits-Ts
            SN     = np.convolve(bits_s, gt)[:len(t_seq)]
            Phi_N  = np.cumsum(SN) * Ts
            mod_signal = np.exp(1j * 2 * np.pi * modulation_index * Phi_N)

            # Canal AWGN
            snr_db_channel = ebn0_db# - 10 * np.log10(os / np.log2(M_ary))
            received = awgn(mod_signal[(pulse_length - 1) * os-1:], snr_db_channel, os)

            # Décodeur Viterbi
            detected_bits = np.zeros(Nbits - decision_delay, dtype=float)
            dbits_idx     = 0
            pathmetric    = np.zeros(states_number)
            survivor_path = np.zeros((states_number, decision_delay), dtype=int)

            for n in range(1, Nbits - pulse_length + 1):
                pathmetric_n = np.zeros(states_number)
                # Métriques de branche cumulées
                bm_cum = np.zeros(branche_number)
                for i in range(branche_number):
                    if n != 1:
                        seg = received[(n - 1) * os: n  * os +1]
                    else:
                        seg = received[0: os+1]
                    # if len(seg) < metric_len:
                    #     seg = np.pad(seg, (0, metric_len - len(seg)))
                    z = np.real(
                        np.exp(-1j * branches[i, 0])
                        * np.trapezoid(seg * np.conj(branch_metrics[i])) * Ts
                    )
                    bm_cum[i] = z

                # ACS (Add-Compare-Select)
                j = 0

                for i in range(0, branche_number, M_ary):
                    candidates = np.array([
                        bm_cum[states_sort[i + k, 2]] + pathmetric[states_sort[i + k, 0]]
                        for k in range(M_ary)
                    ])
                    best_k = np.argmax(candidates)


                    new_state  = states_sort[i + best_k, 1]
                    prev_state = states_sort[i + best_k, 0]
                    pathmetric_n[new_state] = candidates[best_k]

                    if n <= decision_delay:
                        survivor_path[new_state, n - 1] = prev_state
                    else:
                        if i == 0:
                            survivor_path[:, :-1] = survivor_path[:, 1:]
                        survivor_path[new_state, -1] = prev_state
                    j += 1

                pathmetric = pathmetric_n.copy()


                if n > decision_delay:
                    # -------------------- trace back unit ----------------------------
                    max_path = np.max(pathmetric)
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

            # ── Comptage des erreurs
            sl  = 2 + pulse_length
            el  = Nbits - decision_delay - 2 - pulse_length
            sl2 = 1 + pulse_length
            el2 = len(detected_bits) - 3 - pulse_length

            sent = bits_m[sl : el]
            recv = detected_bits[sl2 : el2]

            #n_compare = min(len(sent), len(recv))
            #if n_compare > 0:
            #    errs = int(np.count_nonzero(sent[:n_compare] - recv[:n_compare]))
            #    total_errors += errs
            #    total_bits   += n_compare
            errs = int(np.count_nonzero(sent - recv))
            total_errors += errs
            total_bits   += len(recv)

        # BER simulée pour ce SNR
        ber = total_errors / total_bits if total_bits > 0 else np.nan
        ber_sim[snr_idx] = ber

        error    = abs(ber - pe) if not np.isnan(ber) else np.nan
        mc_iter  = Nbits_div

        print(
            f"Eb/N0 = {ebn0_db:2d} dB | "
            f"BER = {ber:.5e} | "
            f"BER_theo = {pe:.5e} | "
            f"Erreur = {error:.5e} | "
            f"Errors = {total_errors} | "
            f"Iterations = {mc_iter}"
        )

    return snr_range, ber_sim, ber_theory



#  Main

if __name__ == "__main__":

    ebn0_range, ber_sim, ber_theory = run_simulation()

    # Sécurité numérique
    ber_sim    = np.maximum(ber_sim,    1e-12)
    ber_theory = np.maximum(ber_theory, 1e-12)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(
        ebn0_range, ber_sim,
        marker='o', linewidth=2,
        label='Monte Carlo – Viterbi MLSE'
    )
    ax.semilogy(
        ebn0_range, ber_theory,
        linestyle='--', linewidth=2,
        label=r'Theoretical Bound ($d^2_{min}=1.78$)'
    )

    ax.set_xlabel("Eb/N0 (dB)", fontsize=12)
    ax.set_ylabel("Bit Error Rate (BER)", fontsize=12)
    ax.set_title("BER Performance of GMSK over AWGN Channel (Monte Carlo)", fontsize=13)
    ax.grid(True, which='both', linestyle='--', linewidth=0.6)
    ax.set_ylim(1e-6, 1)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.show()


fin = time.time()

duree = fin - debut
print(f"Durée de la simulation : {duree:.3f} secondes")
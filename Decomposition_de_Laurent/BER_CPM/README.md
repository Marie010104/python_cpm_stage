viterbi_laurent

# Récepteur de Laurent (AMP) — GMSK — Simulation BER

## Description
Ce script implémente un récepteur MLSE (Maximum Likelihood Sequence Estimation) simplifié pour une modulation GMSK, basé sur l'approximation à deux composantes de la décomposition de Laurent (`C0 + C1`). Il simule le taux d'erreur binaire (BER) par méthode de Monte Carlo sur une plage de `Eb/N0` et le compare à la courbe théorique.

## Fonctionnement
1. **Impulsion & décomposition de Laurent** : génération de l'impulsion gaussienne GMSK et calcul des impulsions de Laurent `g_0`, `g_1` ainsi que leurs autocorrélations (`pulse_autocorr`).
2. **Treillis CPM** : construction des états et transitions du treillis (`states_branches`, `states_transition`) en fonction de l'indice de modulation `h` et de la longueur `L`.
3. **Pseudo-symboles et biais Ungerboeck** : calcul des pseudo-symboles `a0`/`a1` par branche (`laurent_pseudo_symbols`) et du biais de métrique associé (algorithme d'Ungerboeck).
4. **Filtrage adapté** : corrélation du signal reçu avec `g0` et `g1` (`matched_filter_outputs`).
5. **Décodage Viterbi** (`run_viterbi`) : parcours du treillis avec retard de décision (`decision_delay`), traceback et sortie des bits détectés.
6. **Simulation BER** (`simulate_ber`) : génération de trames bruitées (AWGN), décodage, comptage des erreurs par blocs jusqu'à atteindre un nombre minimal d'erreurs ou un nombre maximal de bits.
7. **Comparaison** avec la BER théorique (`ber_theoretical`, bornée par la distance minimale `dmin2`).

## Paramètres principaux
| Paramètre | Description | Valeur |
|---|---|---|
| `L` | Longueur de l'impulsion | 3 |
| `h` | Indice de modulation | 0.5 |
| `os` | Facteur de suréchantillonnage | 8 |
| `M_ary` | Ordre de la modulation | 2 |
| `decision_delay` | Profondeur de décision du Viterbi | 30–50 |
| `ebn0_range` | Plage de Eb/N0 simulée (dB) | 0 à 10 |

## Dépendances
- numpy
- scipy (`erfc`)
- itertools, fractions (bibliothèque standard)
- matplotlib

## Utilisation
```bash
python laurent_viterbi_gmsk.py
```

## Sorties
- Affichage console du BER simulé vs théorique pour chaque valeur de `Eb/N0`.
- Graphique semi-logarithmique BER vs `Eb/N0` (courbe simulée et courbe théorique).
- Temps d'exécution total affiché en fin de script.


viterbi_laurent_RC

  # Récepteur de Laurent (AMP) — Impulsion RC — Simulation BER

## Description
Variante du récepteur MLSE basé sur la décomposition de Laurent (`C0 + C1`), utilisant une impulsion de mise en forme en **cosinus surélevé (Raised Cosine, RC)** au lieu de l'impulsion gaussienne GMSK. Le script simule le BER par Monte Carlo et le compare à une courbe théorique.

## Fonctionnement
Identique à la version GMSK (voir `README_laurent_viterbi_gmsk.md`), à la différence près que :
- l'impulsion de mise en forme est générée par `create_rc_pulse` (cosinus surélevé) plutôt que par une gaussienne,
- l'interpolation `q_interp` se fait directement sur `[0, L]` (l'impulsion RC n'est pas centrée comme l'impulsion GMSK, donc pas de décalage de `L/2`),
- la distance minimale théorique `dmin2` utilisée pour la BER théorique est différente (adaptée à l'impulsion RC).

Les étapes principales restent : construction du treillis CPM, calcul des pseudo-symboles et du biais Ungerboeck, filtrage adapté, décodage Viterbi, simulation Monte Carlo du BER.

## Paramètres principaux
| Paramètre | Description | Valeur |
|---|---|---|
| `L` | Longueur de l'impulsion | 3 |
| `h` | Indice de modulation | 0.5 |
| `os` | Facteur de suréchantillonnage | 8 |
| `M_ary` | Ordre de la modulation | 2 |
| `dmin2` | Distance minimale (BER théorique) | 1.9 |
| `ebn0_range` | Plage de Eb/N0 simulée (dB) | 0 à 10 |

## Dépendances
- numpy
- scipy (`erfc`, `cumulative_trapezoid`)
- itertools, fractions (bibliothèque standard)
- matplotlib

## Utilisation
```bash
python laurent_viterbi_rc.py
```

## Sorties
- Affichage console du BER simulé vs théorique pour chaque valeur de `Eb/N0`.
- Graphique semi-logarithmique BER vs `Eb/N0` (« Récepteur de Laurent -- Impulsion RC »).
- Temps d'exécution total affiché en fin de script.

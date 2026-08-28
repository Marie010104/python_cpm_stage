Signal_generator_GMSK



# Générateur de signal GMSK — Export fichiers texte

## Description
Ce script génère une trame GMSK complète (pilotes + séquence d'apprentissage + données aléatoires), la module en GMSK, puis exporte le signal transmis ainsi que les symboles et les bits associés dans des fichiers texte (utilisables par exemple pour une simulation matérielle ou un test avec un instrument externe).

## Fonctionnement
1. Génération des bits de données (`genBin`) et mapping en symboles bipolaires (`mapp`).
2. Construction de la trame : pilotes (`genPilotes`) + séquence d'apprentissage (`genTr`) + données.
3. Ajout de zéros de retard pour l'alignement avec le filtre de mise en forme.
4. Génération de l'impulsion gaussienne GMSK (`create_gmsk_pulse`) et suréchantillonnage de la trame.
5. Modulation GMSK par intégration de phase (`np.cumsum`) et calcul du signal complexe transmis.
6. Export du signal en fichier texte avec échantillons I/Q entrelacés, séparés par des tabulations.
7. Export séparé des symboles émis et des bits sources.
8. Tracé de la phase instantanée dépliée `φ(t)` du signal transmis.

## Paramètres principaux
| Paramètre | Description | Valeur |
|---|---|---|
| `L` | Longueur de l'impulsion | 4 |
| `os` | Facteur de suréchantillonnage | 8 |
| `M` | Ordre de la modulation | 2 |
| `Nbits` | Nombre de bits de données par trame | 16384 |
| `Npilotes` | Nombre de symboles pilotes | 64 |
| `modulation_index` | Indice de modulation `h` | 0.5 |
| `trames` | Nombre de trames générées | 1 |

## Fichiers de sortie
| Fichier | Contenu |
|---|---|
| `SignalFichier_Test_gmsk_L=3` | Échantillons I/Q du signal transmis (entrelacés, séparés par tabulations) |
| `SymbolesFichier_Test_gmsk_L=3` | Symboles émis (pilotes + Tr + données) |
| `BitsFichier_Test_gmsk_L=3` | Bits sources générés |

## Dépendances
- numpy
- scipy (`erfc`, `cumulative_trapezoid`)
- matplotlib

## Utilisation
```bash
python gen_signal_gmsk.py
```

## Sortie graphique
Un graphique de la phase instantanée `φ(t)` du signal transmis sur les 500 premières périodes symbole.

(même chose pour signal_generator_RC)

CPM_analyse_GMSK_L=4


# Décodage Viterbi (MLSE) d'un signal GMSK mesuré — L=4

## Description
Ce script charge un signal GMSK **réellement mesuré** (fichier LabVIEW `.lvm`), effectue la synchronisation fréquence/phase à partir des symboles pilotes, décode les données par un récepteur MLSE (algorithme de Viterbi sur le treillis CPM complet, sans approximation de Laurent), puis évalue le BER en fonction du SNR en ajoutant du bruit synthétique au signal mesuré.

## Fonctionnement
1. **Lecture du fichier mesuré** (`read_lvm_iq`) : parsing du fichier `.lvm` (recherche de l'en-tête `X_Value`, extraction des colonnes I et Q) et reconstruction du signal complexe.
2. **Lecture des bits de référence** émis (fichier texte séparé).
3. **Construction du treillis CPM** (`states_branches`, `states_transition`) pour `L=4`, `h=0.5`, `M=2`.
4. **Précalcul des métriques de branche** (`compute_branch_metric`) : forme d'onde de référence associée à chaque branche du treillis.
5. **Boucle sur la plage de SNR** :
   - ajout de bruit AWGN synthétique au signal mesuré (`awgn`),
   - synchronisation fréquence/phase sur les symboles pilotes (`sync_freq_phase`),
   - décodage MLSE par Viterbi (`decode_frame_gmsk`) avec retard de décision,
   - comparaison bits émis / bits détectés et calcul du BER.
6. Tracé de la courbe BER vs `Eb/N0`.

## Paramètres principaux
| Paramètre | Description | Valeur |
|---|---|---|
| `L` | Longueur de l'impulsion | 4 |
| `os` | Facteur de suréchantillonnage | 8 |
| `M` | Ordre de la modulation | 2 |
| `modulation_index` | Indice de modulation `h` | 0.5 |
| `decision_delay` | Profondeur de décision du Viterbi | 50 |
| `Nbits_payload` | Nombre de bits de données par trame | 16384 |
| `snr_range` | Plage de SNR simulée (dB) | 0 à 10 |

## Fichiers d'entrée requis
| Fichier | Contenu |
|---|---|
| `Test_GMSK_L4_1.lvm` | Signal GMSK mesuré (format LabVIEW) |
| `BitsFichier_Test_gmsk_L=4` | Bits de référence émis |

## Dépendances
- numpy, pandas
- scipy (`erfc`, `cumulative_trapezoid`)
- itertools, fractions (bibliothèque standard)
- matplotlib

## Utilisation
```bash
python viterbi_lvm_gmsk_L4.py
```
⚠️ Les fichiers d'entrée (`Test_GMSK_L4_1.lvm` et `BitsFichier_Test_gmsk_L=4`) doivent être présents dans le répertoire d'exécution.

## Sorties
- Affichage console du BER pour chaque valeur de SNR testée.
- Graphique semi-logarithmique BER vs `Eb/N0` (« BER Performance of GMSK »).


(même chose pour CPM_analyse_RC et CPM_analyse_GMSK_L=3)

Pulse_laurent

# Impulsions de Laurent — GMSK

## Description
Ce script calcule et affiche les impulsions de la décomposition de Laurent (AMP — Amplitude Modulated Pulse) d'un signal GMSK. La décomposition de Laurent permet d'exprimer un signal CPM (Continuous Phase Modulation) comme une somme de composantes PAM (Pulse Amplitude Modulation), chacune associée à une impulsion `C_k(t)`.

## Fonctionnement
1. Construction de l'impulsion gaussienne GMSK (`create_gmsk_pulse`) et de son intégrale.
2. Interpolation de l'intégrale `q(t)` pour évaluation en tout point du temps.
3. Calcul des impulsions de Laurent `g_k(t)` (`Laurent_PAM`) à partir des combinaisons binaires possibles (`dec2bin`) et de la fonction `s(t)` (`s_func`).
4. Affichage des `2^(L-1)` impulsions obtenues sur un même graphique.

## Paramètres principaux
| Paramètre | Description | Valeur |
|---|---|---|
| `L` | Longueur de l'impulsion (mémoire du CPM) | 3 |
| `h` | Indice de modulation | 0.5 |
| `Ns` | Facteur de suréchantillonnage | 8 |
| `M`  | Ordre de la modulation | 2 |
| `BT` | Produit bande-temps du filtre gaussien | 0.3 |

## Dépendances
- numpy
- scipy (`scipy.special.erfc`, `scipy.integrate.cumulative_trapezoid`)
- matplotlib

## Utilisation
```bash
python laurent_pulses_plot.py
```

## Sortie
Un graphique matplotlib affichant les `K = 2^(L-1)` impulsions de Laurent `C_0(t), C_1(t), ...` en fonction du temps normalisé `t/T`.

Signal_reconstitue
# Validation de la décomposition de Laurent — Signal GMSK

## Description
Ce script vérifie numériquement la validité de la décomposition de Laurent d'un signal GMSK en comparant :
- le signal GMSK généré **directement** (par intégration de phase classique),
- le signal **reconstruit** à partir des composantes de Laurent (`C0` seul, `C0+C1`, puis toutes les composantes).

Il quantifie l'erreur relative entre les deux approches et affiche les composantes I/Q des signaux superposés.

## Fonctionnement
1. Génération d'une trame complète : symboles pilotes (`genPilotes`), séquence d'apprentissage (`genTr`) et données aléatoires (`genBin` + `mapp`).
2. Construction de l'impulsion gaussienne GMSK et des impulsions de Laurent associées (`Laurent_PAM`, `s_func`, `dec2bin`).
3. Calcul des pseudo-symboles `a_k(t)` pour chaque composante de Laurent à partir de la phase cumulée.
4. Reconstruction progressive du signal par sommation des contributions de chaque composante (suréchantillonnage + convolution).
5. Génération du signal de référence par la méthode directe (intégration de phase classique).
6. Calcul de l'erreur relative (norme L2) entre signal direct et reconstructions partielles.
7. Tracé comparatif des parties réelle (I) et imaginaire (Q) des signaux.

## Paramètres principaux
| Paramètre | Description | Valeur |
|---|---|---|
| `L` | Longueur de l'impulsion | 3 |
| `h` | Indice de modulation | 0.5 |
| `os_` | Facteur de suréchantillonnage | 8 |
| `M` | Ordre de la modulation | 2 |
| `Nbits` | Nombre de bits de données | 512 |
| `Npilotes` | Nombre de symboles pilotes | 64 |

## Dépendances
- numpy
- scipy (`erfc`, `cumulative_trapezoid`)
- matplotlib

## Utilisation
```bash
python laurent_validation_gmsk.py
```

## Sorties
- Affichage console de l'erreur relative pour chaque reconstruction (`C0 seul`, `C0 + C1`, toutes les composantes).
- Deux graphiques comparant le signal direct et les reconstructions de Laurent sur les composantes I et Q.

import pandas as pd
import matplotlib.pyplot as plt

# Lecture des fichiers
python_data = pd.read_csv("RC_L=3_python.txt")
matlab_data = pd.read_csv("RC_L=3_matlab.txt")

#python_data_exp = pd.read_csv("GMSK_experimental_L=3.txt")
plt.figure(figsize=(8,6))

# Courbe Python
plt.semilogy(
    python_data["EbN0_dB"],
    python_data["BER"],
    'o-',
    linewidth=2,
    label="Courbe Python"
)

# Courbe MATLAB
plt.semilogy(
    matlab_data["EbN0_dB"],
    matlab_data["BER"],
    's-',
    linewidth=2,
    label="Courbe MATLAB"
)
'''
#courbe expérimentale

plt.semilogy(
    python_data_exp["EbN0_dB"],
    python_data_exp["BER"],
    'r-',
    linewidth=2,
    label="Courbe expérimentale"
)

'''
# Courbe théorique
plt.semilogy(
    python_data["EbN0_dB"],
    python_data["BER_theo"],
    '--',
    linewidth=2,
    label="Courbe théorique"
)



plt.grid(True, which="both", linestyle=":")
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BER")
plt.title("Comparaison des performances RC")
plt.legend()
plt.tight_layout()
plt.show()

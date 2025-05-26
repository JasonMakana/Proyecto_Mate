#M = data[(data['label'] == 'male')]
#B = data[(data['label'] == 'female')]
#!/usr/bin/env python
# coding: utf-8

# In[3]:


# In[4]:


from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# In[5]:

CH1 = pd.read_excel('C:/Users/jason/Documents/Universidad/Semestre 4/Mate. Aplic. Comun/CH1_PALABRAS_NYP.xlsx')
CH2 = pd.read_excel('C:/Users/jason/Documents/Universidad/Semestre 4/Mate. Aplic. Comun/CH2_PALABRAS_NYP.xlsx')
CH3 = pd.read_excel('C:/Users/jason/Documents/Universidad/Semestre 4/Mate. Aplic. Comun/CH3_PALABRAS_NYP.xlsx')
CH4 = pd.read_excel('C:/Users/jason/Documents/Universidad/Semestre 4/Mate. Aplic. Comun/CH4_PALABRAS_NYP.xlsx')
CH5 = pd.read_excel('C:/Users/jason/Documents/Universidad/Semestre 4/Mate. Aplic. Comun/CH5_PALABRAS_NYP.xlsx')
CH6 = pd.read_excel('C:/Users/jason/Documents/Universidad/Semestre 4/Mate. Aplic. Comun/CH6_PALABRAS_NYP.xlsx')
CH7 = pd.read_excel('C:/Users/jason/Documents/Universidad/Semestre 4/Mate. Aplic. Comun/CH7_PALABRAS_NYP.xlsx')
CH8 = pd.read_excel('C:/Users/jason/Documents/Universidad/Semestre 4/Mate. Aplic. Comun/CH8_PALABRAS_NYP.xlsx')

#Aqui la ruta del archivo
#CH1 = pd.read_excel('C:/Users/edgar/Downloads/CH1_PALABRAS_NYP.xlsx')
#CH2 = pd.read_excel('C:/Users/edgar/Downloads/CH2_PALABRAS_NYP.xlsx')
#CH3 = pd.read_excel('C:/Users/edgar/Downloads/CH3_PALABRAS_NYP.xlsx')
#CH4 = pd.read_excel('C:/Users/edgar/Downloads/CH4_PALABRAS_NYP.xlsx')
#CH5 = pd.read_excel('C:/Users/edgar/Downloads/CH5_PALABRAS_NYP.xlsx')
#CH6 = pd.read_excel('C:/Users/edgar/Downloads/CH6_PALABRAS_NYP.xlsx')
#CH7 = pd.read_excel('C:/Users/edgar/Downloads/CH7_PALABRAS_NYP.xlsx')
#CH8 = pd.read_excel('C:/Users/edgar/Downloads/CH8_PALABRAS_NYP.xlsx')

# In[13]:


#señal no suicida
CH1NO_SUICIDA = CH1[CH1.iloc[:, 1] == 0]
CH2NO_SUICIDA = CH2[CH2.iloc[:, 1] == 0]
CH3NO_SUICIDA = CH3[CH3.iloc[:, 1] == 0]
CH4NO_SUICIDA = CH4[CH4.iloc[:, 1] == 0]
CH5NO_SUICIDA = CH5[CH5.iloc[:, 1] == 0]
CH6NO_SUICIDA = CH6[CH6.iloc[:, 1] == 0]
CH7NO_SUICIDA = CH7[CH7.iloc[:, 1] == 0]
CH8NO_SUICIDA = CH8[CH8.iloc[:, 1] == 0]

CH1NO_SUICIDA = CH1NO_SUICIDA.iloc[:, 2:]
CH2NO_SUICIDA = CH2NO_SUICIDA.iloc[:, 2:]
CH3NO_SUICIDA = CH3NO_SUICIDA.iloc[:, 2:]
CH4NO_SUICIDA = CH4NO_SUICIDA.iloc[:, 2:]
CH5NO_SUICIDA = CH5NO_SUICIDA.iloc[:, 2:]
CH6NO_SUICIDA = CH6NO_SUICIDA.iloc[:, 2:]
CH7NO_SUICIDA = CH7NO_SUICIDA.iloc[:, 2:]
CH8NO_SUICIDA = CH8NO_SUICIDA.iloc[:, 2:]

#señales suicidas
CH1SUICIDA = CH1[CH1.iloc[:, 1] == 1]
CH2SUICIDA = CH2[CH2.iloc[:, 1] == 1]
CH3SUICIDA = CH3[CH3.iloc[:, 1] == 1]
CH4SUICIDA = CH4[CH4.iloc[:, 1] == 1]
CH5SUICIDA = CH5[CH5.iloc[:, 1] == 1]
CH6SUICIDA = CH6[CH6.iloc[:, 1] == 1]
CH7SUICIDA = CH7[CH7.iloc[:, 1] == 1]
CH8SUICIDA = CH8[CH8.iloc[:, 1] == 1]

CH1SUICIDA  = CH1SUICIDA .iloc[:, 2:]
CH2SUICIDA  = CH2SUICIDA .iloc[:, 2:]
CH3SUICIDA  = CH3SUICIDA .iloc[:, 2:]
CH4SUICIDA  = CH4SUICIDA .iloc[:, 2:]
CH5SUICIDA  = CH5SUICIDA .iloc[:, 2:]
CH6SUICIDA  = CH6SUICIDA .iloc[:, 2:]
CH7SUICIDA  = CH7SUICIDA .iloc[:, 2:]
CH8SUICIDA  = CH8SUICIDA .iloc[:, 2:]


# In[14]:


import matplotlib.pyplot as plt

x = CH1SUICIDA.iloc[3, :]  # una sola fila = una serie

plt.figure(figsize=(12, 4))
plt.plot(x.values)  # valores de la serie
plt.title('Señal CH1SUICIDA - Fila 1')
plt.xlabel('Índice de columna')
plt.ylabel('Amplitud')
plt.grid(True)
plt.show()


# In[15]:


#Funciona para eliminar los atípicos
def limpiar_senal(signal):
    signal = np.array(signal)
    Q1 = np.percentile(signal, 25)
    Q3 = np.percentile(signal, 75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    signal_cleaned = signal.copy()
    outliers_idx = (signal < lower) | (signal > upper)

    # Interpolación lineal para valores atípicos
    indices = np.arange(len(signal))
    signal_cleaned[outliers_idx] = np.interp(
        indices[outliers_idx], 
        indices[~outliers_idx], 
        signal[~outliers_idx]
    )

    return signal_cleaned


# In[16]:


CH1NO_SUICIDA_LIMPIA = np.array([limpiar_senal(fila.values) for _, fila in CH1NO_SUICIDA.iterrows()])
CH2NO_SUICIDA_LIMPIA = np.array([limpiar_senal(fila.values) for _, fila in CH2NO_SUICIDA.iterrows()])
CH3NO_SUICIDA_LIMPIA = np.array([limpiar_senal(fila.values) for _, fila in CH3NO_SUICIDA.iterrows()])
CH4NO_SUICIDA_LIMPIA = np.array([limpiar_senal(fila.values) for _, fila in CH4NO_SUICIDA.iterrows()])
CH5NO_SUICIDA_LIMPIA = np.array([limpiar_senal(fila.values) for _, fila in CH5NO_SUICIDA.iterrows()])
CH6NO_SUICIDA_LIMPIA = np.array([limpiar_senal(fila.values) for _, fila in CH6NO_SUICIDA.iterrows()])
CH7NO_SUICIDA_LIMPIA = np.array([limpiar_senal(fila.values) for _, fila in CH7NO_SUICIDA.iterrows()])
CH8NO_SUICIDA_LIMPIA = np.array([limpiar_senal(fila.values) for _, fila in CH8NO_SUICIDA.iterrows()])



CH1SUICIDA_LIMPIA = np.array([limpiar_senal(fila.values) for _, fila in CH1SUICIDA.iterrows()])
CH2SUICIDA_LIMPIA = np.array([limpiar_senal(fila.values) for _, fila in CH2SUICIDA.iterrows()])
CH3SUICIDA_LIMPIA = np.array([limpiar_senal(fila.values) for _, fila in CH3SUICIDA.iterrows()])
CH4SUICIDA_LIMPIA = np.array([limpiar_senal(fila.values) for _, fila in CH4SUICIDA.iterrows()])
CH5SUICIDA_LIMPIA = np.array([limpiar_senal(fila.values) for _, fila in CH5SUICIDA.iterrows()])
CH6SUICIDA_LIMPIA = np.array([limpiar_senal(fila.values) for _, fila in CH6SUICIDA.iterrows()])
CH7SUICIDA_LIMPIA = np.array([limpiar_senal(fila.values) for _, fila in CH7SUICIDA.iterrows()])
CH8SUICIDA_LIMPIA = np.array([limpiar_senal(fila.values) for _, fila in CH8SUICIDA.iterrows()])



print(CH1SUICIDA_LIMPIA)


# In[17]:


CH1=np.vstack([CH1NO_SUICIDA_LIMPIA,CH1SUICIDA_LIMPIA])
CH2=np.vstack([CH2NO_SUICIDA_LIMPIA,CH2SUICIDA_LIMPIA])
CH3=np.vstack([CH3NO_SUICIDA_LIMPIA,CH3SUICIDA_LIMPIA])
CH4=np.vstack([CH4NO_SUICIDA_LIMPIA,CH4SUICIDA_LIMPIA])
CH5=np.vstack([CH5NO_SUICIDA_LIMPIA,CH5SUICIDA_LIMPIA])
CH6=np.vstack([CH6NO_SUICIDA_LIMPIA,CH6SUICIDA_LIMPIA])
CH7=np.vstack([CH7NO_SUICIDA_LIMPIA,CH7SUICIDA_LIMPIA])
CH8=np.vstack([CH8NO_SUICIDA_LIMPIA,CH8SUICIDA_LIMPIA])

print(CH1SUICIDA_LIMPIA.shape)                                                                      


# In[18]:


CH1_1 = CH1
CH2_1 = CH2
CH3_1 = CH3
CH4_1 = CH4
CH5_1 = CH5
CH6_1 = CH6
CH7_1 = CH7
CH8_1 = CH8


# In[19]:


CH1_POSITIVAS = CH1[1:550, :]
CH2_POSITIVAS = CH2[1:550, :]
CH3_POSITIVAS= CH3[1:550, :]
CH4_POSITIVAS = CH4[1:550, :]
CH5_POSITIVAS = CH5[1:550, :]
CH6_POSITIVAS = CH6[1:550, :]
CH7_POSITIVAS = CH7[1:550, :]
CH8_POSITIVAS = CH8[1:550, :]

CH1_NEGATIVAS = CH1[551:1099, :]
CH2_NEGATIVAS = CH2[551:1099, :]
CH3_NEGATIVAS = CH3[551:1099, :]
CH4_NEGATIVAS = CH4[551:1099, :]
CH5_NEGATIVAS = CH5[551:1099, :]
CH6_NEGATIVAS = CH6[551:1099, :]
CH7_NEGATIVAS = CH7[551:1099, :]
CH8_NEGATIVAS = CH8[551:1099, :]

# === Agrupar canales por hemisferio y emoción ===

DERECHO_POSITIVOS = np.vstack([CH1_POSITIVAS, CH2_POSITIVAS, CH3_POSITIVAS, CH4_POSITIVAS])
DERECHO_NEGATIVAS = np.vstack([CH1_NEGATIVAS, CH2_NEGATIVAS, CH3_NEGATIVAS, CH4_NEGATIVAS])

IZQUIERDO_POSITIVOS = np.vstack([CH5_POSITIVAS, CH6_POSITIVAS, CH7_POSITIVAS, CH8_POSITIVAS])
IZQUIERDO_NEGATIVAS = np.vstack([CH5_NEGATIVAS, CH6_NEGATIVAS, CH7_NEGATIVAS, CH8_NEGATIVAS])

# === Promediar señales de hemisferio izquierdo y derecho ===

DERECHO = np.vstack([CH1_1, CH2_1, CH3_1, CH4_1])
IZQUIERDO = np.vstack([CH5_1, CH6_1, CH7_1, CH8_1])

# Promedio general
promedio_izquierdo = np.mean(IZQUIERDO, axis=0)
promedio_derecho = np.mean(DERECHO, axis=0)

# Promedios por emoción
promedio_izquierdo_positivas = np.mean(IZQUIERDO_POSITIVOS, axis=0)
promedio_izquierdo_negativas = np.mean(IZQUIERDO_NEGATIVAS, axis=0)
promedio_derecho_positivas = np.mean(DERECHO_POSITIVOS, axis=0)
promedio_derecho_negativas = np.mean(DERECHO_NEGATIVAS, axis=0)

# === Promedios por canal CH1 a CH8 ===

promedio_ch1_positivas = np.mean(CH1_POSITIVAS, axis=0)
promedio_ch1_negativas = np.mean(CH1_NEGATIVAS, axis=0)

promedio_ch2_positivas = np.mean(CH2_POSITIVAS, axis=0)
promedio_ch2_negativas = np.mean(CH2_NEGATIVAS, axis=0)

promedio_ch3_positivas = np.mean(CH3_POSITIVAS, axis=0)
promedio_ch3_negativas = np.mean(CH3_NEGATIVAS, axis=0)

promedio_ch4_positivas = np.mean(CH4_POSITIVAS, axis=0)
promedio_ch4_negativas = np.mean(CH4_NEGATIVAS, axis=0)

promedio_ch5_positivas = np.mean(CH5_POSITIVAS, axis=0)
promedio_ch5_negativas = np.mean(CH5_NEGATIVAS, axis=0)

promedio_ch6_positivas = np.mean(CH6_POSITIVAS, axis=0)
promedio_ch6_negativas = np.mean(CH6_NEGATIVAS, axis=0)

promedio_ch7_positivas = np.mean(CH7_POSITIVAS, axis=0)
promedio_ch7_negativas = np.mean(CH7_NEGATIVAS, axis=0)

promedio_ch8_positivas = np.mean(CH8_POSITIVAS, axis=0)
promedio_ch8_negativas = np.mean(CH8_NEGATIVAS, axis=0)

# === Gráfica de todos los canales ===

tiempo = np.arange(promedio_izquierdo.shape[0])
plt.figure(figsize=(14, 6))

# CH1 y CH2
plt.plot(tiempo, promedio_ch1_positivas, label='CH1_P', color='orange')
plt.plot(tiempo, promedio_ch1_negativas, label='CH1_N', color='navajowhite')
plt.plot(tiempo, promedio_ch2_positivas, label='CH2_P', color='darkorange')
plt.plot(tiempo, promedio_ch2_negativas, label='CH2_N', color='bisque')

# CH3 y CH4
plt.plot(tiempo, promedio_ch3_positivas, label='CH3_P', color='blue')
plt.plot(tiempo, promedio_ch3_negativas, label='CH3_N', color='lightblue')
plt.plot(tiempo, promedio_ch4_positivas, label='CH4_P', color='green')
plt.plot(tiempo, promedio_ch4_negativas, label='CH4_N', color='lightgreen')

# CH5 y CH6
plt.plot(tiempo, promedio_ch5_positivas, label='CH5_P', color='red')
plt.plot(tiempo, promedio_ch5_negativas, label='CH5_N', color='salmon')
plt.plot(tiempo, promedio_ch6_positivas, label='CH6_P', color='purple')
plt.plot(tiempo, promedio_ch6_negativas, label='CH6_N', color='plum')

# CH7 y CH8
plt.plot(tiempo, promedio_ch7_positivas, label='CH7_P', color='teal')
plt.plot(tiempo, promedio_ch7_negativas, label='CH7_N', color='paleturquoise')
plt.plot(tiempo, promedio_ch8_positivas, label='CH8_P', color='brown')
plt.plot(tiempo, promedio_ch8_negativas, label='CH8_N', color='sandybrown')

# Configuración del gráfico
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.title('Promedio de señales por canal (positivas vs negativas)')
plt.legend(loc='upper right', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()

# In[25]:


import numpy as np
import matplotlib.pyplot as plt

fs = 500  # frecuencia de muestreo
freq_min = 0.5
freq_max = 100

# Organizar señales por canal
canales = {
    'CH3': (promedio_ch3_positivas, promedio_ch3_negativas),
    'CH4': (promedio_ch4_positivas, promedio_ch4_negativas),
    #'CH5': (promedio_ch5_positivas, promedio_ch5_negativas),  # si quieres usar
    'CH6': (promedio_ch6_positivas, promedio_ch6_negativas)
}

colores = {
    'positivas': ['blue', 'green', 'purple'],  # asigna colores por canal si quieres
    'negativas': ['lightblue', 'lightgreen', 'plum']
}

fig, axs = plt.subplots(len(canales), 1, figsize=(10, 4*len(canales)), sharex=True)

if len(canales) == 1:
    axs = [axs]  # para que sea iterable si solo 1 canal

for i, (canal, (senal_pos, senal_neg)) in enumerate(canales.items()):
    N = senal_pos.size
    freqs = np.fft.fftfreq(N, d=1/fs)
    idxs_positivas = freqs >= 0
    freqs_pos = freqs[idxs_positivas]

    # FFT positivas
    fft_pos = np.fft.fft(senal_pos)[idxs_positivas]
    mag_pos = np.abs(fft_pos) / N

    # FFT negativas
    fft_neg = np.fft.fft(senal_neg)[idxs_positivas]
    mag_neg = np.abs(fft_neg) / N

    # Filtrar rango de frecuencia
    idxs_rango = (freqs_pos >= freq_min) & (freqs_pos <= freq_max)

    axs[i].plot(freqs_pos[idxs_rango], mag_pos[idxs_rango], label=f'{canal} positivas', color=colores['positivas'][i])
    axs[i].plot(freqs_pos[idxs_rango], mag_neg[idxs_rango], label=f'{canal} negativas', color=colores['negativas'][i])

    axs[i].set_title(f'FFT {canal}')
    axs[i].set_ylabel('Magnitud normalizada')
    axs[i].legend()
    axs[i].grid(True)

axs[-1].set_xlabel('Frecuencia (Hz)')

plt.tight_layout()
plt.show()


# Aplicar la simetria del CH6 Y CH3

# In[38]:


from scipy.signal import welch

# Suponiendo que tienes las señales en arrays 1D:
F3 = promedio_ch3_negativas
F4 = promedio_ch6_positivas
fs = 500

# --- 1. Calcular la potencia con Welch ---
f3_freqs, f3_psd = welch(F3, fs=fs, nperseg=fs*2)  # promedio espectral en ventanas de 2s
f4_freqs, f4_psd = welch(F4, fs=fs, nperseg=fs*2)

# --- 2. Opcional: limitar a una banda (por ejemplo alfa: 8–12 Hz) ---
banda_min = 8
banda_max = 12
idx_banda = (f3_freqs >= banda_min) & (f3_freqs <= banda_max)

# --- 3. Potencia total en la banda seleccionada ---
potencia_f3 = np.sum(f3_psd[idx_banda])
potencia_f4 = np.sum(f4_psd[idx_banda])

# --- 4. Calcular asimetría ---
asimetria = np.log(potencia_f4) - np.log(potencia_f3)

print(f"Asimetría (log(F4) - log(F3)) en banda {banda_min}-{banda_max} Hz: {asimetria:.4f}")



# In[45]:


import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# Diccionario con los datos de cada canal
canales = {
    "CH1": (CH1_POSITIVAS, CH1_NEGATIVAS),
    "CH2": (CH2_POSITIVAS, CH2_NEGATIVAS),
    "CH3": (CH3_POSITIVAS, CH3_NEGATIVAS),
    "CH4": (CH4_POSITIVAS, CH4_NEGATIVAS),
    "CH5": (CH5_POSITIVAS, CH5_NEGATIVAS),
    "CH6": (CH6_POSITIVAS, CH6_NEGATIVAS),
    "CH7": (CH7_POSITIVAS, CH7_NEGATIVAS),
    "CH8": (CH8_POSITIVAS, CH8_NEGATIVAS)
}

# Para guardar resultados
valores_p = {}
significativos = []

def get_alfa_power(senal, fs=500):
    """
    Calcula la potencia en la banda alfa (8-13 Hz) de una señal.
    
    Parámetros:
        senal: array de forma (n_canales, n_muestras) o (n_muestras,)
        fs: frecuencia de muestreo en Hz (por defecto 500)
    
    Retorna:
        potencia_alfa: potencia total en la banda alfa
    """
    from scipy.signal import welch

    # Si es 1D, lo convertimos a 2D
    if senal.ndim == 1:
        senal = senal.reshape(1, -1)

    potencia_alfa = []
    for canal in senal:
        freqs, psd = welch(canal, fs=fs, nperseg=1024)
        idx = (freqs >= 8) & (freqs <= 13)
        potencia = np.trapz(psd[idx], freqs[idx])
        potencia_alfa.append(potencia)

    return np.array(potencia_alfa)

# Evaluar cada canal
for canal, (positivas, negativas) in canales.items():
    p_alpha_pos = get_alfa_power(positivas)
    p_alpha_neg = get_alfa_power(negativas)

    p_valor, _ = ttest_ind(p_alpha_pos, p_alpha_neg)
    valores_p[canal] = p_valor

    if p_valor < 0.05:
        print(f"✅ {canal} distingue emociones (p = {p_valor:.4f})")
        significativos.append(True)
    else:
        print(f"❌ {canal} no distingue emociones (p = {p_valor:.4f})")
        significativos.append(False)

# Visualización
plt.figure(figsize=(10, 5))
plt.bar(valores_p.keys(), valores_p.values(), color=['green' if s else 'red' for s in significativos])
plt.axhline(0.05, color='black', linestyle='--', label='p = 0.05')
plt.title("Test t: Canales que distinguen emociones")
plt.xlabel("Canal EEG")
plt.ylabel("Valor p")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[53]:


from scipy.signal import welch
import numpy as np

def get_beta_power(signals, fs=500):
    """.

    Parámetros:
    - signals: array de forma (n_ensayos, n_muestras)
    - fs: frecuencia de muestreo (por defecto 250 Hz)

    Retorna:
    - beta_power: array de potencias beta, uno por ensayo
    """
    beta_power = []

    for trial in signals:
        freqs, psd = welch(trial, fs=fs, nperseg=fs*2)
        beta_band = (freqs >= 4) & (freqs <= 8)
        power = np.mean(psd[beta_band])
        beta_power.append(power)

    return np.array(beta_power)


# In[55]:




# In[67]:


import numpy as np
from scipy.signal import welch
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# ------------ CONFIGURACIÓN ------------

# Frecuencia de muestreo
FS = 500

# Número de ensayos esperados
NUM_ENSAYOS = 1100

# Rebanado: del 0 al 549 = negativas, del 550 al 1099 = positivas
NEG = slice(0, 550)
POS = slice(550, 1100)

# ------------ FUNCIÓN DE POTENCIA BETA ------------

def get_beta_power(signals, fs=FS):
    """
    Calcula la potencia media en la banda beta (13–30 Hz) para cada ensayo.
    Entrada: signals → array de forma (n_ensayos, n_muestras)
    Salida: array de potencias beta (uno por ensayo)
    """
    beta_power = []
    for trial in signals:
        freqs, psd = welch(trial, fs=fs, nperseg=fs*2)
        beta_band = (freqs >= 13) & (freqs <= 30)
        power = np.mean(psd[beta_band])
        beta_power.append(power)
    return np.array(beta_power)

# ------------ PARES DE CANALES ------------

pares = [
    ("CH1", "CH8"),
    ("CH2", "CH7"),
    ("CH3", "CH6"),
    ("CH4", "CH5")
]

# ------------ DICCIONARIO DE CANALES ------------

# Asegúrate de tener definidos estos arrays de forma (1100, muestras)
datos = {
    "CH1": CH1_POSITIVAS,
    "CH2": CH2_POSITIVAS, 
    "CH3": CH3_POSITIVAS, 
    "CH4": CH4_POSITIVAS, 
    "CH5": CH5_POSITIVAS, 
    "CH6": CH6_POSITIVAS, 
    "CH7": CH7_POSITIVAS, 
    "CH8": CH8_POSITIVAS, 
}

# ------------ ANÁLISIS DE ASIMETRÍA ------------

valores_p = []
etiquetas = []

for izq, der in pares:
    print(f"\n🔍 Analizando par: {izq}-{der}")

    # Obtener potencias beta
    beta_izq_pos = get_beta_power(datos[izq][POS])
    beta_der_pos = get_beta_power(datos[der][POS])
    beta_izq_neg = get_beta_power(datos[izq][NEG])
    beta_der_neg = get_beta_power(datos[der][NEG])

    # Evitar log(0) usando un valor mínimo pequeño
    eps = 1e-10
    beta_izq_pos = np.maximum(beta_izq_pos, eps)
    beta_der_pos = np.maximum(beta_der_pos, eps)
    beta_izq_neg = np.maximum(beta_izq_neg, eps)
    beta_der_neg = np.maximum(beta_der_neg, eps)

    # Calcular asimetría: log(derecho) - log(izquierdo)
    asim_pos = np.log(beta_der_pos) - np.log(beta_izq_pos)
    asim_neg = np.log(beta_der_neg) - np.log(beta_izq_neg)

    # Validación: evitar NaNs
    if not (np.all(np.isfinite(asim_pos)) and np.all(np.isfinite(asim_neg))):
        print("⚠️ Datos inválidos en logaritmos. Se omite este par.")
        valores_p.append(np.nan)
        etiquetas.append(f"{izq}-{der}")
        continue

    # Test t
    p, _ = ttest_ind(asim_pos, asim_neg)
    valores_p.append(p)
    etiquetas.append(f"{izq}-{der}")

    if p < 0.05:
        print(f"✅ Asimetría BETA {izq}-{der} distingue emociones (p = {p:.4f})")
    else:
        print(f"❌ Asimetría BETA {izq}-{der} NO distingue emociones (p = {p:.4f})")

# ------------ VISUALIZACIÓN ------------

plt.figure(figsize=(8, 4))
colors = ['green' if (p < 0.05) else 'red' for p in valores_p]
plt.bar(etiquetas, valores_p, color=colors)
plt.axhline(0.05, linestyle='--', color='black', label='p = 0.05')
plt.title("Asimetría Beta por par de canales")
plt.ylabel("Valor p")
plt.xlabel("Pares de canales")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[66]:


import numpy as np
from scipy.signal import welch
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# ------------ CONFIGURACIÓN ------------

# Frecuencia de muestreo
FS = 500

# Número de ensayos esperados
NUM_ENSAYOS = 1100

# Rebanado: del 0 al 549 = negativas, del 550 al 1099 = positivas
NEG = slice(0, 550)
POS = slice(550, 1100)

# ------------ FUNCIÓN DE POTENCIA BETA ------------

def get_beta_power(signals, fs=FS):
    """
    Calcula la potencia media en la banda beta (13–30 Hz) para cada ensayo.
    Entrada: signals → array de forma (n_ensayos, n_muestras)
    Salida: array de potencias beta (uno por ensayo)
    """
    beta_power = []
    for trial in signals:
        freqs, psd = welch(trial, fs=fs, nperseg=fs*2)
        beta_band = (freqs >= 8) & (freqs <= 13)
        power = np.mean(psd[beta_band])
        beta_power.append(power)
    return np.array(beta_power)

# ------------ PARES DE CANALES ------------

pares = [
    ("CH1", "CH8"),
    ("CH2", "CH7"),
    ("CH3", "CH6"),
    ("CH4", "CH5")
]

# ------------ DICCIONARIO DE CANALES ------------

# Asegúrate de tener definidos estos arrays de forma (1100, muestras)
datos = {
    "CH1": CH1,
    "CH2": CH2,
    "CH3": CH3,
    "CH4": CH4,
    "CH5": CH5,
    "CH6": CH6,
    "CH7": CH7,
    "CH8": CH8
}

from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# ------------ ANÁLISIS DE ASIMETRÍA ------------

valores_p = []
etiquetas = []

# Para análisis global
todas_asim_pos = []
todas_asim_neg = []

for izq, der in pares:
    print(f"\n🔍 Analizando par: {izq}-{der}")

    # Obtener potencias beta
    beta_izq_pos = get_beta_power(datos[izq][POS])
    beta_der_pos = get_beta_power(datos[der][POS])
    beta_izq_neg = get_beta_power(datos[izq][NEG])
    beta_der_neg = get_beta_power(datos[der][NEG])

    # Evitar log(0)
    eps = 1e-10
    beta_izq_pos = np.maximum(beta_izq_pos, eps)
    beta_der_pos = np.maximum(beta_der_pos, eps)
    beta_izq_neg = np.maximum(beta_izq_neg, eps)
    beta_der_neg = np.maximum(beta_der_neg, eps)

    # Calcular asimetría
    asim_pos = np.log(beta_der_pos) - np.log(beta_izq_pos)
    asim_neg = np.log(beta_der_neg) - np.log(beta_izq_neg)

    # Guardar para análisis global
    todas_asim_pos.append(asim_pos)
    todas_asim_neg.append(asim_neg)

    # Validación de datos válidos
    if not (np.all(np.isfinite(asim_pos)) and np.all(np.isfinite(asim_neg))):
        print("⚠️ Datos inválidos en logaritmos. Se omite este par.")
        valores_p.append(np.nan)
        etiquetas.append(f"{izq}-{der}")
        continue

    # Test t entre positivas y negativas
    p, _ = ttest_ind(asim_pos, asim_neg)
    valores_p.append(p)
    etiquetas.append(f"{izq}-{der}")

    if p < 0.05:
        print(f"✅ Asimetría BETA {izq}-{der} distingue emociones (p = {p:.4f})")
    else:
        print(f"❌ Asimetría BETA {izq}-{der} NO distingue emociones (p = {p:.4f})")

# ===== Análisis global de asimetría (opcional) =====

# Convertir a arrays planos
asim_pos_all = np.concatenate(todas_asim_pos)
asim_neg_all = np.concatenate(todas_asim_neg)

# Limpiar NaNs
asim_pos_all = asim_pos_all[~np.isnan(asim_pos_all)]
asim_neg_all = asim_neg_all[~np.isnan(asim_neg_all)]

# Test t global
p_global, _ = ttest_ind(asim_pos_all, asim_neg_all)
print(f"\n📊 Test t global entre emociones (TODAS las asimetrías): p = {p_global:.4f}")

# ===== Boxplot global =====
plt.figure(figsize=(6, 4))
plt.boxplot([asim_pos_all, asim_neg_all], labels=['Positivas', 'Negativas'], patch_artist=True,
            boxprops=dict(facecolor='lightblue'), medianprops=dict(color='black'))
plt.title("Distribución global de asimetría (log(beta_der) - log(beta_izq))")
plt.ylabel("Asimetría")
plt.grid(True)
plt.tight_layout()
plt.show()
    
# ------------ VISUALIZACIÓN ------------

plt.figure(figsize=(8, 4))
colors = ['green' if (p < 0.05) else 'red' for p in valores_p]
plt.bar(etiquetas, valores_p, color=colors)
plt.axhline(0.05, linestyle='--', color='black', label='p = 0.05')
plt.title("Asimetría Beta por par de canales")
plt.ylabel("Valor p")
plt.xlabel("Pares de canales")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[72]:


# Eje de tiempo (1701 puntos)
tiempo = np.arange(promedio_izquierdo.shape[0])

# Crear figura
plt.figure(figsize=(10, 4))

# Graficar ambos promedios
plt.plot(tiempo, promedio_izquierdo, label='Hemisferio Izquierdo_emociones positivas', color='blue')
plt.plot(tiempo, promedio_derecho, label='Hemisferio Derecho-emociones negativas', color='red')

# Etiquetas y leyenda
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.title('Promedio de señales por hemisferio')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Mostrar la figura
plt.show()


# In[73]:



# Parámetros
fs = 500  # frecuencia de muestreo (ajusta según tus datos)
N = promedio_izquierdo.size

# FFT de las señales promedio
fft_izquierdo = np.fft.fft(promedio_izquierdo)
fft_derecho = np.fft.fft(promedio_derecho)

# Frecuencias asociadas
freqs = np.fft.fftfreq(N, d=1/fs)

# Solo frecuencias positivas
idxs_positivas = freqs >= 0
freqs = freqs[idxs_positivas]
fft_izquierdo = fft_izquierdo[idxs_positivas]
fft_derecho = fft_derecho[idxs_positivas]

# Magnitud normalizada
mag_izquierdo = np.abs(fft_izquierdo) / N
mag_derecho = np.abs(fft_derecho) / N

# Filtrar para rango 0.5 Hz a 100 Hz
freq_min = 0.5
freq_max = 100
idxs_rango = (freqs >= freq_min) & (freqs <= freq_max)

# Graficar solo el rango seleccionado
plt.figure(figsize=(10,4))
plt.plot(freqs[idxs_rango], mag_izquierdo[idxs_rango], label='FFT Izquierdo', color='blue')
plt.plot(freqs[idxs_rango], mag_derecho[idxs_rango], label='FFT Derecho', color='red')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud')
plt.title(f'Espectro de frecuencia ({freq_min} - {freq_max} Hz)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[49]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, norm

# Señales
senal_izq = promedio_izquierdo
senal_der = promedio_derecho


# --- 2. Visualización de ambas señales ---
plt.figure(figsize=(12, 5))

# IZQUIERDO
plt.subplot(1, 2, 1)
mu_izq, std_izq = norm.fit(senal_izq)
plt.hist(senal_izq, bins=50, density=True, alpha=0.6, color='blue', label='Histograma Izq')
x = np.linspace(min(senal_izq), max(senal_izq), 100)
plt.plot(x, norm.pdf(x, mu_izq, std_izq), 'r', lw=2, label='Normal ajustada')
plt.title('Distribución Hemisferio Izquierdo')
plt.xlabel('Valor')
plt.ylabel('Densidad')
plt.legend()
plt.grid(True)

# DERECHO
plt.subplot(1, 2, 2)
mu_der, std_der = norm.fit(senal_der)
plt.hist(senal_der, bins=50, density=True, alpha=0.6, color='green', label='Histograma Der')
x = np.linspace(min(senal_der), max(senal_der), 100)
plt.plot(x, norm.pdf(x, mu_der, std_der), 'r', lw=2, label='Normal ajustada')
plt.title('Distribución Hemisferio Derecho')
plt.xlabel('Valor')
plt.ylabel('Densidad')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# In[50]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, norm

# Señales promedio
senal_izq = promedio_izquierdo
senal_der = promedio_derecho

# --- 1. Prueba de normalidad ---
def evaluar_normalidad(senal, nombre):
    stat, p = shapiro(senal)
    print(f'[{nombre}] Estadística de Shapiro-Wilk: {stat:.4f}, p-valor: {p:.4f}')
    if p > 0.05:
        print(f"✅ [{nombre}] La señal parece seguir una distribución normal (p > 0.05)")
    else:
        print(f"❌ [{nombre}] La señal NO parece seguir una distribución normal (p <= 0.05)")

evaluar_normalidad(senal_izq, "Izquierdo")
evaluar_normalidad(senal_der, "Derecho")

# --- 2. Visualización en una sola gráfica ---
plt.figure(figsize=(10, 5))

# Histograma izquierdo
plt.hist(senal_izq, bins=50, density=True, alpha=0.5, color='blue', label='Izquierdo')

# Histograma derecho
plt.hist(senal_der, bins=50, density=True, alpha=0.5, color='green', label='Derecho')

# Curva normal ajustada izquierdo
mu_izq, std_izq = norm.fit(senal_izq)
x_izq = np.linspace(min(senal_izq), max(senal_izq), 100)
plt.plot(x_izq, norm.pdf(x_izq, mu_izq, std_izq), 'b--', linewidth=2, label='Normal Izquierdo')

# Curva normal ajustada derecho
mu_der, std_der = norm.fit(senal_der)
x_der = np.linspace(min(senal_der), max(senal_der), 100)
plt.plot(x_der, norm.pdf(x_der, mu_der, std_der), 'g--', linewidth=2, label='Normal Derecho')

# Estética
plt.title('Distribución de señales: Izquierdo vs Derecho')
plt.xlabel('Valor')
plt.ylabel('Densidad')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# # Combinar las señales para cada canal
# CH1 = np.vstack([CH1SUICIDA_LIMPIA, CH1NO_SUICIDA_LIMPIA ])
# CH2 = np.vstack([CH2SUICIDA_LIMPIA, CH2NO_SUICIDA_LIMPIA ])
# CH3 = np.vstack([CH3SUICIDA_LIMPIA, CH3NO_SUICIDA_LIMPIA ])
# CH4 = np.vstack([CH4SUICIDA_LIMPIA, CH4NO_SUICIDA_LIMPIA ])
# CH5 = np.vstack([CH5SUICIDA_LIMPIA, CH5NO_SUICIDA_LIMPIA ])
# CH6 = np.vstack([CH6SUICIDA_LIMPIA, CH6NO_SUICIDA_LIMPIA ])
# CH7 = np.vstack([CH7SUICIDA_LIMPIA, CH7NO_SUICIDA_LIMPIA ])
# CH8 = np.vstack([CH8SUICIDA_LIMPIA, CH8NO_SUICIDA_LIMPIA ])
# 
# # Convertir a DataFrame para cada canal (si es necesario para exportar a Excel)
# import pandas as pd
# 
# # Guardar como Excel, por ejemplo, para CH1
# pd.DataFrame(CH1).to_excel('CH1_PALABRAS.xlsx', index=False)
# pd.DataFrame(CH2).to_excel('CH2_PALABRAS.xlsx', index=False)
# pd.DataFrame(CH3).to_excel('CH3_PALABRAS.xlsx', index=False)
# pd.DataFrame(CH4).to_excel('CH4_PALABRAS.xlsx', index=False)
# pd.DataFrame(CH5).to_excel('CH5_PALABRAS.xlsx', index=False)
# pd.DataFrame(CH6).to_excel('CH6_PALABRAS.xlsx', index=False)
# pd.DataFrame(CH7).to_excel('CH7_PALABRAS.xlsx', index=False)
# pd.DataFrame(CH8).to_excel('CH8_PALABRAS.xlsx', index=False)

# In[ ]:





# In[52]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, norm

# Señales promedio
senal_izq = promedio_izquierdo
senal_der = promedio_derecho

# Función para evaluar normalidad
def evaluar_normalidad(senal, nombre):
    stat, p = shapiro(senal)
    print(f'[{nombre}] Estadística de Shapiro-Wilk: {stat:.4f}, p-valor: {p:.4f}')
    if p > 0.05:
        print(f"✅ [{nombre}] La señal parece seguir una distribución normal (p > 0.05)")
    else:
        print(f"❌ [{nombre}] La señal NO parece seguir una distribución normal (p <= 0.05)")

# Evaluar señales base
evaluar_normalidad(senal_izq, "Izquierdo")
evaluar_normalidad(senal_der, "Derecho")

# === NUEVA SEÑAL DE PRUEBA ===
# (puedes cambiar esta señal por cualquier otra)
senal_prueba = IZQUIERDO[20, :]  # ejemplo
media_prueba = np.mean(senal_prueba)

# --- Visualización ---
plt.figure(figsize=(10, 5))

# Histograma izquierdo
plt.hist(senal_izq, bins=50, density=True, alpha=0.5, color='blue', label='Izquierdo')

# Histograma derecho
plt.hist(senal_der, bins=50, density=True, alpha=0.5, color='green', label='Derecho')

# Curva normal ajustada izquierdo
mu_izq, std_izq = norm.fit(senal_izq)
x_izq = np.linspace(min(senal_izq), max(senal_izq), 100)
plt.plot(x_izq, norm.pdf(x_izq, mu_izq, std_izq), 'b--', linewidth=2, label='Normal Izquierdo')

# Curva normal ajustada derecho
mu_der, std_der = norm.fit(senal_der)
x_der = np.linspace(min(senal_der), max(senal_der), 100)
plt.plot(x_der, norm.pdf(x_der, mu_der, std_der), 'g--', linewidth=2, label='Normal Derecho')

# Señal de prueba (media)
plt.axvline(media_prueba, color='red', linestyle='-', linewidth=2, label='Media señal prueba')

# Estética
plt.title('Distribución de señales con señal de prueba')
plt.xlabel('Valor')
plt.ylabel('Densidad')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Clasificación simple basada en cercanía de medias
dist_izq = abs(media_prueba - mu_izq)
dist_der = abs(media_prueba - mu_der)
if dist_izq < dist_der:
    print("🧠 La señal de prueba se parece más al grupo: IZQUIERDO")
else:
    print("🧠 La señal de prueba se parece más al grupo: DERECHO")


# In[ ]:







# === FFT por canal CH1 a CH8 ===

import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

canales = {
    "CH1": (CH1_POSITIVAS, CH1_NEGATIVAS, 'orange', 'navajowhite'),
    "CH2": (CH2_POSITIVAS, CH2_NEGATIVAS, 'darkorange', 'bisque'),
    "CH3": (CH3_POSITIVAS, CH3_NEGATIVAS, 'blue', 'lightblue'),
    "CH4": (CH4_POSITIVAS, CH4_NEGATIVAS, 'green', 'lightgreen'),
    "CH5": (CH5_POSITIVAS, CH5_NEGATIVAS, 'red', 'salmon'),
    "CH6": (CH6_POSITIVAS, CH6_NEGATIVAS, 'purple', 'plum'),
    "CH7": (CH7_POSITIVAS, CH7_NEGATIVAS, 'teal', 'paleturquoise'),
    "CH8": (CH8_POSITIVAS, CH8_NEGATIVAS, 'brown', 'sandybrown'),
}

fs = 500  # frecuencia de muestreo
freq_min = 0.5
freq_max = 100
N = CH1_POSITIVAS.shape[1]
freqs = np.fft.rfftfreq(N, d=1/fs)

fig, axs = plt.subplots(4, 2, figsize=(15, 12))
axs = axs.flatten()

for i, (canal, (positivas, negativas, color_pos, color_neg)) in enumerate(canales.items()):
    # FFT
    fft_pos = np.fft.rfft(np.mean(positivas, axis=0))
    fft_neg = np.fft.rfft(np.mean(negativas, axis=0))

    mag_pos = np.abs(fft_pos) / N
    mag_neg = np.abs(fft_neg) / N

    idx_rango = (freqs >= freq_min) & (freqs <= freq_max)

    axs[i].plot(freqs[idx_rango], mag_pos[idx_rango], label=f'{canal} positivas', color=color_pos)
    axs[i].plot(freqs[idx_rango], mag_neg[idx_rango], label=f'{canal} negativas', color=color_neg)
    axs[i].set_title(f'FFT {canal}')
    axs[i].set_xlabel('Frecuencia (Hz)')
    axs[i].set_ylabel('Magnitud normalizada')
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.show()

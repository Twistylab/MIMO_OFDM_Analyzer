import numpy as np
import commpy as cp

from static_method import rayleigh_impulse_response


M = 16
SNR = 20
N0 = 10**(-SNR/10)

modem = cp.QAMModem(M)

input_bits = np.random.randint(0, 2, size=10000)

input_signal = modem.modulate(input_bits) / np.sqrt(modem.Es)

# Сигнал после прохождения канала
output_signal = np.zeros_like(input_signal)

# h = rayleigh_impulse_response()
h = np.asarray([1., 0.5], dtype='complex')

for k in range(len(input_signal)):
    for m in range(len(h)):
        # Свёртка с ИХ канала
        if k-m >= 0:
            output_signal[k] += h[m] * input_signal[k-m]
        # Добавление шума
    output_signal[k] += (np.random.randn() + 1j * np.random.randn()) * np.sqrt(N0 / 2)

# Сигнал после согласованного фильтра
signal_after_mf = np.zeros_like(output_signal)

# ИХ согласованного фильтра
h_mf = h[::-1].conj()

for k in range(len(output_signal)):
    for m in range(len(h_mf)):
        # Свёртка с ИХ согласованного фильтра
        if k-m >= 0:
            signal_after_mf[k] += h_mf[m] * output_signal[k-m]

# Автокорреляционная функция ИХ канала
R_hh = np.zeros(2*len(h) - 1, dtype='complex')

for k in range(len(R_hh)):
    for m in range(len(h_mf)):
        # Свёртка с ИХ согласованного фильтра
        if (k-m >= 0) and (k-m < len(h)):
            R_hh[k] += h_mf[m] * h[k-m]

delay = np.abs(R_hh).argmax()

psi = np.zeros((len(h), len(h)), dtype='complex')

for l in range(len(h)):
    for j in range(len(h)):
        psi[l, j] += R_hh[len(h) - 1 + (l - j)]

psi += np.eye(len(h)) * N0

p = np.zeros_like(h)

for i in range(len(p)):
    p[i] = R_hh[len(h) - 1 + delay - i]

c_fff = np.linalg.solve(psi, p)

g = np.zeros(2*len(h) - 1, dtype='complex')
for k in range(len(g)):
    for j in range(len(h)):
        if 0 <= k - j < len(h):
            g[k] += c_fff[j] * h[k - j]

c_fbf = g[delay + 1:]

eqv_signal = np.zeros_like(output_signal)

for I in range(len(signal_after_mf)):
    for j in range(len(c_fff)):
        if I-j >= 0:
            eqv_signal[I - delay] += c_fff[j] * signal_after_mf[I - j]

    for j in range(len(c_fbf)):
        if I - j - delay - 1 >= 0:
            eqv_signal[I - delay] -= c_fbf[j] * eqv_signal[I - delay - j - 1]

    eqv_signal[I] = (modem.modulate(
        modem.demodulate(eqv_signal[I] * np.sqrt(modem.Es), demod_type='hard')
    ) / np.sqrt(modem.Es))[0]

output_bits = modem.demodulate(eqv_signal * np.sqrt(modem.Es), demod_type='hard')

ber = np.mean((output_bits + input_bits) % 2)

print(ber)
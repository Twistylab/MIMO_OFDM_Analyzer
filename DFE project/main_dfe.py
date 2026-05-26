import numpy as np
import matplotlib.pyplot as plt

from multithreaded_computing_part import compute_ber_multiprocessing

if __name__ == '__main__':
    decision_methods = {
        'use_dfe_hard': True,
        'use_dfe_fec_hard': True,
        'use_dfe_fec_soft': True,
    }

    const_params = {
        'modulation_size': 16,
        'bits_number': 1000,
        'impulse_response_length': 10,
        'delay': 10,
    }

    ber_without_fec_arr = []
    ber_with_fec_hard_arr = []
    ber_with_fec_soft_arr = []

    SNR_arr = np.arange(0, 15, 2)
    for SNR in SNR_arr:
        D_noise = 10 ** (-SNR / 10)

        ber_results = compute_ber_multiprocessing(
            noise_variance=D_noise,
            **const_params,
            **decision_methods
        )

        ber_without_fec_arr.append(ber_results['ber_dfe_hard'])
        ber_with_fec_hard_arr.append(ber_results['ber_dfe_fec_hard'])
        ber_with_fec_soft_arr.append(ber_results['ber_dfe_fec_soft'])

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.semilogy(SNR_arr, ber_without_fec_arr, color='blue', label='dfe without fec')
    ax.semilogy(SNR_arr, ber_with_fec_hard_arr, color='red', label='dfe with fec hard')
    ax.semilogy(SNR_arr, ber_with_fec_soft_arr, color='green', label='dfe with fec soft')
    ax.set_xlabel(r'SNR, dB', fontsize=20)
    ax.set_ylabel(r'BER', fontsize=20)
    ax.grid()
    ax.legend()
    plt.show()

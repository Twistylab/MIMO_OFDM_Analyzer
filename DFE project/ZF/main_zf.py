import numpy as np
import matplotlib.pyplot as plt

from multithreaded_computing_part_zf import compute_ber_multiprocessing

if __name__ == '__main__':
    decision_methods = {
        'use_zf_hard': True,
        'use_zf_fec_hard': True,
        'use_zf_fec_soft': True,
    }

    const_params = {
        'modulation_size': 16,
        'bits_number': 1000,
        'impulse_response_length': 2,
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

        ber_without_fec_arr.append(ber_results['ber_zf_hard'])
        ber_with_fec_hard_arr.append(ber_results['ber_zf_fec_hard'])
        ber_with_fec_soft_arr.append(ber_results['ber_zf_fec_soft'])

    np.save('results/ber_zf_without_fec_arr', ber_without_fec_arr)
    np.save('results/ber_zf_with_fec_hard_arr', ber_with_fec_hard_arr)
    np.save('results/ber_zf_with_fec_soft_arr', ber_with_fec_soft_arr)
    np.save('results/snr', SNR_arr)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.semilogy(SNR_arr, ber_without_fec_arr, color='blue', label='zf without fec')
    ax.semilogy(SNR_arr, ber_with_fec_hard_arr, color='red', label='zf with fec hard')
    ax.semilogy(SNR_arr, ber_with_fec_soft_arr, color='green', label='zf with fec soft')
    ax.set_xlabel(r'SNR, dB', fontsize=20)
    ax.set_ylabel(r'BER', fontsize=20)
    ax.grid()
    ax.legend()
    plt.show()

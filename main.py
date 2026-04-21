import numpy as np
import matplotlib.pyplot as plt

from time import time

from ber_multithreaded_computing import compute_ber_multiprocessing


if __name__=='__main__':
    start = time()
    M = 16

    num_of_tr = 2
    num_of_rec = 2

    SNR_arr = np.arange(0, 30, 2)

    num_of_iter = 1

    decision_methods = {
        'use_ml_hard': True,
        'use_ml_fec_hard': True,
        'use_ml_fec_soft': True,
        'use_zf_hard': True,
        'use_zf_fec_hard': True,
        'use_zf_fec_soft': True,
        'use_mmse_hard': True,
        'use_mmse_fec_hard': True,
        'use_mmse_fec_soft': True,
    }

    zf_ber = np.zeros((num_of_iter, len(SNR_arr), 3))
    mmse_ber = np.zeros((num_of_iter, len(SNR_arr), 3))
    ml_ber = np.zeros((num_of_iter, len(SNR_arr), 3))

    for i in range(num_of_iter):
        print(f'Итерация {i + 1}')
        for s in range(len(SNR_arr)):
            print(f'\nПодсчёт SNR {SNR_arr[s]} дБ\n')

            ber_results = compute_ber_multiprocessing(
                size_of_modulation=M,
                signal_noise_ratio=SNR_arr[s],
                Nt=num_of_tr,
                Nr=num_of_rec,
                use_awgn=True,
                use_rayleigh_fading_with_mp=True,
                **decision_methods
            )

            ml_ber[i, s] = [
                ber_results['ber_ml_fec_soft'],
                ber_results['ber_ml_fec_hard'],
                ber_results['ber_ml_hard']
            ]

            if num_of_tr <= num_of_rec:
                zf_ber[i, s] = [
                    ber_results['ber_zf_fec_soft'],
                    ber_results['ber_zf_fec_hard'],
                    ber_results['ber_zf_hard']
                ]

                mmse_ber[i, s] = [
                    ber_results['ber_mmse_fec_soft'],
                    ber_results['ber_mmse_fec_hard'],
                    ber_results['ber_mmse_hard']
                ]
            print('\n')

    zf_ber = np.mean(zf_ber, axis=0)
    ml_ber = np.mean(ml_ber, axis=0)
    mmse_ber = np.mean(mmse_ber, axis=0)

    end = time()

    print('Время выполнения: ', round((end - start) / 60, 2), ' минут.')

    fig1 = plt.figure(figsize=(20, 20))

    ax1 = fig1.add_subplot(221)

    ax1.set_title(f'BER vs SNR (MIMO{num_of_tr}x{num_of_rec} QAM{M})\nAll methods', fontsize=16)
    if decision_methods['use_zf_hard']:
        ax1.semilogy(SNR_arr, zf_ber[:, 2], color='orange', linestyle='--', marker='+', label='zf without fec')
    if decision_methods['use_zf_fec_hard']:
        ax1.semilogy(SNR_arr, zf_ber[:, 1], color='lime', linestyle='-.', marker='+', label='zf with fec hard')
    if decision_methods['use_zf_fec_soft']:
        ax1.semilogy(SNR_arr, zf_ber[:, 0], color='darkgreen', linestyle=':', marker='+', label='zf with fec soft')
    if decision_methods['use_ml_hard']:
        ax1.semilogy(SNR_arr, ml_ber[:, 2], color='red', linestyle='--', marker='.', label='ml without fec')
    if decision_methods['use_ml_fec_hard']:
        ax1.semilogy(SNR_arr, ml_ber[:, 1], color='firebrick', linestyle='-.', marker='.', label='ml with fec hard')
    if decision_methods['use_ml_fec_soft']:
        ax1.semilogy(SNR_arr, ml_ber[:, 0], color='black', linestyle=':', marker='.', label='ml with fec soft')
    if decision_methods['use_mmse_hard']:
        ax1.semilogy(SNR_arr, mmse_ber[:, 2], color='aqua', linestyle='--', marker='*', label='mmse without fec')
    if decision_methods['use_mmse_fec_hard']:
        ax1.semilogy(SNR_arr, mmse_ber[:, 1], color='blue', linestyle='-.', marker='*', label='mmse with fec hard')
    if decision_methods['use_mmse_fec_soft']:
        ax1.semilogy(SNR_arr, mmse_ber[:, 0], color='steelblue', linestyle=':', marker='*', label='mmse with fec soft')
    ax1.set_ylabel('BER', fontsize=12)
    ax1.set_xlabel('SNR, dB', fontsize=12)
    ax1.set_ylim(1e-5, 1)
    ax1.legend()
    ax1.grid()

    ax2 = fig1.add_subplot(222)

    ax2.set_title(f'BER vs SNR (MIMO{num_of_tr}x{num_of_rec} QAM{M})\nZero-Forcing', fontsize=16)
    if decision_methods['use_zf_hard']:
        ax2.semilogy(SNR_arr, zf_ber[:, 2], color='orange', linestyle='--', marker='+', label='zf without fec')
    if decision_methods['use_zf_fec_hard']:
        ax2.semilogy(SNR_arr, zf_ber[:, 1], color='lime', linestyle='-.', marker='+', label='zf with fec hard')
    if decision_methods['use_zf_fec_soft']:
        ax2.semilogy(SNR_arr, zf_ber[:, 0], color='darkgreen', linestyle=':', marker='+', label='zf with fec soft')
    ax2.set_ylabel('BER', fontsize=12)
    ax2.set_xlabel('SNR, dB', fontsize=12)
    ax2.set_ylim(1e-5, 1)
    ax2.legend()
    ax2.grid()

    ax3 = fig1.add_subplot(223)

    ax3.set_title(f'BER vs SNR (MIMO{num_of_tr}x{num_of_rec} QAM{M})\nMaximum Likelihood', fontsize=16)
    if decision_methods['use_ml_hard']:
        ax3.semilogy(SNR_arr, ml_ber[:, 2], color='red', linestyle='--', marker='.', label='ml without fec')
    if decision_methods['use_ml_fec_hard']:
        ax3.semilogy(SNR_arr, ml_ber[:, 1], color='firebrick', linestyle='-.', marker='.', label='ml with fec hard')
    if decision_methods['use_ml_fec_soft']:
        ax3.semilogy(SNR_arr, ml_ber[:, 0], color='black', linestyle=':', marker='.', label='ml with fec soft')
    ax3.set_ylabel('BER', fontsize=12)
    ax3.set_xlabel('SNR, dB', fontsize=12)
    ax3.set_ylim(1e-5, 1)
    ax3.legend()
    ax3.grid()

    ax4 = fig1.add_subplot(224)

    ax4.set_title(f'BER vs SNR (MIMO{num_of_tr}x{num_of_rec} QAM{M})\nMinimum Mean Square Error', fontsize=16)
    if decision_methods['use_mmse_hard']:
        ax4.semilogy(SNR_arr, mmse_ber[:, 2], color='aqua', linestyle='--', marker='*', label='mmse without fec')
    if decision_methods['use_mmse_fec_hard']:
        ax4.semilogy(SNR_arr, mmse_ber[:, 1], color='blue', linestyle='-.', marker='*', label='mmse with fec hard')
    if decision_methods['use_mmse_fec_soft']:
        ax4.semilogy(SNR_arr, mmse_ber[:, 0], color='steelblue', linestyle=':', marker='*', label='mmse with fec soft')
    ax4.set_ylabel('BER', fontsize=12)
    ax4.set_xlabel('SNR, dB', fontsize=12)
    ax4.set_ylim(1e-5, 1)
    ax4.legend()
    ax4.grid()
    plt.savefig(f'ber & evm/BER vs SNR All methods.png')

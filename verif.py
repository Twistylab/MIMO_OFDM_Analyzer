import numpy as np
import matplotlib.pyplot as plt

from ber_multithreaded_computing import compute_ber_multiprocessing


if __name__ == '__main__':
    M = 4

    num_of_tr = 1
    num_of_rec = 1

    SNR_arr = np.arange(-10, 7)

    num_of_iter = 1

    decision_methods = {
        'use_ml_hard': False,
        'use_ml_fec_hard': False,
        'use_ml_fec_soft': False,
        'use_zf_hard': True,
        'use_zf_fec_hard': True,
        'use_zf_fec_soft': True,
        'use_mmse_hard': False,
        'use_mmse_fec_hard': False,
        'use_mmse_fec_soft': False,
    }

    zf_metrics = np.zeros((num_of_iter, len(SNR_arr), 3))
    mmse_metrics = np.zeros((num_of_iter, len(SNR_arr), 3))
    ml_metrics = np.zeros((num_of_iter, len(SNR_arr), 3))

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
                use_rayleigh_fading_with_mp=False,
                **decision_methods
            )

            ml_metrics[i, s] = [
                ber_results['ber_ml_fec_soft'],
                ber_results['ber_ml_fec_hard'],
                ber_results['ber_ml_hard']
            ]

            if num_of_tr <= num_of_rec:
                zf_metrics[i, s] = [
                    ber_results['ber_zf_fec_soft'],
                    ber_results['ber_zf_fec_hard'],
                    ber_results['ber_zf_hard']
                ]

                mmse_metrics[i, s] = [
                    ber_results['ber_mmse_fec_soft'],
                    ber_results['ber_mmse_fec_hard'],
                    ber_results['ber_mmse_hard']
                ]

    zf_metrics = np.mean(zf_metrics, axis=0)
    ml_metrics = np.mean(ml_metrics, axis=0)
    mmse_metrics = np.mean(mmse_metrics, axis=0)

    SNR_arr_without_enc = np.load('ber my vs bertool/bertool/snr.npy')
    ber_th_without_enc = np.load('ber my vs bertool/bertool/ber.npy',)
    SNR_arr_with_enc = np.load('ber my vs bertool/bertool/snr_hard.npy')
    ber_th_with_enc = np.load('ber my vs bertool/bertool/ber_hard.npy', )
    SNR_arr_with_enc_soft = np.load('ber my vs bertool/bertool/snr_soft.npy')
    ber_th_with_enc_soft = np.load('ber my vs bertool/bertool/ber_soft.npy', )

    fig1 = plt.figure(figsize=(8, 8))

    ax1 = fig1.add_subplot(111)

    ax1.set_title('BER vs SNR (SISO QAM4)', fontsize=16)
    ax1.semilogy(SNR_arr, zf_metrics[:, 2], color='red', linestyle='', marker='.', label='my without enc')
    ax1.semilogy(SNR_arr, zf_metrics[:, 1], color='aqua', linestyle='', marker='.', label='my with enc')
    ax1.semilogy(SNR_arr, zf_metrics[:, 0], color='lime', linestyle='', marker='.', label='my with enc soft')
    ax1.semilogy(SNR_arr_without_enc[np.where(ber_th_without_enc > 1e-5)],
                 ber_th_without_enc[np.where(ber_th_without_enc > 1e-5)], color='firebrick', linestyle='-',
                 label='bertool without enc')
    ax1.semilogy(SNR_arr_with_enc[np.where(ber_th_with_enc > 1e-5)], ber_th_with_enc[np.where(ber_th_with_enc > 1e-5)],
                 color='blue', linestyle='-', label='bertool with enc')
    ax1.semilogy(SNR_arr_with_enc_soft[np.where(ber_th_with_enc_soft > 1e-5)],
                 ber_th_with_enc_soft[np.where(ber_th_with_enc_soft > 1e-5)], color='green', linestyle='-',
                 label='bertool with enc soft')
    ax1.set_ylabel('BER', fontsize=12)
    ax1.set_xlabel('SNR, dB', fontsize=12)
    ax1.legend()
    ax1.grid()
    plt.savefig(f'ber my vs bertool/my program vs bertool.png')
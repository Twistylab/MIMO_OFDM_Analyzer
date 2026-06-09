import matplotlib.pyplot as plt
import numpy as np


def load_results(eq_type):
    results = {
        'still': np.load(f'{eq_type.upper()}/results/ber_{eq_type}_without_fec_arr.npy'),
        'hard': np.load(f'{eq_type.upper()}/results/ber_{eq_type}_with_fec_hard_arr.npy'),
        'soft': np.load(f'{eq_type.upper()}/results/ber_{eq_type}_with_fec_soft_arr.npy'),
        'snr': np.load(f'{eq_type.upper()}/results/snr.npy'),
    }
    return results


if __name__ == '__main__':
    zf_ber = load_results('zf')
    dfe_ber = load_results('dfe')

    fig = plt.figure(figsize=(16, 10))

    ax = fig.add_subplot(1, 1, 1)

    ax.set_title('DFE vs ZF', fontsize=20)
    ax.semilogy(zf_ber['snr'], zf_ber['still'], color='orange', label='zf still')
    ax.semilogy(zf_ber['snr'], zf_ber['hard'], color='lime', label='zf hard')
    ax.semilogy(zf_ber['snr'], zf_ber['soft'], color='darkgreen', label='zf soft')
    ax.semilogy(dfe_ber['snr'], dfe_ber['still'], color='red', label='dfe still')
    ax.semilogy(dfe_ber['snr'], dfe_ber['hard'], color='firebrick', label='dfe hard')
    ax.semilogy(dfe_ber['snr'], dfe_ber['soft'], color='black', label='dfe soft')
    ax.set_xlabel(r'SNR, dB', fontsize=16)
    ax.set_ylabel(r'BER', fontsize=16)
    ax.grid()
    ax.legend()

    fig.savefig('dfe_vs_zf.png')

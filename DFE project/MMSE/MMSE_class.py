import numpy as np
import commpy as cp

from commpy.channelcoding import Trellis, conv_encode, viterbi_decode


_trellis = Trellis(np.array([7]), np.array([[0o133, 0o171]]))


def generate_impulse_response(impulse_response_length):
    return 10 ** (-np.linspace(0, 10, impulse_response_length) / 20)

def convolutional_matrix(impulse_response):
    impulse_response_length = len(impulse_response)

    conv_matrix = np.zeros((impulse_response_length, impulse_response_length))
    for n in range(impulse_response_length):
        for m in range(n + 1):
            conv_matrix[n, m] = impulse_response[n - m]

    return conv_matrix


class BERAnalyzerWithMMSE(cp.QAMModem):
    def __init__(self, modulation_size, bits_number, noise_variance, impulse_response_length, delay,
                 use_mmse_hard=False, use_mmse_fec_hard=False, use_mmse_fec_soft=False):

        super().__init__(modulation_size)

        self.modulation_size = modulation_size

        self.use_mmse_hard, self.use_mmse_fec_hard, self.use_mmse_fec_soft = use_mmse_hard, use_mmse_fec_hard, use_mmse_fec_soft

        if not (use_mmse_hard or use_mmse_fec_hard or use_mmse_fec_soft):
            raise ValueError("All methods is False. Nothing to do...")

        self.impulse_response_length = impulse_response_length
        self.delay = delay
        self.impulse_response = generate_impulse_response(self.impulse_response_length)

        self.noise_variance = noise_variance
        self.bits_number = bits_number

        self.weights = None
        self.weights_mmse()

        self.input_bits = None
        self.input_bits_enc = None
        self.input_signal = None

        self.output_bits = None
        self.output_bits_enc = None
        self.output_signal = None

        self.equalized_signal = None

        self.evm = []
        self.ber = 0

        self._count_ber = 0
        self._count_data = 0

    def weights_mmse(self):
        conv_matrix = convolutional_matrix(self.impulse_response)
        autocorrelation_matrix = conv_matrix.T @ conv_matrix

        self.weights = np.linalg.inv(
            autocorrelation_matrix + self.noise_variance * np.eye(self.impulse_response_length)) @ np.flip(
            self.impulse_response)

    def apply_multipath_propagation(self):
        length_clear_signal = len(self.input_signal)
        length_isi_signal = length_clear_signal + self.impulse_response_length - 1
        self.output_signal = np.zeros(length_isi_signal, dtype=complex)
        for k in range(length_isi_signal):
            for i in range(self.impulse_response_length):
                # Свёртка с ИХ канала
                if 0 <= k - i <= length_clear_signal - 1:
                    self.output_signal[k] += self.impulse_response[i] * self.input_signal[k - i]
            # Добавление шума
            self.output_signal[k] += (np.random.randn() + 1j * np.random.randn()) * np.sqrt(self.noise_variance / 2)

    def apply_mmse_equalizer(self):
        length_isi_signal = len(self.output_signal)
        self.equalized_signal = np.zeros(length_isi_signal, dtype=complex)
        for k in range(length_isi_signal):
            for i in range(self.impulse_response_length):
                if k - i >= 0:
                    self.equalized_signal[k] += self.weights[i] * self.output_signal[k - i]

    def compute_log_likelihood_ratio(self, entire_signal):
        llr = []

        bits_constellation = self.demodulate(self.constellation, demod_type='hard').reshape(self.modulation_size, int(np.log2(self.modulation_size)))

        for n_signal in range(len(entire_signal)):
            signal_grid = entire_signal[n_signal] * np.ones_like(self.constellation)

            ln_d_grid = np.abs(self.constellation / np.sqrt(self.Es) - signal_grid) ** 2 / (self.noise_variance / 2)

            for bit in range(len(bits_constellation[0])):
                ln_d0 = ln_d_grid[np.where(bits_constellation[:, bit] == 0)]
                sum_d0 = (-1) * np.min(ln_d0) + np.log(np.sum(np.exp(np.min(ln_d0) - ln_d0)))

                ln_d1 = ln_d_grid[np.where(bits_constellation[:, bit] == 1)]
                sum_d1 = (-1) * np.min(ln_d1) + np.log(np.sum(np.exp(np.min(ln_d1) - ln_d1)))

                llr.append(sum_d1 - sum_d0)

        return np.asarray(llr)

    def compute_ber(self, recursion_depth=0) -> float:
        if recursion_depth > 100:
            return self._count_ber / self._count_data

        recursion_depth += 1

        count_dynamic = 0
        sl = 0

        while count_dynamic < 100 and self._count_ber < 100:
            sl += 1

            count_dynamic = np.sum(
                (self.input_bits + self.output_bits)[:sl] % 2)

            if (sl > len(self.output_bits)
                    and (self._count_ber + count_dynamic) < 100):
                self._count_data += len(self.output_bits)
                self._count_ber += count_dynamic
                self.generate_random_transmission()

                return self.compute_ber(recursion_depth)
            elif (self._count_ber + count_dynamic) == 100:
                self._count_data += sl
                self._count_ber += count_dynamic

        return self._count_ber / self._count_data

    def compute_evm(self):
        return np.sqrt(np.mean(np.abs((self.input_signal - self.equalized_signal[self.delay - 1:])) ** 2))

    def generate_random_transmission(self):
        self.input_bits = np.random.randint(0, 2, self.bits_number)

        self.input_bits_enc = conv_encode(self.input_bits, _trellis, termination='cont')

        self.input_signal = self.modulate(self.input_bits_enc) / np.sqrt(self.Es)

        self.apply_multipath_propagation()

        self.apply_mmse_equalizer()

        if self.use_mmse_fec_hard:
            self.output_bits_enc = self.demodulate(self.equalized_signal[self.delay - 1:] * np.sqrt(self.Es), demod_type='hard')
            self.output_bits = viterbi_decode(self.output_bits_enc, _trellis, decoding_type='hard')
        elif self.use_mmse_fec_soft:
            self.output_bits = viterbi_decode(self.compute_log_likelihood_ratio(self.equalized_signal[self.delay - 1:]), _trellis, decoding_type='soft')
        elif self.use_mmse_hard:
            self.input_bits = self.input_bits_enc
            self.output_bits = self.demodulate(self.equalized_signal[self.delay - 1:] * np.sqrt(self.Es), demod_type='hard')

        self.evm.append(self.compute_evm())

    def run_full_analysis(self):
        self.generate_random_transmission()

        self.ber = self.compute_ber()

        self.evm = np.mean(np.asarray(self.evm))

import numpy as np
import commpy as cp

from numpy import ndarray
from commpy.channelcoding import Trellis, conv_encode, viterbi_decode

from static_method import rayleigh_impulse_response


_trellis = Trellis(np.array([7]), np.array([[0o133, 0o171]]))

_frames_number = 10
_band_width = 240e3
_subcarrier_spacing = 15e3

class QAMModemAdapter(cp.QAMModem):
    """
    Адаптер для класса QAMModem из библиотеки CommPy.

    Наследуется от:
        - commpy.QAMModem: класс из библиотеки commpy, предоставляющий базовую
          функциональность QAM модуляции и демодуляции.

    Args:
        size_of_modulation (int): Размер QAM созвездия;

    Attributes:
        M (int): Размер QAM созвездия;
    """
    def __init__(self, size_of_modulation: int) -> None:
        super().__init__(m=size_of_modulation)
        self.M = size_of_modulation

class OFDMMIMOTransceiver:
    """
    Класс для моделирования MIMO-OFDM передатчика и приёмника с учётом реального радиоканала.

    Реализация:
        - Формирование OFDM-символов.
        - Применение IFFT.
        - Вставка пилотных сигналов для оценки канала.
        - Добавление циклического префикса.
        - Прохождение сигнала через канал с Рэлеевским замиранием и многолучевым распространением.
        - Добавление аддитивного белого Гауссова шума (AWGN).
        - Применение FFT.
        - Оценка матрицы канала и извлечение полезного сигнала.
        - OFDM демодуляция.

    Args:
        D_noise (float): Дисперсия АБГШ.
        Nt (int): Количество передающих антенн.
        Nr (int): Количество приёмных антенн.
        use_awgn (bool): Флаг для добавления АБГШ.
        use_rayleigh_fading_with_mp (bool): Флаг для использования рэлеевского канала с многолучевым распространением.

    Attributes:
        number_of_subcarriers (int): Количество поднесущих OFDM.
        number_of_OFDM_symbols (int): Число OFDM-символов.
        length_of_cp (int): Длина циклического префикса.
        pilot_matrix (ndarray): Матрица пилотных сигналов.
        channel_matrix (ndarray): Оценённая матрица канала (размер Nr x Nt).
        output_signal_f (ndarray): Принятый сигнал в частотной области.

    Особенности:
        - Пилоты вставляются перед каждым OFDM-символом для оценки канала.
        - Пилотные и соответствующие им информационные сигналы проходят через
          один и тот же канал.
    """
    def __init__(self, D_noise: float, Nt: int, Nr: int,
                 use_awgn: bool = True, use_rayleigh_fading_with_mp: bool = True,
                 **kwargs) -> None:

        super().__init__(**kwargs)

        self.use_awgn = use_awgn
        self.use_rayleigh_fading_with_mp = use_rayleigh_fading_with_mp

        self.D_noise = D_noise

        if not use_awgn:
            self.D_noise = 1e-10

        self.Nt = Nt
        self.Nr = Nr

        self.number_of_subcarriers = int(_band_width / _subcarrier_spacing)

        self.number_of_OFDM_symbols = 0

        self.impulse_response_length = len(rayleigh_impulse_response())

        self.length_of_cp = self.impulse_response_length - 1

        self.channel_matrix = np.array([])

        self.pilot_matrix = self.generate_pilot_matrix()

        self.output_signal_f = None

    def generate_pilot_matrix(self) -> ndarray:
        """
        Генерация матрицы пилотных сигналов для заданного числа передающих антенн.
        Returns:
            Матрица пилотных сигналов.
        """
        pilot_matrix = np.ones((self.Nt, self.Nt))

        for nt_i in range(self.Nt - 1):
            for nt_j in range(self.Nt - 1, nt_i, -1):
                pilot_matrix[nt_i, nt_j] = 0

        return pilot_matrix

    def apply_rayleigh_fading_with_multipath_propagation(self, signal: ndarray) -> ndarray:
        """
        Применение к сигналу рэлеевского замирания с многолучевым распространением.
        Args:
            signal: Сигнал, переданный в канал.
        Returns:
            Сигнал, прошедший через канал.
        """
        new_signal = np.zeros((self.number_of_OFDM_symbols * (self.Nt + 1),
                               self.number_of_subcarriers + self.length_of_cp,
                               self.Nr), dtype='complex')

        for symb in range(0, (self.Nt + 1) * self.number_of_OFDM_symbols, (self.Nt + 1)):
            # Создание импульсных характеристик для каждого пути распространения
            h = (np.vstack([
                rayleigh_impulse_response() for _ in range(self.Nt * self.Nr)
            ])).T.reshape(self.impulse_response_length, self.Nr, self.Nt)
            for nr in range(self.Nr):
                for n in range(self.number_of_subcarriers + self.length_of_cp):
                    for m in range(self.impulse_response_length):
                        if n - m >= 0:
                            for nt in range(self.Nt):
                                # Применение ИХ к пилотным сигналам
                                for num_p in range(self.Nt):
                                    new_signal[symb + num_p, n, nr] += h[m, nr, nt] * signal[symb + num_p, n - m, nt]

                                # Применение ИХ к информационному сигналу
                                new_signal[symb + self.Nt, n, nr] += h[m, nr, nt] * signal[symb + self.Nt, n - m, nt]
                        else:
                            break

        return new_signal

    def apply_awgn(self, signal: ndarray) -> ndarray:
        """
        Добавление к информационному сигналу Аддитивного Белого Гауссова Шума (АБГШ).
        Args:
            signal: Сигнал, подлежащий изменению.
        Returns:
            Зашумленный сигнал.
        """
        noise_shape = (self.number_of_OFDM_symbols, self.number_of_subcarriers, self.Nr)
        noise = ((np.random.randn(*noise_shape)
                  + 1j * np.random.randn(*noise_shape))
                 * np.sqrt(self.D_noise / 2))

        signal[self.Nt::self.Nt + 1] += np.fft.ifft(noise, axis = 1)

        return signal

    def transmit_and_receive_ofdm(self, input_signal_f: ndarray) -> None:
        """
        Моделирование MIMO-OFDM передатчика и приёмника.
        Args:
            input_signal_f: Исходный информационный сигнал.
        """
        self.number_of_OFDM_symbols = input_signal_f.shape[0] // self.number_of_subcarriers

        # Переход к OFDM (деление на поднесущие частоты)
        input_OFDM_signal_f_without_pilots = input_signal_f.reshape(
            self.number_of_OFDM_symbols,
            self.number_of_subcarriers,
            self.Nt
        )

        # Пилотный сигнал
        pilot_signal = np.asarray([
            np.vstack([self.pilot_matrix[:, nt] for _ in range(self.number_of_subcarriers)]) for nt in range(self.Nt)
        ])

        # Добавление пилотов
        input_OFDM_signal_f_with_pilots = np.zeros((1, self.number_of_subcarriers, self.Nt), dtype=complex)

        for l in range(self.number_of_OFDM_symbols):
            input_OFDM_signal_f_with_pilots = np.concatenate(
                (
                    input_OFDM_signal_f_with_pilots,
                    pilot_signal,
                    np.array([input_OFDM_signal_f_without_pilots[l]])
                ), axis=0
            )

        input_OFDM_signal_f_with_pilots = input_OFDM_signal_f_with_pilots[1:]

        # Обратное быстрое преобразование Фурье
        OFDM_signal_t = np.fft.ifft(input_OFDM_signal_f_with_pilots, axis = 1)

        # Применение эффектов канала
        if self.use_rayleigh_fading_with_mp:
            # Добавление циклического префикса
            OFDM_signal_t_cp = np.concatenate(
                [OFDM_signal_t[:, -self.length_of_cp:], OFDM_signal_t], axis=1
            )

            # Добавление Рэлеевского замирания с многолучевым распространением (эффект межсимвольной интерференции)
            OFDM_signal_t_cp_rf_mp = self.apply_rayleigh_fading_with_multipath_propagation(OFDM_signal_t_cp)

            # Удаление циклического префикса
            OFDM_signal_t = OFDM_signal_t_cp_rf_mp[:, self.length_of_cp:]

        if self.use_awgn:
            # Добавление АБГШ
            OFDM_signal_t = self.apply_awgn(OFDM_signal_t)

        # Прямое быстрое преобразование Фурье
        output_OFDM_signal_f = np.fft.fft(OFDM_signal_t, axis = 1)

        # Удаление пилотов с извлечением матрицы канала
        output_OFDM_signal_f_without_pilots = np.zeros(
            (
                self.number_of_OFDM_symbols,
                self.number_of_subcarriers,
                self.Nr
            ), dtype=complex
        )

        channel_matrix = np.zeros(
            (
                self.number_of_OFDM_symbols,
                self.number_of_subcarriers,
                self.Nr, self.Nt
            ), dtype=complex
        )

        for k in range(self.number_of_OFDM_symbols):
            for l in range(self.number_of_subcarriers):
                channel_matrix[k, l] =  output_OFDM_signal_f[
                    int((self.Nt + 1) * k): int((self.Nt + 1) * k) + self.Nt, l
                ].T @ np.linalg.inv(self.pilot_matrix)
            output_OFDM_signal_f_without_pilots[k] = output_OFDM_signal_f[int((self.Nt + 1) * k) + self.Nt]

        self.channel_matrix = channel_matrix.reshape(self.number_of_OFDM_symbols * self.number_of_subcarriers,
                                                     self.Nr,
                                                     self.Nt)

        # Переход из OFDM
        self.output_signal_f = output_OFDM_signal_f_without_pilots.reshape(
            self.number_of_OFDM_symbols * self.number_of_subcarriers,
            self.Nr
        )


class DataProcessing(OFDMMIMOTransceiver, QAMModemAdapter):
    """
    Класс обработки данных с поддержкой ZF, MMSE эквалайзеров, ML детекции, помехоустойчивого кодирования,
    разных типов решений: жёсткие решения без помехоустойчивого кодирования,
    жёсткие и мягкие решения с помехоустойчивым кодированием.

    Наследуется от:
        - OFDMMIMOTransceiver: обеспечивает OFDM-модуляцию/демодуляцию и моделирование канала
        - QAMModemAdapter: предоставляет QAM-модуляцию/демодуляцию

    Реализация:
        - Помехоустойчивое кодирование битового вектора.
        - Квадратурно Амплитудная Модуляция.
        - Отправка модулированного сигнала.
        - Оптимальный приём принятого сигнала.

    Args:
        signal_noise_ratio (int): Отношение сигнал/шум в дБ.
        use_*_* (bool): Флаги выбора методов эквализациии и типов решений.

    Attributes:
        SNR: Отношение сигнал/шум в дБ.
        output_signal_after_processing: Сигнал после ZF, MMSE эквализации или ML детекции.
        output_vector_of_bits: Вектор битов после демодуляции.
        input_signal: Модулированный сигнал до передачи.
        input_vector_of_bits_enc: Закодированный битовый вектор.
        grid_constellation_bits: Решётка битов для всех комбинаций созвездия.
        grid_constellation: Решётка модуляционных символов для всех комбинаций.
        number_of_bits: Общее количество битов в передаче.
        number_of_bits_enc: Количество битов после свёрточного кодирования.
        convolutional_code_rate_denominator: Знаменатель скорости кода (1/n).
    """
    def __init__(self, signal_noise_ratio: int,
                 use_ml_hard: bool = False, use_ml_fec_hard: bool = False, use_ml_fec_soft: bool = False,
                 use_zf_hard: bool = False, use_zf_fec_hard: bool = False, use_zf_fec_soft: bool = False,
                 use_mmse_hard: bool = False, use_mmse_fec_hard: bool = False, use_mmse_fec_soft: bool = False,
                 **kwargs) -> None:

        D_noise = 10 ** (-signal_noise_ratio / 10)

        super().__init__(D_noise=D_noise, **kwargs)

        self.use_ml_hard = use_ml_hard
        self.use_ml_fec_hard = use_ml_fec_hard
        self.use_ml_fec_soft = use_ml_fec_soft

        self.use_zf_hard = use_zf_hard
        self.use_zf_fec_hard = use_zf_fec_hard
        self.use_zf_fec_soft = use_zf_fec_soft

        self.use_mmse_hard = use_mmse_hard
        self.use_mmse_fec_hard = use_mmse_fec_hard
        self.use_mmse_fec_soft = use_mmse_fec_soft

        if self.Nt > self.Nr:
            self.use_zf_hard, self.use_zf_fec_hard, self.use_zf_fec_soft = False, False, False
            self.use_mmse_hard, self.use_mmse_fec_hard, self.use_mmse_fec_soft= False, False, False

        self.output_signal_after_processing = None

        self.output_vector_of_bits = None

        self.SNR = signal_noise_ratio

        self.number_of_bits = int(_band_width // _subcarrier_spacing * np.log2(self.M) * self.Nt * _frames_number)

        self.convolutional_code_rate_denominator = _trellis.n

        self.number_of_bits_enc = self.number_of_bits * self.convolutional_code_rate_denominator

        if (self.use_ml_hard or self.use_ml_fec_hard or self.use_ml_fec_soft)\
                or (self.use_zf_fec_soft or self.use_mmse_fec_soft):
            self.grid_constellation_bits, self.grid_constellation = self.generate_constellation_combinations()

        self.input_signal = None
        self.input_vector_of_bits_enc = np.array([[]])

    def generate_constellation_combinations(self) -> tuple:
        """
        Генерация матрицы модуляционных символов и соответствующих им битовых векторов
        из всех возможных комбинаций QAM-созвездия для заданного количества передающих антенн.
        Returns:
            Картеж битовой и модуляционной сеток.
        """
        grid_constellation = np.meshgrid(*([self.constellation] * self.Nt), indexing='ij')
        grid_constellation = np.stack(grid_constellation, axis=-1).reshape(-1, self.Nt)

        grid_constellation_bits = np.vstack(
            [
                [self.demodulate(grid_constellation[comb], demod_type='hard')]
                for comb in range(grid_constellation.shape[0])
            ]
        )

        return grid_constellation_bits, grid_constellation / np.sqrt(self.Es)

    def compute_log_likelihood_ratio(self, entire_signal: ndarray, ml_flag: bool) -> ndarray:
        """
        Вычисление матрицы мягких решений (Log-Likelihood Ratio) для принятого сигнала.
        Args:
            entire_signal: Принятый сигнал.
            ml_flag: Флаг для сигнала, принятого при помощи Maximum Likelihood (ML) детектора.
        Returns:
            Матрица мягких решений.
        """
        llr = [[]]

        for n_signal in range(len(entire_signal)):
            signal_grid = entire_signal[n_signal] * np.ones((self.M ** self.Nt, entire_signal.shape[-1]))

            if ml_flag:
                H_qam_grid = np.zeros((self.M ** self.Nt, self.Nr), dtype='complex')

                for comb in range(len(self.grid_constellation)):
                    H_qam_grid[comb] = self.channel_matrix[n_signal] @ self.grid_constellation[comb]

                ln_d_grid = (np.sum(np.abs(H_qam_grid - signal_grid)**2, axis=1)
                             / (self.D_noise / 2))
            else:
                ln_d_grid = (np.sum(np.abs(self.grid_constellation - signal_grid) ** 2, axis=1)
                             / (self.D_noise / 2))

            for bit in range(len(self.grid_constellation_bits[0])):
                ln_d0 = ln_d_grid[np.where(self.grid_constellation_bits[:, bit] == 0)]
                sum_d0 = (-1) * np.min(ln_d0) + np.log(np.sum(np.exp(np.min(ln_d0) - ln_d0)))

                ln_d1 = ln_d_grid[np.where(self.grid_constellation_bits[:, bit] == 1)]
                sum_d1 = (-1) * np.min(ln_d1) + np.log(np.sum(np.exp(np.min(ln_d1) - ln_d1)))

                llr[n_signal].append(sum_d1 - sum_d0)
            llr.append([])

        return np.asarray(llr[:-1])

    def equalization(self, W: ndarray) -> ndarray:
        """
        Эквализация принятого сигнала при помощи матрицы эквализации.
        Args:
            W: Матрица эквализации.
        Returns:
            Эквализированный сигнал.
        """
        x_hat = np.matmul(W, self.output_signal_f.reshape(*self.output_signal_f.shape, 1))

        return x_hat.reshape(*x_hat.shape[:-1])

    def compute_mmse_equalizer_matrix(self) -> ndarray:
        """
        Вычисление матрицы Minimum Mean Square Error эквалайзера  по оценённой матрицы канала.
        Returns:
            Матрица эквализации.
        """
        if self.Nt == 1 and self.Nr == 1:
            W = np.zeros_like(self.channel_matrix)
            for nH, single_H in enumerate(self.channel_matrix):
                W[nH] = np.linalg.inv(
                    np.conj(single_H) @ single_H + 10 ** (-self.SNR / 10) * np.eye(self.Nt)
                ) @ np.conj(single_H)

            return W

        return np.linalg.inv(
            np.conj(np.transpose(self.channel_matrix, axes=(0, 2, 1))) @ self.channel_matrix
            + self.D_noise
            * np.vstack([[np.eye(self.Nt)] for _ in range(self.number_of_OFDM_symbols * self.number_of_subcarriers)])
        ) @ np.conj(np.transpose(self.channel_matrix, axes=(0, 2, 1)))

    def compute_zf_equalizer_matrix(self) -> ndarray:
        """
        Вычисление матрицы Zero-Forcing эквалайзера по оценённой матрицы канала.
        Returns:
            Матрица эквализации.
        """
        if self.Nt == 1 and self.Nr == 1:
            W = np.zeros_like(self.channel_matrix)
            for nH, single_H in enumerate(self.channel_matrix):
                W[nH] = np.linalg.inv((np.conj(single_H) @ single_H)) @ np.conj(single_H)

            return W

        return np.linalg.inv(
            np.conj(np.transpose(self.channel_matrix, axes=(0, 2, 1))) @ self.channel_matrix
        ) @ np.conj(np.transpose(self.channel_matrix, axes=(0, 2, 1)))

    def perform_maximum_likelihood_detection(self) -> ndarray:
        """
        Оценка жёсткими решениями принятого сигнала при помощи Maximum Likelihood (ML) детекции.
        Returns:
            Жёсткие решения для принятого сигнала.
        """
        ml_signal = []

        for n_signal in range(len(self.output_signal_f)):
            signal_grid = self.output_signal_f[n_signal] * np.ones((self.M ** self.Nt, self.Nr))

            H_qam_grid = np.zeros((self.M ** self.Nt, self.Nr), dtype='complex')

            for comb in range(len(self.grid_constellation)):
                    H_qam_grid[comb] = self.channel_matrix[n_signal] @ self.grid_constellation[comb]

            ml_signal.append(self.grid_constellation[np.sum(np.abs(H_qam_grid - signal_grid), axis=1).argmin()])

        return np.asarray(ml_signal)

    def process_received_signal(self, signal: ndarray, ml_flag: bool = False,
                                hard_decision: bool = True,
                                fec_hard_decision: bool = True,
                                fec_soft_decision: bool = True) -> ndarray:
        """
        Демодуляция принятого сигнала при помощи разных видов решений.
        Args:
            signal: Принятый сигнал.
            ml_flag: Флаг Maximum Likelihood (ML) детекции принятого сигнала.
            hard_decision: Демодуляция жёсткими решениями.
            fec_hard_decision: Демодуляция жёсткими решениями с помехоустойчивым кодированием.
            fec_soft_decision: Демодуляция мягкими решениями с помехоустойчивым кодированием.
        Returns:
            Биты после демодуляции.
        Raises:
            ValueError: Все виды помечены как False.
        """
        if not (hard_decision or fec_hard_decision or fec_soft_decision):
            raise ValueError('AВсе виды помечены как False.')

        if hard_decision or fec_hard_decision:
            output_vector_of_bits_hard = np.vstack(
                [self.demodulate(signal[:, st] * np.sqrt(self.Es), demod_type='hard')
                 for st in range(self.Nt)]
            )

            if hard_decision:
                return output_vector_of_bits_hard
            else:
                return np.vstack(
                    [viterbi_decode(output_vector_of_bits_hard[bt], _trellis, decoding_type='hard')
                     for bt in range(self.Nt)]
                )
        else:
            if ml_flag:
                output_vector_of_bits_soft_llr = np.array(
                    self.compute_log_likelihood_ratio(self.output_signal_f, ml_flag))
            else:
                output_vector_of_bits_soft_llr = np.array(
                    self.compute_log_likelihood_ratio(signal, ml_flag))
            return np.vstack(
                [
                    viterbi_decode(
                        output_vector_of_bits_soft_llr.reshape(-1, int(np.log2(self.M)))[bt::self.Nt].reshape(
                            self.number_of_bits_enc // self.Nt
                        ), _trellis, decoding_type='soft') for bt in range(self.Nt)
                ]
            )

    def transmit_and_process(self, input_vector_of_bits: ndarray) -> None:
        """
        Модуляция входного вектора битов, отправка и оптимальный приём принятого сигнала.
        Args:
            input_vector_of_bits: Входной битовый вектор.
        """

        self.input_vector_of_bits_enc = np.vstack(
            [
                conv_encode(np.concatenate(input_vector_of_bits[:, bt]), _trellis, termination='cont')
                for bt in range(self.Nt)
            ]
        )

        self.input_signal = np.vstack(
            [self.modulate(self.input_vector_of_bits_enc[bt]) for bt in range(self.Nt)]
        ).T / np.sqrt(self.Es)

        self.transmit_and_receive_ofdm(self.input_signal)

        if self.use_ml_hard or self.use_ml_fec_hard or self.use_ml_fec_soft:
            self.output_signal_after_processing = self.perform_maximum_likelihood_detection()
        elif self.use_zf_hard or self.use_zf_fec_hard or self.use_zf_fec_soft:
            self.output_signal_after_processing = self.equalization(self.compute_zf_equalizer_matrix())
        elif self.use_mmse_hard or self.use_mmse_fec_hard or self.use_mmse_fec_soft:
            self.output_signal_after_processing = self.equalization(self.compute_mmse_equalizer_matrix())

        self.output_vector_of_bits = np.concatenate(self.process_received_signal(
            self.output_signal_after_processing,
            ml_flag=self.use_ml_hard or self.use_ml_fec_hard or self.use_ml_fec_soft,
            hard_decision=self.use_ml_hard or self.use_zf_hard or self.use_mmse_hard,
            fec_hard_decision=self.use_ml_fec_hard or self.use_zf_fec_hard or self.use_mmse_fec_hard,
            fec_soft_decision=self.use_ml_fec_soft or self.use_zf_fec_soft or self.use_mmse_fec_soft
        ))


class DecisionMethodAnalyzer(DataProcessing):
    """
    Класс для анализа различных методов оптимального приёма и демодуляции
    сигнала при помощи метрик BER (Bit Error Rate) и EVM (Error Vector Magnitude).

    Наследуется от:
        - DataProcessing: обеспечивает полный цикл обработки сигнала
          (кодирование, модуляция, прохождение канала, эквализация/детекция, демодуляция)

    Args:
        signal_noise_ratio (int): Отношение сигнал/шум в дБ;
        Nt (int): Число передающих антенн;
        Nr (int): Число приёмных антенн;
        size_of_modulation (int): Размер QAM созвездия;
        use_ml_hard (bool): Флаг для Maximum Likelihood (ML) детекции c жёсткими решениями;
        use_ml_fec_hard (bool): Флаг для Maximum Likelihood (ML) детекции
        c жёсткими решениями с помехоустойчивым кодированием;
        use_ml_fec_soft (bool): Флаг для Maximum Likelihood (ML) детекции
        c мягкими решениями с помехоустойчивым кодированием;
        use_zf_hard (bool): Флаг для Zero-Forcing (ZF) эквализации c жёсткими решениями;
        use_zf_fec_hard (bool): Флаг для Zero-Forcing (ZF) эквализации
        c жёсткими решениями с помехоустойчивым кодированием;
        use_zf_fec_soft (bool): Флаг для Zero-Forcing (ZF) эквализации
        c мягкими решениями с помехоустойчивым кодированием;
        use_mmse_hard (bool): Флаг для Minimum Mean Square Error (MMSE) эквализации c жёсткими решениями;
        use_mmse_fec_hard (bool): Флаг для Minimum Mean Square Error (MMSE) эквализации
        c жёсткими решениями с помехоустойчивым кодированием;
        use_mmse_fec_soft (bool): Флаг для Minimum Mean Square Error (MMSE) эквализации
        c мягкими решениями с помехоустойчивым кодированием;
        use_awgn (bool): Флаг для добавления АБГШ.
        use_rayleigh_fading_with_mp (bool): Флаг для использования рэлеевского канала с многолучевым распространением.

    Attributes:
        input_vector_of_bits: Сгенерированный случайный вектор битов.
        ber: Вычисленное значение BER.
        evm: Вычисленное значение EVM (только для ZF и MMSE).
    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.input_vector_of_bits = None

        self.ber = None

        if (self.use_zf_hard or self.use_zf_fec_hard or self.use_zf_fec_soft) \
                or (self.use_mmse_hard or self.use_mmse_fec_hard or self.use_mmse_fec_soft):
            self.evm = list()

        self._count_ber = 0
        self._count_data = 0

    def generate_random_transmission(self) -> None:
        """
        Симуляция передачи и приёма случайного битового вектора.
        """
        self.input_vector_of_bits = np.random.randint(
            0, 2, (self.number_of_bits // (self.Nt * int(np.log2(self.M))),
                                  self.Nt,
                                  int(np.log2(self.M)))
        )

        self.transmit_and_process(input_vector_of_bits=self.input_vector_of_bits)

        if (self.use_zf_hard or self.use_zf_fec_hard or self.use_zf_fec_soft)\
                or (self.use_mmse_hard or self.use_mmse_fec_hard or self.use_mmse_fec_soft):
            self.evm.append(self.compute_evm())

    def compute_evm(self) -> float:
        """
        Вычисление метрики EVM (Error Vector Magnitude) между начальным сигналом
        и сигналом, прошедшим через канал, после обработки.
        Returns:
            Error Vector Magnitude.
        """
        return np.sqrt(np.mean(np.abs((self.input_signal - self.output_signal_after_processing)) ** 2))

    def compute_ber(self, recursion_depth = 0) -> float:
        """
        Вычисление метрики BER (Bit Error Ratio).
        Работает по принципу набора 100 ошибок на сколь угодно большом количестве данных,
        но с ограниченной глубиной рекурсии.
        Args:
            recursion_depth: Динамическая глубина рекурсии.
        Returns:
            Bit Error Ratio.
        """
        if recursion_depth > 100:
            return self._count_ber / self._count_data

        recursion_depth += 1

        if self.use_ml_hard or self.use_zf_hard or self.use_mmse_hard:
            input_vector_of_bits = np.concatenate(self.input_vector_of_bits_enc)
        else:
            input_vector_of_bits = np.concatenate(np.vstack(
                [
                    np.concatenate(self.input_vector_of_bits[:, bt]) for bt in range(self.Nt)
                ]
            ))

        count_dynamic = 0
        sl = 0

        while count_dynamic < 100 and self._count_ber < 100:
            sl += 1

            count_dynamic = np.sum(
                (input_vector_of_bits + self.output_vector_of_bits)[:sl] % 2)

            if (sl > len(self.output_vector_of_bits)
                    and (self._count_ber + count_dynamic) < 100):
                self._count_data += len(self.output_vector_of_bits)
                self._count_ber += count_dynamic
                self.generate_random_transmission()

                return self.compute_ber(recursion_depth)
            elif (self._count_ber + count_dynamic) == 100:
                self._count_data += sl
                self._count_ber += count_dynamic

        return self._count_ber / self._count_data

    def run_full_analysis(self) -> None:
        """
        Вычисление метрик BER (Bit Error Ratio) и EVM (Error Vector Magnitude) для данных
        обработанных при помощи указанного вида оптимального приёма сигнала.
        """
        self.generate_random_transmission()

        self.ber = self.compute_ber()

        if (self.use_zf_hard or self.use_zf_fec_hard or self.use_zf_fec_soft) \
                or (self.use_mmse_hard or self.use_mmse_fec_hard or self.use_mmse_fec_soft):
            self.evm = np.mean(np.asarray(self.evm))

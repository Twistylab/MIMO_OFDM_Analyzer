import numpy as np


def rayleigh_impulse_response():
    """
    Создаёт импульсную характеристику для канала
    по типу Рэлеевского замирания с многолучевым распространением,
    где модули лучей ИХ распределены по Рэлею, а фаза равномерная.
    :return: Импульсная характеристика.
    """
    # Мощность лучей в дБ
    power_dB = np.asarray([0, -8, -17, -21, -25])

    # Линейная мощность
    power = 10 ** (power_dB / 10)

    # Распределение лучей импульсной характеристики по Гауссу
    h_rayleigh = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)

    return np.sqrt(power) * h_rayleigh

def dft(polynomial):
    """
    Реализует прямое дискретное преобразование Фурье.
    :param polynomial: Вектор, подлежащий преобразованию, из коэффициентов полинома.
    :return: Преобразованный вектор из коэффициентов полинома.
    """
    P = np.asarray(polynomial, dtype='complex')
    N = len(P)
    y = np.zeros(P.shape, dtype=complex)

    for k in range(N):
        for n in range(N):
            y[k] += P[n] * np.exp(-1j * 2 * np.pi * k * n / N)
    return y

def idft(polynomial):
    """
    Реализует обратное дискретное преобразование Фурье.
    :param polynomial: Вектор, подлежащий преобразованию, из коэффициентов полинома.
    :return: Преобразованный вектор из коэффициентов полинома.
    """
    P = np.asarray(polynomial, dtype='complex')
    N = len(P)
    y = np.zeros(P.shape, dtype=complex)

    for k in range(N):
        for n in range(N):
            y[k] += P[n] * np.exp(1j * 2 * np.pi * k * n / N)
    return y / N

def fft(polynomial):
    """
    Реализует прямое быстрое преобразование Фурье.
    :param polynomial: Вектор, подлежащий преобразованию,
    из коэффициентов полинома, число которых равно степени двойки.
    :return: Преобразованный вектор из коэффициентов полинома.
    """
    if len(polynomial) & (len(polynomial) > 1) != 0 or len(polynomial) == 0:
        raise ValueError(f"Длина полинома ({len(polynomial)}) не является степенью двойки")
    polynomial = np.asarray(polynomial, dtype='complex')
    def _fft(P):
        n = len(P)
        if n == 1:
            return P

        P_e, P_o = P[::2], P[1::2]
        y_e, y_o = _fft(P_e), _fft(P_o)
        w = np.exp(-2j * np.pi / n)
        y = np.asarray([0] * n, dtype='complex')
        for l in range(n // 2):
            y[l] = y_e[l] + w ** l * y_o[l]
            y[l + n // 2] = y_e[l] - w ** l * y_o[l]
        return y
    return _fft(polynomial)

def ifft(polynomial):
    """
    Реализует обратное быстрое преобразование Фурье.
    :param polynomial: Вектор, подлежащий преобразованию,
    из коэффициентов полинома, число которых равно степени двойки.
    :return: Преобразованный вектор из коэффициентов полинома.
    """
    if len(polynomial) & (len(polynomial) > 1) != 0 or len(polynomial) == 0:
        raise ValueError(f"Длина полинома ({len(polynomial)}) не является степенью двойки")
    polynomial = np.asarray(polynomial, dtype='complex')
    def _ifft(P):
        n = len(P)
        if n == 1:
            return P

        P_e, P_o = P[::2], P[1::2]
        y_e, y_o = _ifft(P_e), _ifft(P_o)
        w = np.exp(2j * np.pi / n)
        y = np.asarray([0] * n, dtype='complex')
        for l in range(n // 2):
            y[l] = y_e[l] + w ** l * y_o[l]
            y[l + n // 2] = y_e[l] - w ** l * y_o[l]
        return y
    return _ifft(polynomial) / len(polynomial)

from concurrent.futures import ProcessPoolExecutor

from optimal_decision_ber_analysis_parallel import DecisionMethodAnalyzer


def run_single_method(ber_key, method, **kwargs):
    """
    Запускает анализ одного метода оптимального приёма и демодуляции в отдельном процессе.

    Args:
        ber_key: Уникальный идентификатор метода
        ('ber_ml_hard', 'ber_ml_fec_hard', 'ber_ml_fec_soft',
        'ber_zf_hard', 'ber_zf_fec_hard', 'ber_zf_fec_soft',
        'ber_mmse_hard', 'ber_mmse_fec_hard', 'ber_mmse_fec_soft');
        method: Имя метода оптимального приёма и демодуляции
        ('use_ml_hard', 'use_ml_fec_hard', 'use_ml_fec_soft',
        'use_zf_hard', 'use_zf_fec_hard', 'use_zf_fec_soft',
        'use_mmse_hard', 'use_mmse_fec_hard', 'use_mmse_fec_soft');
        **kwargs: Дополнительные аргументы, передаваемые в DecisionMethodAnalyzer;

    Returns:
        Картеж из Идентификатор метода (переданный на входе) и вычисленного значения BER (Bit Error Rate);
    """
    analyzer = DecisionMethodAnalyzer(**{method: True}, **kwargs)
    analyzer.run_full_analysis()

    return ber_key, analyzer.ber


def compute_ber_multiprocessing(use_ml_hard: bool = False, use_ml_fec_hard: bool = False, use_ml_fec_soft: bool = False,
                 use_zf_hard: bool = False, use_zf_fec_hard: bool = False, use_zf_fec_soft: bool = False,
                 use_mmse_hard: bool = False, use_mmse_fec_hard: bool = False, use_mmse_fec_soft: bool = False,
                                signal_noise_ratio: int = None,
                                Nt: int = None, Nr: int = None, size_of_modulation: int = None,
                                use_awgn: bool = None, use_rayleigh_fading_with_mp: bool = None) -> dict:
    """
    Вычисляет BER для нескольких методов оптимального приёма и демодуляции с использованием многопроцессорности.

    Args:
        signal_noise_ratio (int): Отношение сигнал/шум в дБ;
        Nt (int): Число передающих антенн;
        Nr (int): Число приёмных антенн;
        size_of_modulation (int): Размер QAM созвездия;
        use_ml_hard (bool): Флаг для Maximum Likelihood (ML) детекции
        c жёсткими решениями без помехоустойчивого кодирования;
        use_ml_fec_hard (bool): Флаг для Maximum Likelihood (ML) детекции
        c жёсткими решениями с помехоустойчивым кодированием;
        use_ml_fec_soft (bool): Флаг для Maximum Likelihood (ML) детекции
        c мягкими решениями с помехоустойчивым кодированием;
        use_zf_hard (bool): Флаг для Zero-Forcing (ZF) эквализации
        c жёсткими решениями без помехоустойчивого кодирования;
        use_zf_fec_hard (bool): Флаг для Zero-Forcing (ZF) эквализации
        c жёсткими решениями с помехоустойчивым кодированием;
        use_zf_fec_soft (bool): Флаг для Zero-Forcing (ZF) эквализации
        c мягкими решениями с помехоустойчивым кодированием;
        use_mmse_hard (bool): Флаг для Minimum Mean Square Error (MMSE) эквализации
        c жёсткими решениями без помехоустойчивого кодирования;
        use_mmse_fec_hard (bool): Флаг для Minimum Mean Square Error (MMSE) эквализации
        c жёсткими решениями с помехоустойчивым кодированием;
        use_mmse_fec_soft (bool): Флаг для Minimum Mean Square Error (MMSE) эквализации
        c мягкими решениями с помехоустойчивым кодированием;
        use_awgn (bool): Флаг для добавления АБГШ.
        use_rayleigh_fading_with_mp (bool): Флаг для использования рэлеевского канала с многолучевым распространением.

    Returns:
        dict: Словарь с результатами BER для всех методов. Ключи:
            - 'ber_ml_hard': BER для ML с жёсткими решениями
            - 'ber_ml_fec_hard': BER для ML с помехоустойчивым кодированием и жёсткими решениями
            - 'ber_ml_fec_soft': BER для ML с помехоустойчивым кодированием и мягкими решениями
            - 'ber_zf_hard': BER для ZF с жёсткими решениями
            - 'ber_zf_fec_hard': BER для ZF с помехоустойчивым кодированием и жёсткими решениями
            - 'ber_zf_fec_soft': BER для ZF с помехоустойчивым кодированием и мягкими решениями
            - 'ber_mmse_hard': BER для MMSE с жёсткими решениями
            - 'ber_mmse_fec_hard': BER для MMSE с помехоустойчивым кодированием и жёсткими решениями
            - 'ber_mmse_fec_soft': BER для MMSE с помехоустойчивым кодированием и мягкими решениями
        Для неактивных методов значение равно None.
    """

    methods = {
        'ber_ml_hard': {'use_ml_hard': use_ml_hard},
        'ber_ml_fec_hard': {'use_ml_fec_hard': use_ml_fec_hard},
        'ber_ml_fec_soft': {'use_ml_fec_soft': use_ml_fec_soft},
        'ber_zf_hard': {'use_zf_hard': use_zf_hard},
        'ber_zf_fec_hard': {'use_zf_fec_hard': use_zf_fec_hard},
        'ber_zf_fec_soft': {'use_zf_fec_soft': use_zf_fec_soft},
        'ber_mmse_hard': {'use_mmse_hard': use_mmse_hard},
        'ber_mmse_fec_hard': {'use_mmse_fec_hard': use_mmse_fec_hard},
        'ber_mmse_fec_soft': {'use_mmse_fec_soft': use_mmse_fec_soft},
    }

    active_methods = []
    for ber_key, method_dict in methods.items():
        method_key = list(method_dict.keys())[0]
        if method_dict[method_key]:
            active_methods.append((ber_key, method_key))

    kwargs = {
        'signal_noise_ratio': signal_noise_ratio,
        'size_of_modulation': size_of_modulation,
        'Nt': Nt,
        'Nr': Nr,
        'use_awgn': use_awgn,
        'use_rayleigh_fading_with_mp': use_rayleigh_fading_with_mp
    }

    with ProcessPoolExecutor(max_workers=len(active_methods)) as executor:
        futures = []
        for ber_key, method in active_methods:
            future = executor.submit(run_single_method, ber_key, method, **kwargs)
            futures.append(future)

        results = {}
        for future in futures:
            ber_key, ber = future.result()
            print(ber_key, ' ', ber)
            results[ber_key] = ber

    for key in methods.keys():
        if key not in results:
            results[key] = None

    return results

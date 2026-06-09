import numpy as np

from concurrent.futures import ProcessPoolExecutor
from MMSE_class import BERAnalyzerWithMMSE


def run_single_method(ber_key, method, **kwargs):
    analyzer = BERAnalyzerWithMMSE(**{method: True}, **kwargs)
    analyzer.run_full_analysis()

    return ber_key, analyzer.ber

def compute_ber_multiprocessing(use_mmse_hard: bool = False, use_mmse_fec_hard: bool = False, use_mmse_fec_soft: bool = False,
                                modulation_size: int = 16, bits_number: int = 10000, noise_variance: float = None, impulse_response_length: int = 10, delay: int = 10) -> dict:

    methods = {
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
        'modulation_size': modulation_size,
        'noise_variance': noise_variance,
        'bits_number': bits_number,
        'impulse_response_length': impulse_response_length,
        'delay': delay,
    }

    with ProcessPoolExecutor(max_workers=len(active_methods)) as executor:
        futures = []
        for ber_key, method in active_methods:
            future = executor.submit(run_single_method, ber_key, method, **kwargs)
            futures.append(future)

        results = {}
        for future in futures:
            ber_key, ber = future.result()
            print(int((-10) * np.log10(noise_variance)), 'дБ ', ber_key, ' ', ber)
            results[ber_key] = ber

    for key in methods.keys():
        if key not in results:
            results[key] = None

    return results

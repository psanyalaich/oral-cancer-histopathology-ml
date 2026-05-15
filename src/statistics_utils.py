import numpy as np
from scipy.stats import t

def confidence_interval(values, confidence=0.95):

    values = np.asarray(values)

    n = len(values)
    mean = np.mean(values)
    sem = values.std(ddof=1) / np.sqrt(n)

    margin = sem * t.ppf((1 + confidence) / 2, n - 1)

    return (
        mean,
        mean - margin,
        mean + margin,
    )
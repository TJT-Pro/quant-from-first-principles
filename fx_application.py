import csv
import statistics
from typing import List, Dict

import spectral_analysis as sa


def load_log_returns(csv_path: str, column: str = "log_return") -> List[float]:
    values = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row[column]))
    return values


def band_average_power(power: List[float], freqs: List[float], low: float, high: float) -> float:
    values = [p for f, p in zip(freqs, power) if low <= f <= high]
    if not values:
        raise ValueError("no frequency bins fall inside the requested band")
    return statistics.mean(values)


def low_vs_high_frequency_ratio(x: List[float], sample_rate: float = 1.0,
                                 low_band: tuple = (0.001, 0.01),
                                 high_band: tuple = (0.4, 0.5)) -> Dict:
    """The ratio of average spectral power in a low-frequency band to
    a high-frequency band: a value near 1 describes a flat ("white")
    spectrum, equal power at every frequency, the signature of an
    uncorrelated (memoryless) process; a value well above 1 describes
    a "red" spectrum, power concentrated at low frequencies, the
    frequency-domain signature of a persistent, slowly-varying
    (positively autocorrelated) process. This is a direct spectral
    analogue of the ACF-decay diagnostics used elsewhere in this
    project family, arrived at through a completely different route
    (the Fourier transform's energy distribution across frequency,
    rather than the time-domain autocorrelation function).
    """
    result = sa.periodogram(x, sample_rate)
    low_power = band_average_power(result["power"], result["frequencies"], *low_band)
    high_power = band_average_power(result["power"], result["frequencies"], *high_band)
    return {"low_band_power": low_power, "high_band_power": high_power, "ratio": low_power / high_power}


def extract_volatility_trend(squared_returns: List[float], sample_rate: float = 1.0,
                              cutoff_period_days: float = 20.0) -> List[float]:
    """Extract a smoothed, slowly-varying volatility trend from the
    (necessarily very noisy, since each squared return is a one-sample
    estimate of the underlying variance) squared-return series, by
    keeping only frequencies whose period exceeds `cutoff_period_days`
    and removing everything faster. This is a direct, practical use of
    `spectral_analysis.spectral_filter`: rather than a moving average
    (which is itself a particular, not-especially-sharp low-pass
    filter with its own frequency response), this applies an ideal
    brick-wall filter directly in the frequency domain, keeping
    exactly the slow-moving component the low-vs-high-frequency-ratio
    analysis above already showed carries disproportionate power.
    """
    cutoff_freq = 1.0 / cutoff_period_days
    return sa.spectral_filter(squared_returns, sample_rate, low=0.0, high=cutoff_freq)


def weekly_cycle_power(returns: List[float], sample_rate: float = 1.0) -> Dict:
    """Checks specifically for a 5-trading-day (one calendar week)
    periodic component, at frequency 1/5 = 0.2 cycles/day: the natural
    first place to look for a deterministic calendar effect (a
    systematic day-of-week pattern) in daily financial data, as
    opposed to the broadband, non-periodic persistence a "red
    spectrum" describes. The power at this specific frequency is
    compared against the average power of its immediate neighbouring
    bins, since a genuine periodic component should show up as a
    narrow spike well above the surrounding noise floor, not merely as
    one bin among many with similar power.
    """
    result = sa.periodogram(returns, sample_rate)
    freqs, power = result["frequencies"], result["power"]
    target_freq = 0.2
    idx = min(range(len(freqs)), key=lambda i: abs(freqs[i] - target_freq))
    neighbors = [power[j] for j in range(max(1, idx - 5), min(len(power), idx + 6)) if j != idx]
    neighbor_avg = statistics.mean(neighbors)
    return {
        "frequency": freqs[idx],
        "power": power[idx],
        "neighbor_average_power": neighbor_avg,
        "ratio_to_neighbors": power[idx] / neighbor_avg,
    }

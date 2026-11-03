import argparse
import math
import os
import sys
import time

import dft
import fft
import convolution as conv
import spectral_analysis as sa
import fx_application as fx

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


def cmd_transform(args):
    x = [float(v) for v in args.signal.split(",")]
    if args.method == "dft":
        result = dft.dft(x)
    else:
        result = fft.fft(x)
    print(f"{args.method.upper()} of {x}")
    for k, v in enumerate(result):
        print(f"  X[{k}] = {v.real:+.4f} {'+' if v.imag >= 0 else '-'} {abs(v.imag):.4f}i")


def cmd_speedtest(args):
    import random
    rng = random.Random(1)
    x = [rng.gauss(0, 1) for _ in range(args.n)]

    t0 = time.time()
    dft.dft(x)
    t_dft = time.time() - t0

    t0 = time.time()
    fft.fft(x)
    t_fft = time.time() - t0

    print(f"N = {args.n}")
    print(f"naive DFT: {t_dft:.4f}s")
    print(f"FFT:       {t_fft:.4f}s")
    print(f"speedup:   {t_dft / t_fft:.1f}x")


def cmd_convolve(args):
    x = [float(v) for v in args.x.split(",")]
    h = [float(v) for v in args.h.split(",")]
    linear = conv.linear_convolution(x, h)
    fast = conv.fft_convolution(x, h)
    print(f"linear convolution:        {[round(v, 6) for v in linear]}")
    print(f"FFT convolution (padded):  {[round(v, 6) for v in fast]}")

    if args.show_pitfall and len(x) == len(h):
        circular = [v.real for v in conv.circular_convolution_via_fft(x, h)]
        print(f"circular (unpadded, WRONG for linear conv): {[round(v, 6) for v in circular]}")


def cmd_periodogram(args):
    x = [float(v) for v in args.signal.split(",")]
    result = sa.periodogram(x, args.sample_rate)
    print("frequency   power")
    for f, p in zip(result["frequencies"], result["power"]):
        print(f"  {f:.4f}   {p:.6f}")


def cmd_fxspectrum(args):
    returns = fx.load_log_returns(DATA_PATH)
    n = args.n
    x_raw = returns[:n]
    x_sq = [r ** 2 for r in x_raw]

    print(f"spectral analysis of the first {n} EUR/USD daily returns\n")
    ratio_raw = fx.low_vs_high_frequency_ratio(x_raw)
    ratio_sq = fx.low_vs_high_frequency_ratio(x_sq)
    print(f"raw returns:     low/high frequency power ratio = {ratio_raw['ratio']:.3f}  "
          f"({'flat (white) spectrum' if ratio_raw['ratio'] < 1.5 else 'red spectrum'})")
    print(f"squared returns: low/high frequency power ratio = {ratio_sq['ratio']:.3f}  "
          f"({'flat (white) spectrum' if ratio_sq['ratio'] < 1.5 else 'red spectrum'})")

    weekly = fx.weekly_cycle_power(x_raw)
    print(f"\nweekly-cycle check (0.2 cycles/day): power/neighbor-average ratio = "
          f"{weekly['ratio_to_neighbors']:.3f}  "
          f"({'no evidence of a weekly cycle' if weekly['ratio_to_neighbors'] < 3 else 'possible weekly cycle'})")

    dom_raw = sa.dominant_frequencies(x_raw, n_top=args.top)
    print(f"\ntop {args.top} frequencies in raw returns:")
    for d in dom_raw:
        print(f"  freq={d['frequency']:.5f}  period={d['period']:.2f} days  power={d['power']:.3e}")


def cmd_fxtrend(args):
    returns = fx.load_log_returns(DATA_PATH)
    squared = [r ** 2 for r in returns[:args.n]]
    trend = fx.extract_volatility_trend(squared, cutoff_period_days=args.cutoff)
    print(f"volatility trend extracted from the first {args.n} squared returns, "
          f"cutoff period = {args.cutoff} days")
    print(f"raw squared returns:  min={min(squared):.3e}  max={max(squared):.3e}")
    print(f"smoothed trend:       min={min(trend):.3e}  max={max(trend):.3e}")
    if min(trend) < 0:
        print(f"\nnote: the smoothed trend dips slightly negative ({min(trend):.3e}), "
              f"even though squared returns are non-negative by construction; see the README.")


def cmd_verify(args):
    import random
    import numpy as np
    from scipy.signal import fftconvolve

    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}")

    rng = random.Random(1)
    x = [rng.gauss(0, 1) for _ in range(16)]
    ours = dft.dft(x)
    theirs = np.fft.fft(x)
    check("naive DFT matches numpy.fft.fft",
          all(math.isclose(a.real, b.real, abs_tol=1e-8) and math.isclose(a.imag, b.imag, abs_tol=1e-8)
              for a, b in zip(ours, theirs)))

    x2 = [rng.gauss(0, 1) for _ in range(128)]
    ours2 = fft.fft(x2)
    theirs2 = np.fft.fft(x2)
    check("Cooley-Tukey FFT matches numpy.fft.fft",
          all(math.isclose(a.real, b.real, abs_tol=1e-6) and math.isclose(a.imag, b.imag, abs_tol=1e-6)
              for a, b in zip(ours2, theirs2)))

    ours3 = fft.ifft(ours2)
    theirs3 = np.fft.ifft(theirs2)
    check("IFFT matches numpy.fft.ifft",
          all(math.isclose(a.real, b.real, abs_tol=1e-8) for a, b in zip(ours3, theirs3)))

    x4 = [rng.gauss(0, 1) for _ in range(64)]
    time_energy = sum(v ** 2 for v in x4)
    freq_energy = sum(abs(v) ** 2 for v in fft.fft(x4)) / len(x4)
    check("Parseval's theorem holds to numerical precision",
          math.isclose(time_energy, freq_energy, abs_tol=1e-8))

    x5 = [rng.gauss(0, 1) for _ in range(50)]
    h5 = [rng.gauss(0, 1) for _ in range(20)]
    ours5 = conv.fft_convolution(x5, h5)
    theirs5 = fftconvolve(x5, h5)
    check("fft_convolution matches scipy.signal.fftconvolve",
          all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(ours5, theirs5)))

    ours6 = conv.linear_convolution(x5, h5)
    check("fft_convolution matches direct linear_convolution",
          all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(ours5, ours6)))

    xc = [1.0, 2.0, 3.0]
    hc = [1.0, 1.0, 0.0]
    circ = [v.real for v in conv.circular_convolution_via_fft(xc, hc)]
    lin = conv.linear_convolution(xc, hc[:2])
    check("unpadded circular convolution genuinely differs from linear convolution",
          not math.isclose(circ[0], lin[0], abs_tol=1e-6))

    n = 256
    freq_bin = 12
    xs = [math.sin(2 * math.pi * freq_bin * t / n) for t in range(n)]
    ours8 = sa.periodogram(xs)
    numpy_fft = np.fft.fft(xs)
    numpy_power = (np.abs(numpy_fft) ** 2 / n)[:n // 2 + 1]
    check("periodogram matches numpy-FFT-based recomputation",
          all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(ours8["power"], numpy_power)))

    n9 = 256
    x9 = [math.sin(2 * math.pi * 5 * t / n9) + 0.5 * math.sin(2 * math.pi * 50 * t / n9) for t in range(n9)]
    filtered = sa.spectral_filter(x9, low=0.0, high=20 / n9)
    filtered_power = sa.periodogram(filtered)["power"]
    check("low-pass filter removes the high-frequency component and keeps the low one",
          filtered_power[50] < 1e-10 and filtered_power[5] > 1.0)

    returns = fx.load_log_returns(DATA_PATH)
    n10 = 4096
    raw_ratio = fx.low_vs_high_frequency_ratio(returns[:n10])
    sq_ratio = fx.low_vs_high_frequency_ratio([r ** 2 for r in returns[:n10]])
    check("real EUR/USD raw returns show a close-to-flat spectrum", raw_ratio["ratio"] < 1.5)
    check("real EUR/USD squared returns show a red (low-frequency-heavy) spectrum", sq_ratio["ratio"] > 2.0)

    passed = sum(1 for _, ok in checks if ok)
    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(description="Fourier analysis engine CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("transform", help="compute the DFT or FFT of a signal")
    p1.add_argument("--signal", type=str, required=True, help="comma-separated real values")
    p1.add_argument("--method", choices=["dft", "fft"], default="fft")
    p1.set_defaults(func=cmd_transform)

    p2 = sub.add_parser("speedtest", help="compare naive DFT vs FFT timing")
    p2.add_argument("--n", type=int, default=1024)
    p2.set_defaults(func=cmd_speedtest)

    p3 = sub.add_parser("convolve", help="linear vs FFT convolution, with the circular-convolution pitfall")
    p3.add_argument("--x", type=str, required=True)
    p3.add_argument("--h", type=str, required=True)
    p3.add_argument("--show-pitfall", action="store_true")
    p3.set_defaults(func=cmd_convolve)

    p4 = sub.add_parser("periodogram", help="compute the periodogram of a signal")
    p4.add_argument("--signal", type=str, required=True)
    p4.add_argument("--sample-rate", type=float, default=1.0)
    p4.set_defaults(func=cmd_periodogram)

    p5 = sub.add_parser("fxspectrum", help="spectral analysis of real EUR/USD returns")
    p5.add_argument("--n", type=int, default=4096)
    p5.add_argument("--top", type=int, default=5)
    p5.set_defaults(func=cmd_fxspectrum)

    p6 = sub.add_parser("fxtrend", help="extract a smoothed volatility trend from squared returns")
    p6.add_argument("--n", type=int, default=4096)
    p6.add_argument("--cutoff", type=float, default=20.0)
    p6.set_defaults(func=cmd_fxtrend)

    p7 = sub.add_parser("verify", help="cross-check implementation against analytical/oracle values")
    p7.set_defaults(func=cmd_verify)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

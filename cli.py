import argparse
import math
import os
import sys

import entropy as ent
import divergence as div
import huffman_coding as hc
import channel_capacity as cc
import entropy_rate as er
import fx_application as fx

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


def cmd_entropy(args):
    counts = args.symbols.split(",")
    h = ent.entropy_from_counts(counts)
    n_distinct = len(set(counts))
    print(f"symbols: {counts}")
    print(f"entropy: {h:.6f} bits")
    print(f"max possible ({n_distinct} distinct symbols): {ent.max_entropy(n_distinct):.6f} bits")


def cmd_mutualinfo(args):
    x = args.x.split(",")
    y = args.y.split(",")
    pairs = list(zip(x, y))
    mi = div.mutual_information_from_pairs(pairs)
    nmi = div.normalized_mutual_information(pairs)
    print(f"I(X;Y) = {mi:.6f} bits")
    print(f"normalized I(X;Y) = {nmi:.6f}")


def cmd_huffman(args):
    symbols = list(args.text)
    encoded, codes = hc.huffman_encode(symbols)
    result = hc.verify_source_coding_bound(symbols)
    print(f"text: {args.text!r}")
    print("codes:")
    for s, c in sorted(codes.items()):
        print(f"  {s!r}: {c}")
    print(f"\nentropy: {result['entropy']:.4f} bits/symbol")
    print(f"average code length: {result['average_code_length']:.4f} bits/symbol")
    print(f"naive fixed-length would need: {math.ceil(math.log2(len(codes)))} bits/symbol")
    decoded = hc.huffman_decode(encoded, codes)
    print(f"decode matches original: {decoded == symbols}")


def cmd_channel(args):
    print(f"BSC capacity at p={args.p}: {cc.bsc_capacity(args.p):.4f} bits/use")
    print(f"Gaussian channel capacity at SNR={args.snr}: {cc.gaussian_channel_capacity(args.snr):.4f} bits/use\n")

    print("repetition code simulation:")
    for n in (1, 3, 5, 9, 15, 25):
        result = cc.simulate_repetition_code(args.p, n, 20000, seed=1)
        print(f"  n={n:3d}  rate={result['rate']:.3f}  empirical_error_rate={result['empirical_error_rate']:.4f}")


def cmd_entropyrate(args):
    p = [[args.p00, 1 - args.p00], [1 - args.p11, args.p11]]
    exact = er.exact_markov_entropy_rate(p)
    seq = er.simulate_markov_chain(p, n_steps=args.n_steps, seed=1)
    print(f"transition matrix: {p}")
    print(f"exact entropy rate: {exact:.6f} bits\n")
    for k in (1, 2, 4, 8):
        est = er.block_entropy_rate_estimate(seq, k)
        print(f"block-entropy estimate (k={k}): {est:.6f}")
    cond_est = er.conditional_entropy_rate_estimate(seq, order=1)
    print(f"conditional-entropy estimate (order=1): {cond_est:.6f}")


def cmd_fxinfo(args):
    returns = fx.load_log_returns(DATA_PATH)
    signs = fx.sign_sequence(returns)

    sign_result = fx.sign_entropy_analysis(returns)
    print(f"sign entropy: {sign_result['entropy']:.6f} bits ({sign_result['fraction_of_max']*100:.2f}% of max)\n")

    print("mutual information of return sign at various lags, with permutation-test significance:")
    lags_to_test = (1, 2, 5, 10, 20)
    for lag in lags_to_test:
        perm = fx.permutation_null_mi(signs, lag, n_permutations=300, seed=1)
        sig = "significant" if perm["exceeds_95th_percentile"] else "not significant"
        print(f"  lag={lag:3d}  MI={perm['observed_mi']:.6f}  null_95th={perm['null_95th_percentile']:.6f}  "
              f"({sig}, p={perm['p_value']:.3f})")

    corrected = fx.multi_lag_significance_with_correction(signs, list(lags_to_test), n_permutations=300, seed=1)
    print(f"\n  Bonferroni-corrected threshold across {corrected['n_tests']} lags: "
          f"alpha={corrected['corrected_alpha']:.4f}")
    print(f"  any lag survives correction: {corrected['any_survives_correction']}")

    print("\nmutual information of 20-day rolling volatility state at various lags:")
    vol_states = fx.rolling_volatility_states(returns, window=20)
    for lag in (1, 5, 10, 20, 40):
        perm = fx.permutation_null_mi(vol_states, lag, n_permutations=300, seed=1)
        sig = "significant" if perm["exceeds_95th_percentile"] else "not significant"
        print(f"  lag={lag:3d}  MI={perm['observed_mi']:.6f}  null_95th={perm['null_95th_percentile']:.6f}  "
              f"({sig}, p={perm['p_value']:.3f})")

    print("\nHuffman compression:")
    sign_huffman = fx.huffman_compression_analysis(signs)
    print(f"  sign sequence: entropy={sign_huffman['entropy']:.4f}  "
          f"avg_code_length={sign_huffman['average_code_length']:.4f} (2-symbol alphabet: always 1 bit)")
    vol_huffman = fx.huffman_compression_analysis(vol_states)
    print(f"  vol-state sequence: entropy={vol_huffman['entropy']:.4f}  "
          f"avg_code_length={vol_huffman['average_code_length']:.4f}")

    print("\nKL divergence from a fitted Gaussian:")
    kl_result = fx.kl_divergence_from_gaussian(returns)
    print(f"  D(empirical || Gaussian) = {kl_result['kl_divergence_bits']:.4f} bits")


def cmd_verify(args):
    import random
    import numpy as np
    from scipy.stats import entropy as scipy_entropy

    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}")

    # 1. entropy vs scipy.stats.entropy
    p = {"a": 0.5, "b": 0.3, "c": 0.2}
    ours = ent.entropy(p)
    theirs = scipy_entropy(list(p.values()), base=2)
    check("entropy matches scipy.stats.entropy", math.isclose(ours, theirs, abs_tol=1e-10))

    # 2. KL divergence vs scipy
    q = {"a": 0.6, "b": 0.2, "c": 0.2}
    ours_kl = div.kl_divergence(p, q)
    theirs_kl = scipy_entropy(list(p.values()), list(q.values()), base=2)
    check("KL divergence matches scipy.stats.entropy (relative entropy mode)",
          math.isclose(ours_kl, theirs_kl, abs_tol=1e-10))

    # 3. mutual information formula consistency (entropy-based vs KL-based)
    rng = random.Random(1)
    pairs = [(rng.choice(["a", "b", "c"]), rng.choice([1, 2])) for _ in range(5000)]
    mi_entropy = div.mutual_information_from_pairs(pairs)
    mi_kl = div.mutual_information_via_kl(pairs)
    check("mutual information: entropy-based formula matches KL-based formula",
          math.isclose(mi_entropy, mi_kl, abs_tol=1e-8))

    # 4. Huffman source coding bound, checked directly
    symbols = [rng.choice("abcdefgh") for _ in range(5000)]
    result = hc.verify_source_coding_bound(symbols)
    check("Huffman coding satisfies Shannon's source coding bound (H <= L < H+1)",
          result["satisfies_lower_bound"] and result["satisfies_upper_bound"])

    # 5. Huffman decode correctness
    encoded, codes = hc.huffman_encode(symbols)
    decoded = hc.huffman_decode(encoded, codes)
    check("Huffman decode recovers the original sequence exactly", decoded == symbols)

    # 6. BSC capacity known values
    check("BSC capacity at p=0.5 is exactly 0 (pure noise)",
          math.isclose(cc.bsc_capacity(0.5), 0.0, abs_tol=1e-10))
    check("BSC capacity at p=0 is exactly 1 (noiseless)",
          math.isclose(cc.bsc_capacity(0.0), 1.0, abs_tol=1e-10))

    # 7. entropy rate: exact Markov formula vs empirical block-entropy convergence
    transition = [[0.9, 0.1], [0.3, 0.7]]
    exact_rate = er.exact_markov_entropy_rate(transition)
    seq = er.simulate_markov_chain(transition, n_steps=100000, seed=1)
    cond_estimate = er.conditional_entropy_rate_estimate(seq, order=1)
    check("empirical conditional-entropy-rate estimate matches exact Markov entropy rate",
          abs(cond_estimate - exact_rate) < 0.01)

    # 8. stationary distribution vs numpy eigenvector
    pi_ours = er.stationary_distribution(transition)
    eigvals, eigvecs = np.linalg.eig(np.array(transition).T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    vec = np.real(eigvecs[:, idx])
    vec = vec / vec.sum()
    check("stationary distribution matches numpy eigenvector",
          all(math.isclose(pi_ours[i], vec[i], abs_tol=1e-6) for i in range(2)))

    # 9. real EUR/USD application
    returns = fx.load_log_returns(DATA_PATH)
    sign_result = fx.sign_entropy_analysis(returns)
    check("real EUR/USD sign entropy exceeds 99% of theoretical maximum",
          sign_result["fraction_of_max"] > 0.99)

    signs = fx.sign_sequence(returns)
    sign_perm = fx.permutation_null_mi(signs, lag=1, n_permutations=300, seed=1)
    check("real EUR/USD sign shows no significant lag-1 mutual information",
          not sign_perm["exceeds_95th_percentile"])

    vol_states = fx.rolling_volatility_states(returns, window=20)
    vol_perm = fx.permutation_null_mi(vol_states, lag=1, n_permutations=300, seed=1)
    check("real EUR/USD rolling volatility shows significant lag-1 mutual information",
          vol_perm["exceeds_95th_percentile"] and vol_perm["p_value"] < 0.01)

    kl_result = fx.kl_divergence_from_gaussian(returns)
    check("real EUR/USD returns show a real, positive KL divergence from Gaussian",
          kl_result["kl_divergence_bits"] > 0.01)

    corrected = fx.multi_lag_significance_with_correction(signs, [1, 2, 5, 10, 20], n_permutations=300, seed=1)
    check("no lag survives Bonferroni correction for real EUR/USD sign (no genuine multi-lag dependence)",
          not corrected["any_survives_correction"])

    passed = sum(1 for _, ok in checks if ok)
    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(description="Information theory engine CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("entropy", help="compute the entropy of a symbol sequence")
    p1.add_argument("--symbols", type=str, required=True, help="comma-separated symbols")
    p1.set_defaults(func=cmd_entropy)

    p2 = sub.add_parser("mutualinfo", help="compute mutual information between two sequences")
    p2.add_argument("--x", type=str, required=True)
    p2.add_argument("--y", type=str, required=True)
    p2.set_defaults(func=cmd_mutualinfo)

    p3 = sub.add_parser("huffman", help="Huffman-encode a text string")
    p3.add_argument("--text", type=str, required=True)
    p3.set_defaults(func=cmd_huffman)

    p4 = sub.add_parser("channel", help="channel capacity and repetition-code demonstration")
    p4.add_argument("--p", type=float, default=0.1, help="BSC crossover probability")
    p4.add_argument("--snr", type=float, default=15.0, help="Gaussian channel SNR")
    p4.set_defaults(func=cmd_channel)

    p5 = sub.add_parser("entropyrate", help="entropy rate estimation on a synthetic Markov chain")
    p5.add_argument("--p00", type=float, default=0.9)
    p5.add_argument("--p11", type=float, default=0.7)
    p5.add_argument("--n-steps", type=int, default=100000)
    p5.set_defaults(func=cmd_entropyrate)

    p6 = sub.add_parser("fxinfo", help="information-theoretic analysis of real EUR/USD returns")
    p6.set_defaults(func=cmd_fxinfo)

    p7 = sub.add_parser("verify", help="cross-check implementation against scipy/numpy oracles and known theory")
    p7.set_defaults(func=cmd_verify)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

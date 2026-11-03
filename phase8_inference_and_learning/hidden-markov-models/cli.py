import argparse
import math
import os
import statistics
import sys

import hmm
import model_selection as ms
import fx_application as fx

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


def cmd_simulate(args):
    means = [float(x) for x in args.means.split(",")]
    variances = [float(x) for x in args.variances.split(",")]
    k = len(means)
    transition = [[float(x) for x in row.split(",")] for row in args.transition.split(";")]
    initial = [1.0 / k] * k
    model = hmm.GaussianHMM(means, variances, transition, initial)
    obs, states = model.simulate(args.length, seed=args.seed)
    print(f"simulated {args.length} observations from a {k}-state HMM")
    if args.length <= 30:
        print("observations:", [round(o, 3) for o in obs])
        print("true states: ", states)


def cmd_fit(args):
    means = [float(x) for x in args.means.split(",")]
    variances = [float(x) for x in args.variances.split(",")]
    k = len(means)
    transition = [[float(x) for x in row.split(",")] for row in args.transition.split(";")]
    initial = [1.0 / k] * k
    true_model = hmm.GaussianHMM(means, variances, transition, initial)
    obs, true_states = true_model.simulate(args.length, seed=args.seed)

    result = ms.fit_best_of_n_restarts(obs, k, args.restarts, seed=args.seed)
    fitted = result["model"]
    print(f"true means: {means}")
    print(f"fitted means (unordered): {[round(m, 3) for m in fitted.means]}")
    print(f"log-likelihood: {result['log_likelihoods'][0]:.2f} -> {result['log_likelihoods'][-1]:.2f} "
          f"over {len(result['log_likelihoods'])} EM iterations")

    path = fitted.viterbi(obs)
    order = sorted(range(k), key=lambda i: fitted.means[i])
    true_order = sorted(range(k), key=lambda i: means[i])
    label_map = {order[i]: true_order[i] for i in range(k)}
    mapped_path = [label_map[s] for s in path]
    accuracy = sum(1 for a, b in zip(mapped_path, true_states) if a == b) / len(true_states)
    print(f"Viterbi state-recovery accuracy: {accuracy:.4f}")


def cmd_selectstates(args):
    returns = fx.load_log_returns(DATA_PATH)
    k_range = [int(x) for x in args.k_range.split(",")]
    result = ms.select_n_states(returns, k_range, args.restarts, seed=1)
    print(f"{'K':>3}  {'log-lik':>12}  {'AIC':>12}  {'BIC':>12}")
    for r in result["results"]:
        print(f"{r['n_states']:3d}  {r['log_likelihood']:12.2f}  {r['aic']:12.2f}  {r['bic']:12.2f}")
    print(f"\nbest by AIC: K={result['best_aic_k']}")
    print(f"best by BIC: K={result['best_bic_k']}")


def cmd_fxregimes(args):
    returns = fx.load_log_returns(DATA_PATH)
    result = fx.fit_regime_model(returns, args.n_states, args.restarts, seed=1)
    model = result["model"]
    order = result["state_order_low_to_high_vol"]
    path = model.viterbi(returns)

    print(f"fitted {args.n_states}-state regime model on {len(returns)} EUR/USD log returns")
    print(f"log-likelihood: {result['log_likelihoods'][-1]:.2f}\n")

    summary = fx.regime_summary(model, path, order)
    durations = fx.regime_duration_statistics(path, order)
    for s in summary:
        rank = s["rank"]
        print(f"regime {rank} (volatility rank {rank}):")
        print(f"  mean daily return: {s['mean']:+.6f}")
        print(f"  daily sd:          {s['sd']:.6f}  (annualised: {s['sd']*math.sqrt(252):.4f})")
        print(f"  stationary probability: {s['stationary_probability']:.4f}")
        print(f"  fraction of decoded path: {s['viterbi_fraction']:.4f}")
        print(f"  model-implied expected duration: {s['expected_duration_days']:.1f} days")
        d = durations.get(rank)
        if d:
            print(f"  actual decoded episodes: {d['n_episodes']}, "
                  f"mean length {d['mean_duration']:.1f} days, max {d['max_duration']} days")
        print()


def cmd_verify(args):
    import random
    import numpy as np
    from scipy import stats as sstats

    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}")

    # 1. Gaussian pdf vs scipy
    ok = True
    for x, mean, var in [(0.5, 0.0, 1.0), (-1.2, 0.3, 2.5), (3.0, 3.0, 0.5)]:
        ours = hmm.gaussian_pdf(x, mean, var)
        theirs = sstats.norm.pdf(x, loc=mean, scale=math.sqrt(var))
        if not math.isclose(ours, theirs, abs_tol=1e-9):
            ok = False
    check("gaussian_pdf matches scipy.stats.norm.pdf", ok)

    # 2. forward algorithm log-likelihood vs brute-force enumeration
    #    over all possible state sequences (feasible only for a short
    #    sequence and few states, but an exact, code-independent check)
    model = hmm.GaussianHMM([0.0, 3.0], [1.0, 1.0], [[0.7, 0.3], [0.4, 0.6]], [0.6, 0.4])
    obs, _ = model.simulate(6, seed=1)
    forward_ll = model.log_likelihood(obs)

    import itertools
    total_prob = 0.0
    for path in itertools.product(range(2), repeat=len(obs)):
        prob = model.initial[path[0]] * hmm.gaussian_pdf(obs[0], model.means[path[0]], model.variances[path[0]])
        for t in range(1, len(obs)):
            prob *= model.transition[path[t - 1]][path[t]]
            prob *= hmm.gaussian_pdf(obs[t], model.means[path[t]], model.variances[path[t]])
        total_prob += prob
    check("forward algorithm log-likelihood matches brute-force enumeration over all state paths",
          math.isclose(forward_ll, math.log(total_prob), abs_tol=1e-8))

    # 3. Viterbi path matches brute-force max over all state sequences
    viterbi_path = tuple(model.viterbi(obs))
    best_prob = -1.0
    best_path = None
    for path in itertools.product(range(2), repeat=len(obs)):
        prob = model.initial[path[0]] * hmm.gaussian_pdf(obs[0], model.means[path[0]], model.variances[path[0]])
        for t in range(1, len(obs)):
            prob *= model.transition[path[t - 1]][path[t]]
            prob *= hmm.gaussian_pdf(obs[t], model.means[path[t]], model.variances[path[t]])
        if prob > best_prob:
            best_prob = prob
            best_path = path
    check("Viterbi path matches brute-force max over all state paths", viterbi_path == best_path)

    # 4. gamma matches brute-force marginal posterior P(S_t=i | X_1:T)
    alpha, scales = model.forward(obs)
    beta = model.backward(obs, scales)
    gamma = model.gamma(alpha, beta)
    t_check = 3
    marginal = [0.0, 0.0]
    for path in itertools.product(range(2), repeat=len(obs)):
        prob = model.initial[path[0]] * hmm.gaussian_pdf(obs[0], model.means[path[0]], model.variances[path[0]])
        for t in range(1, len(obs)):
            prob *= model.transition[path[t - 1]][path[t]]
            prob *= hmm.gaussian_pdf(obs[t], model.means[path[t]], model.variances[path[t]])
        marginal[path[t_check]] += prob
    marginal = [m / total_prob for m in marginal]
    check(f"gamma at t={t_check} matches brute-force marginal posterior",
          math.isclose(gamma[t_check][0], marginal[0], abs_tol=1e-8))

    # 5. EM log-likelihood never decreases (checked on a longer real fit)
    true_model = hmm.GaussianHMM([0.0, 4.0], [1.0, 1.0], [[0.9, 0.1], [0.15, 0.85]], [0.5, 0.5])
    long_obs, _ = true_model.simulate(2000, seed=2)
    init_model = hmm.random_init(long_obs, 2, seed=3)
    fit_result = init_model.fit(long_obs, n_iter=100)
    lls = fit_result["log_likelihoods"]
    check("Baum-Welch log-likelihood is non-decreasing across iterations",
          all(b >= a - 1e-8 for a, b in zip(lls, lls[1:])))

    # 6. EM recovers known parameters
    fitted = fit_result["model"]
    fitted_means_sorted = sorted(fitted.means)
    check("Baum-Welch recovers known means from synthetic data",
          math.isclose(fitted_means_sorted[0], 0.0, abs_tol=0.3) and
          math.isclose(fitted_means_sorted[1], 4.0, abs_tol=0.3))

    # 7. n_free_parameters vs direct count for a small case
    check("n_free_parameters matches direct parameter count for K=2",
          ms.n_free_parameters(2) == (2 - 1) + 2 * (2 - 1) + 2 + 2)

    # 8. stationary distribution vs numpy eigenvector
    transition = [[0.9, 0.1], [0.2, 0.8]]
    ours = fx._stationary_distribution(transition)
    eigvals, eigvecs = np.linalg.eig(np.array(transition).T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    vec = np.real(eigvecs[:, idx])
    vec = vec / vec.sum()
    check("stationary distribution matches numpy eigenvector",
          all(math.isclose(ours[i], vec[i], abs_tol=1e-6) for i in range(2)))

    # 9. real EUR/USD data: fitted regimes show a substantial
    #    volatility separation and are persistent
    returns = fx.load_log_returns(DATA_PATH)
    real_result = fx.fit_regime_model(returns, n_states=2, n_restarts=3, seed=1)
    real_model = real_result["model"]
    order = real_result["state_order_low_to_high_vol"]
    check("real EUR/USD 2-state fit shows a substantial volatility separation",
          real_model.variances[order[1]] / real_model.variances[order[0]] > 1.3)
    check("real EUR/USD regimes are persistent (self-transition > 0.95)",
          all(real_model.transition[i][i] > 0.95 for i in range(2)))

    passed = sum(1 for _, ok in checks if ok)
    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(description="Hidden Markov model engine CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("simulate", help="simulate from a Gaussian HMM")
    p1.add_argument("--means", type=str, required=True, help="comma-separated state means")
    p1.add_argument("--variances", type=str, required=True, help="comma-separated state variances")
    p1.add_argument("--transition", type=str, required=True, help="rows separated by ';', entries by ','")
    p1.add_argument("--length", type=int, default=20)
    p1.add_argument("--seed", type=int, default=1)
    p1.set_defaults(func=cmd_simulate)

    p2 = sub.add_parser("fit", help="simulate then fit via Baum-Welch, check recovery")
    p2.add_argument("--means", type=str, required=True)
    p2.add_argument("--variances", type=str, required=True)
    p2.add_argument("--transition", type=str, required=True)
    p2.add_argument("--length", type=int, default=2000)
    p2.add_argument("--restarts", type=int, default=5)
    p2.add_argument("--seed", type=int, default=1)
    p2.set_defaults(func=cmd_fit)

    p3 = sub.add_parser("selectstates", help="AIC/BIC selection of the number of hidden states on EUR/USD")
    p3.add_argument("--k-range", type=str, default="1,2,3")
    p3.add_argument("--restarts", type=int, default=3)
    p3.set_defaults(func=cmd_selectstates)

    p4 = sub.add_parser("fxregimes", help="fit and summarise EUR/USD volatility regimes")
    p4.add_argument("--n-states", type=int, default=2)
    p4.add_argument("--restarts", type=int, default=5)
    p4.set_defaults(func=cmd_fxregimes)

    p5 = sub.add_parser("verify", help="cross-check implementation against analytical/oracle/brute-force values")
    p5.set_defaults(func=cmd_verify)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

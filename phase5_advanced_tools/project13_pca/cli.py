"""Command-line interface for PCA from scratch.

Usage
-----
    python3 cli.py pca       --data PATH [--lags N] [--k K] [--std]
    python3 cli.py scree     --data PATH [--lags N] [--std]
    python3 cli.py reconstruct --data PATH [--lags N] [--k K]
    python3 cli.py whiten    --data PATH [--lags N] [--k K]
    python3 cli.py verify

Examples
--------
    python3 cli.py pca    --data data/DEXUSEU_returns.csv --lags 5 --k 3
    python3 cli.py scree  --data data/DEXUSEU_returns.csv --lags 5
    python3 cli.py reconstruct --data data/DEXUSEU_returns.csv --lags 5 --k 2
    python3 cli.py verify
"""

import argparse
import math
import os
import sys

import pca as pc

_HERE = os.path.dirname(os.path.abspath(__file__))
_P12  = os.path.join(_HERE, "..", "project12_matrix_engine")
sys.path.insert(0, _P12)
import matrix_engine as mx


def load_and_build(args):
    series = pc.load_csv(args.data)
    lags   = getattr(args, "lags", 1)
    X      = pc.lag_matrix(series, n_lags=lags)
    return X


def cmd_pca(args):
    X = load_and_build(args)
    n, p = len(X), len(X[0])
    k    = min(args.k, p)
    r    = pc.pca(X, n_components=k, standardise_data=args.std)

    print("PCA  n={:,}  p={}  k={}".format(n, p, k))
    print()
    print("{:>4}  {:>14}  {:>14}  {:>14}".format(
        "PC", "eigenvalue", "explained", "cumulative"))
    for i in range(k):
        print("{:>4}  {:>14.6f}  {:>13.2%}  {:>13.2%}".format(
            i+1,
            r["eigenvalues"][i],
            r["explained_ratio"][i],
            r["cumulative_ratio"][i],
        ))
    print()
    k95 = pc.n_components_for_variance(r, 0.95)
    print("Components for 95% variance: {}".format(k95))


def cmd_scree(args):
    X   = load_and_build(args)
    n, p = len(X), len(X[0])
    r   = pc.pca(X, n_components=p, standardise_data=args.std)
    print("Scree plot  n={:,}  p={}".format(n, p))
    print()
    print("{:>4}  {:>14}  {:>14}".format("PC", "eigenvalue", "explained"))
    for i in range(p):
        bar = "#" * int(r["explained_ratio"][i] * 40)
        print("{:>4}  {:>14.6f}  {:>13.2%}  {}".format(
            i+1, r["eigenvalues"][i], r["explained_ratio"][i], bar))


def cmd_reconstruct(args):
    X   = load_and_build(args)
    n, p = len(X), len(X[0])
    r   = pc.pca(X, n_components=p)
    print("Reconstruction error  n={:,}  p={}".format(n, p))
    print()
    print("{:>4}  {:>16}".format("k", "MSE"))
    for k in range(1, min(args.k + 1, p + 1)):
        err = pc.reconstruction_error(X, r, n_components=k)
        print("{:>4}  {:>16.8e}".format(k, err))


def cmd_whiten(args):
    X   = load_and_build(args)
    n, p = len(X), len(X[0])
    k   = min(args.k, p)
    r   = pc.pca(X, n_components=k)
    w   = pc.whitened_scores(r)
    print("Whitened scores  n={:,}  k={}".format(n, k))
    for c in range(k):
        col_var = sum(w[i][c]**2 for i in range(n)) / (n - 1)
        print("  PC{}: variance = {:.6f} (should be 1.0)".format(c+1, col_var))


def cmd_verify(args):
    checks = []

    # ---- Preprocessing ----
    X = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    Xc, means = pc.centre(X)
    checks.append((abs(means[0] - 2.0) < 1e-10 and abs(means[1] - 5.0) < 1e-10,
                   "column means correct"))
    col0_mean = sum(Xc[i][0] for i in range(3)) / 3
    checks.append((abs(col0_mean) < 1e-10, "centre gives zero-mean columns"))

    # ---- Covariance matrix ----
    Xperfect = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    Xpc, _ = pc.centre(Xperfect)
    C = pc.covariance_matrix(Xpc)
    checks.append((abs(C[0][1] - 1.0) < 1e-8, "cov of perfectly correlated = 1"))
    checks.append((abs(C[0][1] - C[1][0]) < 1e-12, "covariance matrix symmetric"))

    # ---- Jacobi eigenvalues ----
    Csym = [[2.0, 1.0], [1.0, 2.0]]
    ev, V = pc.jacobi_eigen(Csym)
    checks.append((abs(sum(ev) - mx.trace(Csym)) < 1e-8,
                   "sum of eigenvalues = trace"))
    checks.append((allclose(sorted(ev), [1.0, 3.0], tol=1e-8),
                   "eigenvalues of [[2,1],[1,2]] = 1 and 3"))

    # VᵀV = I
    Vt  = mx.transpose(V)
    VtV = mx.matmul(Vt, V)
    I2  = mx.identity(2)
    err = max(abs(VtV[i][j] - I2[i][j]) for i in range(2) for j in range(2))
    checks.append((err < 1e-8, "Jacobi eigenvectors orthonormal"))

    # Av = λv for each eigenpair
    for k in range(2):
        vk  = [V[r][k] for r in range(2)]
        Cvk = mx.matvec(Csym, vk)
        lv  = mx.vec_scale(ev[k], vk)
        err_k = mx.norm(mx.vec_sub(Cvk, lv))
        checks.append((err_k < 1e-7, "Cv = λv for eigenpair {}".format(k)))

    # ---- Full PCA ----
    import random
    rng = random.Random(42)
    X2 = [[rng.gauss(0,2) + rng.gauss(0,0.5),
            rng.gauss(0,2) - rng.gauss(0,0.5)] for _ in range(200)]
    r  = pc.pca(X2, n_components=2)
    checks.append((abs(sum(r["explained_ratio"]) - 1.0) < 1e-6,
                   "explained ratios sum to 1"))
    checks.append((r["eigenvalues"][0] > r["eigenvalues"][1],
                   "eigenvalues sorted descending"))
    checks.append((r["cumulative_ratio"][-1] > r["cumulative_ratio"][0],
                   "cumulative ratio increases"))

    # Scores zero-mean
    n_obs = len(r["scores"])
    mean0 = sum(r["scores"][i][0] for i in range(n_obs)) / n_obs
    checks.append((abs(mean0) < 1e-6, "PC1 scores have zero mean"))

    # Component orthogonality
    comp = r["components"]
    dot01 = sum(comp[j][0] * comp[j][1] for j in range(2))
    checks.append((abs(dot01) < 1e-6, "PC1 and PC2 orthogonal"))

    # ---- Reconstruction ----
    X3 = [[1.0, 2.0], [3.0, 4.0], [5.0, 0.0]]
    r3 = pc.pca(X3, n_components=2)
    err_full = pc.reconstruction_error(X3, r3, n_components=2)
    checks.append((err_full < 1e-10, "full reconstruction exact"))

    # ---- Whitening ----
    r_w  = pc.pca(X2, n_components=2)
    w    = pc.whitened_scores(r_w)
    var0 = sum(w[i][0]**2 for i in range(n_obs)) / (n_obs - 1)
    checks.append((abs(var0 - 1.0) < 0.05, "whitened PC1 variance ≈ 1"))

    # ---- Lag matrix ----
    s = list(range(10))
    L = pc.lag_matrix(s, n_lags=3)
    checks.append((len(L) == 8, "lag_matrix length = n - n_lags + 1"))
    checks.append((L[0] == [2.0, 1.0, 0.0], "lag_matrix first row correct"))

    passed = sum(1 for ok, _ in checks if ok)
    for ok, label in checks:
        print("{:>5}  {}".format("PASS" if ok else "FAIL", label))
    print("-" * 60)
    print("{}/{} checks passed".format(passed, len(checks)))
    if passed != len(checks):
        sys.exit(1)


def allclose(a, b, tol=1e-6):
    if isinstance(a, list):
        return all(abs(x - y) <= tol for x, y in zip(a, b))
    return abs(a - b) <= tol


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub    = parser.add_subparsers(dest="command")

    for cmd in ("pca", "scree", "reconstruct", "whiten"):
        p = sub.add_parser(cmd)
        p.add_argument("--data", required=True)
        p.add_argument("--lags", type=int, default=1)
        p.add_argument("--k", type=int, default=5)
        p.add_argument("--std", action="store_true")
        fn = {"pca": cmd_pca, "scree": cmd_scree,
              "reconstruct": cmd_reconstruct, "whiten": cmd_whiten}[cmd]
        p.set_defaults(func=fn)

    p = sub.add_parser("verify")
    p.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()

import argparse
import math
import os
import sys

import linear_algebra as la
import jacobi_eigen as je
import svd
import fx_application as fx

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


def _parse_matrix(text: str) -> la.Matrix:
    return [[float(x) for x in row.split(",")] for row in text.strip().split(";")]


def cmd_decompose(args):
    a = _parse_matrix(args.matrix)
    u, s, v = svd.singular_value_decompose(a)
    print(f"singular values: {[round(x, 6) for x in s]}")
    print("U:")
    for row in u:
        print("  ", [round(x, 4) for x in row])
    print("V:")
    for row in v:
        print("  ", [round(x, 4) for x in row])
    print(f"rank estimate: {svd.rank_estimate(s)}")
    print(f"condition number: {svd.condition_number(s):.4f}")


def cmd_lowrank(args):
    a = _parse_matrix(args.matrix)
    u, s, v = svd.singular_value_decompose(a)
    approx = svd.low_rank_approximation(u, s, v, args.k)
    error = la.frobenius_norm(la.subtract(a, approx))
    print(f"rank-{args.k} approximation:")
    for row in approx:
        print("  ", [round(x, 4) for x in row])
    print(f"Frobenius reconstruction error: {error:.6f}")
    expected = math.sqrt(sum(si ** 2 for si in s[args.k:]))
    print(f"expected error from Eckart-Young (sqrt of dropped singular values squared): {expected:.6f}")


def cmd_pseudoinverse(args):
    a = _parse_matrix(args.matrix)
    u, s, v = svd.singular_value_decompose(a)
    a_plus = svd.pseudoinverse(u, s, v)
    print("Moore-Penrose pseudoinverse:")
    for row in a_plus:
        print("  ", [round(x, 4) for x in row])


def cmd_pca_vs_svd(args):
    returns = fx.load_log_returns(DATA_PATH)
    windows = [int(x) for x in args.windows.split(",")]
    x = fx.build_volatility_matrix(returns, windows)
    print(f"volatility matrix: {len(x)} rows x {len(x[0])} columns (windows={windows})\n")

    result = fx.compare_pca_methods(x)
    print("variance explained (SVD-based PCA):", [round(v, 4) for v in result["svd"]["variance_explained"]])
    print("variance explained (rescaled covariance PCA):",
          [round(v, 4) for v in result["rescaled"]["variance_explained"]])
    print()
    print(f"leading-vector gap, naive covariance PCA vs SVD:    {result['naive_vs_svd_leading_vector_gap']:.6e}")
    print(f"leading-vector gap, rescaled covariance PCA vs SVD: {result['rescaled_vs_svd_leading_vector_gap']:.2e}")


def cmd_verify(args):
    import random
    import numpy as np

    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}")

    rng = random.Random(1)
    n = 5
    m = [[rng.gauss(0, 1) for _ in range(n)] for _ in range(n)]
    a = la.matmul(m, la.transpose(m))
    vals, vecs = je.sorted_eigen(a)
    numpy_vals, numpy_vecs = np.linalg.eigh(np.array(a))
    numpy_vals_sorted = sorted(numpy_vals, reverse=True)
    check("Jacobi eigenvalues match numpy.linalg.eigh",
          all(math.isclose(v1, v2, abs_tol=1e-6) for v1, v2 in zip(vals, numpy_vals_sorted)))

    a2 = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(6)]
    u, s, v = svd.singular_value_decompose(a2)
    numpy_u, numpy_s, numpy_vt = np.linalg.svd(np.array(a2), full_matrices=False)
    check("singular values match numpy.linalg.svd",
          all(math.isclose(s1, s2, abs_tol=1e-6) for s1, s2 in zip(s, numpy_s)))

    recon = svd.reconstruct(u, s, v)
    check("SVD reconstruction matches original matrix",
          la.frobenius_norm(la.subtract(a2, recon)) < 1e-6)

    a_plus = svd.pseudoinverse(u, s, v)
    numpy_pinv = np.linalg.pinv(np.array(a2))
    check("pseudoinverse matches numpy.linalg.pinv",
          all(math.isclose(a_plus[i][j], numpy_pinv[i][j], abs_tol=1e-5)
              for i in range(len(a_plus)) for j in range(len(a_plus[0]))))

    all_match = True
    for k in range(len(s) + 1):
        approx = svd.low_rank_approximation(u, s, v, k)
        error = la.frobenius_norm(la.subtract(a2, approx))
        expected = math.sqrt(sum(si ** 2 for si in s[k:]))
        if not math.isclose(error, expected, abs_tol=1e-6):
            all_match = False
            break
    check("Eckart-Young error matches theory at every rank k", all_match)

    cond_ours = svd.condition_number(s)
    cond_numpy = np.linalg.cond(np.array(a2))
    check("condition number matches numpy.linalg.cond",
          math.isclose(cond_ours, cond_numpy, rel_tol=1e-4))

    returns = fx.load_log_returns(DATA_PATH)
    x = fx.build_volatility_matrix(returns, windows=[5, 10, 20, 60])
    result = fx.compare_pca_methods(x)
    check("real EUR/USD: rescaled covariance PCA matches SVD to within 1e-8",
          result["rescaled_vs_svd_leading_vector_gap"] < 1e-8)
    check("real EUR/USD: naive covariance PCA shows a real, measurable gap from SVD",
          result["naive_vs_svd_leading_vector_gap"] > 1e-4)
    check("real EUR/USD: first principal component dominates (explains > 70% of variance)",
          result["svd"]["variance_explained"][0] > 0.7)

    x_centered = fx.center_columns(x)
    x_np = np.array(x_centered)
    cov_np = (x_np.T @ x_np) / (len(x_centered) - 1)
    numpy_eigvals = sorted(np.linalg.eigvalsh(cov_np), reverse=True)
    our_eigvals = result["svd"]["eigenvalues"]
    check("SVD-derived eigenvalues match numpy covariance eigenvalues on real data",
          all(math.isclose(a, b, rel_tol=1e-3) for a, b in zip(our_eigvals, numpy_eigvals)))

    passed = sum(1 for _, ok in checks if ok)
    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(description="SVD engine CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("decompose", help="compute the SVD of a matrix")
    p1.add_argument("--matrix", type=str, required=True, help="rows separated by ';', entries by ','")
    p1.set_defaults(func=cmd_decompose)

    p2 = sub.add_parser("lowrank", help="best rank-k approximation")
    p2.add_argument("--matrix", type=str, required=True)
    p2.add_argument("--k", type=int, required=True)
    p2.set_defaults(func=cmd_lowrank)

    p3 = sub.add_parser("pseudoinverse", help="Moore-Penrose pseudoinverse")
    p3.add_argument("--matrix", type=str, required=True)
    p3.set_defaults(func=cmd_pseudoinverse)

    p4 = sub.add_parser("pcavssvd", help="PCA via covariance vs PCA via SVD, on real EUR/USD volatility data")
    p4.add_argument("--windows", type=str, default="5,10,20,60")
    p4.set_defaults(func=cmd_pca_vs_svd)

    p5 = sub.add_parser("verify", help="cross-check implementation against numpy oracles and known theorems")
    p5.set_defaults(func=cmd_verify)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

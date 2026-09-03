"""Command-line front end for the Matrix Engine."""
import argparse
import ast
import math
import sys

import matrix_engine as mx


def parse_matrix(s):
    try:
        A = ast.literal_eval(s)
        return [[float(v) for v in row] for row in A]
    except Exception:
        sys.exit("cannot parse matrix: {!r}".format(s))


def parse_vector(s):
    try:
        v = ast.literal_eval(s)
        return [float(x) for x in v]
    except Exception:
        sys.exit("cannot parse vector: {!r}".format(s))


def print_matrix(A, label=""):
    if label:
        print("{}:".format(label))
    for row in A:
        print("  [{}]".format("  ".join("{:12.8g}".format(v) for v in row)))


def print_vector(v, label=""):
    if label:
        print("{}: [{}]".format(label, "  ".join("{:.8g}".format(x) for x in v)))
    else:
        print("[{}]".format("  ".join("{:.8g}".format(x) for x in v)))


def cmd_matmul(args):
    A = parse_matrix(args.A)
    B = parse_matrix(args.B)
    C = mx.matmul(A, B)
    print_matrix(C, "A @ B")


def cmd_solve(args):
    A = parse_matrix(args.A)
    b = parse_vector(args.b)
    x = mx.solve(A, b)
    print_vector(x, "x")
    Ax = mx.matvec(A, x)
    r  = mx.norm(mx.vec_sub(Ax, b))
    print("residual ||Ax-b||: {:.3e}".format(r))


def cmd_inverse(args):
    A  = parse_matrix(args.A)
    Ai = mx.inverse(A)
    print_matrix(Ai, "A⁻¹")


def cmd_det(args):
    A = parse_matrix(args.A)
    print("det(A) = {:.12g}".format(mx.det_lu(A)))


def cmd_lu(args):
    A = parse_matrix(args.A)
    L, U, P, n_swaps = mx.lu_decompose(A)
    print_matrix(L, "L")
    print_matrix(U, "U")
    print_matrix(P, "P")
    print("swaps: {}".format(n_swaps))


def cmd_qr(args):
    A = parse_matrix(args.A)
    Q, R = mx.qr_decompose(A)
    print_matrix(Q, "Q")
    print_matrix(R, "R")


def cmd_lstsq(args):
    A = parse_matrix(args.A)
    b = parse_vector(args.b)
    x = mx.least_squares(A, b)
    print_vector(x, "x_hat")
    print("residual ||Ax-b||: {:.6g}".format(mx.residual_norm(A, b, x)))


def cmd_eigen(args):
    A  = parse_matrix(args.A)
    ev = mx.eigenvalues_2x2(A)
    print("eigenvalues:", ev)


def cmd_power(args):
    A = parse_matrix(args.A)
    lam, v = mx.power_iteration(A)
    print("dominant eigenvalue: {:.10g}".format(lam))
    print_vector(v, "eigenvector")


def cmd_cond(args):
    A   = parse_matrix(args.A)
    kap = mx.condition_number(A)
    print("condition number κ(A) ≈ {:.6g}".format(kap))


def cmd_verify(args):
    checks = []

    # Basic ops
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[5.0, 6.0], [7.0, 8.0]]

    checks.append((mx.add(A, B) == [[6,8],[10,12]],
                   "matrix addition"))
    checks.append((mx.subtract(B, A) == [[4,4],[4,4]],
                   "matrix subtraction"))
    checks.append((mx.transpose(A) == [[1,3],[2,4]],
                   "transpose"))
    checks.append((mx.matmul(A, B) == [[19,22],[43,50]],
                   "2×2 matrix multiply"))
    checks.append((mx.matvec(A, [1,1]) == [3.0, 7.0],
                   "matvec"))
    checks.append((abs(mx.trace(A) - 5.0) < 1e-10,
                   "trace = 5"))

    # Determinant
    checks.append((abs(mx.determinant(A) - (-2.0)) < 1e-10,
                   "det([[1,2],[3,4]]) = -2 (cofactor)"))
    checks.append((abs(mx.det_lu(A) - (-2.0)) < 1e-10,
                   "det([[1,2],[3,4]]) = -2 (LU)"))

    A3 = [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,10.0]]
    checks.append((abs(mx.det_lu(A3) - (-3.0)) < 1e-8,
                   "det of 3×3 = -3"))

    # LU: PA = LU
    L, U, P, _ = mx.lu_decompose(A3)
    PA  = mx.matmul(P, A3)
    LU  = mx.matmul(L, U)
    err = max(abs(PA[i][j] - LU[i][j])
              for i in range(3) for j in range(3))
    checks.append((err < 1e-10, "LU: PA = LU"))

    # Solve
    Asys = [[3.0, 1.0, -1.0],
            [2.0, 4.0, 1.0],
            [-1.0, 2.0, 5.0]]
    bsys = [4.0, 1.0, 1.0]
    x    = mx.solve(Asys, bsys)
    Ax   = mx.matvec(Asys, x)
    res  = mx.norm(mx.vec_sub(Ax, bsys))
    checks.append((res < 1e-10, "solve: residual < 1e-10"))

    # Inverse: A @ A⁻¹ = I
    Ai  = mx.inverse(A3)
    AAi = mx.matmul(A3, Ai)
    I3  = mx.identity(3)
    err = max(abs(AAi[i][j] - I3[i][j])
              for i in range(3) for j in range(3))
    checks.append((err < 1e-10, "A @ A⁻¹ = I"))

    # Eigenvalues
    Asym = [[2.0, 1.0], [1.0, 2.0]]
    ev   = mx.eigenvalues_2x2(Asym)
    checks.append((abs(sum(ev) - mx.trace(Asym)) < 1e-10,
                   "sum of eigenvalues = trace"))
    checks.append((abs(ev[0] * ev[1] - mx.det_lu(Asym)) < 1e-10,
                   "product of eigenvalues = det"))

    # Power iteration
    lam, v = mx.power_iteration(Asym)
    checks.append((abs(lam - 3.0) < 1e-6,
                   "power iteration dominant eigenvalue = 3"))
    Av = mx.matvec(Asym, v)
    lv = mx.vec_scale(lam, v)
    err = mx.norm(mx.vec_sub(Av, lv))
    checks.append((err < 1e-5, "Av = λv for dominant eigenpair"))

    # QR
    Aqr = [[1.0,2.0],[3.0,4.0],[5.0,6.0]]
    Q, R = mx.qr_decompose(Aqr)
    # A = QR
    QR   = mx.matmul(Q, R)
    err  = max(abs(QR[i][j] - Aqr[i][j])
               for i in range(3) for j in range(2))
    checks.append((err < 1e-10, "QR: A = QR"))
    # Q orthonormal
    QtQ  = mx.matmul(mx.transpose(Q), Q)
    I2   = mx.identity(2)
    err  = max(abs(QtQ[i][j] - I2[i][j]) for i in range(2) for j in range(2))
    checks.append((err < 1e-10, "QR: QᵀQ = I"))

    # Least squares
    Als = [[1.0,0.0],[1.0,1.0],[1.0,2.0]]
    bls = [1.0, 2.0, 4.0]
    xls = mx.least_squares(Als, bls)
    AtA = mx.matmul(mx.transpose(Als), Als)
    Atb = mx.matvec(mx.transpose(Als), bls)
    lhs = mx.matvec(AtA, xls)
    err = mx.norm(mx.vec_sub(lhs, Atb))
    checks.append((err < 1e-8, "least squares satisfies normal equations"))

    # Condition number
    kap_I = mx.condition_number(mx.identity(3))
    checks.append((abs(kap_I - 1.0) < 0.1, "κ(I) ≈ 1"))
    kap_d = mx.condition_number([[2.0, 0.0], [0.0, 1.0]])
    checks.append((abs(kap_d - 2.0) < 0.5, "κ([[2,0],[0,1]]) ≈ 2"))

    passed = sum(1 for ok, _ in checks if ok)
    for ok, label in checks:
        print("{:>5}  {}".format("PASS" if ok else "FAIL", label))
    print("-" * 60)
    print("{}/{} checks passed".format(passed, len(checks)))
    if passed != len(checks):
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub    = parser.add_subparsers(dest="command")

    for cmd, help_text in [
        ("matmul",  "matrix multiply A @ B"),
        ("solve",   "solve Ax = b"),
        ("inverse", "matrix inverse"),
        ("det",     "determinant"),
        ("lu",      "LU decomposition"),
        ("qr",      "QR decomposition"),
        ("lstsq",   "least-squares solve"),
        ("eigen",   "eigenvalues of 2×2"),
        ("power",   "power iteration"),
        ("cond",    "condition number"),
        ("verify",  "cross-check all results"),
    ]:
        p = sub.add_parser(cmd, help=help_text)
        if cmd not in ("verify",):
            p.add_argument("--A", required=True)
        if cmd in ("matmul", "qr"):
            p.add_argument("--B", required=(cmd == "matmul"))
        if cmd in ("solve", "lstsq"):
            p.add_argument("--b", required=True)
        fn = {
            "matmul": cmd_matmul, "solve": cmd_solve,
            "inverse": cmd_inverse, "det": cmd_det, "lu": cmd_lu,
            "qr": cmd_qr, "lstsq": cmd_lstsq, "eigen": cmd_eigen,
            "power": cmd_power, "cond": cmd_cond, "verify": cmd_verify,
        }[cmd]
        p.set_defaults(func=fn)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()

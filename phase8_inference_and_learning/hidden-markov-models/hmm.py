import math
import random
import statistics
from typing import List, Tuple, Dict, Optional


def gaussian_pdf(x: float, mean: float, var: float) -> float:
    if var <= 0:
        raise ValueError("var must be positive")
    return math.exp(-0.5 * (x - mean) ** 2 / var) / math.sqrt(2 * math.pi * var)


def gaussian_log_pdf(x: float, mean: float, var: float) -> float:
    if var <= 0:
        raise ValueError("var must be positive")
    return -0.5 * math.log(2 * math.pi * var) - 0.5 * (x - mean) ** 2 / var


class GaussianHMM:
    """A hidden Markov model with K discrete latent states and
    Gaussian emissions: at each time t, an unobserved state
    S_t in {0,...,K-1} evolves according to a first-order Markov chain
    with transition matrix A (A[i][j] = P(S_t=j | S_{t-1}=i)) and
    initial distribution pi, and the observed value X_t is drawn from
    N(mean[S_t], var[S_t]). The states themselves are never observed;
    everything this class computes is inferred purely from the
    observation sequence X_1,...,X_T.
    """

    def __init__(self, means: List[float], variances: List[float],
                 transition: List[List[float]], initial: List[float]):
        k = len(means)
        if len(variances) != k or len(transition) != k or len(initial) != k:
            raise ValueError("means, variances, transition, and initial must have consistent length K")
        if any(len(row) != k for row in transition):
            raise ValueError("transition must be a K x K matrix")
        self.n_states = k
        self.means = list(means)
        self.variances = list(variances)
        self.transition = [list(row) for row in transition]
        self.initial = list(initial)

    def forward(self, observations: List[float]) -> Tuple[List[List[float]], List[float]]:
        """The scaled forward algorithm. Raw (unscaled) forward
        probabilities alpha_t(j) = P(X_1,...,X_t, S_t=j) decay
        geometrically with t (each is a sum of products of
        probabilities, all less than 1), underflowing a float within a
        few hundred observations for any realistic problem. Rabiner's
        (1989) scaling trick renormalises alpha_t to sum to 1 at every
        t, dividing by a scaling factor c_t = sum_j alpha_t(j)
        (computed *before* the unscaled values would have underflowed,
        since the recursion uses only the previous step's already-
        renormalised values); the log-likelihood of the whole sequence
        is then recovered exactly as sum_t log(c_t), since each c_t is
        precisely the total probability mass the raw recursion would
        have carried forward from step t-1 to step t.
        """
        t_len = len(observations)
        k = self.n_states
        alpha = [[0.0] * k for _ in range(t_len)]
        scales = [0.0] * t_len

        for j in range(k):
            alpha[0][j] = self.initial[j] * gaussian_pdf(observations[0], self.means[j], self.variances[j])
        scales[0] = sum(alpha[0])
        if scales[0] <= 0:
            raise ValueError("initial forward probabilities collapsed to zero; check parameters")
        alpha[0] = [a / scales[0] for a in alpha[0]]

        for t in range(1, t_len):
            for j in range(k):
                total = sum(alpha[t - 1][i] * self.transition[i][j] for i in range(k))
                alpha[t][j] = total * gaussian_pdf(observations[t], self.means[j], self.variances[j])
            scales[t] = sum(alpha[t])
            if scales[t] <= 0:
                raise ValueError(f"forward probabilities collapsed to zero at t={t}; check parameters")
            alpha[t] = [a / scales[t] for a in alpha[t]]

        return alpha, scales

    def backward(self, observations: List[float], scales: List[float]) -> List[List[float]]:
        """The scaled backward algorithm: beta_t(i) = P(X_{t+1},...,X_T
        | S_t=i), rescaled at every step by the *same* scaling factors
        c_t computed during the forward pass, which keeps
        alpha_t(i)*beta_t(i) proportional to the correct state
        posterior P(S_t=i | X_1:T) once renormalised (see `gamma`),
        rather than requiring a separate, independently-derived set of
        backward scaling factors.
        """
        t_len = len(observations)
        k = self.n_states
        beta = [[0.0] * k for _ in range(t_len)]
        beta[t_len - 1] = [1.0] * k

        for t in range(t_len - 2, -1, -1):
            for i in range(k):
                beta[t][i] = sum(
                    self.transition[i][j] * gaussian_pdf(observations[t + 1], self.means[j], self.variances[j])
                    * beta[t + 1][j]
                    for j in range(k)
                )
            beta[t] = [b / scales[t + 1] for b in beta[t]]

        return beta

    def log_likelihood(self, observations: List[float]) -> float:
        _, scales = self.forward(observations)
        return sum(math.log(c) for c in scales)

    def viterbi(self, observations: List[float]) -> List[int]:
        """The most likely single state sequence given the
        observations (as opposed to the marginal state posteriors
        `gamma` gives at each time separately), by dynamic programming
        in log-space:

            delta_t(j) = max_i [delta_{t-1}(i) + log A[i][j]] + log B_j(X_t)

        with delta_1(j) = log(pi_j) + log(B_j(X_1)), and a backpointer
        at each step recording which i achieved the max, so the full
        optimal path can be recovered by tracing backpointers from the
        final time step's best state back to the first. Working in log
        -space avoids the same underflow problem the forward algorithm
        has, and is exact here (no scaling-and-rescaling needed) since
        taking a running max, unlike a running sum, is unaffected by
        an additive shift applied uniformly to every candidate.
        """
        t_len = len(observations)
        k = self.n_states
        log_a = [[math.log(p) if p > 0 else float("-inf") for p in row] for row in self.transition]
        log_pi = [math.log(p) if p > 0 else float("-inf") for p in self.initial]

        delta = [[0.0] * k for _ in range(t_len)]
        psi = [[0] * k for _ in range(t_len)]

        for j in range(k):
            delta[0][j] = log_pi[j] + gaussian_log_pdf(observations[0], self.means[j], self.variances[j])

        for t in range(1, t_len):
            for j in range(k):
                candidates = [delta[t - 1][i] + log_a[i][j] for i in range(k)]
                best_i = max(range(k), key=lambda i: candidates[i])
                delta[t][j] = candidates[best_i] + gaussian_log_pdf(observations[t], self.means[j], self.variances[j])
                psi[t][j] = best_i

        path = [0] * t_len
        path[t_len - 1] = max(range(k), key=lambda j: delta[t_len - 1][j])
        for t in range(t_len - 2, -1, -1):
            path[t] = psi[t + 1][path[t + 1]]
        return path

    def gamma(self, alpha: List[List[float]], beta: List[List[float]]) -> List[List[float]]:
        """P(S_t=i | X_1:T) at every t, from the scaled forward and
        backward passes. Because both alpha and beta were rescaled
        (by the same c_t at each t, for beta), their product
        alpha_t(i)*beta_t(i) is only proportional to, not already
        equal to, the true posterior; renormalising across i at each t
        recovers the exact posterior regardless of the particular
        scaling convention used.
        """
        t_len = len(alpha)
        k = self.n_states
        result = [[0.0] * k for _ in range(t_len)]
        for t in range(t_len):
            unnormalized = [alpha[t][i] * beta[t][i] for i in range(k)]
            total = sum(unnormalized)
            result[t] = [u / total for u in unnormalized]
        return result

    def xi(self, observations: List[float], alpha: List[List[float]],
           beta: List[List[float]]) -> List[List[List[float]]]:
        """P(S_t=i, S_{t+1}=j | X_1:T) for t=0,...,T-2, proportional to
        alpha_t(i) * A[i][j] * B_j(X_{t+1}) * beta_{t+1}(j), renormalised
        so that summing over both i and j at each t gives 1. This is
        exactly the joint posterior needed to re-estimate the
        transition matrix in the Baum-Welch M-step: the expected
        number of i->j transitions is the sum of xi_t(i,j) over t.
        """
        t_len = len(observations)
        k = self.n_states
        result = []
        for t in range(t_len - 1):
            unnormalized = [[0.0] * k for _ in range(k)]
            for i in range(k):
                for j in range(k):
                    unnormalized[i][j] = (alpha[t][i] * self.transition[i][j]
                                           * gaussian_pdf(observations[t + 1], self.means[j], self.variances[j])
                                           * beta[t + 1][j])
            total = sum(sum(row) for row in unnormalized)
            result.append([[u / total for u in row] for row in unnormalized])
        return result

    def baum_welch_step(self, observations: List[float]) -> "GaussianHMM":
        """One iteration of the Baum-Welch (EM) algorithm: an E-step
        (compute gamma and xi from the current parameters via forward-
        backward) followed by an M-step (re-estimate every parameter
        as a gamma/xi-weighted maximum-likelihood update):

            pi_j_new = gamma_1(j)
            A[i][j]_new = sum_t xi_t(i,j) / sum_t gamma_t(i)
            mean_j_new = sum_t gamma_t(j) X_t / sum_t gamma_t(j)
            var_j_new = sum_t gamma_t(j) (X_t - mean_j_new)^2 / sum_t gamma_t(j)

        each of which is the closed-form weighted MLE for that
        parameter's role in a Gaussian/categorical model, with
        gamma_t(j) standing in for "how much of observation t belongs
        to state j" in place of a hard, observed assignment. Baum-Welch
        is the specific instance of the EM algorithm for HMMs, and each
        step of it is guaranteed not to decrease the observed-data
        log-likelihood (a general property of EM, verified directly on
        this implementation in the test suite rather than only cited).
        """
        alpha, scales = self.forward(observations)
        beta = self.backward(observations, scales)
        gamma = self.gamma(alpha, beta)
        xi = self.xi(observations, alpha, beta)

        t_len = len(observations)
        k = self.n_states

        new_initial = list(gamma[0])

        new_transition = [[0.0] * k for _ in range(k)]
        for i in range(k):
            denom = sum(gamma[t][i] for t in range(t_len - 1))
            for j in range(k):
                numer = sum(xi[t][i][j] for t in range(t_len - 1))
                new_transition[i][j] = numer / denom if denom > 0 else self.transition[i][j]

        new_means = [0.0] * k
        new_variances = [0.0] * k
        for j in range(k):
            weight_total = sum(gamma[t][j] for t in range(t_len))
            new_means[j] = sum(gamma[t][j] * observations[t] for t in range(t_len)) / weight_total
            new_variances[j] = sum(gamma[t][j] * (observations[t] - new_means[j]) ** 2
                                    for t in range(t_len)) / weight_total
            new_variances[j] = max(new_variances[j], 1e-10)

        return GaussianHMM(new_means, new_variances, new_transition, new_initial)

    def fit(self, observations: List[float], n_iter: int = 100, tol: float = 1e-6) -> Dict:
        """Repeated Baum-Welch iterations until the observed-data
        log-likelihood improves by less than `tol` between iterations,
        or `n_iter` is reached. The full log-likelihood trajectory is
        returned so that monotone non-decrease (the defining guarantee
        of EM) can be checked directly rather than assumed.
        """
        model = self
        log_likelihoods = [model.log_likelihood(observations)]
        for _ in range(n_iter):
            model = model.baum_welch_step(observations)
            ll = model.log_likelihood(observations)
            log_likelihoods.append(ll)
            if ll - log_likelihoods[-2] < tol:
                break
        return {"model": model, "log_likelihoods": log_likelihoods}


    def simulate(self, t_len: int, seed: Optional[int] = None) -> Tuple[List[float], List[int]]:
        """Simulate a sequence of observations and their (normally
        hidden) generating states, for use in validating the inference
        methods above against a known ground truth before trusting
        them on real, ambiguous data.
        """
        rng = random.Random(seed)
        states = []
        observations = []
        current = rng.choices(range(self.n_states), weights=self.initial)[0]
        for t in range(t_len):
            states.append(current)
            observations.append(rng.gauss(self.means[current], math.sqrt(self.variances[current])))
            current = rng.choices(range(self.n_states), weights=self.transition[current])[0]
        return observations, states


def random_init(observations: List[float], n_states: int, seed: Optional[int] = None) -> GaussianHMM:
    """A data-driven random initialisation for Baum-Welch: means are
    seeded at randomly chosen observations (rather than, say, all at
    the sample mean, which would make every state start identical and
    unable to differentiate during EM, a real failure mode of
    symmetric initialisation in any mixture-type model), variances all
    start at the overall sample variance, and the transition matrix
    starts with a mild self-transition preference (favouring
    persistence, a reasonable generic prior for most real regime-
    switching applications) plus a small random perturbation so that
    different random seeds do not all begin from an identical matrix.
    """
    rng = random.Random(seed)
    sample_var = statistics.variance(observations)
    seed_points = rng.sample(observations, n_states)
    means = list(seed_points)
    variances = [sample_var] * n_states

    transition = []
    for i in range(n_states):
        row = [rng.uniform(0.5, 1.5) for _ in range(n_states)]
        row[i] *= 3.0
        total = sum(row)
        transition.append([x / total for x in row])

    initial = [1.0 / n_states] * n_states
    return GaussianHMM(means, variances, transition, initial)

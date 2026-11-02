# EM estimation of the one-dimensional Generalized Hyperbolic distribution.
#
# X = mu + gamma * Z + sigma * sqrt(Z) * epsilon
# Z follows GIG(lambda, chi=1, psi).
#
# The restriction chi=1 identifies the parameterization. It does not impose
# E[Z]=1. The algorithm keeps only likelihood-nondecreasing EM updates.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats
from scipy.special import kve


def _log_bessel_k(order: float | np.ndarray, argument: np.ndarray) -> np.ndarray:
    """Return log(K_order(argument)) using exponentially scaled Bessel K."""
    argument = np.asarray(argument, dtype=float)
    scaled = np.maximum(kve(order, argument), np.finfo(float).tiny)
    return np.log(scaled) - argument


def _d_log_bessel_k_d_order(order: float, argument: np.ndarray) -> np.ndarray:
    """Fourth-order central derivative of log K with respect to its order."""
    h = 1.0e-4 * max(1.0, abs(float(order)))
    return (
        -_log_bessel_k(order + 2.0 * h, argument)
        + 8.0 * _log_bessel_k(order + h, argument)
        - 8.0 * _log_bessel_k(order - h, argument)
        + _log_bessel_k(order - 2.0 * h, argument)
    ) / (12.0 * h)


@dataclass(frozen=True)
class GHEMResult:
    lambda_: float
    chi: float
    psi: float
    mu: float
    gamma: float
    sigma: float
    loglik: float
    iterations: int
    converged: bool
    monotone: bool
    largest_loglik_drop: float
    loglik_path: tuple[float, ...]

    @property
    def scipy_parameters(self) -> tuple[float, float, float, float, float]:
        """Map the chi=1 NMVM parameters to scipy.stats.genhyperbolic."""
        b = self.gamma / self.sigma
        a = float(np.sqrt(self.psi + b * b))
        return self.lambda_, a, b, self.mu, self.sigma


def _observed_loglik(
    data: np.ndarray,
    lambda_: float,
    psi: float,
    mu: float,
    gamma: float,
    sigma: float,
) -> float:
    if psi <= 0.0 or sigma <= 0.0:
        return -np.inf
    b = gamma / sigma
    a = np.sqrt(psi + b * b)
    values = stats.genhyperbolic.logpdf(
        data,
        p=lambda_,
        a=a,
        b=b,
        loc=mu,
        scale=sigma,
    )
    return float(np.sum(values)) if np.all(np.isfinite(values)) else -np.inf


def _conditional_moments(
    data: np.ndarray,
    lambda_: float,
    psi: float,
    mu: float,
    gamma: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nu = lambda_ - 0.5
    chi_post = 1.0 + np.square((data - mu) / sigma)
    psi_post = psi + np.square(gamma / sigma)
    argument = np.sqrt(chi_post * psi_post)
    log_k_nu = _log_bessel_k(nu, argument)
    e_z = np.sqrt(chi_post / psi_post) * np.exp(
        _log_bessel_k(nu + 1.0, argument) - log_k_nu
    )
    e_inv_z = np.sqrt(psi_post / chi_post) * np.exp(
        _log_bessel_k(nu - 1.0, argument) - log_k_nu
    )
    e_log_z = (
        0.5 * np.log(chi_post / psi_post)
        + _d_log_bessel_k_d_order(nu, argument)
    )
    return e_z, e_inv_z, e_log_z


def _update_location_skew_scale(
    data: np.ndarray,
    e_z: np.ndarray,
    e_inv_z: np.ndarray,
) -> tuple[float, float, float]:
    n = float(data.size)
    a_sum = float(np.sum(e_inv_z))
    c_sum = float(np.sum(e_z))
    x_sum = float(np.sum(data))
    dx_sum = float(np.sum(e_inv_z * data))
    determinant = a_sum * c_sum - n * n
    if determinant <= 1.0e-10:
        raise FloatingPointError("Singular GH location/skewness M-step.")
    mu = (c_sum * dx_sum - n * x_sum) / determinant
    gamma = (a_sum * x_sum - n * dx_sum) / determinant
    residual = data - mu
    sigma2 = float(
        np.mean(e_inv_z * residual * residual)
        - 2.0 * gamma * np.mean(residual)
        + gamma * gamma * np.mean(e_z)
    )
    sigma = float(np.sqrt(max(sigma2, 1.0e-10)))
    return float(mu), float(gamma), sigma


def _update_mixing_parameters(
    lambda_old: float,
    psi_old: float,
    e_z: np.ndarray,
    e_log_z: np.ndarray,
) -> tuple[float, float]:
    n = float(e_z.size)
    sum_e_z = float(np.sum(e_z))
    sum_e_log_z = float(np.sum(e_log_z))

    def negative_q(parameters: np.ndarray) -> float:
        lambda_new, log_psi_new = map(float, parameters)
        psi_new = np.exp(log_psi_new)
        root = np.sqrt(psi_new)
        log_normalizer = (
            0.5 * lambda_new * log_psi_new
            - np.log(2.0)
            - float(_log_bessel_k(lambda_new, np.array([root]))[0])
        )
        q_value = (
            n * log_normalizer
            + (lambda_new - 1.0) * sum_e_log_z
            - 0.5 * psi_new * sum_e_z
        )
        return -q_value if np.isfinite(q_value) else 1.0e100

    result = optimize.minimize(
        negative_q,
        x0=np.array([lambda_old, np.log(psi_old)], dtype=float),
        method="L-BFGS-B",
        bounds=[(-30.0, 30.0), (-18.0, 18.0)],
        options={"ftol": 1.0e-12, "gtol": 1.0e-8, "maxiter": 300},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        return float(lambda_old), float(psi_old)
    return float(result.x[0]), float(np.exp(result.x[1]))


def fit_GH_EM(
    data: np.ndarray,
    *,
    max_iter: int = 1000,
    tolerance: float = 1.0e-7,
) -> GHEMResult:
    """Fit a univariate GH distribution by safeguarded generalized EM."""
    observations = np.asarray(data, dtype=float)
    observations = observations[np.isfinite(observations)]
    if observations.size < 20:
        raise ValueError("At least 20 finite observations are required.")

    mu = float(np.median(observations))
    sigma = float(max(np.std(observations), 1.0e-3))
    sample_skew = float(stats.skew(observations, bias=False))
    gamma = float(np.sign(sample_skew) * 0.15 * sigma)
    lambda_ = 1.0
    psi = 1.0
    current_loglik = _observed_loglik(
        observations, lambda_, psi, mu, gamma, sigma
    )
    path = [current_loglik]
    converged = False

    for _iteration in range(1, max_iter + 1):
        e_z, e_inv_z, e_log_z = _conditional_moments(
            observations, lambda_, psi, mu, gamma, sigma
        )
        proposed_mu, proposed_gamma, proposed_sigma = (
            _update_location_skew_scale(observations, e_z, e_inv_z)
        )
        proposed_lambda, proposed_psi = _update_mixing_parameters(
            lambda_, psi, e_z, e_log_z
        )
        proposed = np.array(
            [
                proposed_lambda,
                np.log(proposed_psi),
                proposed_mu,
                proposed_gamma,
                np.log(proposed_sigma),
            ],
            dtype=float,
        )
        current = np.array(
            [lambda_, np.log(psi), mu, gamma, np.log(sigma)], dtype=float
        )

        accepted = False
        step = 1.0
        candidate_loglik = -np.inf
        while step >= 2.0**-20:
            trial = current + step * (proposed - current)
            (
                trial_lambda,
                trial_log_psi,
                trial_mu,
                trial_gamma,
                trial_log_sigma,
            ) = map(float, trial)
            candidate_loglik = _observed_loglik(
                observations,
                trial_lambda,
                np.exp(trial_log_psi),
                trial_mu,
                trial_gamma,
                np.exp(trial_log_sigma),
            )
            if candidate_loglik >= current_loglik - 1.0e-8:
                lambda_ = trial_lambda
                psi = float(np.exp(trial_log_psi))
                mu = trial_mu
                gamma = trial_gamma
                sigma = float(np.exp(trial_log_sigma))
                accepted = True
                break
            step *= 0.5

        if not accepted:
            break
        path.append(candidate_loglik)
        relative_gain = (candidate_loglik - current_loglik) / (
            1.0 + abs(current_loglik)
        )
        current_loglik = candidate_loglik
        if 0.0 <= relative_gain < tolerance:
            converged = True
            break

    differences = np.diff(np.asarray(path))
    largest_drop = float(np.min(differences)) if differences.size else 0.0
    return GHEMResult(
        lambda_=float(lambda_),
        chi=1.0,
        psi=float(psi),
        mu=float(mu),
        gamma=float(gamma),
        sigma=float(sigma),
        loglik=float(current_loglik),
        iterations=len(path) - 1,
        converged=converged,
        monotone=bool(largest_drop >= -1.0e-7),
        largest_loglik_drop=largest_drop,
        loglik_path=tuple(float(value) for value in path),
    )

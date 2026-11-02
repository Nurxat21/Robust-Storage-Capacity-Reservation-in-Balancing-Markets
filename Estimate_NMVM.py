# Functions for estimating the finite-regime log-NMVM candidate.

import csv
from pathlib import Path

import numpy as np
from scipy import stats


def write_csv(file, rows):
    # Save one list of dictionaries as a CSV table.
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prepare_latent(xi):
    # Standardize the positive log-exposure observations.
    log_xi = np.log(xi)
    log_mean = float(np.mean(log_xi))
    log_sd = float(np.std(log_xi, ddof=0))
    standardized_log_xi = (log_xi - log_mean) / log_sd
    return standardized_log_xi, log_mean, log_sd


def finite_regime_cdf(x, probabilities, z_states, mu, gamma, sigma):
    # CDF of X = mu + gamma Z + sigma sqrt(Z) epsilon.
    cdf = 0.0
    for probability, z_state in zip(probabilities, z_states):
        location = mu + gamma * z_state
        scale = sigma * np.sqrt(z_state)
        cdf += probability * stats.norm.cdf(
            x,
            loc=location,
            scale=scale,
        )
    return float(cdf)


def finite_regime_ppf(
    probability,
    probabilities,
    z_states,
    mu,
    gamma,
    sigma,
):
    # Invert the finite-regime CDF by bisection.
    lower = -12.0
    upper = 12.0
    for _ in range(120):
        middle = 0.5 * (lower + upper)
        middle_cdf = finite_regime_cdf(
            middle,
            probabilities,
            z_states,
            mu,
            gamma,
            sigma,
        )
        if middle_cdf < probability:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def fit_finite_regime_em(
    data,
    z_states,
    initial_probabilities,
    max_iter=5000,
    tol=1e-8,
):
    # EM estimation for X | Z=z_j ~ N(mu + gamma z_j, sigma^2 z_j).
    data = np.asarray(data, dtype=float)
    z_states = np.asarray(z_states, dtype=float)
    probabilities = np.asarray(initial_probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()

    mu = float(np.mean(data))
    gamma = 0.0
    sigma = float(np.std(data, ddof=0))
    previous_log_likelihood = -np.inf
    log_likelihood_change = np.inf
    converged = False

    for iteration in range(1, max_iter + 1):
        # E-step
        component_density = []
        for probability, z_state in zip(probabilities, z_states):
            location = mu + gamma * z_state
            scale = sigma * np.sqrt(z_state)
            component_density.append(
                probability
                * stats.norm.pdf(data, loc=location, scale=scale)
            )
        component_density = np.vstack(component_density).T
        total_density = np.maximum(
            component_density.sum(axis=1, keepdims=True),
            1e-300,
        )
        responsibility = component_density / total_density
        log_likelihood = float(np.sum(np.log(total_density[:, 0])))

        # M-step
        probabilities = responsibility.mean(axis=0)
        design_matrix = []
        response = []
        weights = []
        for j, z_state in enumerate(z_states):
            design_matrix.append(
                np.column_stack(
                    [
                        np.ones_like(data),
                        np.full_like(data, z_state),
                    ]
                )
            )
            response.append(data)
            weights.append(responsibility[:, j] / z_state)
        design_matrix = np.vstack(design_matrix)
        response = np.concatenate(response)
        weights = np.concatenate(weights)
        weighted_design = design_matrix * weights[:, None]
        coefficients = (
            np.linalg.pinv(weighted_design.T @ design_matrix)
            @ (weighted_design.T @ response)
        )
        mu = float(coefficients[0])
        gamma = float(coefficients[1])

        residual_sum = 0.0
        for j, z_state in enumerate(z_states):
            residual = data - mu - gamma * z_state
            residual_sum += float(
                np.sum(
                    responsibility[:, j]
                    * residual**2
                    / z_state
                )
            )
        sigma = float(
            np.sqrt(max(residual_sum / len(data), 1e-12))
        )

        if np.isfinite(previous_log_likelihood):
            log_likelihood_change = abs(
                log_likelihood - previous_log_likelihood
            )
        if (
            np.isfinite(previous_log_likelihood)
            and log_likelihood_change <= tol
        ):
            converged = True
            break
        previous_log_likelihood = log_likelihood

    return {
        "p": probabilities,
        "z_states": z_states,
        "mu": mu,
        "gamma": gamma,
        "sigma": sigma,
        "loglik_z": previous_log_likelihood,
        "iterations": iteration,
        "ll_change": log_likelihood_change,
        "converged": converged,
    }


def profit(
    q,
    xi,
    r=8.0,
    c=4.0,
    s=0.0,
    ell=12.0,
):
    # Storage-reservation operating value.
    return (
        r * np.minimum(q, xi)
        + s * np.maximum(q - xi, 0.0)
        - ell * np.maximum(xi - q, 0.0)
        - c * q
    )

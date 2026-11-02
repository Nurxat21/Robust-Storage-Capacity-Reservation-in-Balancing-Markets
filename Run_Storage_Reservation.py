# This file reproduces the numerical analysis of the paper.
# The 2024 observations are used for estimation.
# The 2025 observations are retained as the out-of-sample test.
# All input data and numerical results are stored in CSV format.

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import Bounds, LinearConstraint, minimize

import Estimate_NMVM as NMVM
from Estimate_GH import fit_GH_EM


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "RTE_aFRR_2024_2025.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
Q_MAX = 25.0
THETA = 0.80
ALPHA = 0.90
N_SCENARIOS = 5000
Q_GRID = np.linspace(0.0, Q_MAX, 5001)
BOOTSTRAP_REPLICATIONS = 2000
RNG_SEED = 20260729
STRESS_GRID = np.array([0.5, 1.0, 2.0])
QP_VALIDATION_SCENARIOS = 30
BASE_R = 8.0
BASE_C = 4.0
BASE_S = 0.0
BASE_ELL = 12.0


def read_market_data(
    *,
    drop_source_missing: bool = True,
) -> pd.DataFrame:
    # Read the complete CSV data.
    data = pd.read_csv(DATA_FILE)
    data["source_missing_both_volumes"] = (
        pd.to_numeric(
            data["source_missing_both_volumes"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        .astype(bool)
    )

    # Use the original French market date to define the train and test years.
    timestamp_text = data["start_date"].astype(str)
    data["market_date"] = pd.to_datetime(
        timestamp_text.str.slice(0, 10),
        errors="coerce",
    )
    data["market_year"] = data["market_date"].dt.year
    data["market_quarter"] = data["market_date"].dt.quarter
    data["activated_volume_mw"] = pd.to_numeric(
        data["activated_volume_mw"],
        errors="coerce",
    )
    data["price_intensity_eur_per_mwh"] = pd.to_numeric(
        data["price_intensity_eur_per_mwh"],
        errors="coerce",
    )
    data = data.loc[
        data["market_year"].isin([2024, 2025])
        & np.isfinite(data["activated_volume_mw"])
        & (data["activated_volume_mw"] >= 0)
    ].copy()
    if drop_source_missing:
        data = data.loc[~data["source_missing_both_volumes"]].copy()

    # Convert physical aFRR activation from MW to the system-equivalent GW scale.
    data["xi_gw"] = data["activated_volume_mw"] / 1000.0
    return data


def quantile_scenarios(values: np.ndarray, probs: np.ndarray) -> np.ndarray:
    return np.quantile(values, probs, method="linear")


def zero_inflated_quantiles(
    positive_ppf,
    zero_probability: float,
    probs: np.ndarray,
) -> np.ndarray:
    """Combine an estimated positive distribution with an observed zero mass."""

    output = np.zeros_like(probs, dtype=float)
    positive = probs > zero_probability
    adjusted = (
        (probs[positive] - zero_probability)
        / max(1.0 - zero_probability, np.finfo(float).eps)
    )
    adjusted = np.clip(adjusted, 1e-8, 1.0 - 1e-8)
    output[positive] = positive_ppf(adjusted)
    return output


def finite_regime_ks(
    observations: np.ndarray,
    fit: dict,
) -> float:
    ordered = np.sort(observations)
    cdf = np.array(
        [
            NMVM.finite_regime_cdf(
                float(value),
                fit["p"],
                fit["z_states"],
                fit["mu"],
                fit["gamma"],
                fit["sigma"],
            )
            for value in ordered
        ]
    )
    n = ordered.size
    upper = np.arange(1, n + 1) / n
    lower = np.arange(n) / n
    return float(np.max(np.maximum(upper - cdf, cdf - lower)))


def fit_training_candidates(
    train: pd.DataFrame,
    probs: np.ndarray,
) -> tuple[
    dict[str, np.ndarray],
    list[dict],
    dict,
    float,
    float,
]:
    """Construct 2024-only empirical and fitted candidate distributions."""

    train_all = train["xi_gw"].to_numpy(dtype=float)
    train_positive = train_all[train_all > 0]
    z, y_mean, y_sd = NMVM.prepare_latent(train_positive)
    # The standardized log sample is theoretically invariant to a common
    # exposure-scale change. Rounding only machine-level noise makes that
    # invariance exact for the deterministic EM path.
    z = np.round(z, 12)
    zero_probability = float(np.mean(train_all == 0))

    candidates: dict[str, np.ndarray] = {
        "2024 full": quantile_scenarios(train_all, probs),
    }
    for quarter in range(1, 5):
        values = train.loc[
            train["market_quarter"] == quarter,
            "xi_gw",
        ].to_numpy(dtype=float)
        candidates[f"2024 Q{quarter}"] = quantile_scenarios(values, probs)

    diagnostics: list[dict] = []
    fitted_parameters: dict[str, dict] = {}

    def record(
        candidate: str,
        method: str,
        iterations: int | str,
        terminal_change: float | str,
        ks_distance: float | str,
        valid: bool,
        included: bool,
    ) -> None:
        diagnostics.append(
            {
                "candidate": candidate,
                "method": method,
                "iterations": iterations,
                "terminal_change": terminal_change,
                "ks_distance": ks_distance,
                "valid": "yes" if valid else "no",
                "included": "yes" if included else "no",
            }
        )

    train_mean = float(np.mean(train_all))
    train_sd = float(np.std(train_all, ddof=0))
    q_normal = float(
        np.clip(
            train_mean + train_sd * stats.norm.ppf(THETA),
            0.0,
            Q_MAX,
        )
    )
    record(
        "Normal benchmark",
        "moment benchmark",
        "--",
        "--",
        "--",
        True,
        False,
    )

    lognormal_ks = float(stats.kstest(z, "norm").statistic)
    lognormal = zero_inflated_quantiles(
        lambda probability: np.exp(
            stats.norm.ppf(
                probability,
                loc=y_mean,
                scale=y_sd,
            )
        ),
        zero_probability,
        probs,
    )
    lognormal_included = True
    if lognormal_included:
        candidates["Lognormal"] = lognormal
    record(
        "Lognormal",
        "closed-form MLE",
        "--",
        "--",
        lognormal_ks,
        True,
        lognormal_included,
    )
    fitted_parameters["Lognormal"] = {
        "zero_probability": zero_probability,
        "log_mean": y_mean,
        "log_sd": y_sd,
        "ks_distance": lognormal_ks,
    }

    finite_fit = NMVM.fit_finite_regime_em(
        z,
        STRESS_GRID,
        np.ones(STRESS_GRID.size) / STRESS_GRID.size,
        max_iter=5000,
        tol=1e-6,
    )
    finite_ks = finite_regime_ks(z, finite_fit)
    finite_valid = bool(
        finite_fit["converged"]
        and np.isfinite(finite_ks)
        and finite_fit["sigma"] > 0
        and np.all(finite_fit["p"] >= 0)
        and np.isclose(np.sum(finite_fit["p"]), 1.0)
    )
    finite_included = finite_valid
    if finite_included:
        finite_positive_ppf = lambda probability: np.exp(
            y_mean
            + y_sd
            * np.array(
                [
                    NMVM.finite_regime_ppf(
                        float(prob),
                        finite_fit["p"],
                        finite_fit["z_states"],
                        finite_fit["mu"],
                        finite_fit["gamma"],
                        finite_fit["sigma"],
                    )
                    for prob in np.atleast_1d(probability)
                ]
            )
        )
        candidates["Three-state log-NMVM"] = zero_inflated_quantiles(
            finite_positive_ppf,
            zero_probability,
            probs,
        )
    record(
        "Three-state log-NMVM",
        "FR-EM",
        finite_fit["iterations"],
        finite_fit["ll_change"],
        finite_ks,
        finite_valid,
        finite_included,
    )
    fitted_parameters["Three-state log-NMVM"] = {
        "zero_probability": zero_probability,
        "states": STRESS_GRID.tolist(),
        "probabilities": finite_fit["p"].tolist(),
        "mu": float(finite_fit["mu"]),
        "gamma": float(finite_fit["gamma"]),
        "sigma": float(finite_fit["sigma"]),
        "ks_distance": finite_ks,
    }

    gh_fit = fit_GH_EM(
        z,
        max_iter=1000,
        tolerance=1.0e-7,
    )
    gh_parameters = gh_fit.scipy_parameters
    gh_p, gh_a, gh_b, gh_loc, gh_scale = gh_parameters
    gh_gamma_log_scale = float(y_sd * gh_fit.gamma)
    gh_sigma_log_scale = float(y_sd * gh_fit.sigma)
    gh_u1 = float(
        gh_gamma_log_scale + 0.5 * gh_sigma_log_scale**2
    )
    gh_u2 = float(
        2.0 * gh_gamma_log_scale + 2.0 * gh_sigma_log_scale**2
    )
    gh_exponential_moments_valid = bool(
        gh_u1 < 0.5 * gh_fit.psi
        and gh_u2 < 0.5 * gh_fit.psi
    )
    gh_ks = float(
        stats.kstest(
            z,
            "genhyperbolic",
            args=gh_parameters,
        ).statistic
    )
    gh_valid = bool(
        np.all(np.isfinite(gh_parameters))
        and gh_fit.converged
        and gh_fit.monotone
        and gh_a > abs(gh_b)
        and gh_scale > 0
        and abs(gh_p + 0.5) > 1e-6
        and gh_exponential_moments_valid
        and np.isfinite(
            stats.genhyperbolic.logpdf(
                z,
                *gh_parameters,
            )
        ).all()
    )
    gh_included = gh_valid
    if gh_included:
        candidates["Log-GH"] = zero_inflated_quantiles(
            lambda probability: np.exp(
                y_mean
                + y_sd
                * stats.genhyperbolic.ppf(
                    probability,
                    *gh_parameters,
                )
            ),
            zero_probability,
            probs,
        )
    record(
        "Log-GH",
        "safeguarded GH-EM",
        gh_fit.iterations,
        gh_fit.largest_loglik_drop,
        gh_ks,
        gh_valid,
        gh_included,
    )
    fitted_parameters["Log-GH"] = {
        "zero_probability": zero_probability,
        "lambda": gh_fit.lambda_,
        "chi_fixed_for_identification": gh_fit.chi,
        "psi": gh_fit.psi,
        "mu_standardized": gh_fit.mu,
        "gamma_standardized": gh_fit.gamma,
        "sigma_standardized": gh_fit.sigma,
        "log_scale_gamma": gh_gamma_log_scale,
        "log_scale_sigma": gh_sigma_log_scale,
        "u1": gh_u1,
        "u2": gh_u2,
        "exponential_moment_domain_valid": gh_exponential_moments_valid,
        "scipy_p": gh_p,
        "scipy_a": gh_a,
        "scipy_b": gh_b,
        "scipy_loc": gh_loc,
        "scipy_scale": gh_scale,
        "iterations": gh_fit.iterations,
        "converged": gh_fit.converged,
        "monotone_loglikelihood": gh_fit.monotone,
        "largest_loglik_drop": gh_fit.largest_loglik_drop,
        "terminal_loglikelihood": gh_fit.loglik,
        "ks_distance": gh_ks,
    }

    q_empirical = float(
        np.clip(np.quantile(train_all, THETA), 0.0, Q_MAX)
    )
    return (
        candidates,
        diagnostics,
        fitted_parameters,
        q_normal,
        q_empirical,
    )


def upper_tail_mean(loss: np.ndarray, alpha: float = ALPHA) -> float:
    tail_count = max(1, int(np.ceil((1.0 - alpha) * loss.size)))
    threshold_index = loss.size - tail_count
    tail = np.partition(loss, threshold_index)[threshold_index:]
    return float(np.mean(tail))


def precompute_candidate_curves(
    candidates: dict[str, np.ndarray],
    *,
    r: float = BASE_R,
    c: float = BASE_C,
    s: float = BASE_S,
    ell: float = BASE_ELL,
    alpha: float = ALPHA,
) -> dict[str, dict[str, np.ndarray | float]]:
    curves: dict[str, dict[str, np.ndarray | float]] = {}
    for name, sample in candidates.items():
        expected = np.empty(Q_GRID.size)
        cvar = np.empty(Q_GRID.size)
        for index, q in enumerate(Q_GRID):
            payoff = NMVM.profit(
                float(q),
                sample,
                r=r,
                c=c,
                s=s,
                ell=ell,
            )
            expected[index] = float(np.mean(payoff))
            cvar[index] = upper_tail_mean(-payoff, alpha=alpha)
        curves[name] = {
            "expected": expected,
            "cvar": cvar,
            "best": float(np.max(expected)),
        }
    return curves


def solve_subset(
    names: list[str],
    curves: dict[str, dict[str, np.ndarray | float]],
    weights: tuple[float, float, float] = (
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 3.0,
    ),
) -> dict[str, float]:
    expected = np.vstack(
        [np.asarray(curves[name]["expected"]) for name in names]
    )
    cvar = np.vstack(
        [np.asarray(curves[name]["cvar"]) for name in names]
    )
    best = np.array([float(curves[name]["best"]) for name in names])
    regret = best[:, None] - expected

    objectives = np.column_stack(
        [
            -np.min(expected, axis=0),
            np.max(cvar, axis=0),
            np.max(regret, axis=0),
        ]
    )
    boundary_indices = np.argmin(objectives, axis=0)
    ideal = np.min(objectives, axis=0)
    anchors = objectives[boundary_indices, :]
    scales = np.max(anchors, axis=0) - ideal
    full_ranges = np.max(objectives, axis=0) - ideal
    scales = np.where(scales > 1e-10, scales, full_ranges)
    scales = np.where(scales > 1e-10, scales, 1.0)
    weight_array = np.asarray(weights, dtype=float)
    if np.any(weight_array < 0) or not np.isclose(weight_array.sum(), 1.0):
        raise ValueError("Achievement weights must be nonnegative and sum to one.")
    score = np.sum(
        weight_array
        * ((objectives - ideal) / scales) ** 2,
        axis=1,
    )
    mo_index = int(np.argmin(score))

    return {
        "q_ep": float(Q_GRID[int(boundary_indices[0])]),
        "q_cvar": float(Q_GRID[int(boundary_indices[1])]),
        "q_regret": float(Q_GRID[int(boundary_indices[2])]),
        "q_mo": float(Q_GRID[mo_index]),
        "z1_ideal": float(ideal[0]),
        "z2_ideal": float(ideal[1]),
        "z3_ideal": float(ideal[2]),
        "d1": float(scales[0]),
        "d2": float(scales[1]),
        "d3": float(scales[2]),
        "j1": float(objectives[mo_index, 0]),
        "j2": float(objectives[mo_index, 1]),
        "j3": float(objectives[mo_index, 2]),
    }


def validate_finite_scenario_qp(
    candidates: dict[str, np.ndarray],
    *,
    scenario_count: int = QP_VALIDATION_SCENARIOS,
) -> list[dict]:
    """Solve a reduced finite-scenario instance in the explicit QP variables.

    The main experiment retains 5,000 quantiles per candidate and exploits the
    single capacity dimension.  This validation uses the same candidate set on
    a smaller deterministic support so that the full epigraph QP can be solved
    independently and compared with exhaustive search on the identical support.
    """

    names = list(candidates)
    probabilities = (np.arange(scenario_count) + 0.5) / scenario_count
    support = np.vstack(
        [
            np.quantile(candidates[name], probabilities, method="linear")
            for name in names
        ]
    )
    reduced_candidates = {
        name: support[index]
        for index, name in enumerate(names)
    }
    reduced_curves = precompute_candidate_curves(reduced_candidates)
    grid_solution = solve_subset(names, reduced_curves)

    candidate_count = len(names)
    index = 0
    q_index = index
    index += 1
    h_index = np.arange(
        index,
        index + candidate_count * scenario_count,
    ).reshape(candidate_count, scenario_count)
    index += candidate_count * scenario_count
    rho_index = index
    index += 1
    zeta_index = index
    index += 1
    regret_index = index
    index += 1
    eta_index = np.arange(index, index + candidate_count)
    index += candidate_count
    u_index = np.arange(
        index,
        index + candidate_count * scenario_count,
    ).reshape(candidate_count, scenario_count)
    index += candidate_count * scenario_count
    normalized_index = np.arange(index, index + 3)
    index += 3
    variable_count = index

    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_constraint(
        coefficients: list[tuple[int, float]],
        *,
        lower_bound: float = -np.inf,
        upper_bound: float = np.inf,
    ) -> None:
        row = np.zeros(variable_count)
        for column, coefficient in coefficients:
            row[column] += coefficient
        rows.append(row)
        lower.append(lower_bound)
        upper.append(upper_bound)

    overage_cost = 4.0
    underage_cost = 16.0
    utilized_slope = 8.0
    shortage_cost = 12.0
    cvar_factor = 1.0 / ((1.0 - ALPHA) * scenario_count)
    candidate_best = np.array(
        [
            float(reduced_curves[name]["best"])
            for name in names
        ]
    )

    for candidate in range(candidate_count):
        for scenario in range(scenario_count):
            xi = float(support[candidate, scenario])
            add_constraint(
                [
                    (h_index[candidate, scenario], 1.0),
                    (q_index, underage_cost),
                ],
                lower_bound=shortage_cost * xi,
            )
            add_constraint(
                [
                    (h_index[candidate, scenario], 1.0),
                    (q_index, -overage_cost),
                ],
                lower_bound=-utilized_slope * xi,
            )
        add_constraint(
            [(rho_index, 1.0)]
            + [
                (h_index[candidate, scenario], -1.0 / scenario_count)
                for scenario in range(scenario_count)
            ],
            lower_bound=0.0,
        )
        add_constraint(
            [(zeta_index, 1.0), (eta_index[candidate], -1.0)]
            + [
                (u_index[candidate, scenario], -cvar_factor)
                for scenario in range(scenario_count)
            ],
            lower_bound=0.0,
        )
        for scenario in range(scenario_count):
            add_constraint(
                [
                    (u_index[candidate, scenario], 1.0),
                    (h_index[candidate, scenario], -1.0),
                    (eta_index[candidate], 1.0),
                ],
                lower_bound=0.0,
            )
        add_constraint(
            [(regret_index, 1.0)]
            + [
                (h_index[candidate, scenario], -1.0 / scenario_count)
                for scenario in range(scenario_count)
            ],
            lower_bound=float(candidate_best[candidate]),
        )

    ideal = np.array(
        [
            grid_solution["z1_ideal"],
            grid_solution["z2_ideal"],
            grid_solution["z3_ideal"],
        ]
    )
    scales = np.array(
        [
            grid_solution["d1"],
            grid_solution["d2"],
            grid_solution["d3"],
        ]
    )
    epigraph_indices = [rho_index, zeta_index, regret_index]
    for objective, epigraph_index in enumerate(epigraph_indices):
        add_constraint(
            [
                (normalized_index[objective], scales[objective]),
                (epigraph_index, -1.0),
            ],
            lower_bound=-ideal[objective],
        )

    matrix = np.vstack(rows)
    linear_constraint = LinearConstraint(
        matrix,
        np.asarray(lower),
        np.asarray(upper),
    )
    variable_lower = np.full(variable_count, -np.inf)
    variable_upper = np.full(variable_count, np.inf)
    variable_lower[q_index] = 0.0
    variable_upper[q_index] = Q_MAX
    variable_lower[u_index.ravel()] = 0.0
    variable_lower[normalized_index] = 0.0
    bounds = Bounds(variable_lower, variable_upper)

    initial_q = Q_MAX / 2.0
    initial = np.zeros(variable_count)
    initial[q_index] = initial_q
    initial[rho_index] = -np.inf
    initial[zeta_index] = -np.inf
    initial[regret_index] = -np.inf
    for candidate in range(candidate_count):
        loss = -NMVM.profit(initial_q, support[candidate])
        initial[h_index[candidate]] = loss
        initial[rho_index] = max(
            initial[rho_index],
            float(np.mean(loss)),
        )
        eta = float(np.quantile(loss, ALPHA, method="higher"))
        excess = np.maximum(loss - eta, 0.0)
        initial[eta_index[candidate]] = eta
        initial[u_index[candidate]] = excess
        initial[zeta_index] = max(
            initial[zeta_index],
            eta + float(np.mean(excess)) / (1.0 - ALPHA),
        )
        initial[regret_index] = max(
            initial[regret_index],
            float(candidate_best[candidate] + np.mean(loss)),
        )
    initial[normalized_index] = np.maximum(
        (
            np.array(
                [
                    initial[rho_index],
                    initial[zeta_index],
                    initial[regret_index],
                ]
            )
            - ideal
        )
        / scales,
        0.0,
    )

    weights = np.full(3, 1.0 / 3.0)

    def objective(vector: np.ndarray) -> float:
        deviations = vector[normalized_index]
        return float(np.dot(weights, deviations**2))

    def gradient(vector: np.ndarray) -> np.ndarray:
        output = np.zeros(variable_count)
        output[normalized_index] = (
            2.0 * weights * vector[normalized_index]
        )
        return output

    start = time.perf_counter()
    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=[linear_constraint],
        options={
            "ftol": 1.0e-10,
            "maxiter": 2000,
            "disp": False,
        },
    )
    runtime = time.perf_counter() - start
    if not result.success:
        raise RuntimeError(f"Finite-scenario QP validation failed: {result.message}")

    qp_q = float(result.x[q_index])
    grid_q = float(grid_solution["q_mo"])
    return [
        {
            "scenarios_per_candidate": scenario_count,
            "candidate_count": candidate_count,
            "qp_q": qp_q,
            "grid_q": grid_q,
            "absolute_difference": abs(qp_q - grid_q),
            "objective_value": float(result.fun),
            "iterations": int(result.nit),
            "runtime_seconds": runtime,
            "solver": "SciPy SLSQP",
            "status": "converged",
        }
    ]


def representative_weight_sensitivity(
    names: list[str],
    curves: dict[str, dict[str, np.ndarray | float]],
    test_values: np.ndarray,
    oracle_value: float,
) -> list[dict]:
    designs = [
        ("Mean boundary", (1.0, 0.0, 0.0)),
        ("CVaR boundary", (0.0, 1.0, 0.0)),
        ("Regret boundary", (0.0, 0.0, 1.0)),
        ("Equal weights", (1.0 / 3.0,) * 3),
        ("Mean emphasis", (0.5, 0.25, 0.25)),
        ("CVaR emphasis", (0.25, 0.5, 0.25)),
        ("Regret emphasis", (0.25, 0.25, 0.5)),
    ]
    rows: list[dict] = []
    for label, weights in designs:
        solution = solve_subset(
            names,
            curves,
            weights=weights,
        )
        metrics = test_metrics(
            solution["q_mo"],
            test_values,
            oracle_value,
        )
        rows.append(
            {
                "preference": label,
                "lambda_1": weights[0],
                "lambda_2": weights[1],
                "lambda_3": weights[2],
                "q": metrics["q"],
                "test_profit": metrics["test_profit"],
                "test_cvar": metrics["test_cvar"],
                "test_regret": metrics["test_regret"],
            }
        )
    return rows


def simplex_frontier(
    names: list[str],
    curves: dict[str, dict[str, np.ndarray | float]],
    step: float = 0.1,
) -> list[dict]:
    """Evaluate a deterministic simplex grid for the achievement weights."""

    denominator = int(round(1.0 / step))
    rows: list[dict] = []
    for first in range(denominator + 1):
        for second in range(denominator + 1 - first):
            third = denominator - first - second
            weights = (
                first / denominator,
                second / denominator,
                third / denominator,
            )
            solution = solve_subset(
                names,
                curves,
                weights=weights,
            )
            rows.append(
                {
                    "lambda_1": weights[0],
                    "lambda_2": weights[1],
                    "lambda_3": weights[2],
                    "q": solution["q_mo"],
                    "j1": solution["j1"],
                    "j2": solution["j2"],
                    "j3": solution["j3"],
                }
            )
    return rows


def test_metrics(
    q: float,
    test_values: np.ndarray,
    oracle_value: float,
    *,
    r: float = BASE_R,
    c: float = BASE_C,
    s: float = BASE_S,
    ell: float = BASE_ELL,
    alpha: float = ALPHA,
) -> dict[str, float]:
    payoff = NMVM.profit(
        q,
        test_values,
        r=r,
        c=c,
        s=s,
        ell=ell,
    )
    return {
        "q": float(q),
        "test_profit": float(np.mean(payoff)),
        "test_cvar": upper_tail_mean(-payoff, alpha=alpha),
        "test_regret": float(oracle_value - np.mean(payoff)),
        "test_unmet": float(np.mean(np.maximum(test_values - q, 0.0))),
        "test_idle": float(np.mean(np.maximum(q - test_values, 0.0))),
    }


def economic_parameter_sensitivity(
    names: list[str],
    candidates: dict[str, np.ndarray],
    test_values: np.ndarray,
) -> list[dict]:
    """Vary every normalized payoff coefficient and the CVaR level."""

    cases = [
        ("Lower utilization margin", 6.0, 4.0, 0.0, 12.0, 0.90),
        ("Higher reservation cost", 8.0, 6.0, 0.0, 12.0, 0.90),
        ("Lower shortage cost", 8.0, 4.0, 0.0, 8.0, 0.90),
        ("Baseline", 8.0, 4.0, 0.0, 12.0, 0.90),
        ("Idle-capacity recovery", 8.0, 4.0, 2.0, 12.0, 0.90),
        ("Higher shortage cost", 8.0, 4.0, 0.0, 20.0, 0.90),
        ("Higher tail confidence", 8.0, 4.0, 0.0, 12.0, 0.95),
        ("Extreme tail confidence", 8.0, 4.0, 0.0, 12.0, 0.99),
    ]
    rows: list[dict] = []
    for label, r, c, s, ell, alpha in cases:
        theta = (r + ell - c) / (r + ell - s)
        curves = precompute_candidate_curves(
            candidates,
            r=r,
            c=c,
            s=s,
            ell=ell,
            alpha=alpha,
        )
        solution = solve_subset(names, curves)
        oracle_q = float(
            np.clip(
                np.quantile(test_values, theta),
                0.0,
                Q_MAX,
            )
        )
        oracle_value = float(
            np.mean(
                NMVM.profit(
                    oracle_q,
                    test_values,
                    r=r,
                    c=c,
                    s=s,
                    ell=ell,
                )
            )
        )
        metrics = test_metrics(
            solution["q_mo"],
            test_values,
            oracle_value,
            r=r,
            c=c,
            s=s,
            ell=ell,
            alpha=alpha,
        )
        rows.append(
            {
                "case": label,
                "r": r,
                "c": c,
                "s": s,
                "ell": ell,
                "alpha": alpha,
                "theta": theta,
                "q_ep": solution["q_ep"],
                "q_cvar": solution["q_cvar"],
                "q_mo": solution["q_mo"],
                "test_profit": metrics["test_profit"],
                "test_cvar": metrics["test_cvar"],
            }
        )
    return rows


def finite_model_robustness_increment(
    nominal_solution: dict[str, float],
    robust_solution: dict[str, float],
    test_values: np.ndarray,
    oracle_value: float,
) -> list[dict]:
    """Compare matched nominal and finite-model robust policy selections."""

    pairs = [
        (
            "Expected profit",
            nominal_solution["q_ep"],
            robust_solution["q_ep"],
        ),
        (
            "CVaR",
            nominal_solution["q_cvar"],
            robust_solution["q_cvar"],
        ),
        (
            "Maximum regret",
            nominal_solution["q_regret"],
            robust_solution["q_regret"],
        ),
        (
            "Equal-weight MO",
            nominal_solution["q_mo"],
            robust_solution["q_mo"],
        ),
    ]
    rows: list[dict] = []
    for criterion, nominal_q, robust_q in pairs:
        nominal = test_metrics(nominal_q, test_values, oracle_value)
        robust = test_metrics(robust_q, test_values, oracle_value)
        rows.append(
            {
                "criterion": criterion,
                "nominal_q": nominal_q,
                "robust_q": robust_q,
                "delta_q": robust_q - nominal_q,
                "delta_profit": (
                    robust["test_profit"] - nominal["test_profit"]
                ),
                "delta_cvar": robust["test_cvar"] - nominal["test_cvar"],
                "delta_regret": (
                    robust["test_regret"] - nominal["test_regret"]
                ),
            }
        )
    return rows


def paired_daily_block_bootstrap(
    test: pd.DataFrame,
    policies: list[tuple[str, float]],
) -> list[dict]:
    day_arrays = [
        group["xi_gw"].to_numpy(dtype=float)
        for _, group in test.groupby("market_date", sort=True)
    ]
    rng = np.random.default_rng(RNG_SEED)
    normal_q = dict(policies)["Normal rule"]
    rows: dict[str, dict[str, list[float]]] = {
        name: {"profit_difference": [], "cvar_difference": []}
        for name, _ in policies
    }

    for _ in range(BOOTSTRAP_REPLICATIONS):
        selected = rng.integers(0, len(day_arrays), size=len(day_arrays))
        sample = np.concatenate([day_arrays[index] for index in selected])
        normal_payoff = NMVM.profit(normal_q, sample)
        normal_profit = float(np.mean(normal_payoff))
        normal_cvar = upper_tail_mean(-normal_payoff)
        for name, q in policies:
            payoff = NMVM.profit(q, sample)
            rows[name]["profit_difference"].append(
                float(np.mean(payoff)) - normal_profit
            )
            rows[name]["cvar_difference"].append(
                upper_tail_mean(-payoff) - normal_cvar
            )

    output: list[dict] = []
    for name, _ in policies:
        profit_difference = np.array(rows[name]["profit_difference"])
        cvar_difference = np.array(rows[name]["cvar_difference"])
        output.append(
            {
                "policy": name,
                "delta_profit": float(np.mean(profit_difference)),
                "profit_ci": (
                    f"[{np.quantile(profit_difference, 0.025):.3f}, "
                    f"{np.quantile(profit_difference, 0.975):.3f}]"
                ),
                "delta_cvar": float(np.mean(cvar_difference)),
                "cvar_ci": (
                    f"[{np.quantile(cvar_difference, 0.025):.3f}, "
                    f"{np.quantile(cvar_difference, 0.975):.3f}]"
                ),
            }
        )
    return output


def frequency_samples(data: pd.DataFrame) -> dict[str, np.ndarray]:
    fifteen = data["xi_gw"].to_numpy(dtype=float)
    daily = (
        data.groupby("market_date", sort=True)["xi_gw"]
        .max()
        .to_numpy(dtype=float)
    )
    weekly_key = data["market_date"].dt.to_period("W-SUN")
    weekly = (
        data.assign(week=weekly_key)
        .groupby("week", sort=True)["xi_gw"]
        .max()
        .to_numpy(dtype=float)
    )
    return {
        "15-min": fifteen,
        "Daily": daily,
        "Weekly": weekly,
    }


def descriptive_rows(
    samples: dict[str, np.ndarray],
) -> list[dict]:
    rows: list[dict] = []
    for frequency, values in samples.items():
        quantiles = np.quantile(
            values,
            [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99],
        )
        rows.append(
            {
                "frequency": frequency,
                "n": int(values.size),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=0)),
                "q_1": float(quantiles[0]),
                "q_5": float(quantiles[1]),
                "q_10": float(quantiles[2]),
                "median": float(quantiles[3]),
                "q_90": float(quantiles[4]),
                "q_95": float(quantiles[5]),
                "q_99": float(quantiles[6]),
                "maximum": float(np.max(values)),
                "skewness": float(stats.skew(values, bias=False)),
                "kurtosis": float(
                    stats.kurtosis(
                        values,
                        fisher=False,
                        bias=False,
                    )
                ),
            }
        )
    return rows


def plot_distribution_and_quantiles(
    samples: dict[str, np.ndarray],
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6), sharey=True)
    for axis, (frequency, values) in zip(axes, samples.items()):
        transformed = np.log1p(values)
        axis.hist(
            transformed,
            bins=55,
            density=True,
            color="0.45",
            edgecolor="white",
            linewidth=0.25,
        )
        axis.axvline(
            np.log1p(np.quantile(values, 0.5)),
            color="0.1",
            linewidth=1.0,
            label="Median",
        )
        axis.axvline(
            np.log1p(np.quantile(values, 0.95)),
            color="0.1",
            linestyle="--",
            linewidth=1.0,
            label=r"$Q_{95}$",
        )
        axis.set_title(frequency)
        axis.set_xlabel(r"$\log(1+\widetilde{\xi})$")
        axis.grid(alpha=0.2, linewidth=0.4)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Density")
    axes[-1].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "Figure_Distribution.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    theta_grid = np.array([0.80, 0.90, 0.95, 0.99])
    x = np.arange(theta_grid.size)
    width = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.75), sharey=True)
    for axis, (frequency, values) in zip(axes, samples.items()):
        empirical = np.quantile(values, theta_grid)
        normal = np.clip(
            np.mean(values)
            + np.std(values, ddof=0)
            * stats.norm.ppf(theta_grid),
            1e-8,
            None,
        )
        axis.bar(
            x - width / 2,
            empirical,
            width,
            color="0.35",
            label="Empirical",
        )
        axis.bar(
            x + width / 2,
            normal,
            width,
            color="0.80",
            label="Normal",
        )
        axis.set_title(frequency)
        axis.set_xticks(x)
        axis.set_xticklabels([f"{value:.2f}" for value in theta_grid])
        axis.set_xlabel(r"Critical fractile $\theta$")
        axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.2, linewidth=0.4)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Critical capacity (GW-eq., log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=7.5,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    fig.savefig(
        FIGURES / "Figure_Quantile_Distortion.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_temporal_holdout(
    train_values: np.ndarray,
    test_values: np.ndarray,
    policy_rows: list[dict],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.25))

    for values, label, color, style in [
        (train_values, "2024 training", "0.20", "-"),
        (test_values, "2025 test", "0.55", "--"),
    ]:
        transformed = np.sort(np.log1p(values))
        probability = np.arange(1, transformed.size + 1) / transformed.size
        axes[0].plot(
            transformed,
            probability,
            label=label,
            color=color,
            linestyle=style,
            linewidth=1.1,
        )
    axes[0].set_xlabel(r"$\log(1+\xi^{sys})$")
    axes[0].set_ylabel("Empirical CDF")
    axes[0].legend(frameon=False, fontsize=8)

    displayed = {
        "Normal rule": ("Normal", "o"),
        "Nominal EP": ("Nominal EP", "v"),
        "Nominal CVaR": ("Nominal CVaR", "D"),
        "Nominal MO": ("Nominal MO", "P"),
        "FC robust EP": ("Robust EP", "s"),
        "FC robust CVaR": ("Robust CVaR", "^"),
        "FC robust regret": ("Robust regret", "h"),
        "FC robust MO": ("Robust MO", "X"),
    }
    plotted = [
        row
        for row in policy_rows
        if row["policy"] in displayed
    ]
    for row in plotted:
        label, marker = displayed[row["policy"]]
        axes[1].scatter(
            row["test_profit"],
            row["test_cvar"],
            marker=marker,
            s=26,
            color="0.25",
            label=label,
        )
    axes[1].set_xlabel("2025 expected profit index")
    axes[1].set_ylabel("2025 CVaR loss index")
    axes[1].legend(
        frameon=False,
        fontsize=6.4,
        loc="upper left",
        ncol=2,
        handletextpad=0.4,
        labelspacing=0.3,
    )

    for axis in axes:
        axis.grid(alpha=0.23, linewidth=0.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_robust_frontier(
    frontier_rows: list[dict],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    j1 = np.array([row["j1"] for row in frontier_rows])
    j2 = np.array([row["j2"] for row in frontier_rows])
    j3 = np.array([row["j3"] for row in frontier_rows])
    capacity = np.array([row["q"] for row in frontier_rows])

    fig, axis = plt.subplots(figsize=(5.0, 3.4))
    scatter = axis.scatter(
        j1,
        j2,
        c=j3,
        s=24 + 4 * capacity,
        cmap="cividis",
        edgecolor="0.25",
        linewidth=0.25,
    )
    axis.set_xlabel(r"Worst-case negative expected-profit index $J_1$")
    axis.set_ylabel(r"Worst-case CVaR-loss index $J_2$")
    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label(r"Maximum-regret index $J_3$")
    capacity_handles = [
        axis.scatter(
            [],
            [],
            s=24 + 4 * reference,
            facecolor="none",
            edgecolor="0.25",
            linewidth=0.5,
            label=rf"$q={reference:.1f}$",
        )
        for reference in (0.5, 1.5, 2.5)
    ]
    axis.legend(
        handles=capacity_handles,
        title="Capacity (GW)",
        frameon=False,
        fontsize=7,
        title_fontsize=7,
        loc="best",
    )
    axis.grid(alpha=0.2, linewidth=0.4)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    diagnostics,
    fitted_parameters,
    critical_rows,
    policy_rows,
    bootstrap_rows,
    sensitivity_rows,
    weight_rows,
    frontier_rows,
    data_quality_rows,
    description_rows,
    robustness_increment_rows,
    qp_validation_rows,
    economic_sensitivity_rows,
):
    # Save every numerical result as a CSV file.
    RESULTS.mkdir(parents=True, exist_ok=True)

    parameter_rows = []
    for candidate, parameters in fitted_parameters.items():
        parameter_rows.append({
            "candidate": candidate,
            "parameters": repr(parameters),
        })

    output_files = {
        "Fit_Diagnostics.csv": diagnostics,
        "Fitted_Parameters.csv": parameter_rows,
        "Critical_Fractile.csv": critical_rows,
        "Policy_Performance.csv": policy_rows,
        "Bootstrap_Test.csv": bootstrap_rows,
        "Ambiguity_Sensitivity.csv": sensitivity_rows,
        "Weight_Sensitivity.csv": weight_rows,
        "Simplex_Frontier.csv": frontier_rows,
        "Data_Quality_Sensitivity.csv": data_quality_rows,
        "Descriptive_Statistics.csv": description_rows,
        "Robustness_Increment.csv": robustness_increment_rows,
        "QP_Validation.csv": qp_validation_rows,
        "Economic_Sensitivity.csv": economic_sensitivity_rows,
    }
    for file_name, rows in output_files.items():
        NMVM.write_csv(RESULTS / file_name, rows)


def main():
    data = read_market_data(drop_source_missing=True)
    legacy_data = read_market_data(drop_source_missing=False)
    train = data.loc[data["market_year"] == 2024].copy()
    test = data.loc[data["market_year"] == 2025].copy()
    legacy_train = legacy_data.loc[
        legacy_data["market_year"] == 2024
    ].copy()
    legacy_test = legacy_data.loc[
        legacy_data["market_year"] == 2025
    ].copy()
    train_values = train["xi_gw"].to_numpy(dtype=float)
    test_values = test["xi_gw"].to_numpy(dtype=float)
    probs = (np.arange(N_SCENARIOS) + 0.5) / N_SCENARIOS

    (
        candidates,
        diagnostics,
        fitted_parameters,
        q_normal,
        q_empirical,
    ) = fit_training_candidates(
            train,
            probs,
    )
    samples = frequency_samples(data)
    description_rows = descriptive_rows(samples)
    plot_distribution_and_quantiles(samples)
    critical_rows: list[dict] = [
        {
            "candidate": "Empirical 2024",
            "q_0_8": q_empirical,
            "error": 0.0,
            "abs_error": 0.0,
            "role": "reference",
        },
        {
            "candidate": "Normal benchmark",
            "q_0_8": q_normal,
            "error": q_normal - q_empirical,
            "abs_error": abs(q_normal - q_empirical),
            "role": "benchmark",
        },
    ]
    for name, sample in candidates.items():
        if name.startswith("2024"):
            continue
        fitted_q = float(np.quantile(sample, THETA))
        critical_rows.append(
            {
                "candidate": name,
                "q_0_8": fitted_q,
                "error": fitted_q - q_empirical,
                "abs_error": abs(fitted_q - q_empirical),
                "role": "candidate",
            }
        )
    critical_rows = critical_rows[:1] + sorted(
        critical_rows[1:],
        key=lambda row: row["abs_error"],
    )
    curves = precompute_candidate_curves(candidates)

    empirical_names = [
        name
        for name in candidates
        if name == "2024 full" or name.startswith("2024 Q")
    ]
    fitted_names = [
        name
        for name in candidates
        if name not in empirical_names
    ]
    combined_names = list(candidates)
    nominal_solution = solve_subset(["2024 full"], curves)
    combined_solution = solve_subset(combined_names, curves)
    economic_sensitivity_rows = economic_parameter_sensitivity(
        combined_names,
        candidates,
        test_values,
    )

    selected_policies: list[tuple[str, float]] = [
        ("Normal rule", q_normal),
        ("Nominal EP", nominal_solution["q_ep"]),
        ("Nominal CVaR", nominal_solution["q_cvar"]),
        ("Nominal MO", nominal_solution["q_mo"]),
    ]
    if "Log-GH" in candidates:
        selected_policies.append(
            (
                "Log-GH EP",
                float(np.quantile(candidates["Log-GH"], THETA)),
            )
        )
    selected_policies.extend(
        [
            ("FC robust EP", combined_solution["q_ep"]),
            ("FC robust CVaR", combined_solution["q_cvar"]),
            ("FC robust regret", combined_solution["q_regret"]),
            ("FC robust MO", combined_solution["q_mo"]),
        ]
    )

    oracle_payoff = np.array(
        [np.mean(NMVM.profit(q, test_values)) for q in Q_GRID]
    )
    oracle_index = int(np.argmax(oracle_payoff))
    oracle_q = float(Q_GRID[oracle_index])
    oracle_value = float(oracle_payoff[oracle_index])

    weight_rows = representative_weight_sensitivity(
        combined_names,
        curves,
        test_values,
        oracle_value,
    )
    frontier_rows = simplex_frontier(
        combined_names,
        curves,
        step=0.1,
    )
    robustness_increment_rows = finite_model_robustness_increment(
        nominal_solution,
        combined_solution,
        test_values,
        oracle_value,
    )
    qp_validation_rows = validate_finite_scenario_qp(candidates)

    policy_rows = [
        {
            "policy": name,
            **test_metrics(q, test_values, oracle_value),
        }
        for name, q in selected_policies
    ]
    policy_rows.append(
        {
            "policy": "2025 oracle",
            **test_metrics(oracle_q, test_values, oracle_value),
        }
    )

    bootstrap_rows = paired_daily_block_bootstrap(
        test,
        selected_policies,
    )

    sensitivity_rows: list[dict] = []
    for label, names in [
        ("Empirical regimes", empirical_names),
        ("Fitted models", fitted_names),
        ("Combined", combined_names),
    ]:
        solution = solve_subset(names, curves)
        metrics = test_metrics(
            solution["q_mo"],
            test_values,
            oracle_value,
        )
        sensitivity_rows.append(
            {
                "ambiguity_set": label,
                "candidate_count": len(names),
                "q": metrics["q"],
                "test_profit": metrics["test_profit"],
                "test_cvar": metrics["test_cvar"],
                "test_regret": metrics["test_regret"],
            }
        )

    legacy_train_values = legacy_train["xi_gw"].to_numpy(dtype=float)
    legacy_q_empirical = float(
        np.clip(
            np.quantile(legacy_train_values, THETA),
            0.0,
            Q_MAX,
        )
    )
    legacy_q_normal = float(
        np.clip(
            np.mean(legacy_train_values)
            + np.std(legacy_train_values, ddof=0)
            * stats.norm.ppf(THETA),
            0.0,
            Q_MAX,
        )
    )
    data_quality_rows = [
        {
            "treatment": "Drop source-missing (primary)",
            "train_n": len(train),
            "test_n": len(test),
            "train_zero_n": int(np.sum(train_values == 0)),
            "q_empirical": q_empirical,
            "q_normal": q_normal,
        },
        {
            "treatment": "Set source-missing to zero",
            "train_n": len(legacy_train),
            "test_n": len(legacy_test),
            "train_zero_n": int(np.sum(legacy_train_values == 0)),
            "q_empirical": legacy_q_empirical,
            "q_normal": legacy_q_normal,
        },
    ]

    plot_temporal_holdout(
        train_values,
        test_values,
        policy_rows,
        FIGURES / "Figure_Temporal_Holdout.pdf",
    )
    plot_robust_frontier(
        frontier_rows,
        FIGURES / "Figure_Robust_Frontier.pdf",
    )
    write_outputs(
        diagnostics,
        fitted_parameters,
        critical_rows,
        policy_rows,
        bootstrap_rows,
        sensitivity_rows,
        weight_rows,
        frontier_rows,
        data_quality_rows,
        description_rows,
        robustness_increment_rows,
        qp_validation_rows,
        economic_sensitivity_rows,
    )

    metadata = [
        {
            "train_start": train["market_date"].min().date().isoformat(),
            "train_end": train["market_date"].max().date().isoformat(),
            "train_n": len(train),
            "train_zero_n": int(np.sum(train_values == 0)),
            "test_start": test["market_date"].min().date().isoformat(),
            "test_end": test["market_date"].max().date().isoformat(),
            "test_n": len(test),
            "test_zero_n": int(np.sum(test_values == 0)),
            "source_missing_dropped": int(
                legacy_data["source_missing_both_volumes"].sum()
            ),
            "q_grid_min": float(Q_GRID[0]),
            "q_grid_max": float(Q_GRID[-1]),
            "q_grid_step": float(Q_GRID[1] - Q_GRID[0]),
            "candidate_count": len(candidates),
            "empirical_candidate_count": len(empirical_names),
            "fitted_candidate_count": len(fitted_names),
            "bootstrap_replications": BOOTSTRAP_REPLICATIONS,
            "random_seed": RNG_SEED,
            "candidate_inclusion_rule": "prespecified family plus convergence and parameter validity",
            "ks_role": "descriptive diagnostic only",
            "simplex_step": 0.1,
            "qp_validation_scenarios_per_candidate": QP_VALIDATION_SCENARIOS,
        }
    ]
    NMVM.write_csv(RESULTS / "Metadata.csv", metadata)

    print("Training candidates:", list(candidates))
    print("Diagnostics:")
    for row in diagnostics:
        print(row)
    print("Out-of-sample policy table:")
    for row in policy_rows:
        print(row)
    print("Ambiguity-set sensitivity:")
    for row in sensitivity_rows:
        print(row)
    print("Weight sensitivity:")
    for row in weight_rows:
        print(row)
    print("Data-quality sensitivity:")
    for row in data_quality_rows:
        print(row)
    print("Finite-model robustness increment:")
    for row in robustness_increment_rows:
        print(row)
    print("QP validation:")
    for row in qp_validation_rows:
        print(row)
    print("Economic-parameter sensitivity:")
    for row in economic_sensitivity_rows:
        print(row)


if __name__ == "__main__":
    main()

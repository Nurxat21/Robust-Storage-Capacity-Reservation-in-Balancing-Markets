# Replication package

## Paper

This package reproduces the empirical tables and figures reported in:

> Robust Storage Capacity Reservation in Balancing Markets: Non-Gaussian
> Risk, Model Ambiguity and French aFRR Evidence

The calculations use French 15-minute automatic frequency restoration reserve
(aFRR) activation observations for 2024 and 2025. The physical exposure is

$$
\xi_t^{sys}=A_t^+ + A_t^-,
\qquad
\widetilde{\xi}_t^{sys}=\xi_t^{sys}/1000,
$$

where $A_t^+$ and $A_t^-$ are upward and downward activated power. The
source is the RTE Balancing Energy API, endpoint `standard_afrr_data`.

## File structure

- `code/Run_Storage_Reservation.py`: main program.
- `code/Estimate_NMVM.py`: finite-regime log-NMVM estimation functions.
- `code/Estimate_GH.py`: likelihood-safeguarded GH-EM estimator.
- `data/RTE_aFRR_2024_2025.csv`: complete model input.
- `results/*.csv`: numerical results reported in the paper.
- `figures/*.pdf`: vector figures reported in the paper.

The model code reads one CSV file directly. No API credential, access token,
raw JSON response or download routine is required.

## Data

`RTE_aFRR_2024_2025.csv` retains:

- the original interval start and end time;
- upward and downward activated aFRR volumes;
- upward and downward weighted average activation prices;
- total activated power in MW;
- the auxiliary price-intensity measure in EUR/MWh;
- `source_missing_both_volumes`, which identifies intervals for which both
  source activation-volume fields were missing.

The verified input contains 70,176 intervals. The primary analysis removes the
75 source-missing intervals, leaving 35,094 training observations from 2024 and
35,007 previously unused test observations from 2025.

## Environment

The verified environment was:

- Python 3.13.9
- NumPy 2.3.5
- pandas 2.3.3
- SciPy 1.16.3
- Matplotlib 3.10.6

Install the dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

## Run

From the root of this package, execute:

```bash
python3 code/Run_Storage_Reservation.py
```

The program uses the fixed seed `20260729`, estimates all candidate
distributions from 2024, evaluates the selected policies on the 2025 temporal
holdout, performs the paired daily-block bootstrap and sensitivity analyses,
and rewrites all CSV result tables and PDF figures.

The run may take several minutes because it includes EM estimation, a simplex
weight grid, 2,000 bootstrap replications and a finite-scenario QP validation.

## Main numerical settings

- capacity interval: `[0, 25]` system-equivalent GW;
- capacity grid step: `0.005`;
- CVaR confidence level: `0.90`;
- deterministic support size per candidate: `5,000`;
- QP-validation support size per candidate: `30`;
- paired daily-block bootstrap replications: `2,000`;
- simplex-weight step: `0.1`.

All operating-value, CVaR and regret values are normalized objective indices.
They are not site-specific euro-profit estimates.

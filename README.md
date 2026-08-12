# Air Quality Prediction — Linear, Kernel & Neural Approaches

Predicting hourly air pollutant concentrations (CO, NO₂, NOx, C₆H₆) from sensor readings, comparing linear, kernel-based, and neural regression methods — and confronting the models with a realistic time-series evaluation.

> Group project — built with [Elhabib Touisse](https://github.com/elhabib-touisse).

## Dataset

- **Source:** [UCI Air Quality dataset](https://archive.ics.uci.edu/dataset/360/air+quality) — hourly measurements from a gas multisensor device deployed in an Italian city, alongside ground-truth pollutant concentrations from a reference analyzer.
- **Targets:** CO(GT), C6H6(GT), NOx(GT), NO2(GT) — the four reference pollutant concentrations.
- **Features:** the corresponding sensor readings (PT08.S1–S5) plus temperature, relative humidity and absolute humidity.
- Missing values (encoded as `-200` in the raw data) were identified and excluded from the modeling target; sensor features were mean-imputed.

## Methodology

**1. Exploratory analysis**
PCA and a correlation circle confirm three natural variable groups: ground-truth pollutant measurements, their corresponding sensors, and environmental variables (T, RH, AH) — with sensors correlating with both pollutants and weather, consistent with known cross-sensitivities of low-cost gas sensors. Hourly aggregation reveals two daily pollution peaks (morning and evening), consistent with traffic patterns.

**2. Model comparison, from linear to non-linear**

| Model | Idea | Result |
|---|---|---|
| Linear Regression | Unregularized least squares | Good baseline (R² close to 1), but systematic curvature in predicted-vs-true plots at high concentrations |
| Ridge | L2-penalized least squares, α tuned by validation MSE | Near-identical to linear regression (optimal α very small) |
| Lasso | L1-penalized (sparsity-inducing) | Same story — optimal α small enough that it barely differs from OLS |
| Kernel Ridge (linear kernel) | Confirms equivalence with Ridge in the linear case | As expected, matches Ridge |
| Kernel Ridge (RBF kernel) | Non-linear kernel, α and γ tuned via grid search | **Clear improvement** — confirms the true relationship is non-linear |
| Dense Neural Network | Multi-output MLP (Keras/TensorFlow), architecture tuned sequentially (layers, neurons, learning rate, epochs) | **Best performance**, confirming non-linearity; best architecture found: 3 layers × 500 neurons |

**3. Catching a data leakage bug**
The initial random train/val/test split ignored the fact that this is *hourly time-series data* — meaning the model could implicitly learn from "future" observations. This was corrected using `TimeSeriesSplit` (strict chronological ordering for train/val/test), which is the methodologically correct evaluation for this kind of data. As expected, performance drops once the model can no longer see the future — a more honest, if less flattering, estimate of real-world performance.

## Key Findings

- The true pollutant-vs-sensor relationship is genuinely non-linear: every linear method (OLS, Ridge, Lasso) plateaus at the same performance, while kernel RBF and the neural network both improve substantially and agree with each other.
- Kernel Ridge (RBF) is the best performance-to-complexity trade-off: it nearly matches the neural network's accuracy with far less tuning and training cost.
- Naive random train/test splitting silently inflates performance on time-series data by leaking future information — a methodological pitfall this project explicitly identifies and corrects.

## Tech Stack

- **Data manipulation:** pandas, numpy, skrub
- **Machine Learning:** scikit-learn (Linear/Ridge/Lasso/KernelRidge, GridSearchCV, TimeSeriesSplit, pipelines)
- **Deep Learning:** TensorFlow / Keras (dense neural network)
- **Visualization:** matplotlib, seaborn

## Repository Contents

- `notebook.ipynb` — full analysis: cleaning, EDA, PCA, model comparison, hyperparameter tuning, time-series correction
- `AirQualityUCI.csv` — the underlying dataset
- `requirements.txt` — dependencies to reproduce the analysis

## Getting Started

```bash
git clone https://github.com/<your-username>/air-quality-prediction.git
cd air-quality-prediction
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## Authors

**Valentin Cherin** — [GitHub](https://github.com/vacheee)
**Elhabib Touisse** — [GitHub](https://github.com/elhabib-touisse)

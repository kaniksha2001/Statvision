# StatVision — Statistical Analysis Toolkit

StatVision is a desktop-style statistical analysis application built with
Streamlit, comparable in scope to SPSS, Minitab, JMP and Numiqo. It covers the
full workflow — from importing and cleaning data, through descriptive
statistics, probability distributions, visualization and statistical inference,
to exporting a consolidated PDF / Excel report — and is designed for users with
little to no programming experience.

Every statistical routine is fully implemented on top of SciPy, Statsmodels,
NumPy and scikit-learn. There are no mock calculations or placeholder buttons.

## Features

The toolkit is organised into thirteen pages, reachable from the sidebar:

1. **Home** — load a CSV / Excel / TSV file or the bundled sample dataset.
2. **Data Exploration** — overview metrics, data inspection (search, filter,
   sort, column metadata), a data-quality report and an automated profiling
   report.
3. **Data Cleaning** — missing-value handling (drop / mean / median / mode /
   constant imputation), duplicate detection & removal, transformations
   (standardize, normalize, min-max, log) and feature engineering (rename,
   create, bin, encode).
4. **Descriptive Statistics** — central tendency, dispersion, position and
   shape, plus an exportable summary table.
5. **Visualization** — histogram, density, KDE, box, bar, pie, scatter, line,
   area, heatmap, correlation heatmap, pair plot and joint plot, plus normality
   (Q-Q, P-P) and outlier diagnostics. Charts download as PNG / HTML.
6. **Probability Distributions** — six discrete and seven continuous
   distributions with interactive parameter sliders, PMF/PDF and CDF plots, and
   a multi-distribution comparison view.
7. **Goodness of Fit** — Chi-square, Kolmogorov-Smirnov and Anderson-Darling
   tests with observed-vs-expected, distribution-fit, Q-Q and residual plots.
8. **Normality Testing** — Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling,
   Jarque-Bera and D'Agostino K² tests with diagnostic plots.
9. **Confidence Intervals** — for the mean, a proportion and the variance at
   90 / 95 / 99% confidence.
10. **Hypothesis Testing** — one-sample / independent / paired t-tests,
    one-way ANOVA, Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis and the
    chi-square test of independence.
11. **Correlation Analysis** — Pearson, Spearman and Kendall coefficients with
    a correlation matrix, heatmap and scatter matrix.
12. **Regression Analysis** — simple and multiple OLS regression with R²,
    adjusted R², the coefficient table and residual diagnostics.
13. **Reporting** — collect tables, charts and test results from any page and
    export them as a multi-page PDF or a multi-sheet Excel workbook.

A light and a dark theme are available from the sidebar.

## Requirements

- Python 3.12+
- The packages listed in `requirements.txt`

## Installation

```bash
# (optional) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## Running

From the project root (the folder containing `app.py`):

```bash
streamlit run app.py
```

Streamlit will open the app in your browser. On the **Home** page, upload your
own dataset or click *Load sample dataset* to explore the bundled
`sample_data/sample_customers.csv` (300 rows of synthetic customer data).

## Project structure

```
statvision/
├── app.py                     # entry point: sidebar nav, theming, dispatch
├── requirements.txt
├── .streamlit/config.toml     # server & base theme configuration
├── sample_data/
│   └── sample_customers.csv   # bundled demo dataset
└── statvision/                # application package
    ├── config/                # settings (navigation) & styling
    ├── core/                  # session-state & shared utilities
    ├── data_exploration/      # I/O, overview, data-quality
    ├── cleaning/              # missing, duplicates, transforms, features
    ├── statistics/            # descriptive statistics
    ├── visualizations/        # Plotly / Matplotlib chart builders
    ├── distributions/         # probability-distribution registry
    ├── inference/             # GOF, normality, CIs, tests, correlation, regression
    ├── reporting/             # PDF & Excel report builders
    └── pages_impl/            # one module per UI page + shared components
```

The package is deliberately layered: the `pages_impl` UI modules call into the
analytical packages (`statistics`, `inference`, `distributions`, …), which in
turn depend only on the scientific libraries. Business logic contains no
Streamlit calls, so it is straightforward to test or reuse outside the app.

## Notes

- PNG chart export uses `kaleido`. If it is unavailable, the interactive HTML
  download still works.
- All numeric routines use sample conventions (e.g. `ddof=1` for variance and
  standard deviation) consistent with inferential statistics.

[index.html](https://github.com/user-attachments/files/30109250/index.html)# Predictive Sales Forecast

**Intern:** Haresh S
**Intern ID:** CITS7117
**Domain:** Data Analytics
**Duration:** 4 Weeks (16 July 2026 – 13 August 2026)
**Organization:** CodTech IT Solutions Private Limited

## Project Scope

This project forecasts future monthly sales using historical trend and seasonality patterns, comparing two forecasting approaches and producing a 6-month forward forecast with confidence intervals to support sales and marketing planning.

## Dataset

A **simulated 4-year monthly retail sales dataset** (Jan 2022 – Dec 2025, 48 months), consistent with CodTech's guidance that dummy datasets are acceptable for Data Science & Analytics projects. The simulation includes:
- A steady upward **trend** (business growth over time)
- **Seasonality** — dip in February, steady mid-year growth, strong peak in Nov-Dec (holiday season)
- **Promotional campaign effects** — ~20% of months include a simulated marketing promotion with a sales lift
- Random noise to mimic real-world variability

## Methodology

### 1. Exploratory Analysis
Seasonal decomposition (additive model) separated the series into trend, seasonal, and residual components to confirm the presence of both a growth trend and a clear yearly seasonal cycle.

### 2. Forecasting Models Compared
Two models were trained on the first 40 months and tested on the final 8 months (out-of-sample):

1. **Linear Regression** — using a time trend variable, month dummy variables (to capture seasonality), and a promotion flag
2. **Holt-Winters Exponential Smoothing** — triple exponential smoothing with additive trend and seasonality

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| **Linear Regression (trend + season + promo)** | 2,779.9 | 3,640.9 | **2.62%** |
| Holt-Winters Exponential Smoothing | 4,932.3 | 7,169.4 | 4.49% |

**Linear Regression was selected as the final model** — a MAPE of 2.6% is a strong result for retail sales forecasting (under 10% is generally considered good), and it outperformed Holt-Winters because it could directly incorporate the promotional campaign signal, which pure time-series smoothing cannot see.

### 3. Future Forecast
The winning model was retrained on the full 48 months of data to produce a genuine 6-month forward forecast (Jan–Jun 2026) with a 90% confidence interval.

## Key Insights

- **Clear seasonality:** February is consistently the weakest month (~12% below trend); November-December are the strongest (+22% to +28% above trend) — classic holiday-driven retail pattern
- **Promotions work:** months with an active promotion show an 8-18% sales lift over baseline
- **Forecast for next 6 months (Jan-Jun 2026):** sales are projected to grow from ~$95,200 in January to ~$106,400 by June, continuing the established growth trend
- **Model accuracy of 2.62% MAPE** means forecasts are reliable enough to support inventory and budget planning decisions with reasonable confidence

## Business Recommendations

1. **Plan inventory and staffing around confirmed seasonality**, especially the Nov-Dec peak and the February dip — don't treat these as anomalies, they're structural patterns.
2. **Use the promotion flag finding to justify a regular promotional calendar.** Since promotions show a consistent measurable lift, a planned quarterly promotion calendar (rather than ad hoc campaigns) could smooth revenue and hit targets more predictably.
3. **Use the 6-month forecast for budget-setting**, but treat the confidence interval as the real planning range, not just the point forecast — plan for the lower bound in conservative scenarios (e.g., inventory commitments) and the upper bound for stretch targets.
4. **Combine this forecast with the CLV and Attribution projects.** Since marketing spend and promotions clearly move sales, forecast accuracy could improve further by feeding in actual planned campaign schedules rather than a flat "no promo" assumption for future months.
5. **Re-run the forecast monthly as new actuals come in** — a rolling forecast approach will catch trend shifts (e.g., a slowdown) faster than a static one-time forecast.

## Repository Structure

```
predictive-sales-forecast/
├── data/
│   └── monthly_sales.csv              # Simulated 48-month sales dataset
├── scripts/
│   ├── 01_generate_dataset.py         # Dataset simulation
│   └── 02_forecasting_models.py       # Decomposition + 2 models + evaluation + forecast
├── outputs/
│   ├── charts/                        # All 6 visualizations
│   ├── model_comparison.csv
│   └── next_6_months_forecast.csv
└── README.md
```

## Tools Used

Python (pandas, numpy, scikit-learn, statsmodels, matplotlib, seaborn)

[Uploading index.html…]


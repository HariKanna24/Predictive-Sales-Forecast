"""
Predictive Sales Forecast - Step 1: Simulated Retail Sales Dataset
Intern: Haresh S | Intern ID: CITS7117 | CodTech IT Solutions
Project: Predictive Sales Forecast

NOTE ON DATA: This project uses a simulated (dummy) monthly retail sales dataset,
consistent with CodTech's guidance that dummy datasets are acceptable for
Data Science & Analytics projects. The simulation includes realistic trend,
seasonality, and promotional-lift patterns typical of retail sales data.
"""

import pandas as pd
import numpy as np

np.random.seed(7)

# 4 years of monthly data
dates = pd.date_range(start='2022-01-01', end='2025-12-01', freq='MS')
n = len(dates)

# Base trend: steady growth over time
trend = np.linspace(50000, 95000, n)

# Yearly seasonality: peak in Nov-Dec (holiday season), dip in Feb
month_seasonality = {
    1: -0.05, 2: -0.12, 3: -0.02, 4: 0.0, 5: 0.02, 6: 0.04,
    7: 0.03, 8: 0.01, 9: 0.05, 10: 0.10, 11: 0.22, 12: 0.28
}
seasonality = np.array([month_seasonality[d.month] for d in dates]) * trend

# Promotional campaigns: random months with a sales boost (simulate marketing campaigns)
promo_flag = np.zeros(n)
promo_months_idx = np.random.choice(range(n), size=int(n * 0.2), replace=False)
promo_flag[promo_months_idx] = 1
promo_effect = promo_flag * trend * np.random.uniform(0.08, 0.18, n)

# Random noise
noise = np.random.normal(0, trend * 0.04, n)

sales = trend + seasonality + promo_effect + noise
sales = np.round(sales, 2)

df = pd.DataFrame({
    'date': dates,
    'sales': sales,
    'promotion_active': promo_flag.astype(int),
    'month': dates.month,
    'year': dates.year
})

df.to_csv("../data/monthly_sales.csv", index=False)

print("Dataset generated:", df.shape)
print(df.head(12))
print("\nSales summary:")
print(df['sales'].describe())

"""
Predictive Sales Forecast - Step 2: Forecasting Models & Evaluation
Intern: Haresh S | Intern ID: CITS7117 | CodTech IT Solutions
Project: Predictive Sales Forecast
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 5.5)
plt.rcParams['font.size'] = 10

df = pd.read_csv("../data/monthly_sales.csv", parse_dates=['date'])
df.set_index('date', inplace=True)

# ---------------------------------------------------------
# 1. TRAIN-TEST SPLIT (last 8 months as test set - out-of-sample forecast test)
# ---------------------------------------------------------
train = df.iloc[:-8]
test = df.iloc[-8:]

# ---------------------------------------------------------
# 2. SEASONAL DECOMPOSITION (for EDA / understanding the series)
# ---------------------------------------------------------
decomposition = seasonal_decompose(df['sales'], model='additive', period=12)
fig = decomposition.plot()
fig.set_size_inches(10, 8)
fig.suptitle('Sales Time Series Decomposition', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("../outputs/charts/01_seasonal_decomposition.png", dpi=150, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# 3. MODEL 1: LINEAR REGRESSION (trend + month dummies + promo)
# ---------------------------------------------------------
def build_features(data):
    X = pd.DataFrame(index=data.index)
    X['t'] = np.arange(len(data)) if data is train else None
    return X

month_categories = pd.CategoricalDtype(categories=range(1, 13))

train_feat = pd.DataFrame({'t': np.arange(len(train))}, index=train.index)
train_feat = pd.concat([train_feat, pd.get_dummies(train['month'].astype(month_categories), prefix='m', drop_first=True)], axis=1)
train_feat['promo'] = train['promotion_active'].values

test_feat = pd.DataFrame({'t': np.arange(len(train), len(train) + len(test))}, index=test.index)
test_feat = pd.concat([test_feat, pd.get_dummies(test['month'].astype(month_categories), prefix='m', drop_first=True)], axis=1)
test_feat['promo'] = test['promotion_active'].values
# align columns (in case a month category doesn't appear in test)
test_feat = test_feat.reindex(columns=train_feat.columns, fill_value=0)

lr = LinearRegression()
lr.fit(train_feat, train['sales'])
lr_pred = lr.predict(test_feat)

# ---------------------------------------------------------
# 4. MODEL 2: HOLT-WINTERS EXPONENTIAL SMOOTHING (triple exponential smoothing)
# ---------------------------------------------------------
hw_model = ExponentialSmoothing(train['sales'], trend='add', seasonal='add', seasonal_periods=12).fit()
hw_pred = hw_model.forecast(len(test))

# ---------------------------------------------------------
# 5. EVALUATION
# ---------------------------------------------------------
def evaluate(name, actual, pred):
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    return {'Model': name, 'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape}

results = pd.DataFrame([
    evaluate('Linear Regression (trend+season+promo)', test['sales'].values, lr_pred),
    evaluate('Holt-Winters Exponential Smoothing', test['sales'].values, hw_pred.values)
])
print(results.to_string(index=False))
results.to_csv("../outputs/model_comparison.csv", index=False)

# ---------------------------------------------------------
# 6. FUTURE FORECAST (next 6 months, using best model retrained on full data)
# ---------------------------------------------------------
best_model_name = results.loc[results['MAPE (%)'].idxmin(), 'Model']
print(f"\nBest model by MAPE: {best_model_name}")

# Retrain Linear Regression (the winning model) on the FULL dataset for genuine future forecast
future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=6, freq='MS')

full_feat = pd.DataFrame({'t': np.arange(len(df))}, index=df.index)
full_feat = pd.concat([full_feat, pd.get_dummies(df['month'].astype(month_categories), prefix='m', drop_first=True)], axis=1)
full_feat['promo'] = df['promotion_active'].values

future_months = pd.Series(future_dates.month, index=future_dates)
future_feat = pd.DataFrame({'t': np.arange(len(df), len(df) + 6)}, index=future_dates)
future_feat = pd.concat([future_feat, pd.get_dummies(future_months.astype(month_categories), prefix='m', drop_first=True)], axis=1)
future_feat['promo'] = 0  # assume no promo planned; a business can re-run with promo=1 to see uplift
future_feat = future_feat.reindex(columns=full_feat.columns, fill_value=0)

lr_full = LinearRegression()
lr_full.fit(full_feat, df['sales'])
future_forecast = lr_full.predict(future_feat)

# Residual-based 90% CI from in-sample fit error
in_sample_pred = lr_full.predict(full_feat)
resid_std = (df['sales'] - in_sample_pred).std()
ci_upper = future_forecast + 1.645 * resid_std
ci_lower = future_forecast - 1.645 * resid_std

forecast_df = pd.DataFrame({
    'date': future_dates,
    'forecast': future_forecast,
    'lower_90ci': ci_lower,
    'upper_90ci': ci_upper
})
forecast_df.to_csv("../outputs/next_6_months_forecast.csv", index=False)
print("\nNext 6 months forecast:")
print(forecast_df)

# ---------------------------------------------------------
# 7. VISUALIZATIONS
# ---------------------------------------------------------

# Chart 2: Historical sales trend with promo markers
fig, ax = plt.subplots()
ax.plot(df.index, df['sales'], color='#2E86AB', linewidth=2, label='Monthly Sales')
promo_points = df[df['promotion_active'] == 1]
ax.scatter(promo_points.index, promo_points['sales'], color='#E63946', zorder=5, label='Promotion Active', s=40)
ax.set_title('Historical Monthly Sales Trend (2022-2025)', fontsize=13, fontweight='bold')
ax.set_ylabel('Sales ($)')
ax.legend()
plt.tight_layout()
plt.savefig("../outputs/charts/02_historical_sales_trend.png", dpi=150)
plt.close()

# Chart 3: Actual vs Predicted (test period) - both models
fig, ax = plt.subplots()
ax.plot(test.index, test['sales'], color='black', marker='o', linewidth=2, label='Actual')
ax.plot(test.index, lr_pred, color='#F4A261', marker='s', linestyle='--', label='Linear Regression')
ax.plot(test.index, hw_pred.values, color='#2E86AB', marker='^', linestyle='--', label='Holt-Winters')
ax.set_title('Actual vs Forecasted Sales (Test Period - Last 8 Months)', fontsize=13, fontweight='bold')
ax.set_ylabel('Sales ($)')
plt.xticks(rotation=30)
ax.legend()
plt.tight_layout()
plt.savefig("../outputs/charts/03_actual_vs_predicted.png", dpi=150)
plt.close()

# Chart 4: Model accuracy comparison
fig, ax = plt.subplots()
results.set_index('Model')['MAPE (%)'].plot(kind='barh', ax=ax, color=['#F4A261', '#2E86AB'])
ax.set_title('Model Accuracy Comparison (Lower MAPE = Better)', fontsize=13, fontweight='bold')
ax.set_xlabel('MAPE (%)')
plt.tight_layout()
plt.savefig("../outputs/charts/04_model_accuracy_comparison.png", dpi=150)
plt.close()

# Chart 5: Future forecast with confidence interval
fig, ax = plt.subplots()
ax.plot(df.index[-12:], df['sales'].iloc[-12:], color='black', linewidth=2, marker='o', label='Historical')
ax.plot(forecast_df['date'], forecast_df['forecast'], color='#2E86AB', linewidth=2, marker='s', label='Forecast')
ax.fill_between(forecast_df['date'], forecast_df['lower_90ci'], forecast_df['upper_90ci'],
                 color='#2E86AB', alpha=0.2, label='90% Confidence Interval')
ax.set_title('Next 6-Month Sales Forecast', fontsize=13, fontweight='bold')
ax.set_ylabel('Sales ($)')
plt.xticks(rotation=30)
ax.legend()
plt.tight_layout()
plt.savefig("../outputs/charts/05_future_forecast.png", dpi=150)
plt.close()

# Chart 6: Monthly seasonality pattern (avg index by month)
fig, ax = plt.subplots()
monthly_avg = df.groupby('month')['sales'].mean()
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
ax.bar(month_names, monthly_avg.values, color='#3EAE7E')
ax.set_title('Average Sales by Month (Seasonality Pattern)', fontsize=13, fontweight='bold')
ax.set_ylabel('Average Sales ($)')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("../outputs/charts/06_monthly_seasonality.png", dpi=150)
plt.close()

print("\nAll 6 forecasting charts saved to outputs/charts/")

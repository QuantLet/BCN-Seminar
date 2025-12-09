## 1 Data====================================================================================

##Step 1: Data Acquisition-------------------------------------------------------------------------

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Pic Style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# 1. Define time and cypto currencies
start_date = '2020-01-01'
end_date = '2025-12-01' # set here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
tickers = ['BTC-USD', 'ETH-USD']

# 2. Getting data
df = yf.download(tickers, start=start_date, end=end_date)['Close']

# 3. Data cleaning
# deal with potential black data
df = df.ffill().dropna()

# 4. rename
df.columns = ['BTC', 'ETH']

print("Data Overview：")
print(df.head())
print(f"\nNumber of Lines of Data: {len(df)}")

##Step 2: Log-Transformation-------------------------------------------------------------------------

# Calculate the log price
df['ln_BTC'] = np.log(df['BTC'])
df['ln_ETH'] = np.log(df['ETH'])

# plotting
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Fig1: Original Price
ax1 = axes[0]
ax2 = ax1.twinx()
ax1.plot(df.index, df['BTC'], 'g-', label='BTC Price')
ax2.plot(df.index, df['ETH'], 'b-', label='ETH Price')
ax1.set_title('Raw Prices: BTC (Green) vs ETH (Blue)', fontsize=14)
ax1.set_ylabel('BTC Price ($)')
ax2.set_ylabel('ETH Price ($)')

# Fig2: Logged price）
axes[1].plot(df.index, df['ln_BTC'], 'g', label='Log BTC')
axes[1].plot(df.index, df['ln_ETH'], 'b', label='Log ETH')
axes[1].set_title('Log-Prices: Exploring the Trend', fontsize=14)
axes[1].set_ylabel('Log Price')
axes[1].legend()

plt.tight_layout()
plt.show()


##Step 3: Spurious Regression-------------------------------------------------------------------------
# Pearson coefficient
correlation = df['BTC'].corr(df['ETH'])
log_correlation = df['ln_BTC'].corr(df['ln_ETH'])

print(f"Correlation of origaion price: {correlation:.4f}")
print(f"Correlation of logged price: {log_correlation:.4f}")

# Scatter Plot
plt.figure(figsize=(10, 8))
sns.regplot(x='ln_BTC', y='ln_ETH', data=df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title(f'Scatter Plot: Log BTC vs Log ETH (Corr: {log_correlation:.2f})', fontsize=14)
plt.xlabel('Log BTC Price')
plt.ylabel('Log ETH Price')
plt.show()


##Step 4: Differencing-------------------------------------------------------------------------
# Daily regure (1st differencing)
df['ret_BTC'] = df['BTC'].pct_change()
df['ret_ETH'] = df['ETH'].pct_change()

# drop the first nan
df_ret = df.dropna()

# plot the return figure
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_ret.index, df_ret['ret_BTC'], 'g', alpha=0.6, label='BTC Returns')
ax.plot(df_ret.index, df_ret['ret_ETH'], 'b', alpha=0.6, label='ETH Returns')
ax.set_title('Daily Returns: Stationary Series (Noise)', fontsize=14)
ax.set_ylabel('Daily % Change')
ax.legend()
plt.show()

# calculate the correlation of the return rate
ret_correlation = df_ret['ret_BTC'].corr(df_ret['ret_ETH'])
print(f"correlation of the return rate: {ret_correlation:.4f}")


## 2 Unit Root Tests====================================================================================

##Step1 Define functions--------------------------------------------------------------------------------
from statsmodels.tsa.stattools import adfuller


def adf_test(series, title=''):

    print(f'Augmented Dickey-Fuller Test: {title}')
    # adfuller Return: (adf_stat, pvalue, usedlag, nobs, critical_values, icbest)
    result = adfuller(series.dropna(), autolag='AIC')

    labels = ['ADF Statistic', 'p-value', '# Lags Used', '# Observations']
    out = pd.Series(result[0:4], index=labels)

    for key, val in result[4].items():
        out[f'Critical Value ({key})'] = val

    print(out)

    # rule the conclusion
    if result[1] <= 0.05:
        print("Conclusion: P-Value < 0.05，reject H0 -> The data is Stationary")
    else:
        print("Conclusion: P-Value > 0.05，cannot reject H0 -> The data is Non-Stationary")
    print('-' * 40)

## Step2  Log Prices-------------------------------------------------------------------------------
# check BTC log price
adf_test(df['ln_BTC'], title='Log BTC Prices')

# check ETH log price
adf_test(df['ln_ETH'], title='Log ETH Prices')


## Step3 check Returns------------------------------------------------------------------------------------
# check BTC return 1st differencing
adf_test(df['ret_BTC'], title='BTC Returns (1st Difference)')

# check ETH return
adf_test(df['ret_ETH'], title='ETH Returns (1st Difference)')


## 3 Cointegration Discovery=======================================================================================
## Step1 OLS ---------------------------------------------------------------------------------------------
import statsmodels.api as sm

# 1. data
Y = df['ln_ETH']
X = df['ln_BTC']

# 2. Constant
X_with_const = sm.add_constant(X)

# 3. OLS
model = sm.OLS(Y, X_with_const)
results = model.fit()

# 4. results
alpha = results.params['const']
beta = results.params['ln_BTC']

print(f"OLS Outcome:")
print(f"Alpha (Residuals): {alpha:.4f}")
print(f"Beta (Hedge Ratio): {beta:.4f}")
print(f"R-squared: {results.rsquared:.4f}")

## Step 2 Residuals------------------------------------------------------------------------------
# calculate residual series
df['spread'] = Y - results.predict(X_with_const)

# visualization
plt.figure(figsize=(14, 6))
df['spread'].plot(color='purple', alpha=0.7)
plt.axhline(df['spread'].mean(), color='black', linestyle='--')
plt.title(f'The Spread (Residuals): Mean Reversion Check (Beta={beta:.2f})', fontsize=14)
plt.ylabel('Log Price Deviation')
plt.xlabel('Date')
plt.show()

## Step3 stationarity check----------------------------------------------------------------------------
print(">>> 执行 Engle-Granger 协整检验 (残差 ADF 测试) <<<")
adf_test(df['spread'], title='Cointegration Residuals')

##visualization----------------------------------------------------------------------
import matplotlib.dates as mdates

# 1. re-calcaulate residuals
df['spread'] = df['ln_ETH'] - (results.params['const'] + results.params['ln_BTC'] * df['ln_BTC'])

# 2. plot
fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(df.index, df['spread'], color='purple', label='Log Price Spread (Residuals)')

ax.axhline(0, color='black', linestyle='--', linewidth=2, label='Long-term Equilibrium')

std_dev = df['spread'].std()
ax.axhline(std_dev, color='green', linestyle=':', alpha=0.5)
ax.axhline(-std_dev, color='green', linestyle=':', alpha=0.5)
ax.fill_between(df.index, std_dev, -std_dev, color='green', alpha=0.1, label='1 Sigma Range')


# 2022/9/15: ETH The Merge (PoW -> PoS)
merge_date = pd.Timestamp('2022-09-15')
ax.axvline(merge_date, color='red', linestyle='-.', alpha=0.8)
ax.text(merge_date, df['spread'].max()*0.9, ' The Merge (PoS)', color='red', rotation=0)

# 2022/11: FTX crash
ftx_date = pd.Timestamp('2022-11-11')
ax.axvline(ftx_date, color='orange', linestyle='-.', alpha=0.8)
ax.text(ftx_date, df['spread'].min()*0.9, ' FTX Crash', color='orange', rotation=0)

ax.set_title(f'Why Cointegration Failed: Residual Analysis (p-value={0.21:.2f})', fontsize=16)
ax.set_ylabel('Deviation from Equilibrium')
ax.legend(loc='upper left')

plt.show()


## Regime-Specific Test=========================================
# 1. data clip
start_slice = '2020-01-01'
end_slice = '2021-04-01'

df_sub = df.loc[start_slice:end_slice].copy()

print(f"Period:from {start_slice} to {end_slice} (data points: {len(df_sub)})")

# 2. ols
X_sub = sm.add_constant(df_sub['ln_BTC'])
Y_sub = df_sub['ln_ETH']

model_sub = sm.OLS(Y_sub, X_sub)
results_sub = model_sub.fit()

print(f"New Beta: {results_sub.params['ln_BTC']:.4f}")
print(f"New R-squared: {results_sub.rsquared:.4f}")

# 3. calculate the new residuals
df_sub['spread'] = Y_sub - results_sub.predict(X_sub)

# 4. ADT Test
print("\n>>> Results <<<")
adf_test(df_sub['spread'], title='Sub-period Cointegration Test')

# 5. plot
plt.figure(figsize=(12, 5))
plt.plot(df_sub.index, df_sub['spread'], color='green')
plt.axhline(0, color='black', linestyle='--')
plt.title(f'The "Golden Era" Spread ({start_slice} to {end_slice})', fontsize=14)
plt.ylabel('Spread')
plt.show()
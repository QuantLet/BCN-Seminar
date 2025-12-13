## 1 Data====================================================================================

##Step 1: Data Acquisition-------------------------------------------------------------------------

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- [Global Transparency Configuration] ---
sns.set_style('whitegrid', {
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none"
})
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['figure.figsize'] = (12, 6)
# --------------------------------

# 1. Define time and crypto currencies
start_date = '2020-01-01'
end_date = '2025-12-01'
tickers = ['BTC-USD', 'ETH-USD']

# 2. Getting data
print("Downloading data...")
df = yf.download(tickers, start=start_date, end=end_date)['Close']

# 3. Data cleaning
df = df.ffill().dropna()

# 4. rename
df.columns = ['BTC', 'ETH']

# --- [Export Dataset to CSV (Keep this function)] ---
csv_filename = 'crypto_data.csv'
df.to_csv(csv_filename)
print(f"Data exported to {csv_filename} successfully.")
# ----------------------------------

print("Data Overview:")
print(df.head())
print(f"\nNumber of Lines of Data: {len(df)}")

##Step 2: Log-Transformation-------------------------------------------------------------------------

# Calculate the log price
df['ln_BTC'] = np.log(df['BTC'])
df['ln_ETH'] = np.log(df['ETH'])

# plotting
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Set figure background to transparent
fig.patch.set_alpha(0.0)

# Fig1: Original Price
ax1 = axes[0]
ax1.patch.set_alpha(0.0)
ax2 = ax1.twinx()
ax1.plot(df.index, df['BTC'], 'g-', label='BTC Price')
ax2.plot(df.index, df['ETH'], 'b-', label='ETH Price')
ax1.set_title('Raw Prices: BTC (Green) vs ETH (Blue)', fontsize=14)
ax1.set_ylabel('BTC Price ($)')
ax2.set_ylabel('ETH Price ($)')

# Fig2: Logged price
axes[1].patch.set_alpha(0.0)
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

print(f"Correlation of original price: {correlation:.4f}")
print(f"Correlation of logged price: {log_correlation:.4f}")

# --- [Restore to default: Standard Rectangular Scatter Plot] ---
plt.figure(figsize=(10, 8)) # Restore to default size

# Set transparency
plt.gcf().patch.set_alpha(0.0)
plt.gca().patch.set_alpha(0.0)

sns.regplot(x='ln_BTC', y='ln_ETH', data=df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})

# Remove forced xlim, ylim, and aspect ratio settings to make it adaptive

plt.title(f'Scatter Plot: Log BTC vs Log ETH (Corr: {log_correlation:.2f})', fontsize=14)
plt.xlabel('Log BTC Price')
plt.ylabel('Log ETH Price')
plt.show()
# ------------------------------------------


##Step 4: Differencing-------------------------------------------------------------------------
# Daily returns (1st differencing)
df['ret_BTC'] = df['BTC'].pct_change()
df['ret_ETH'] = df['ETH'].pct_change()

# drop the first nan
df_ret = df.dropna()

# plot the return figure
fig, ax = plt.subplots(figsize=(14, 6))

# Set transparency
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

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
        print("Conclusion: P-Value < 0.05, reject H0 -> The data is Stationary")
    else:
        print("Conclusion: P-Value > 0.05, cannot reject H0 -> The data is Non-Stationary")
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

# Set transparency
plt.gcf().patch.set_alpha(0.0)
plt.gca().patch.set_alpha(0.0)

df['spread'].plot(color='purple', alpha=0.7)
plt.axhline(df['spread'].mean(), color='black', linestyle='--')
plt.title(f'The Spread (Residuals): Mean Reversion Check (Beta={beta:.2f})', fontsize=14)
plt.ylabel('Log Price Deviation')
plt.xlabel('Date')
plt.show()

## Step3 stationarity check----------------------------------------------------------------------------
print(">>> Performing Engle-Granger Cointegration Test (Residual ADF Test) <<<")
adf_test(df['spread'], title='Cointegration Residuals')

##visualization----------------------------------------------------------------------
import matplotlib.dates as mdates

# 1. re-calculate residuals
df['spread'] = df['ln_ETH'] - (results.params['const'] + results.params['ln_BTC'] * df['ln_BTC'])

# 2. plot
fig, ax = plt.subplots(figsize=(14, 7))

# Set transparency
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

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

print(f"Period: from {start_slice} to {end_slice} (data points: {len(df_sub)})")

# 2. ols
X_sub = sm.add_constant(df_sub['ln_BTC'])
Y_sub = df_sub['ln_ETH']

model_sub = sm.OLS(Y_sub, X_sub)
results_sub = model_sub.fit()

print(f"New Beta: {results_sub.params['ln_BTC']:.4f}")
print(f"New R-squared: {results_sub.rsquared:.4f}")

# 3. calculate the new residuals
df_sub['spread'] = Y_sub - results_sub.predict(X_sub)

# 4. ADF Test
print("\n>>> Results <<<")
adf_test(df_sub['spread'], title='Sub-period Cointegration Test')

# 5. plot
plt.figure(figsize=(12, 5))

# Set transparency
plt.gcf().patch.set_alpha(0.0)
plt.gca().patch.set_alpha(0.0)

plt.plot(df_sub.index, df_sub['spread'], color='green')
plt.axhline(0, color='black', linestyle='--')
plt.title(f'The "Golden Era" Spread ({start_slice} to {end_slice})', fontsize=14)
plt.ylabel('Spread')
plt.show()

## Visualization (Animation) ==============================
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# 1. Read data (If previous df is still in memory, skip this step)
try:
    df = pd.read_csv('crypto_data.csv', index_col=0, parse_dates=True)
    # Recalculate log prices
    if 'ln_BTC' not in df.columns:
        print("Recalculating log prices...")
        df['ln_BTC'] = np.log(df['BTC'])
        df['ln_ETH'] = np.log(df['ETH'])
except Exception as e:
    print(f"Error reading or processing data: {e}")
    # Generate dummy data for demonstration
    df = pd.DataFrame({
        'ln_BTC': np.linspace(5, 12, 2200),
        'ln_ETH': np.linspace(5, 12, 2200) + np.random.normal(0, 0.5, 2200)
    })

# ==========================================
# ⚡️ Core Logic for Speed Control ⚡️
# ==========================================
TARGET_DURATION = 6  # Target duration: 6 seconds
FPS = 30             # Frame rate: 30 fps
TOTAL_FRAMES = TARGET_DURATION * FPS  # Total frames = 180 frames

# Calculate step size: How many new points per frame?
# If there are 2160 data points, step size is approx 2160 / 180 = 12
data_len = len(df)
step = max(1, data_len // TOTAL_FRAMES)

# Generate list of keyframe indices (e.g., 0, 12, 24, 36...)
frame_indices = list(range(0, data_len, step))
# Ensure the last frame includes all data
if frame_indices[-1] != data_len:
    frame_indices.append(data_len)

print(f"Total Data Points: {data_len}, Target Duration: {TARGET_DURATION}s")
print(f"Total Frames Generated: {len(frame_indices)}, New Points per Frame: {step}")
# ==========================================

# 2. Set canvas
fig, ax = plt.subplots(figsize=(8, 8))

# Global transparency
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# 3. Lock axis ranges (Must use full data to calculate ranges)
x_min, x_max = df['ln_BTC'].min() - 0.5, df['ln_BTC'].max() + 0.5
y_min, y_max = df['ln_ETH'].min() - 0.5, df['ln_ETH'].max() + 0.5

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel('Log BTC Price')
ax.set_ylabel('Log ETH Price')
ax.set_title(f'Scatter Plot Animation ({TARGET_DURATION}s version)', color='black')

# Initialize scatter object
scat = ax.scatter([], [], alpha=0.6, c='#1f77b4', edgecolors='none')


# 4. Update function
def update(frame_idx):
    # frame_idx is the index preset in frame_indices
    current_data = df.iloc[:frame_idx]

    if len(current_data) > 0:
        offsets = current_data[['ln_BTC', 'ln_ETH']].values
        scat.set_offsets(offsets)
    return scat,


# 5. Create animation
# frames=frame_indices: Pass in the calculated list of skip indices directly
ani = animation.FuncAnimation(fig, update, frames=frame_indices, interval=1000 / FPS, blit=True)

# 6. Save
print("Rendering 6-second high-speed animation...")
try:
    # fps=FPS (30) determines playback speed
    ani.save('scatter_fast_6s.gif', writer='pillow', fps=FPS, savefig_kwargs={'transparent': True, 'facecolor': 'none'})
    print("Success! Saved as scatter_fast_6s.gif")
except Exception as e:
    print(f"Save failed: {e}")

plt.close()
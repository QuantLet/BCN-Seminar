import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from pathlib import Path
from statistics import NormalDist

# colors for plotting
cblue = "#0072B2"   
corange = "#E69F00"   

def read_csvs(coin): # function to read csv files from folder
    base_dir = Path(__file__).resolve().parent
    path = base_dir / "data" / coin
    files = sorted(path.glob("*.csv"))

    df = pd.concat((pd.read_csv(f, skiprows=1) for f in files), ignore_index=True)
    df.loc[df["unix"] > 10**12, "unix"] //= 1000 # make sure that unix in seconds
    df['date'] = pd.to_datetime(df['date']) # convert date to datetime
    return df.sort_values(["date"]).reset_index(drop=True) 

btc = read_csvs("btc") # bitcoin
eth = read_csvs("eth") # ethereum

m1_data = pd.merge(
    btc[['unix', 'date', 'close', 'Volume BTC', 'Volume USD']].rename(columns={'close': 'BTC_close', 'Volume BTC': 'BTC_vol', 'Volume USD': "BTC_USD_vol"}),
    eth[['date', 'close', 'Volume ETH', 'Volume USD']].rename(columns={'close': 'ETH_close', 'Volume ETH': 'ETH_vol', 'Volume USD': "ETH_USD_vol"}),
    on="date",
    how="right" 
) # merge to get minute data

m1_data = m1_data.sort_values("date").reset_index(drop=True) # sort by date

# keep only one row if there are duplicates
m1_data = m1_data.drop_duplicates(subset="unix", keep="first")

# calculate annualzed realized variance and volatility 
rv_data = m1_data[["unix", "date", "BTC_close", "ETH_close"]].copy()

rv_data["bin_5m"] = (rv_data["unix"] // 300) * 300 # create 5-minute bins (5 minutes = 300 seconds)

rv_data = (
        rv_data.groupby("bin_5m")
         .last()
         .reset_index().drop(columns='bin_5m')
    ) # take last value in each 5-minute bin

# compute squared log-returns
rv_data['BTC_r2'] = np.log(rv_data['BTC_close']).diff()**2
rv_data['ETH_r2'] = np.log(rv_data['ETH_close']).diff()**2

# sum up squared log-returns for every day
rv_data = (
    rv_data.groupby(rv_data["date"].dt.date)[["BTC_r2", "ETH_r2"]]
    .sum(min_count=1)          
    .reset_index()
    .rename(columns={"date": "day"})
    # calculate annualized variance and volatility
    .assign(
        BTC_rv = lambda x: x["BTC_r2"] * 365,
        ETH_rv = lambda x: x["ETH_r2"] * 365
        )
    # drop intermediate columns
    .drop(columns=["BTC_r2", "ETH_r2"])   
)  

# calculate daily volume
volume_data = m1_data[["date", "BTC_vol", "BTC_USD_vol", "ETH_vol", "ETH_USD_vol"]].copy()

volume = (
    volume_data
    .groupby(volume_data["date"].dt.date)[["BTC_vol", "BTC_USD_vol", "ETH_vol", "ETH_USD_vol"]]
    .sum(min_count=1)          
    .reset_index()
    .rename(columns={"date": "day"})
)

daily_data = pd.merge(rv_data, volume, on="day", how="left")


# 5-min returns from minute data
tmp = m1_data[["unix", "date", "BTC_close", "ETH_close"]].copy()
tmp["bin_5m"] = (tmp["unix"] // 300) * 300
tmp = tmp.groupby("bin_5m", as_index=False).last()
tmp["day"] = pd.to_datetime(tmp["date"]).dt.date

tmp["BTC_r"] = np.log(tmp["BTC_close"]).diff()
tmp["ETH_r"] = np.log(tmp["ETH_close"]).diff()

# daily RV / J / TJ 
ALPHA = 0.9999
C_THETA = 3.0

def threshold_jump_stats(r, alpha=ALPHA, c_theta=C_THETA):
    r = r.dropna().to_numpy(dtype=float)
    m = len(r)
    if m < 3:
        return pd.Series({"RV": np.nan, "BV": np.nan, "J": np.nan, "TJ": np.nan, "Z": np.nan})

    r2 = r ** 2
    abs_r = np.abs(r)

    rv = float(np.sum(r2))
    bv = float((np.pi / 2.0) * np.sum(abs_r[1:] * abs_r[:-1]))

    # Practical threshold approximation for local filtering
    local_var = float(np.median(r2))
    theta = (c_theta ** 2) * max(local_var, 1e-12)
    keep = r2 <= theta

    tbv = float((np.pi / 2.0) * np.sum(abs_r[1:] * abs_r[:-1] * keep[1:] * keep[:-1]))

    j_raw = max(rv - bv, 0.0)
    tj_raw = max(rv - tbv, 0.0)

    # Jump significance z-test (Barndorff-Nielsen / Shephard style scaling)
    c = ((np.pi / 2.0) ** 2 + np.pi - 5.0)
    denom = np.sqrt(max(c * (tbv ** 2) / max(m, 1), 1e-12))
    z = (rv - tbv) / denom

    z_crit = NormalDist().inv_cdf(alpha)
    sig = z > z_crit

    j_sig = j_raw if sig else 0.0
    tj_sig = tj_raw if sig else 0.0

    return pd.Series({"RV": rv, "BV": bv, "J": j_sig, "TJ": tj_sig, "Z": z})

btc_jump = tmp.groupby("day")["BTC_r"].apply(threshold_jump_stats).reset_index()
btc_jump = btc_jump.pivot(index="day", columns="level_1", values="BTC_r").reset_index()
btc_jump.columns = ["day", "BTC_BV", "BTC_J", "BTC_RV_from_5m", "BTC_TJ", "BTC_Z"]

eth_jump = tmp.groupby("day")["ETH_r"].apply(threshold_jump_stats).reset_index()
eth_jump = eth_jump.pivot(index="day", columns="level_1", values="ETH_r").reset_index()
eth_jump.columns = ["day", "ETH_BV", "ETH_J", "ETH_RV_from_5m", "ETH_TJ", "ETH_Z"]

jump_daily = btc_jump.merge(eth_jump, on="day", how="inner")

# add jump variables to modeling dataset
daily_data = daily_data.merge(
    jump_daily[["day", "BTC_J", "BTC_TJ", "ETH_J", "ETH_TJ"]],
    on="day",
    how="left",
)

# drop first and last row (incomplete data)
daily_data = daily_data.iloc[1:-1].reset_index(drop=True)

# plot volume
plt.figure(figsize=(14, 4.2), dpi=300)

plt.plot(daily_data["day"], np.log(daily_data["BTC_USD_vol"]), label="BTC", color=cblue, linewidth=0.5)
plt.plot(daily_data["day"], np.log(daily_data["ETH_USD_vol"]), label="ETH", color=corange, linewidth=0.5)

plt.xlim(daily_data["day"].min(), daily_data["day"].max())
plt.ylim(0, np.log(daily_data["BTC_USD_vol"]).max() * 1.2)  

ax = plt.gca() 
ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_minor_locator(mdates.YearLocator(base=1))
ax.tick_params(axis='x', which='minor', length=0)
plt.grid(True, which='both', axis='x', alpha=0.25)

plt.xlabel("Day")
plt.ylabel("log Volume (USD)")
plt.legend(loc="upper right", frameon=False)

plt.grid(True, alpha=0.25)
plt.savefig("volume.png", transparent=True, bbox_inches='tight')



# plot realized volatility
plt.figure(figsize=(14, 4.2), dpi=300)

plt.plot(daily_data["day"], np.log(daily_data["BTC_rv"]), label="BTC", color=cblue, linewidth=0.5)
plt.plot(daily_data["day"], np.log(daily_data["ETH_rv"]), label="ETH", color=corange, linewidth=0.5)

plt.xlim(daily_data["day"].min(), daily_data["day"].max())
# plt.ylim(0, np.log(daily_data["BTC_rvol"]).max() * 1.1)
ax = plt.gca() 
ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_minor_locator(mdates.YearLocator(base=1))
ax.tick_params(axis='x', which='minor', length=0)
plt.grid(True, which='both', axis='x', alpha=0.25)

plt.xlabel("Day")
plt.ylabel("log Realized Variance")
plt.legend(loc="upper right", frameon=False)

plt.grid(True, alpha=0.25)
plt.savefig("rized_vola.png", transparent=True, bbox_inches='tight')


# train-test split
# 70 / 15 / 15 split + small purge gap
n = len(daily_data)
purge = 30
i_train_end = int(n * 0.70)
i_val_end   = int(n * 0.85)

train_end_day = daily_data.loc[i_train_end - 1, "day"]
val_start_day = daily_data.loc[i_train_end + purge, "day"]
val_end_day   = daily_data.loc[i_val_end - 1, "day"]
test_start_day = daily_data.loc[i_val_end + purge, "day"]


# HAR + MLP helpers 
def qlike(y, yhat):
    y = np.maximum(np.asarray(y, dtype=float), 1e-12)
    yhat = np.maximum(np.asarray(yhat, dtype=float), 1e-12)
    return np.mean(np.log(yhat) + (y / yhat))


def score_forecast(y, yhat):
    y = np.maximum(np.asarray(y, dtype=float), 1e-12)
    yhat = np.maximum(np.asarray(yhat, dtype=float), 1e-12)

    mse = np.mean((y - yhat) ** 2)
    hrmse = np.sqrt(np.mean(((y - yhat) / y) ** 2))
    q = qlike(y, yhat)
    return {"mse": mse, "hrmse": hrmse, "qlike": q}


def build_dataset_log(df, rv_col, h, train_end_day, val_start_day, val_end_day, test_start_day, jump_col=None):
    cols = ["day", rv_col]
    if jump_col is not None:
        cols.append(jump_col)

    d = df[cols].copy()
    d["rv_log"] = np.log(np.maximum(d[rv_col].astype(float), 1e-12))

    # HAR regressors
    d["X_d"] = d["rv_log"].shift(1)
    d["X_w"] = d["rv_log"].shift(1).rolling(7).mean()
    d["X_m"] = d["rv_log"].shift(1).rolling(30).mean()
    feat_cols = ["X_d", "X_w", "X_m"]

    # Optional jump regressors
    if jump_col is not None:
        d["j_log"] = np.log(np.maximum(d[jump_col].astype(float), 1e-12))
        d["X_jd"] = d["j_log"].shift(1)
        d["X_jw"] = d["j_log"].shift(1).rolling(7).mean()
        d["X_jm"] = d["j_log"].shift(1).rolling(30).mean()
        feat_cols += ["X_jd", "X_jw", "X_jm"]

    # Normalized horizon target (average future log-RV)
    d["y_log"] = sum(d["rv_log"].shift(-k) for k in range(1, h + 1)) / h
    d = d.dropna().reset_index(drop=True)

    train = d[d["day"] <= train_end_day]
    val = d[(d["day"] >= val_start_day) & (d["day"] <= val_end_day)]
    test = d[d["day"] >= test_start_day]

    return {
        "X_train": train[feat_cols],
        "y_train": train["y_log"],
        "X_val": val[feat_cols],
        "y_val": val["y_log"],
        "X_test": test[feat_cols],
        "y_test": test["y_log"],
    }


def fit_har(ds, h, hac_lags):
    X_train = sm.add_constant(ds["X_train"], has_constant="add")
    y_train = ds["y_train"].to_numpy(dtype=float)
    return sm.OLS(y_train, X_train).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags[h]})


def predict_har(model, X_df):
    X = sm.add_constant(X_df, has_constant="add")
    return model.predict(X).astype(float)


def fit_mlp(ds, params, seed=42):
    X_train = ds["X_train"].to_numpy(dtype=float)
    y_train = ds["y_train"].to_numpy(dtype=float)

    x_scaler = StandardScaler().fit(X_train)
    X_train_s = x_scaler.transform(X_train)

    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).ravel()

    model = MLPRegressor(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        alpha=params["alpha"],
        learning_rate_init=params["learning_rate_init"],
        activation=params.get("activation", "relu"),
        solver="adam",
        learning_rate=params.get("learning_rate", "adaptive"),
        batch_size=params.get("batch_size", "auto"),
        early_stopping=True,
        validation_fraction=params.get("validation_fraction", 0.2),
        n_iter_no_change=params.get("n_iter_no_change", 60),
        max_iter=params.get("max_iter", 6000),
        random_state=seed,
    )


    model.fit(X_train_s, y_train_s)

    return {"model": model, "x_scaler": x_scaler, "y_scaler": y_scaler, "params": params, "seed": seed}


def predict_mlp(bundle, X_df):
    X = bundle["x_scaler"].transform(X_df.to_numpy(dtype=float))
    yhat_s = bundle["model"].predict(X)
    return bundle["y_scaler"].inverse_transform(yhat_s.reshape(-1, 1)).ravel()

# HAR models: HAR, HAR-RVJ, HAR-RVTJ
daily_model_data = daily_data.copy()

asset_cfg = {
    "BTC": {"rv": "BTC_rv", "j": "BTC_J", "tj": "BTC_TJ"},
    "ETH": {"rv": "ETH_rv", "j": "ETH_J", "tj": "ETH_TJ"},
}
har_specs = {
    "HAR": None,
    "HAR_RVJ": "j",
    "HAR_RVTJ": "tj",
}
horizons = [1]
hac_lags = {1: 7}
rows = []

for model_name, jump_key in har_specs.items():
    for asset, cfg in asset_cfg.items():
        jump_col = cfg[jump_key] if jump_key is not None else None

        for h in horizons:
            ds = build_dataset_log(
                daily_model_data,
                rv_col=cfg["rv"],
                h=h,
                train_end_day=train_end_day,
                val_start_day=val_start_day,
                val_end_day=val_end_day,
                test_start_day=test_start_day,
                jump_col=jump_col,
            )

            if len(ds["y_train"]) == 0:
                continue

            model = fit_har(ds, h=h, hac_lags=hac_lags)

            y_log = ds["y_test"].to_numpy(dtype=float)
            if len(y_log) == 0:
                continue

            yhat_log = predict_har(model, ds["X_test"])
            y_sq = np.exp(y_log)
            yhat_sq = np.exp(yhat_log)
            scores = score_forecast(y_sq, yhat_sq)

            rows.append({
                "model": model_name,
                "asset": asset,
                **scores,
            })

har_eval = pd.DataFrame(rows).sort_values(["model", "asset"]).reset_index(drop=True)
har_eval = har_eval[["model", "asset", "mse", "hrmse", "qlike"]]
print(har_eval)

# MLP 
mlp_specs = {
    "MLP_HAR": None,
    "MLP_HAR_RVJ": "j",
    "MLP_HAR_RVTJ": "tj",
}
mlp_param_grid = [
    {"hidden_layer_sizes": (32, 16), "alpha": 1e-4, "learning_rate_init": 1e-3},
    {"hidden_layer_sizes": (64, 32), "alpha": 1e-4, "learning_rate_init": 1e-3},
    {"hidden_layer_sizes": (64, 32, 16), "alpha": 1e-4, "learning_rate_init": 5e-4},
    {"hidden_layer_sizes": (128, 64), "alpha": 1e-4, "learning_rate_init": 5e-4},
    {"hidden_layer_sizes": (64, 32), "alpha": 1e-3, "learning_rate_init": 5e-4},
    {"hidden_layer_sizes": (32, 16), "alpha": 1e-5, "learning_rate_init": 2e-3},
    {"hidden_layer_sizes": (64, 32), "alpha": 1e-4, "learning_rate_init": 1e-3, "activation": "tanh"},
    {"hidden_layer_sizes": (32, 16), "alpha": 1e-4, "learning_rate_init": 1e-3, "activation": "tanh"},
]
mlp_seeds = [11, 42]

rows = []

for model_name, jump_key in mlp_specs.items():
    for asset, cfg in asset_cfg.items():
        jump_col = cfg[jump_key] if jump_key is not None else None

        for h in horizons:
            ds = build_dataset_log(
                daily_model_data,
                rv_col=cfg["rv"],
                h=h,
                train_end_day=train_end_day,
                val_start_day=val_start_day,
                val_end_day=val_end_day,
                test_start_day=test_start_day,
                jump_col=jump_col,
            )

            if len(ds["y_train"]) == 0 or len(ds["y_val"]) == 0:
                continue

            best_bundle = None
            best_val_qlike = np.inf

            for params in mlp_param_grid:
                for seed in mlp_seeds:
                    bundle = fit_mlp(ds, params=params, seed=seed)
                    yhat_val_log = predict_mlp(bundle, ds["X_val"])
                    y_val_sq = np.exp(ds["y_val"].to_numpy(dtype=float))
                    yhat_val_sq = np.exp(yhat_val_log)
                    val_qlike = qlike(y_val_sq, yhat_val_sq)

                    if val_qlike < best_val_qlike:
                        best_val_qlike = val_qlike
                        best_bundle = bundle

            if best_bundle is None:
                continue

            y_log = ds["y_test"].to_numpy(dtype=float)
            if len(y_log) == 0:
                continue

            yhat_log = predict_mlp(best_bundle, ds["X_test"])
            y_sq = np.exp(y_log)
            yhat_sq = np.exp(yhat_log)
            scores = score_forecast(y_sq, yhat_sq)

            rows.append({
                "model": model_name,
                "asset": asset,
                "best_params": f'{best_bundle["params"]} | seed={best_bundle["seed"]} | val_qlike={best_val_qlike:.6f}',
                **scores,
            })

mlp_eval = pd.DataFrame(rows).sort_values(["model", "asset"]).reset_index(drop=True)
mlp_eval = mlp_eval[["model", "asset", "best_params", "mse", "hrmse", "qlike"]]
print(mlp_eval)

# Compare HAR vs MLP and select  best 
def _norm_eval(df, default_spec=""):
    x = df.copy()
    spec_col = next((c for c in ["best_params", "params", "spec"] if c in x.columns), None)
    x["model_spec"] = x[spec_col].astype(str) if spec_col is not None else default_spec
    keep = ["model", "model_spec", "asset", "mse", "hrmse", "qlike"]
    return x[keep]


e_har = _norm_eval(har_eval, default_spec="OLS-HAC")
e_har.loc[e_har["model"] == "HAR", "model_spec"] = "OLS-HAC | HAR"
e_har.loc[e_har["model"] == "HAR_RVJ", "model_spec"] = "OLS-HAC | HAR-RVJ"
e_har.loc[e_har["model"] == "HAR_RVTJ", "model_spec"] = "OLS-HAC | HAR-RVTJ"

e_mlp = _norm_eval(mlp_eval, default_spec="MLP")

all_eval = pd.concat([e_har, e_mlp], ignore_index=True)
all_eval["class"] = np.where(all_eval["model"].str.startswith("MLP"), "MLP", "HAR")

best_per_class_single = (
    all_eval.sort_values(["class", "qlike"])
    .groupby("class", as_index=False)
    .first()[["class", "model", "model_spec", "asset", "mse", "hrmse", "qlike"]]
    .rename(
        columns={
            "model": "best_model",
            "model_spec": "best_spec",
            "mse": "test_mse",
            "hrmse": "test_hrmse",
            "qlike": "test_qlike",
        }
    )
    .sort_values("class")
    .reset_index(drop=True)
)

print(best_per_class_single)
#!/usr/bin/env python3
"""
DDoS + Blockchain Integrity Quantlet (CIC-IDS2017 / ISCX flow CSVs)
---------------------------------------------------------------

This script does TWO things:

1) BEFORE blockchain (Network-only EDA + baseline classifier: DDoS vs BENIGN)
   - 4–5 distribution-shift plots (BENIGN vs DDoS) for key flow features:
       * Flow Packets/s
       * Flow Bytes/s
       * Total Fwd Packets
       * Flow Duration
       * Fwd Packet Length Mean
   - Correlation / redundancy:
       * Correlation heatmap (top-N important features)
       * Top correlated pairs (CSV)
   - Simple rule-of-thumb table:
       * Mean/median by Label for the key flow features (CSV)
   - Baseline classifier outputs (RandomForest):
       * Confusion matrix (PNG)
       * ROC curve + AUC (PNG + TXT)
       * Feature importances (PNG + CSV)

2) BLOCKCHAIN layer (Integrity / tamper-evidence on the log)
   - Adds: record_hash, prev_hash (hash-chain), block_id, merkle_root
   - Simulates tampering and computes integrity flags:
       * hash_ok, chain_ok, merkle_ok
   - Saves a compact ledger CSV:
       ddos_ledger_with_integrity_checks.csv

How to run (example):
  python ddos_blockchain_updated.py --csv "/path/to/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv" --outdir output

Notes:
- CIC/ISCX CSVs sometimes have leading/trailing spaces in column names; this script strips them.
- If my machine is slow, lower --sample_rows / --ledger_rows.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

# ---------------------------
# Utilities
# ---------------------------
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def sanitize_colname(s: str) -> str:
    """Lowercase + remove non-alphanumerics for fuzzy matching."""
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def resolve_columns(df: pd.DataFrame, desired: List[str]) -> Dict[str, Optional[str]]:
    """
    Map desired canonical names -> actual df columns using fuzzy matching.
    Returns dict {canonical_name: actual_column_name or None}.
    """
    actual = list(df.columns)
    actual_s = {sanitize_colname(c): c for c in actual}

    mapping: Dict[str, Optional[str]] = {}
    for want in desired:
        if want in df.columns:
            mapping[want] = want
            continue
        key = sanitize_colname(want)
        if key in actual_s:
            mapping[want] = actual_s[key]
            continue

        # partial match
        candidates = []
        for c in actual:
            sc = sanitize_colname(c)
            if key in sc or sc in key:
                candidates.append(c)

        if len(candidates) == 1:
            mapping[want] = candidates[0]
        elif len(candidates) > 1:
            # pick the shortest candidate (often best)
            mapping[want] = sorted(candidates, key=len)[0]
        else:
            mapping[want] = None

    return mapping

# ---------------------------
# Data cleaning (CIC/ISCX flow CSV)
# ---------------------------
def clean_flow_dataframe(df: pd.DataFrame, label_col: str = "Label") -> pd.DataFrame:
    """Basic, safe cleaning for CIC/ISCX-style flow CSVs.

    What we do:
    - Strip column-name whitespace
    - Keep only {BENIGN, DDoS} (anything containing 'ddos' => DDoS, exact 'BENIGN' => BENIGN)
    - Drop obvious identifiers/leakage columns if present (IPs, ports, Flow ID, Timestamp, Protocol)
    - Convert non-numeric feature columns to numeric when possible; drop columns that remain non-numeric
    - Replace inf/-inf -> NaN, then fill NaN with 0
    - Drop exact duplicate rows
    - Drop constant (zero-variance) feature columns
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if label_col not in df.columns:
        # Try case/space-insensitive matching (e.g., ' Label', 'label', 'Label ')
        target = label_col.strip().lower()
        match = None
        for c in df.columns:
            if str(c).strip().lower() == target:
                match = c
                break
        if match is not None:
            df = df.rename(columns={match: label_col})
        else:
            cols_preview = ", ".join(list(df.columns)[:20])
            raise ValueError(
                f"No '{label_col}' column found. This script expects a CIC/ISCX flow CSV with a Label column. "
                f"First columns seen: {cols_preview}"
            )

    # Keep only BENIGN and DDoS rows
    lab_raw = df[label_col].astype(str).str.strip()
    lab_lower = lab_raw.str.lower()
    keep = lab_lower.eq("benign") | lab_lower.str.contains("ddos")
    df = df.loc[keep].copy()
    df[label_col] = np.where(lab_lower.loc[keep].str.contains("ddos"), "DDoS", "BENIGN")

    # Drop obvious identifier / leakage columns (if present)
    drop_if_present = [
        "Flow ID", "FlowID", "Timestamp",
        "Src IP", "Dst IP", "Source IP", "Destination IP",
        "Src Port", "Dst Port", "Source Port", "Destination Port",
        "Protocol",
    ]
    to_drop = [c for c in drop_if_present if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)

    # Convert / drop non-numeric feature columns
    drop_cols = []
    for c in df.columns:
        if c == label_col:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        converted = pd.to_numeric(df[c], errors="coerce")
        if converted.notna().sum() == 0:
            drop_cols.append(c)
        else:
            df[c] = converted

    if drop_cols:
        df = df.drop(columns=drop_cols)

    # inf/-inf handling and NaNs
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Drop duplicates
    df = df.drop_duplicates()

    # Drop constant feature columns
    const_cols = [c for c in df.columns if c != label_col and df[c].nunique(dropna=False) <= 1]
    if const_cols:
        df = df.drop(columns=const_cols)

    return df

# ---------------------------
# Plotting helpers (before blockchain)
# ---------------------------
def plot_label_distribution(df: pd.DataFrame, label_col: str, out_png: Path) -> None:
    counts = df[label_col].value_counts()
    plt.figure(figsize=(6, 4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Label distribution (BENIGN vs DDoS)")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_hist_compare(
    df: pd.DataFrame,
    label_col: str,
    feature_col: str,
    out_png: Path,
    log1p: bool = True,
    clip_quantile: float = 0.995
) -> None:
    """Overlay histogram for BENIGN vs DDoS; log1p by default (usually required for flow features)."""
    benign = df.loc[df[label_col] == "BENIGN", feature_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    ddos   = df.loc[df[label_col] == "DDoS",   feature_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()

    if log1p:
        benign = np.log1p(np.clip(benign, a_min=0, a_max=None))
        ddos   = np.log1p(np.clip(ddos, a_min=0, a_max=None))
        xlabel = f"log1p({feature_col})"
    else:
        xlabel = feature_col

    # Clip extreme tails for readability (without deleting rows)
    combined = pd.concat([pd.Series(benign), pd.Series(ddos)], ignore_index=True)
    if len(combined) > 0:
        hi = float(np.quantile(combined, clip_quantile))
        benign = np.clip(benign, a_min=None, a_max=hi)
        ddos   = np.clip(ddos,   a_min=None, a_max=hi)

    plt.figure(figsize=(9, 5))
    plt.hist(benign, bins=60, alpha=0.6, density=True, label="BENIGN")
    plt.hist(ddos,   bins=60, alpha=0.6, density=True, label="DDoS")
    plt.title(f"Distribution shift: {feature_col} (BENIGN vs DDoS)")
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_confusion_matrix(cm: np.ndarray, labels: list[str], out_png: Path) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5))

    # Make colors lighter by stretching the max range
    vmax = cm.max() * 2.0  # increase this to 2.5 if you want even lighter
    im = ax.imshow(cm, aspect="equal", cmap="Blues", vmin=0, vmax=vmax)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Baseline confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Always use black text (background is now light)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_roc(y_true: np.ndarray, y_score: np.ndarray, out_png: Path) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.title("Baseline ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    return float(roc_auc)

def plot_feature_importances(imp: pd.Series, out_png: Path, top_k: int = 12) -> None:
    top = imp.head(top_k).sort_values()  # ascending for barh
    plt.figure(figsize=(12, 5))
    plt.barh(top.index.astype(str), top.values)
    plt.title(f"Top {top_k} feature importances (RandomForest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

#def plot_corr_heatmap(X: pd.DataFrame, out_png: Path, method: str = "spearman") -> None:
 #   corr = X.corr(method=method)
  #  plt.figure(figsize=(12, 10))
   # plt.imshow(corr.values, aspect="auto")
    #plt.title(f"Feature correlation heatmap ({method})")
   # plt.colorbar()
   # plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
   # plt.yticks(range(len(corr.columns)), corr.columns, fontsize=6)
   # plt.tight_layout()
   # plt.savefig(out_png, dpi=200)
  #  plt.close()


def plot_corr_heatmap(
    X: pd.DataFrame,
    out_png: Path,
    method: str = "spearman",
    show_every: int = 1,
    dpi: int = 300
) -> None:
    """
    Pretty correlation-matrix heatmap (like common online examples):
    - square cells
    - tight saving
    - optional tick thinning via show_every
    """
    corr = X.corr(method=method)
    n = corr.shape[0]
    if n < 2:
        print("NOTE: Not enough features for correlation heatmap.")
        return

    # scale figure with number of features
    fig_w = max(8, 0.35 * n)
    fig_h = max(8, 0.35 * n)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    sns.heatmap(
        corr,
        cmap="viridis",
        vmin=-1, vmax=1, center=0,
        square=True,                      # <- makes it look “clean”
        linewidths=0.25, linecolor="white",
        cbar_kws={"shrink": 0.85, "pad": 0.02},
        ax=ax
    )

    ax.set_title(f"Feature correlation heatmap ({method})", pad=12)

    # nicer labels (optional)
    xlabels = [c.replace("_", " ") for c in corr.columns]
    ylabels = [c.replace("_", " ") for c in corr.index]
    ax.set_xticklabels(xlabels, rotation=90, fontsize=7)
    ax.set_yticklabels(ylabels, rotation=0, fontsize=7)

    # show fewer tick labels if crowded
    if show_every > 1:
        for i, lab in enumerate(ax.get_xticklabels()):
            if i % show_every != 0:
                lab.set_visible(False)
        for i, lab in enumerate(ax.get_yticklabels()):
            if i % show_every != 0:
                lab.set_visible(False)

    fig.tight_layout(pad=0.2)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)







def save_top_correlated_pairs(X: pd.DataFrame, out_csv: Path, method: str = "spearman", top_k: int = 25) -> None:
    corr = X.corr(method=method).abs()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_ut = corr.where(mask)
    pairs = (
        corr_ut.stack()
        .sort_values(ascending=False)
        .head(top_k)
        .reset_index()
        .rename(columns={"level_0": "Feature_A", "level_1": "Feature_B", 0: "AbsCorr"})
    )
    pairs.to_csv(out_csv, index=False)






#-----------++


# ============================================================
# Two separate outputs:
#   1) dendrogram only (PNG)
#   2) reordered correlation heatmap only (PNG)
#
# Paste these helper functions ABOVE def main(): (module level)
# Then call them in main() where you currently do correlation outputs.
# ============================================================

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import seaborn as sns

def _prep_corr_X(X: pd.DataFrame) -> pd.DataFrame:
    """Numeric-only, safe NaNs/Infs, drop constants."""
    Xn = X.select_dtypes(include=[np.number]).copy()
    Xn = Xn.replace([np.inf, -np.inf], np.nan)
    Xn = Xn.fillna(Xn.median(numeric_only=True))
    nunique = Xn.nunique(dropna=False)
    Xn = Xn.loc[:, nunique > 1]
    return Xn

def _cluster_order_from_corr(corr: pd.DataFrame, cluster_on_abs: bool = True):
    """Return linkage Z + leaf order from a correlation matrix."""
    if cluster_on_abs:
        dist = 1.0 - np.abs(corr.values)
    else:
        dist = 1.0 - corr.values

    dist = np.clip(dist, 0, 2)
    np.fill_diagonal(dist, 0.0)

    Z = linkage(squareform(dist, checks=False), method="average")
    leaves = dendrogram(Z, no_plot=True)["leaves"]
    return Z, leaves

def save_corr_dendrogram(
    X: pd.DataFrame,
    out_png: Path,
    method: str = "spearman",
    cluster_on_abs: bool = True
) -> None:
    """Save dendrogram ONLY."""
    Xn = _prep_corr_X(X)
    if Xn.shape[1] < 2:
        print("NOTE: Not enough numeric features for dendrogram.")
        return

    corr = Xn.corr(method=method)
    Z, _ = _cluster_order_from_corr(corr, cluster_on_abs=cluster_on_abs)

    plt.figure(figsize=(10, max(6, 0.25 * corr.shape[0])))
    dendrogram(
        Z,
        labels=corr.columns.tolist(),
        orientation="left",
        leaf_font_size=7
    )
    plt.title(f"Feature dendrogram (clustered by {'|corr|' if cluster_on_abs else 'corr'}; {method})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def save_reordered_corr_heatmap(
    X: pd.DataFrame,
    out_png: Path,
    method: str = "spearman",
    cluster_on_abs: bool = True,
    show_every: int = 1,   # set to 2 or 3 if labels are crowded
    dpi: int = 300
) -> None:
    Xn = _prep_corr_X(X)
    if Xn.shape[1] < 2:
        print("NOTE: Not enough numeric features for reordered heatmap.")
        return

    corr = Xn.corr(method=method)
    _, leaves = _cluster_order_from_corr(corr, cluster_on_abs=cluster_on_abs)
    corr_ord = corr.iloc[leaves, leaves]

    n = corr_ord.shape[0]
    # scale figure with n (CIC feature names are long)
    fig_w = max(12, 0.45 * n)
    fig_h = max(10, 0.45 * n)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        corr_ord,
        cmap="viridis",
        vmin=-1, vmax=1, center=0,
        square=False,  # IMPORTANT: give labels more room
        cbar_kws={"shrink": 0.9},
        ax=ax
    )

    ax.set_title(f"Reordered correlation heatmap ({method})", pad=14)
    # force square plot area (prevents the “tall rectangle” look)
    ax.set_aspect("equal", adjustable="box")

    # reserve room for long tick labels WITHOUT stretching the saved PNG
    fig.subplots_adjust(left=0.28, bottom=0.28, right=0.95, top=0.92)

    # Show fewer tick labels if crowded
    if show_every > 1:
        xt = ax.get_xticks()
        yt = ax.get_yticks()
        ax.set_xticks(xt[::show_every])
        ax.set_yticks(yt[::show_every])

        ax.set_xticklabels([t.get_text() for t in ax.get_xticklabels()][::show_every], rotation=90, fontsize=9)
        ax.set_yticklabels([t.get_text() for t in ax.get_yticklabels()][::show_every], rotation=0, fontsize=9)
    else:
        ax.tick_params(axis="x", labelrotation=90, labelsize=9)
        ax.tick_params(axis="y", labelrotation=0, labelsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)










#----------++






# ---------------------------
# Helpers: hashing / Merkle
# ---------------------------
def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def canonical_row_dict(row: pd.Series, feature_cols: List[str], round_decimals: int) -> Dict:
    """Create a stable dict representation for hashing (rounded floats)."""
    d: Dict = {}
    for k in feature_cols:
        v = row[k]
        if isinstance(v, (float, np.floating)):
            v = round(float(v), round_decimals)
        d[k] = v
    d["Label"] = row["Label"]
    return d

def hash_record(row: pd.Series, feature_cols: List[str], round_decimals: int) -> str:
    payload = canonical_row_dict(row, feature_cols, round_decimals)
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256_hex(s)

def merkle_root_hex(hash_list_hex: List[str]) -> str:
    """Compute Merkle root from list of hex hashes."""
    nodes = [bytes.fromhex(h) for h in hash_list_hex]
    if not nodes:
        return sha256_hex("")  # empty block
    while len(nodes) > 1:
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])
        new_level = []
        for i in range(0, len(nodes), 2):
            new_level.append(hashlib.sha256(nodes[i] + nodes[i + 1]).digest())
        nodes = new_level
    return nodes[0].hex()

def verify_ledger(
    ledger_df: pd.DataFrame,
    feature_cols: List[str],
    round_decimals: int
) -> pd.DataFrame:
    """
    Verifies:
      - hash_ok: recomputed hash matches record_hash
      - chain_ok: prev_hash links to previous record_hash
      - merkle_ok: recomputed Merkle root matches stored merkle_root (per block)
    Returns a copy with columns: recalc_hash, hash_ok, chain_ok, merkle_ok
    """
    df = ledger_df.copy()

    df["recalc_hash"] = df.apply(lambda r: hash_record(r, feature_cols, round_decimals), axis=1)
    df["hash_ok"] = (df["recalc_hash"] == df["record_hash"])

    # Verify chain using stored record_hash values
    df["chain_ok"] = True
    rh = df["record_hash"].tolist()
    ph = df["prev_hash"].tolist()
    for i in range(1, len(df)):
        if ph[i] != rh[i - 1]:
            df.iloc[i, df.columns.get_loc("chain_ok")] = False

    # Verify Merkle roots per block (use recalc_hash for content-based verification)
    df["merkle_ok"] = True
    for bid, g in df.groupby("block_id"):
        root_now = merkle_root_hex(g["recalc_hash"].tolist())
        expected = g["merkle_root"].iloc[0]
        if root_now != expected:
            df.loc[g.index, "merkle_ok"] = False

    return df

def plot_integrity_summary(df_verified: pd.DataFrame, out_png: Path, title: str) -> None:
    """Bar plot of ok/fail counts for hash_ok/chain_ok/merkle_ok."""
    cols = ["hash_ok", "chain_ok", "merkle_ok"]
    ok_counts = [int(df_verified[c].sum()) for c in cols]
    fail_counts = [int((~df_verified[c]).sum()) for c in cols]

    x = np.arange(len(cols))
    w = 0.4
    plt.figure(figsize=(8, 4.5))
    plt.bar(x - w/2, ok_counts, width=w, label="OK")
    plt.bar(x + w/2, fail_counts, width=w, label="FAIL")
    plt.xticks(x, cols)
    plt.title(title)
    plt.ylabel("Rows")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_chain_breaks(df_verified: pd.DataFrame, out_png: Path) -> None:
    """Line plot of chain_ok over row index (helps show where chain breaks)."""
    y = df_verified["chain_ok"].astype(int).to_numpy()
    plt.figure(figsize=(10, 3.5))
    plt.plot(np.arange(len(y)), y)
    plt.ylim(-0.05, 1.05)
    plt.title("Hash-chain integrity over index (chain_ok)")
    plt.xlabel("Row index in ledger")
    plt.ylabel("chain_ok (1=ok, 0=fail)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_merkle_by_block(df_verified: pd.DataFrame, out_png: Path) -> None:
    """Bar plot: fraction of rows passing merkle_ok per block."""
    frac = df_verified.groupby("block_id")["merkle_ok"].mean().sort_index()
    plt.figure(figsize=(10, 4))
    plt.bar(frac.index.astype(int).astype(str), frac.values)
    plt.title("Merkle integrity by block (fraction merkle_ok)")
    plt.xlabel("block_id")
    plt.ylabel("fraction merkle_ok")
    plt.xticks(rotation=90, fontsize=7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()














# ---------------------------
# Main pipeline
# ---------------------------
def main():
   # parser = argparse.ArgumentParser()
   # parser.add_argument("--csv", type=str, required=True, help="Path to CIC/ISCX DDoS flow CSV")
   # parser.add_argument("--outdir", type=str, default="output", help="Output folder")
   # parser.add_argument("--sample_rows", type=int, default=50000, help="Rows for ML baseline (stratified sampling)")
   # parser.add_argument("--ledger_rows", type=int, default=50000, help="Rows to build ledger demo")
   # parser.add_argument("--block_size", type=int, default=1000, help="Records per block for Merkle roots")
   # parser.add_argument("--round_decimals", type=int, default=6, help="Float rounding for stable hashing")
   # parser.add_argument("--tamper_n", type=int, default=2000, help="How many rows to tamper for the demo")
   # parser.add_argument("--seed", type=int, default=42, help="Random seed")
   # parser.add_argument("--skip_ledger", action="store_true", help="Skip blockchain ledger section")
   # args = parser.parse_args()










    # ---------------------------
    # CLI (zero-arg friendly)
    # ---------------------------
    ROOT = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to CIC/ISCX DDoS flow CSV. If omitted, auto-detects a CSV inside the downloaded repo folder (prefers ./data/)."
    )
    parser.add_argument("--outdir", type=str, default=str(ROOT / "output"), help="Output folder")
    parser.add_argument("--sample_rows", type=int, default=50000, help="Rows for ML baseline (stratified sampling)")
    parser.add_argument("--ledger_rows", type=int, default=50000, help="Rows to build ledger demo")
    parser.add_argument("--block_size", type=int, default=1000, help="Records per block for Merkle roots")
    parser.add_argument("--round_decimals", type=int, default=6, help="Float rounding for stable hashing")
    parser.add_argument("--tamper_n", type=int, default=2000, help="How many rows to tamper for the demo")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip_ledger", action="store_true", help="Skip blockchain ledger section")

    args = parser.parse_args()

    # Auto-select CSV if not provided (search common locations, recursively)
    if args.csv is None:
        data_dir = ROOT / "data"
        junk = {".venv", "venv", "__pycache__", ".git", "output", "outputs"}

        search_bases = [data_dir, ROOT]
        found = []

        for base in search_bases:
            if not base.exists():
                continue
            for f in list(base.rglob("*.csv")) + list(base.rglob("*.CSV")):
                if any(part in junk for part in f.parts):
                    continue
                found.append(f)

        # Prefer CSVs inside ./data/; pick the largest file (usually the real dataset)
        found = list({p.resolve() for p in found})
        in_data = [p for p in found if data_dir in p.parents]
        pool = in_data or found

        # Keep only CIC/ISCX-style flow CSVs (must contain a 'Label' column, case/space-insensitive)
        def _has_label_col(p: Path) -> bool:
            try:
                cols = pd.read_csv(p, nrows=0).columns
                norm = {str(c).strip().lower() for c in cols}
                return "label" in norm
            except Exception:
                return False

        label_pool = [p for p in pool if _has_label_col(p)]
        if not label_pool:
            parser.error(
                "Auto-detect found CSV file(s), but none looks like a CIC/ISCX flow CSV because no 'Label' column was found. "
                "Put the CIC-IDS2017/ISCX flow CSV (with a Label column) inside ./data/ (recommended) or run with --csv /path/to/file.csv."
            )

        pool = label_pool

        if not pool:
            parser.error(
                "No --csv provided and no .csv found inside the downloaded repo folder. "
                "Put a CSV anywhere in the repo (recommended: ./data/) or pass --csv."
            )

        picked = max(pool, key=lambda p: p.stat().st_size)
        args.csv = str(picked)
        print(f"[INFO] Auto-selected CSV: {args.csv}")

    # --- Robust path resolution (works in PyCharm even if Working directory is not the script folder) ---
    script_dir = Path(__file__).resolve().parent

    # Resolve CSV path: try as-given, then relative to the script folder
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        csv_path = (script_dir / args.csv).resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"CSV not found. Tried: '{args.csv}' and '{csv_path}'.\n"
            f"Tip: In PyCharm Run/Debug config, set Working directory to: {script_dir}"
        )
    args.csv = str(csv_path)

    # Resolve output directory: user requested NO subfolder; save outputs next to the script
    # (This keeps paths stable in PyCharm regardless of Working directory.)
    args.outdir = str(script_dir)

    out_dir = Path(args.outdir)
    safe_mkdir(out_dir)

    # 1) Load + clean
    df_raw = pd.read_csv(args.csv, low_memory=False)

    raw_shape = df_raw.shape
    raw_dup = int(df_raw.duplicated().sum())
    raw_num = df_raw.select_dtypes(include=[np.number])
    raw_inf = int(np.isinf(raw_num.to_numpy()).sum()) if raw_num.shape[1] else 0

    df = clean_flow_dataframe(df_raw, label_col="Label")

    diag_path = out_dir / "diagnostics.txt"
    with open(diag_path, "w", encoding="utf-8") as f:
        f.write(f"Loaded: {args.csv}\n")
        f.write(f"Raw shape: {raw_shape}\n")
        f.write(f"Raw duplicates: {raw_dup}\n")
        f.write(f"Raw inf values (numeric cols): {raw_inf}\n")
        f.write(f"Clean shape: {df.shape}\n\n")
        f.write("Label counts:\n")
        f.write(df["Label"].value_counts().to_string())
        f.write("\n")
    print(f"Saved: {diag_path}")

    # 2) BEFORE blockchain: EDA plots (distribution shift + group stats)
    desired_features = [
        "Flow Packets/s",
        "Flow Bytes/s",
        "Total Fwd Packets",
        "Flow Duration",
        "Fwd Packet Length Mean",
    ]
    mapping = resolve_columns(df, desired_features)
    resolved = [mapping[k] for k in desired_features if mapping[k] is not None]
    missing = [k for k in desired_features if mapping[k] is None]
    if missing:
        print("WARNING: Could not find these columns (name mismatch in my file):")
        for m in missing:
            print(f"  - {m}")

    # Make a modest EDA sample for speed (still stratified)
    if args.sample_rows < len(df):
        df_eda = (
            df.groupby("Label", group_keys=False)
            .apply(lambda g: g.sample(
                n=max(1, int(round(args.sample_rows * len(g) / len(df)))),
                random_state=args.seed,
            ))
        )
        if len(df_eda) > args.sample_rows:
            df_eda = df_eda.sample(n=args.sample_rows, random_state=args.seed)
    else:
        df_eda = df.copy()

    # Label distribution plot
    #plot_label_distribution(df_eda, "Label", out_dir / "label_distribution.png")
    #print(f"Saved: {out_dir / 'label_distribution.png'}")

    # Distribution shift plots (BENIGN vs DDoS)
    for canonical in desired_features:
        col = mapping.get(canonical)
        if col is None:
            continue
        out_png = out_dir / f"dist_{sanitize_colname(canonical)[:40]}.png"
        plot_hist_compare(df_eda, "Label", col, out_png, log1p=True)
        print(f"Saved: {out_png}")

    # Group stats table (mean/median by label for key features)
    if resolved:
        stats_df = df_eda.groupby("Label")[resolved].agg(["median", "mean"])
        stats_df.columns = [f"{feat}__{stat}" for feat, stat in stats_df.columns]
        stats_out = out_dir / "group_stats_by_label.csv"
        stats_df.to_csv(stats_out)
        print(f"Saved: {stats_out}")
    else:
        print("NOTE: No key features were resolved; skipping group_stats_by_label.csv.")

    # 3) Baseline IDS model (RandomForest) + outputs
    X = df_eda.drop(columns=["Label"])
    y = (df_eda["Label"] == "DDoS").astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )


    # --- Cross-validation on TRAINING set only (test set remains untouched) ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    rf_cv = RandomForestClassifier(
        n_estimators=200,
        max_depth=18,
        random_state=args.seed,
        n_jobs=1,  # avoid nested parallelism; CV already parallelizes folds
        class_weight="balanced_subsample",
    )

    cv_auc = cross_val_score(rf_cv, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_acc = cross_val_score(rf_cv, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

    print(f"CV ROC-AUC (train only): {cv_auc.mean():.6f} ± {cv_auc.std():.6f}")
    print(f"CV Accuracy (train only): {cv_acc.mean():.6f} ± {cv_acc.std():.6f}")

    #-----



    #-----



    cv_txt = Path(args.outdir) / "cv_results_train_only.txt"
    with open(cv_txt, "w", encoding="utf-8") as f:
        f.write("Cross-validation on TRAIN set only (5-fold StratifiedKFold)\n")
        f.write("Model: RandomForestClassifier(n_estimators=200, max_depth=18)\n\n")

        f.write("ROC-AUC per fold:\n")
        for i, x in enumerate(cv_auc, start=1):
            f.write(f"  Fold {i}: {x:.6f}\n")
        f.write(f"ROC-AUC mean ± std: {cv_auc.mean():.6f} ± {cv_auc.std():.6f}\n\n")

        f.write("Accuracy per fold:\n")
        for i, x in enumerate(cv_acc, start=1):
            f.write(f"  Fold {i}: {x:.6f}\n")
        f.write(f"Accuracy mean ± std: {cv_acc.mean():.6f} ± {cv_acc.std():.6f}\n")

    print(f"Saved: {cv_txt}")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=18,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf.fit(X_train, y_train)

    from sklearn.tree import plot_tree

    one_tree_png = out_dir / "example_decision_tree_depth3.png"







    plt.figure(figsize=(32, 20))
    plot_tree(
        rf.estimators_[0],  # pick one tree
        feature_names=X.columns,
        class_names=["BENIGN", "DDoS"],
        filled=True,
        max_depth=3,  # readable depth
        impurity=False
    )
    plt.tight_layout()
    plt.savefig(one_tree_png, dpi=200)
    plt.close()

    print(f"Saved: {one_tree_png}")

    proba = rf.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    auc_val = float(roc_auc_score(y_test, proba))
    print(f"Baseline IDS AUC: {auc_val:.6f}")

    # Save ROC + AUC
    roc_png = out_dir / "baseline_roc_curve.png"
    plot_roc(y_test, proba, roc_png)
    print(f"Saved: {roc_png}")

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_png = out_dir / "baseline_confusion_matrix.png"
    plot_confusion_matrix(cm, labels=["BENIGN", "DDoS"], out_png=cm_png)
    print(f"Saved: {cm_png}")

    # Feature importances
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    imp_csv = out_dir / "baseline_feature_importances.csv"
    imp.to_csv(imp_csv, header=["importance"])
    print(f"Saved: {imp_csv}")

    imp_png = out_dir / "baseline_top_feature_importances.png"
    plot_feature_importances(imp, imp_png, top_k=12)
    print(f"Saved: {imp_png}")


















    # Correlation / redundancy (heatmap on top-N important features + top correlated pairs)
    topN = min(30, len(imp))
    if topN >= 2:
        top_cols = imp.head(topN).index.tolist()
        X_top = df_eda[top_cols].select_dtypes(include=[np.number]).copy()

        # Correlation / redundancy (top-N important features + top correlated pairs)
        topN = min(30, len(imp))
        if topN >= 2:
            top_cols = imp.head(topN).index.tolist()
            X_top = df_eda[top_cols].select_dtypes(include=[np.number]).copy()

            # (optional) keep my old plain heatmap
            corr_png = out_dir / "correlation_heatmap_top_features.png"
            plot_corr_heatmap(X_top, corr_png, method="spearman")
            print(f"Saved: {corr_png}")

            # NEW: two separate outputs you asked for
            dendro_png = out_dir / "correlation_dendrogram_top_features.png"
            save_corr_dendrogram(X_top, dendro_png, method="spearman", cluster_on_abs=True)
            print(f"Saved: {dendro_png}")

            heat_ord_png = out_dir / "correlation_heatmap_reordered_top_features.png"
            save_reordered_corr_heatmap(X_top, heat_ord_png, method="spearman", cluster_on_abs=True)
            print(f"Saved: {heat_ord_png}")

            # CSV of top correlated pairs
            top_corr_csv = out_dir / "top_correlated_pairs.csv"
            save_top_correlated_pairs(
                df_eda.drop(columns=["Label"]).select_dtypes(include=[np.number]),
                top_corr_csv,
                method="spearman",
                top_k=25
            )
            print(f"Saved: {top_corr_csv}")
        else:
            print("NOTE: Not enough features for correlation outputs; skipping correlation plots.")




    else:
        print("NOTE: Not enough features for correlation outputs; skipping correlation plots.")

    # Metrics text
    metrics_txt = out_dir / "baseline_metrics.txt"
    report = classification_report(y_test, y_pred, target_names=["BENIGN", "DDoS"], digits=6  )
    with open(metrics_txt, "w", encoding="utf-8") as f:
        f.write(f"Baseline model: RandomForestClassifier\n")
        f.write(f"ROC-AUC: {auc_val:.6f}\n\n")
        f.write("Train-only 5-fold CV (StratifiedKFold):\n")
        f.write(f"  CV ROC-AUC mean ± std: {cv_auc.mean():.6f} ± {cv_auc.std():.6f}\n")
        f.write(f"  CV Accuracy mean ± std: {cv_acc.mean():.6f} ± {cv_acc.std():.6f}\n\n")
        f.write("Classification report (threshold=0.5):\n")
        f.write(report)
        f.write("\n\nConfusion matrix [[TN FP],[FN TP]]:\n")
        f.write(np.array2string(cm))
        f.write("\n")
    print(f"Saved: {metrics_txt}")

    # 4) BLOCKCHAIN layer (optional)
    if args.skip_ledger:
        print("Skipping blockchain ledger section (--skip_ledger). Done.")
        return

    ledger_df = df.sample(n=min(args.ledger_rows, len(df)), random_state=args.seed).copy()
    feature_cols = [c for c in ledger_df.columns if c != "Label"]

    # Build initial ledger hashes
    ledger_df["record_hash"] = ledger_df.apply(lambda r: hash_record(r, feature_cols, args.round_decimals), axis=1)

    # Hash chain
    prev = "0" * 64
    prev_hashes = []
    for h in ledger_df["record_hash"].tolist():
        prev_hashes.append(prev)
        prev = h
    ledger_df["prev_hash"] = prev_hashes

    # Block + Merkle roots
    ledger_df = ledger_df.reset_index(drop=True)
    ledger_df["block_id"] = (np.arange(len(ledger_df)) // args.block_size).astype(int)
    roots = {bid: merkle_root_hex(g["record_hash"].tolist()) for bid, g in ledger_df.groupby("block_id")}
    ledger_df["merkle_root"] = ledger_df["block_id"].map(roots)

    print("Ledger columns added: record_hash, prev_hash, block_id, merkle_root")

    # =========================================================
    # DECENTRALIZATION DEMO: simulate multiple nodes + consensus
    # =========================================================

    def rebuild_chain_and_roots(df_node: pd.DataFrame) -> pd.DataFrame:
        """Recompute record_hash, prev_hash, block_id, merkle_root for a node (like a node "re-mining")."""
        df_node = df_node.reset_index(drop=True).copy()

        # recompute record_hash based on content
        df_node["record_hash"] = df_node.apply(lambda r: hash_record(r, feature_cols, args.round_decimals), axis=1)

        # rebuild hash chain
        prev = "0" * 64
        prev_hashes = []
        for h in df_node["record_hash"].tolist():
            prev_hashes.append(prev)
            prev = h
        df_node["prev_hash"] = prev_hashes

        # rebuild blocks + merkle roots
        df_node["block_id"] = (np.arange(len(df_node)) // args.block_size).astype(int)
        roots_local = {bid: merkle_root_hex(g["record_hash"].tolist()) for bid, g in df_node.groupby("block_id")}
        df_node["merkle_root"] = df_node["block_id"].map(roots_local)

        return df_node

    def consensus_check(nodes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Compare merkle_root per block across nodes (majority rule consensus)."""
        all_blocks = sorted(set().union(*[set(nd["block_id"].unique()) for nd in nodes.values()]))

        rows = []
        for bid in all_blocks:
            roots = {}
            for node_name, nd in nodes.items():
                root_val = nd.loc[nd["block_id"] == bid, "merkle_root"].iloc[0]
                roots[node_name] = root_val

            # majority vote root
            roots_series = pd.Series(roots)
            counts = roots_series.value_counts()
            majority_root = counts.index[0]
            agree = int(counts.iloc[0])
            total = int(len(nodes))
            agree_frac = agree / total

            rows.append({
                "block_id": bid,
                "majority_root": majority_root,
                "agree_nodes": agree,
                "total_nodes": total,
                "agree_fraction": agree_frac,
                **{f"root_{k}": v for k, v in roots.items()},
            })

        return pd.DataFrame(rows)

    # --- 1) Create node copies (replicated ledger) ---
    num_nodes = 3  # change to 5 or 7 if you want stronger demo
    nodes = {f"node_{i}": ledger_df.copy() for i in range(num_nodes)}

    # --- 2) Tamper ONLY one node (attacker changes data on one server) ---
    attacked_node = "node_0"
    tamper_n = min(args.tamper_n, len(nodes[attacked_node]))
    tamper_idx = np.random.RandomState(args.seed + 999).choice(nodes[attacked_node].index, size=tamper_n, replace=False)

    # Change Label + small feature modification (same as my attack idea)
    nodes[attacked_node].loc[tamper_idx, "Label"] = "BENIGN"
    if feature_cols:
        nodes[attacked_node].loc[tamper_idx, feature_cols[0]] = nodes[attacked_node].loc[
                                                                    tamper_idx, feature_cols[0]] + 1

    # --- 3) Attacker "covers tracks" by recomputing chain + merkle roots on that node ---
    # This is the KEY: locally the node looks valid, but it diverges from the other nodes.
    nodes[attacked_node] = rebuild_chain_and_roots(nodes[attacked_node])

    # --- 4) Consensus: compare block roots across nodes ---
    cons_df = consensus_check(nodes)
    cons_out = out_dir / "consensus_by_block.csv"
    cons_df.to_csv(cons_out, index=False)
    print(f"Saved: {cons_out}")

    # --- 5) Plot: agreement fraction per block (great slide figure) ---
    plt.figure(figsize=(10, 4))
    plt.plot(cons_df["block_id"], cons_df["agree_fraction"])
    plt.ylim(0, 1.05)
    plt.title("Decentralized consensus: agreement fraction per block")
    plt.xlabel("block_id")
    plt.ylabel("agree_fraction (1.0 = all nodes agree)")
    plt.tight_layout()
    plt.savefig(out_dir / "consensus_agreement_fraction.png", dpi=200)
    plt.close()
    print(f"Saved: {out_dir / 'consensus_agreement_fraction.png'}")

    # Verify clean ledger (should be all OK)
    verified_clean = verify_ledger(ledger_df[["Label"] + feature_cols + ["record_hash", "prev_hash", "block_id", "merkle_root"]], feature_cols, args.round_decimals)
    plot_integrity_summary(verified_clean, out_dir / "integrity_summary_clean.png", "Integrity checks (clean ledger)")
    print(f"Saved: {out_dir / 'integrity_summary_clean.png'}")

    # Tampering simulation:
    # - Modify some rows (Label + a feature)
    # - Recompute ONLY the tampered row hashes (attacker tries to cover tracks)
    # - Do NOT update prev_hash / merkle_root (so chain and Merkle should fail)
    tampered = ledger_df.copy()

    # ADD THIS LINE HERE (before changing labels)
    tampered["original_label"] = tampered["Label"].copy()

    #tamper_idx = tampered.sample(n=min(args.tamper_n, len(tampered)), random_state=args.seed + 7).index
    tamper_idx = np.arange(min(args.tamper_n, len(tampered)))

    # Flip labels (hide DDoS) + tiny feature change
    tampered.loc[tamper_idx, "Label"] = "BENIGN"

    # Count rows that were originally DDoS but were changed to BENIGN
    hidden = tampered[(tampered["original_label"] == "DDoS") & (tampered["Label"] == "BENIGN")]
    print("Hidden DDoS rows:", len(hidden))

    hidden_out = out_dir / "hidden_ddos_rows.csv"
    hidden.to_csv(hidden_out, index=False)
    print(f"Saved: {hidden_out}")

    if feature_cols:
        tampered.loc[tamper_idx, feature_cols[0]] = tampered.loc[tamper_idx, feature_cols[0]] + 1








#---

    # Tampering simulation (attacker modifies logs)
    tampered = ledger_df.copy()
    tampered["original_label"] = tampered["Label"].copy()

    tamper_idx = np.arange(min(args.tamper_n, len(tampered)))

    # Flip labels (hide DDoS) + tiny feature change
    tampered.loc[tamper_idx, "Label"] = "BENIGN"
    if feature_cols:
        tampered.loc[tamper_idx, feature_cols[0]] = tampered.loc[tamper_idx, feature_cols[0]] + 1

    #---








    # Attacker updates ONLY those record_hash values to match the new content (but doesn't fix the chain)
    #tampered.loc[tamper_idx, "record_hash"] = tampered.loc[tamper_idx].apply(
     #   lambda r: hash_record(r, feature_cols, args.round_decimals), axis=1
    #)

    #verified_tampered = verify_ledger(tampered[["Label"] + feature_cols + ["record_hash", "prev_hash", "block_id", "merkle_root"]], feature_cols, args.round_decimals)

    verified_tampered = verify_ledger(
        tampered[["original_label", "Label"] + feature_cols + ["record_hash", "prev_hash", "block_id", "merkle_root"]],
        feature_cols,
        args.round_decimals
    )

    # =========================================================
    # Compare detection: clean vs tampered vs tampered+filtered
    # =========================================================


    def eval_detection(df_eval, name, out_dir, seed=42):
        # Use ONLY numeric flow features (drop blockchain string columns)
        X = df_eval.drop(columns=["Label"])
        X = X.select_dtypes(include=[np.number])
        y = (df_eval["Label"] == "DDoS").astype(int).values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=200, max_depth=18,
            random_state=seed, n_jobs=-1,
            class_weight="balanced_subsample",
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, proba)
        cm = confusion_matrix(y_test, pred)
        rep = classification_report(y_test, pred, target_names=["BENIGN", "DDoS"], digits=4)

        out_txt = Path(out_dir) / f"detection_{name}.txt"
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(f"Detection evaluation: {name}\n")
            f.write(f"AUC: {auc:.6f}\n")
            f.write("Confusion matrix [[TN FP],[FN TP]]:\n")
            f.write(np.array2string(cm) + "\n\n")
            f.write(rep + "\n")
        print(f"Saved: {out_txt}")
        return auc, cm

    # 1) Clean (what you already had, but evaluated on the ledger sample)
    clean_eval = ledger_df.copy()
    clean_eval = clean_eval.drop(columns=["record_hash", "prev_hash", "block_id", "merkle_root"], errors="ignore")
    eval_detection(clean_eval, "clean", args.outdir, seed=args.seed)

    # 2) Tampered, WITHOUT integrity filtering (this is the missing part)
    tampered_eval = tampered.copy()
    tampered_eval = tampered_eval.drop(columns=["record_hash", "prev_hash", "block_id", "merkle_root"], errors="ignore")
    eval_detection(tampered_eval, "tampered_no_checks", args.outdir, seed=args.seed)

    # 3) Tampered, WITH integrity filtering (keep only verified rows)
    tampered_with_flags = tampered.copy()
    tampered_with_flags["hash_ok"] = verified_tampered["hash_ok"].values
    tampered_with_flags["chain_ok"] = verified_tampered["chain_ok"].values
    tampered_with_flags["merkle_ok"] = verified_tampered["merkle_ok"].values

    filtered = tampered_with_flags.loc[
        (tampered_with_flags["hash_ok"]) &
        (tampered_with_flags["chain_ok"]) &
        (tampered_with_flags["merkle_ok"])
        ].copy()

    filtered = filtered.drop(columns=[
        "record_hash", "prev_hash", "block_id", "merkle_root",
        "hash_ok", "chain_ok", "merkle_ok"
    ], errors="ignore")

    if len(filtered) >= 50 and filtered["Label"].nunique() == 2:
        eval_detection(filtered, "tampered_filtered_by_integrity", args.outdir, seed=args.seed)
    else:
        print("NOTE: Not enough verified rows (or only one class) after filtering; skipping filtered detection eval.")








    plot_integrity_summary(verified_tampered, out_dir / "integrity_summary_tampered.png", "Integrity checks (tampered ledger)")
    #plot_chain_breaks(verified_tampered, out_dir / "chain_ok_over_index.png")
    #plot_merkle_by_block(verified_tampered, out_dir / "merkle_ok_by_block.png")

    print(f"Saved: {out_dir / 'integrity_summary_tampered.png'}")
    print(f"Saved: {out_dir / 'chain_ok_over_index.png'}")
    print(f"Saved: {out_dir / 'merkle_ok_by_block.png'}")

    # Save compact results CSV for my repo (keeps size small)
    #out_csv = out_dir / "ddos_ledger_with_integrity_checks.csv"
    #compact = verified_tampered[["Label", "record_hash", "prev_hash", "block_id", "merkle_root", "hash_ok", "chain_ok", "merkle_ok"]].copy()
    #compact.to_csv(out_csv, index=False)
    #print(f"Saved: {out_csv}")

    #out_csv = out_dir / "ddos_ledger_with_integrity_checks.csv"

    #compact = verified_tampered[
    #    ["original_label", "Label", "block_id", "hash_ok", "chain_ok", "merkle_ok",
    #     "record_hash", "prev_hash", "merkle_root"]
    #].copy()

    #compact.to_csv(out_csv, index=False)
    #print(f"Saved: {out_csv}")

    # Save compact results CSV (includes recalc_hash so you can see SHA256 differences)
    out_csv = out_dir / "ddos_ledger_with_integrity_checks.csv"

    compact = verified_tampered[
        ["original_label", "Label", "block_id",
         "record_hash", "recalc_hash",
         "prev_hash", "merkle_root",
         "hash_ok", "chain_ok", "merkle_ok"]
    ].copy()

    compact.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")



    #--- Merkle tree output
    # =========================================================
    # EXTRA OUTPUT FOR SLIDES: 10 rows with DIFFERENT merkle_root
    # (1 representative row per unique Merkle root / per block)
    # =========================================================
    show_cols = ["block_id", "merkle_root", "hash_ok", "chain_ok", "merkle_ok", "record_hash", "recalc_hash"]
    show_cols = [c for c in show_cols if c in compact.columns]

    sample_10 = (
        compact.drop_duplicates(subset=["merkle_root"])
        .sort_values("block_id")
        .head(10)
    )

    sample_out = out_dir / "sample_10_unique_merkle_roots.csv"
    sample_10[show_cols].to_csv(sample_out, index=False)

    print("\n=== 10 UNIQUE MERKLE ROOTS (1 row per block) ===")
    print(sample_10[show_cols].to_string(index=False))
    print(f"\nSaved: {sample_out}")


if __name__ == "__main__":
    main()
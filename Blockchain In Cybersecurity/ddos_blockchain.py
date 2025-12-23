"""
DDoS + Blockchain Integrity Quantlet (CIC-IDS2017)
-------------------------------------------------
This script does two things:
1) Cybersecurity (IDS): trains a simple baseline model to separate BENIGN vs DDoS
   and outputs AUC, ROC curve, label distribution, and feature importances.

2) Blockchain (Integrity): "blockchain-enables" the IDS log by adding:
   - record_hash (SHA-256 over feature values + label)
   - prev_hash (hash chain)
   - block_id (records grouped into blocks)
   - merkle_root (Merkle root per block)

Then it simulates tampering (flipping labels / changing features) and shows how
hash, chain, and Merkle verification detect manipulation.

How to run (example):
python ddos_blockchain_quantlet.py --csv "/path/to/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

Notes:
- The CIC-IDS2017 CSVs sometimes have leading spaces in column names; this script strips them.
- If your machine is slow, lower --ledger_rows or --sample_rows.
"""

import argparse
import hashlib
import json
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, RocCurveDisplay


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


# ---------------------------
# Main pipeline
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to CIC-IDS2017 DDoS CSV")
    parser.add_argument("--sample_rows", type=int, default=50000, help="Rows for ML baseline (sampling)")
    parser.add_argument("--ledger_rows", type=int, default=50000, help="Rows to build ledger demo")
    parser.add_argument("--block_size", type=int, default=1000, help="Records per block for Merkle roots")
    parser.add_argument("--round_decimals", type=int, default=6, help="Float rounding for stable hashing")
    parser.add_argument("--tamper_n", type=int, default=10, help="How many rows to tamper for the demo")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # 1) Load + clean
    df = pd.read_csv(args.csv, low_memory=False)
    df.columns = [c.strip() for c in df.columns]  # remove leading/trailing spaces
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    if "Label" not in df.columns:
        raise ValueError("No 'Label' column found. Check that you opened a CIC-IDS2017 flow CSV.")

    print("Loaded:", args.csv)
    print("Shape:", df.shape)
    print("\nLabel counts:\n", df["Label"].value_counts())

    # 2) Baseline IDS model (BENIGN vs DDoS)
    df_sample = df.sample(n=min(args.sample_rows, len(df)), random_state=args.seed)

    X = df_sample.drop(columns=["Label"])
    y = (df_sample["Label"] == "DDoS").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=120, max_depth=18, random_state=args.seed, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    proba = rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print("\nBaseline IDS AUC:", auc)

    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_imp = imp.head(12)
    print("\nTop features:\n", top_imp)

    # Plots (close the windows after saving screenshots for your slides)
    df["Label"].value_counts().plot(kind="bar")
    plt.title("Label distribution (BENIGN vs DDoS)")
    plt.tight_layout()
    plt.show()

    RocCurveDisplay.from_predictions(y_test, proba)
    plt.title(f"ROC (RandomForest)  AUC={auc:.4f}")
    plt.tight_layout()
    plt.show()

    top_imp.sort_values().plot(kind="barh")
    plt.title("Top feature importances (sampled)")
    plt.tight_layout()
    plt.show()

    # 3) Blockchain layer: record_hash + prev_hash + Merkle roots
    ledger_df = df.head(min(args.ledger_rows, len(df))).copy()
    feature_cols = [c for c in ledger_df.columns if c != "Label"]

    ledger_df["record_hash"] = ledger_df.apply(
        lambda r: hash_record(r, feature_cols, args.round_decimals), axis=1
    )

    # Hash chain
    prev = "0" * 64
    prev_hashes = []
    for h in ledger_df["record_hash"].tolist():
        prev_hashes.append(prev)
        prev = h
    ledger_df["prev_hash"] = prev_hashes

    # Block + Merkle roots
    ledger_df["block_id"] = (np.arange(len(ledger_df)) // args.block_size).astype(int)
    roots = {}
    for bid, g in ledger_df.groupby("block_id"):
        roots[bid] = merkle_root_hex(g["record_hash"].tolist())
    ledger_df["merkle_root"] = ledger_df["block_id"].map(roots)

    print("\nLedger columns added: record_hash, prev_hash, block_id, merkle_root")

    # 4) Tampering simulation + verification
    tampered = ledger_df.copy()
    tamper_idx = tampered.sample(n=min(args.tamper_n, len(tampered)), random_state=args.seed + 7).index

    # Flip labels (hide DDoS) + tiny feature change
    tampered.loc[tamper_idx, "Label"] = "BENIGN"
    if feature_cols:
        tampered.loc[tamper_idx, feature_cols[0]] = tampered.loc[tamper_idx, feature_cols[0]] + 1

    # Recompute hashes for verification
    tampered["recalc_hash"] = tampered.apply(
        lambda r: hash_record(r, feature_cols, args.round_decimals), axis=1
    )
    tampered["hash_ok"] = (tampered["recalc_hash"] == tampered["record_hash"])

    # Verify chain: each prev_hash must equal previous record_hash
    tampered["chain_ok"] = True
    rh = tampered["record_hash"].tolist()
    ph = tampered["prev_hash"].tolist()
    for i in range(1, len(tampered)):
        if ph[i] != rh[i - 1]:
            tampered.at[tampered.index[i], "chain_ok"] = False

    # Verify Merkle roots per block
    tampered["merkle_ok"] = True
    for bid, g in tampered.groupby("block_id"):
        root_now = merkle_root_hex(g["recalc_hash"].tolist())
        expected = g["merkle_root"].iloc[0]
        if root_now != expected:
            tampered.loc[g.index, "merkle_ok"] = False

    summary = {
        "tampered_rows": int(len(tamper_idx)),
        "hash_failed": int((~tampered["hash_ok"]).sum()),
        "chain_failed": int((~tampered["chain_ok"]).sum()),
        "merkle_failed": int((~tampered["merkle_ok"]).sum()),
        "ledger_rows_used": int(len(tampered)),
        "block_size": int(args.block_size),
    }

    print("\nIntegrity summary:", summary)

    # Save a compact results CSV for your repo output/ folder (optional)
    out_path = "ddos_ledger_with_integrity_checks.csv"
    tampered_out = tampered[["Label", "record_hash", "prev_hash", "block_id", "merkle_root", "hash_ok", "chain_ok", "merkle_ok"]].copy()
    tampered_out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

"""
ddos_decentralized_nodes.py
---------------------------------
Decentralized-node simulation add-on for a DDoS / IDS blockchain integrity project.

What it demonstrates:
- N nodes (servers) store identical blockchain-style ledgers (hash-chain + Merkle root).
- One node is tampered (a record is edited) WITHOUT updating stored hashes.
- Verification flags the tampered node.
- Majority-consensus (per-block Merkle roots) identifies deviating nodes.

How to use:
1) Put this file next to my existing project scripts.
2) Run it directly:
   python ddos_decentralized_nodes.py --csv "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
3) Or import functions into my existing file:
   from ddos_decentralized_nodes import simulate_decentralized_nodes, majority_consensus
"""

import argparse
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ----------------------------
# Utils: hashing + merkle tree
# ----------------------------
def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def row_fingerprint(row: pd.Series, cols: List[str]) -> str:
    """
    Stable deterministic representation of a row across selected columns.
    Important: keep cols list consistent across nodes.
    """
    parts = []
    for c in cols:
        v = row[c]
        if pd.isna(v):
            parts.append(f"{c}=<NA>")
        else:
            parts.append(f"{c}={str(v)}")
    return "|".join(parts)


def merkle_root(hashes: List[str]) -> str:
    """
    Standard Merkle root:
    - pairwise hash concatenation
    - duplicate last element if odd length
    """
    if len(hashes) == 0:
        return sha256("EMPTY")
    level = hashes[:]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        next_level = []
        for i in range(0, len(level), 2):
            next_level.append(sha256(level[i] + level[i + 1]))
        level = next_level
    return level[0]


# ----------------------------
# Blockchain structures
# ----------------------------
@dataclass
class Block:
    block_id: int
    start_row: int
    end_row: int
    tx_count: int
    prev_hash: str
    merkle_root: str
    block_hash: str


@dataclass
class NodeLedger:
    node_id: int
    tx_df: pd.DataFrame
    blocks: List[Block]


def build_blockchain(df: pd.DataFrame, cols_for_hash: List[str], block_size: int) -> Tuple[pd.DataFrame, List[Block]]:
    """
    Build a blockchain-style ledger from a DataFrame.

    Returns:
      tx_df: contains cols_for_hash + stored_tx_hash + block_id
      blocks: list of Block objects (with prev_hash, merkle_root, block_hash)
    """
    tx_df = df[cols_for_hash].copy().reset_index(drop=True)

    # Store tx hashes (these represent "signed" / committed events)
    tx_df["stored_tx_hash"] = tx_df.apply(lambda r: sha256(row_fingerprint(r, cols_for_hash)), axis=1)

    blocks: List[Block] = []
    prev = "GENESIS"

    n = len(tx_df)
    n_blocks = int(np.ceil(n / block_size))

    for b in range(n_blocks):
        start = b * block_size
        end = min((b + 1) * block_size, n)

        leaf_hashes = tx_df.loc[start:end - 1, "stored_tx_hash"].tolist()
        mroot = merkle_root(leaf_hashes)

        header = f"block_id={b}|prev={prev}|mroot={mroot}|tx_count={end-start}"
        bhash = sha256(header)

        blocks.append(
            Block(
                block_id=b,
                start_row=start,
                end_row=end - 1,
                tx_count=end - start,
                prev_hash=prev,
                merkle_root=mroot,
                block_hash=bhash,
            )
        )
        prev = bhash

    tx_df["block_id"] = (tx_df.index // block_size).astype(int)
    return tx_df, blocks


def verify_blockchain(tx_df: pd.DataFrame, blocks: List[Block], cols_for_hash: List[str], block_size: int) -> Dict:
    """
    Verification checks:
      1) Row-level integrity: recomputed hash == stored_tx_hash
      2) Block-level integrity: Merkle root recomputation matches stored merkle root
      3) Chain integrity: prev_hash links + block_hash matches recomputation from stored header

    Returns:
      dict with ok + mismatch counts
    """
    # 1) Row-level integrity
    recomputed_tx_hash = tx_df.apply(lambda r: sha256(row_fingerprint(r, cols_for_hash)), axis=1)
    row_mismatch = int((recomputed_tx_hash != tx_df["stored_tx_hash"]).sum())

    # 2) Block Merkle mismatch + 3) chain mismatch
    block_merkle_mismatch = 0
    chain_mismatch = 0

    prev = "GENESIS"
    for blk in blocks:
        # recompute Merkle root based on recomputed row hashes (fresh)
        start = blk.block_id * block_size
        end = min((blk.block_id + 1) * block_size, len(tx_df))
        leafs = recomputed_tx_hash.iloc[start:end].tolist()
        mroot_now = merkle_root(leafs)

        if mroot_now != blk.merkle_root:
            block_merkle_mismatch += 1

        # recompute block hash from stored metadata (not from mroot_now),
        # this tests whether the committed chain is consistent with itself.
        header_now = f"block_id={blk.block_id}|prev={prev}|mroot={blk.merkle_root}|tx_count={blk.tx_count}"
        bhash_now = sha256(header_now)

        if blk.prev_hash != prev or blk.block_hash != bhash_now:
            chain_mismatch += 1

        prev = blk.block_hash

    ok = (row_mismatch == 0) and (block_merkle_mismatch == 0) and (chain_mismatch == 0)

    return {
        "Integrity_passed": bool(ok),
        "Row_mismatch": row_mismatch,
        "Block_merkle_mismatch": int(block_merkle_mismatch),
        "Chain_mismatch": int(chain_mismatch),
    }


def simulate_decentralized_nodes(
    df: pd.DataFrame,
    cols_for_hash: List[str],
    block_size: int = 5000,
    num_nodes: int = 5,
    tamper_node: int = 2,
    tamper_row: int = 1234,
) -> Tuple[List[NodeLedger], pd.DataFrame]:
    """
    Creates N nodes with identical ledgers and then tampers a chosen node by editing one row
    (WITHOUT updating stored_tx_hash), so verification fails.

    Returns:
      nodes: list of NodeLedger
      verify_df: summary verification results per node
    """
    base_tx_df, base_blocks = build_blockchain(df, cols_for_hash, block_size)

    # Create decentralized nodes (each one has its own copy)
    nodes: List[NodeLedger] = []
    for i in range(num_nodes):
        nodes.append(NodeLedger(node_id=i, tx_df=base_tx_df.copy(deep=True), blocks=list(base_blocks)))

    # Tamper 1 node (simulate insider/hacker modifying evidence)
    if 0 <= tamper_node < num_nodes and 0 <= tamper_row < len(nodes[tamper_node].tx_df):
        col_to_change = cols_for_hash[0]  # change first selected column
        old_val = nodes[tamper_node].tx_df.loc[tamper_row, col_to_change]
        nodes[tamper_node].tx_df.loc[tamper_row, col_to_change] = int(old_val) + 1
        print(f"[Tamper] Node {tamper_node}: row {tamper_row}, column '{col_to_change}' modified.")

    # Verify each node independently
    results = []
    for node in nodes:
        v = verify_blockchain(node.tx_df, node.blocks, cols_for_hash, block_size)
        results.append({"node_id": node.node_id, **v})

    verify_df = pd.DataFrame(results).sort_values("node_id").reset_index(drop=True)
    return nodes, verify_df


def majority_consensus(nodes: List[NodeLedger], cols_for_hash: List[str], block_size: int) -> pd.DataFrame:
    """
    Simple consensus mechanism:
      - recompute each node's Merkle root per block from current data
      - take the majority root per block
      - flag nodes deviating from majority

    Returns a DataFrame with columns:
      block_id, node_id, root_matches_majority
    """
    if not nodes:
        raise ValueError("No nodes supplied.")

    n_blocks = len(nodes[0].blocks)
    consensus_rows = []

    for b in range(n_blocks):
        node_roots = {}
        for node in nodes:
            start = b * block_size
            end = min((b + 1) * block_size, len(node.tx_df))
            recomputed_hashes = node.tx_df.iloc[start:end].apply(
                lambda r: sha256(row_fingerprint(r, cols_for_hash)), axis=1
            ).tolist()
            node_roots[node.node_id] = merkle_root(recomputed_hashes)

        roots_list = list(node_roots.values())
        majority_root = max(set(roots_list), key=roots_list.count)

        for nid, root in node_roots.items():
            consensus_rows.append(
                {
                    "block_id": b,
                    "node_id": nid,
                    "root_matches_majority": (root == majority_root),
                }
            )

    return pd.DataFrame(consensus_rows)


def pick_candidate_cols(df: pd.DataFrame) -> List[str]:
    """
    Picks typical CIC-IDS style numeric columns if they exist; otherwise fallback to first 8 columns.
    """
    preferred = [
        " Flow Duration",
        " Total Fwd Packets",
        " Total Backward Packets",
        "Total Length of Fwd Packets",
        " Total Length of Bwd Packets",
        " Fwd Packet Length Mean",
        " Bwd Packet Length Mean",
        " Flow Bytes/s",
        " Flow Packets/s",
    ]
    cols = [c for c in preferred if c in df.columns]
    if len(cols) >= 3:
        return cols

    # fallback: choose numeric columns first (better stability)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) >= 3:
        return num_cols[:8]

    # final fallback: first 8 columns
    return list(df.columns[:8])


def main():
    parser = argparse.ArgumentParser(description="Decentralized nodes simulation for IDS blockchain integrity demo.")
    parser.add_argument("--csv", type=str, default=None, help="Path to my CIC-IDS CSV file. If omitted, the script will auto-detect a CSV in the working directory or script directory.")
    parser.add_argument("--sample", type=int, default=50000, help="Number of rows to sample for speed.")
    parser.add_argument("--block_size", type=int, default=5000, help="Rows per block.")
    parser.add_argument("--nodes", type=int, default=5, help="Number of nodes (servers).")
    parser.add_argument("--tamper_node", type=int, default=2, help="Which node to tamper (0-indexed).")
    parser.add_argument("--tamper_row", type=int, default=1234, help="Which row index to tamper inside that node.")
    parser.add_argument("--outdir", type=str, default="output_decentralized", help="Folder to save CSV/plots.")
    parser.add_argument("--save_outputs", action="store_true", help="Save CSV + plot to outdir.")
    args = parser.parse_args()

    # ----------------------------
    # Auto-detect CSV if not provided
    # ----------------------------
    def _list_csvs(folder: Path):
        return sorted([p for p in folder.glob("*.csv") if p.is_file()], key=lambda p: p.stat().st_size, reverse=True)

    if args.csv is None:
        wd = Path.cwd()
        script_dir = Path(__file__).resolve().parent
        candidates = _list_csvs(wd)
        if not candidates and script_dir != wd:
            candidates = _list_csvs(script_dir)

        if not candidates:
            raise FileNotFoundError(
                "No CSV found. Put my dataset CSV in the same folder as this script (or run from the folder), "
                "or pass --csv with an absolute path."
            )

        # Prefer a DDoS-looking file name, else pick the largest CSV
        ddos_like = [p for p in candidates if "ddos" in p.name.lower()]
        chosen = ddos_like[0] if ddos_like else candidates[0]
        args.csv = str(chosen)
        print(f"[Auto] Using CSV: {chosen}")

    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError as e:
        here = Path.cwd()
        script_dir = Path(__file__).resolve().parent
        csvs_here = sorted([p.name for p in here.glob("*.csv")])
        csvs_script = sorted([p.name for p in script_dir.glob("*.csv")]) if script_dir != here else []

        msg = [
            f"CSV not found: {args.csv}",
            f"Working directory: {here}",
            f"Script directory: {script_dir}",
            "",
            "CSV files in working directory:",
            "  " + ("\n  ".join(csvs_here) if csvs_here else "(none)"),
        ]
        if csvs_script:
            msg += ["", "CSV files next to script:", "  " + "\n  ".join(csvs_script)]
        msg += [
            "",
            "Fix:",
            '  - Run: cd "<folder containing my CSV>"',
            '  - Then run again, OR pass an absolute path: --csv "/full/path/to/my.csv"',
        ]
        raise FileNotFoundError("\n".join(msg)) from e

    # sample for speed
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        print(f"[Info] Sampled {len(df)} rows for the demo.")

    cols_for_hash = pick_candidate_cols(df)
    print("[Info] Columns used for hashing:")
    for c in cols_for_hash:
        print("  -", c)

    nodes, verify_df = simulate_decentralized_nodes(
        df=df,
        cols_for_hash=cols_for_hash,
        block_size=args.block_size,
        num_nodes=args.nodes,
        tamper_node=args.tamper_node,
        tamper_row=args.tamper_row,
    )

    print("\n=== Node Verification Results ===")
    print(verify_df.to_string(index=False))

    consensus_df = majority_consensus(nodes, cols_for_hash, args.block_size)
    bad = consensus_df[consensus_df["root_matches_majority"] == False]

    print("\n=== Consensus Summary ===")
    if len(bad) == 0:
        print("All nodes match the majority for every block ✅")
        mismatch_counts = pd.Series(dtype=int)
    else:
        mismatches = bad.groupby("node_id")["block_id"].count().rename("blocks_mismatched")
        print("Nodes deviating from majority detected (********** Warning **********)")
        print(mismatches.to_string())
        mismatch_counts = mismatches.astype(int)

    # ----------------------------
    # Optional: save outputs (CSV + plot)
    # ----------------------------
    if args.save_outputs:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # Save tables
        verify_df.to_csv(outdir / "node_verification_results.csv", index=False)
        consensus_df.to_csv(outdir / "consensus_per_block.csv", index=False)

        # Plot: blocks mismatched per node (0 means clean)
        if mismatch_counts.empty:
            # compute from consensus_df to include all nodes (even clean)
            mismatch_counts = (
                consensus_df.assign(mismatch=lambda d: ~d["root_matches_majority"])
                .groupby("node_id")["mismatch"]
                .sum()
                .astype(int)
            )
        else:
            # include clean nodes explicitly
            all_nodes = sorted(consensus_df["node_id"].unique().tolist())
            mismatch_counts = mismatch_counts.reindex(all_nodes, fill_value=0)

        plt.figure()
        mismatch_counts.plot(kind="bar")
        plt.title("Consensus mismatches per node (higher = tampered / divergent)")
        plt.xlabel("Node ID")
        plt.ylabel("Blocks mismatched")
        plt.tight_layout()
        #plt.savefig(outdir / "consensus_mismatches_per_node.png", dpi=200)
        plt.close()

        print(f"\n[Saved] CSV + plot saved to: {outdir.resolve()}")



if __name__ == "__main__":
    main()

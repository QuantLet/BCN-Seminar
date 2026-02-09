#Code generated with ChatGPT


import hashlib
import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Helpers
# -----------------------------
def canon(obj: Any) -> str:
    """Stable JSON encoding for deterministic hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def fmt_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


# -----------------------------
# Off-chain storage (simulated EHR/Cloud)
# -----------------------------
OFFCHAIN_DB: Dict[str, Dict[str, Any]] = {}

def store_offchain(doc_id: str, doc: Dict[str, Any]) -> str:
    OFFCHAIN_DB[doc_id] = doc
    return f"ehr://docs/{doc_id}"

def load_offchain(pointer: str) -> Dict[str, Any]:
    doc_id = pointer.split("/")[-1]
    if doc_id not in OFFCHAIN_DB:
        raise KeyError(f"Off-chain document not found: {doc_id}")
    return OFFCHAIN_DB[doc_id]

def doc_hash(pointer: str) -> str:
    return sha256_hex(canon(load_offchain(pointer)))


# -----------------------------
# Simple PoW Blockchain
# -----------------------------
@dataclass
class Tx:
    kind: str
    data: Dict[str, Any]
    ts: float

@dataclass
class Block:
    index: int
    prev_hash: str
    ts: float
    nonce: int
    txs: List[Tx]
    hash: str

class HealthcarePoWChain:
    """
    Demo blockchain for healthcare:
      - PoW blocks (hash must start with difficulty_prefix)
      - On-chain: document proof (hash+pointer), consent events, audit events
      - Off-chain: actual medical document content
    """

    def __init__(self, difficulty_prefix: str = "000"):
        self.difficulty_prefix = difficulty_prefix
        self.chain: List[Block] = []
        self.pending: List[Tx] = []

        # Derived state (replayed from chain)
        self.consent: Dict[Tuple[str, str], bool] = {}            # (patient_id, provider_id) -> granted
        self.doc_registry: Dict[str, Dict[str, Any]] = {}         # doc_id -> {patient_id, pointer, hash, block_index}
        self.audit: List[Dict[str, Any]] = []                     # list of audit rows

        self._genesis()

    def _block_hash(self, index: int, prev_hash: str, ts: float, nonce: int, txs: List[Tx]) -> str:
        payload = {
            "index": index,
            "prev_hash": prev_hash,
            "ts": ts,
            "nonce": nonce,
            "txs": [asdict(t) for t in txs],
        }
        return sha256_hex(canon(payload))

    def _mine(self, txs: List[Tx]) -> Block:
        index = len(self.chain)
        prev_hash = self.chain[-1].hash if self.chain else "0" * 64
        ts = time.time()
        nonce = 0

        while True:
            h = self._block_hash(index, prev_hash, ts, nonce, txs)
            if h.startswith(self.difficulty_prefix):
                return Block(index=index, prev_hash=prev_hash, ts=ts, nonce=nonce, txs=txs, hash=h)
            nonce += 1

    def _genesis(self) -> None:
        g = Tx(kind="GENESIS", data={"note": "healthcare PoW demo chain"}, ts=time.time())
        self.chain.append(self._mine([g]))
        self._replay_state()

    def add_tx(self, kind: str, data: Dict[str, Any]) -> None:
        self.pending.append(Tx(kind=kind, data=data, ts=time.time()))

    def mine_pending(self) -> Block:
        if not self.pending:
            raise ValueError("No pending transactions to mine.")
        block = self._mine(self.pending)
        self.pending = []
        self.chain.append(block)
        self._replay_state()
        return block

    def _replay_state(self) -> None:
        self.consent.clear()
        self.doc_registry.clear()
        self.audit.clear()

        for b in self.chain:
            for tx in b.txs:
                if tx.kind == "REGISTER_DOC":
                    self.doc_registry[tx.data["doc_id"]] = {
                        "patient_id": tx.data["patient_id"],
                        "pointer": tx.data["pointer"],
                        "hash": tx.data["hash"],
                        "block_index": b.index,
                    }
                elif tx.kind == "GRANT_CONSENT":
                    self.consent[(tx.data["patient_id"], tx.data["provider_id"])] = True
                elif tx.kind == "REVOKE_CONSENT":
                    self.consent[(tx.data["patient_id"], tx.data["provider_id"])] = False
                elif tx.kind == "AUDIT":
                    self.audit.append({
                        "ts": tx.ts,
                        "patient_id": tx.data["patient_id"],
                        "provider_id": tx.data["provider_id"],
                        "doc_id": tx.data["doc_id"],
                        "action": tx.data["action"],
                        "block_index": b.index,
                    })

    def is_valid(self) -> bool:
        for i, b in enumerate(self.chain):
            expected = self._block_hash(b.index, b.prev_hash, b.ts, b.nonce, b.txs)
            if expected != b.hash:
                return False
            if not b.hash.startswith(self.difficulty_prefix):
                return False
            if i == 0:
                if b.prev_hash != "0" * 64:
                    return False
            else:
                if b.prev_hash != self.chain[i - 1].hash:
                    return False
        return True

    def has_consent(self, patient_id: str, provider_id: str) -> bool:
        return self.consent.get((patient_id, provider_id), False)

    def verify_doc_integrity(self, doc_id: str) -> Tuple[bool, str]:
        reg = self.doc_registry.get(doc_id)
        if not reg:
            return False, "Document is not registered on-chain (or not mined yet)."
        pointer = reg["pointer"]
        try:
            current_hash = doc_hash(pointer)
        except KeyError:
            return False, "Off-chain document is missing."
        if current_hash == reg["hash"]:
            return True, "Document matches on-chain hash."
        return False, "Hash mismatch! Off-chain document was modified or corrupted."


# -----------------------------
# CLI
# -----------------------------
def prompt(s: str) -> str:
    return input(s).strip()

def show_menu() -> None:
    print("\n=== Healthcare Blockchain Demo (CLI) ===")
    print("1) Create medical document (off-chain) and register proof on-chain (pending)")
    print("2) Grant consent (patient -> provider) (pending)")
    print("3) Revoke consent (pending)")
    print("4) Read document (checks consent + creates AUDIT tx) (pending)")
    print("5) Verify document integrity (hash check)")
    print("6) Show audit log (mined only)")
    print("7) Show chain status")
    print("8) Simulate tampering (modify off-chain document)")
    print("9) Mine pending transactions (create PoW block)")
    print("0) Exit")

def print_block(block: Block) -> None:
    print("Block mined successfully")
    print(f"Index: {block.index}")
    print(f"Timestamp: {fmt_ts(block.ts)}")
    print(f"Nonce: {block.nonce}")
    print(f"Hash: {block.hash}")
    print(f"TX count: {len(block.txs)}")

def main() -> None:
    chain = HealthcarePoWChain(difficulty_prefix="000")  # keep it fast for demo
    print(f"Chain started. Difficulty: {chain.difficulty_prefix}")

    while True:
        show_menu()
        choice = prompt("Selection: ")

        try:
            if choice == "1":
                patient_id = prompt("patient_id: ")
                doc_id = prompt("doc_id: ")
                resource_type = prompt("FHIR resourceType (e.g. Observation): ") or "Observation"
                code = prompt("Code (e.g. LOINC 2093-3): ") or "LOINC 2093-3"
                value = prompt("Value (e.g. 186): ") or "186"
                unit = prompt("Unit (e.g. mg/dL): ") or "mg/dL"

                doc = {
                    "resourceType": resource_type,
                    "code": code,
                    "value": {"value": value, "unit": unit},
                    "createdAt": fmt_ts(time.time()),
                }

                pointer = store_offchain(doc_id, doc)
                h = doc_hash(pointer)

                chain.add_tx("REGISTER_DOC", {
                    "patient_id": patient_id,
                    "doc_id": doc_id,
                    "pointer": pointer,
                    "hash": h,
                })

                print(f"OK. Document stored off-chain: {pointer}")
                print("On-chain transaction pending: REGISTER_DOC (stores hash+pointer only).")
                print("Tip: Select option 9 to mine the block.")

            elif choice == "2":
                patient_id = prompt("patient_id: ")
                provider_id = prompt("provider_id: ")
                chain.add_tx("GRANT_CONSENT", {"patient_id": patient_id, "provider_id": provider_id})
                print("OK. Consent transaction pending.")
                print("Tip: Select option 9 to mine the block.")

            elif choice == "3":
                patient_id = prompt("patient_id: ")
                provider_id = prompt("provider_id: ")
                chain.add_tx("REVOKE_CONSENT", {"patient_id": patient_id, "provider_id": provider_id})
                print("OK. Revoke transaction pending.")
                print("Tip: Select option 9 to mine the block.")

            elif choice == "4":
                patient_id = prompt("patient_id: ")
                provider_id = prompt("provider_id: ")
                doc_id = prompt("doc_id: ")

                if not chain.has_consent(patient_id, provider_id):
                    print("ACCESS DENIED")
                    print("Reason: No valid consent found (or consent tx not mined yet).")
                    continue

                reg = chain.doc_registry.get(doc_id)
                if not reg:
                    print("ACCESS DENIED")
                    print("Reason: Document not registered on-chain (or not mined yet).")
                    continue

                doc = load_offchain(reg["pointer"])

                chain.add_tx("AUDIT", {
                    "patient_id": patient_id,
                    "provider_id": provider_id,
                    "doc_id": doc_id,
                    "action": "READ",
                })

                print("ACCESS GRANTED\n")
                print("Document (off-chain):")
                print(json.dumps(doc, indent=2, ensure_ascii=False))
                print("\nAudit transaction pending.")
                print("Tip: Select option 9 to mine the block.")

            elif choice == "5":
                doc_id = prompt("doc_id: ")
                ok, msg = chain.verify_doc_integrity(doc_id)
                if ok:
                    print("Integrity check PASSED.")
                    print(msg)
                else:
                    print("Integrity check FAILED!")
                    print(msg)

            elif choice == "6":
                if not chain.audit:
                    print("Audit log is empty (or audit tx not mined yet).")
                else:
                    print("Audit log (mined):")
                    for a in chain.audit:
                        print(
                            f"- {fmt_ts(a['ts'])} | patient={a['patient_id']} "
                            f"| provider={a['provider_id']} | doc={a['doc_id']} "
                            f"| action={a['action']} | block={a['block_index']}"
                        )

            elif choice == "7":
                print(f"Chain valid: {chain.is_valid()}")
                print(f"Blocks: {len(chain.chain)}")
                print(f"Pending TXs: {len(chain.pending)}")
                print(f"Registered docs (mined): {list(chain.doc_registry.keys())}")
                cons = {f"{p}->{pr}": v for (p, pr), v in chain.consent.items()}
                print(f"Consents (mined): {cons}")

            elif choice == "8":
                doc_id = prompt("doc_id to tamper: ")
                if doc_id not in OFFCHAIN_DB:
                    print("Off-chain document does not exist.")
                    continue

                field = prompt("Field (e.g. value.value): ") or "value.value"
                new_val = prompt("New value: ")

                if field == "value.value":
                    OFFCHAIN_DB[doc_id]["value"]["value"] = new_val
                else:
                    OFFCHAIN_DB[doc_id][field] = new_val

                print("Off-chain document modified successfully.")
                print("Tip: Use option 5 to verify integrity (should fail if it was mined before).")

            elif choice == "9":
                block = chain.mine_pending()
                print_block(block)

            elif choice == "0":
                print("Bye.")
                break

            else:
                print("Invalid selection.")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()

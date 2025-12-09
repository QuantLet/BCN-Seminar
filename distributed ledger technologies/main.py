"""
Toy replicated-ledger + signature + majority-vote consensus demo with blocks.

Improvements:
- Transaction is a @dataclass with type hints, nonces, tx_id, and full records.
- Replay protection via:
    * per-sender nonce
    * per-node set of seen tx_ids
- Node maintains:
    * balances
    * blockchain (list of Block objects)
    * next_nonce_per_sender
    * seen_tx_ids
- New Block structure:
    * index
    * prev_hash
    * transactions (list of Transaction)
    * deterministic block hash
- Network:
    * runs majority-vote consensus per transaction
    * if accepted, wraps tx in a Block and appends to each node's blockchain
- Optional full blockchain printing at the end.
"""

import json
import random
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Set

from ecdsa import SigningKey, VerifyingKey, NIST256p, BadSignatureError


# ============================================================
# Transaction model
# ============================================================

@dataclass
class Transaction:
    """
    A signed transfer of value from `sender` to `receiver`.

    Fields:
        sender   : logical name of the sender (e.g. "Alice").
        receiver : logical name of the receiver (e.g. "Bob").
        amount   : integer amount of value to transfer.
        nonce    : strictly increasing integer per sender for replay protection.
        signature: ECDSA signature over the serialized payload (may be None before signing).

    Design:
        - `nonce` prevents replays and enforces ordering per sender.
        - `signature` proves authorization.
        - `tx_id` (property) = SHA-256( serialize(payload) || signature ).
    """
    sender: str
    receiver: str
    amount: int
    nonce: int
    signature: bytes | None = None

    def payload(self) -> Dict:
        """
        Return the dictionary that is signed and recorded on-chain,
        excluding the signature itself.
        """
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "amount": self.amount,
            "nonce": self.nonce,
        }

    def serialize(self) -> bytes:
        """
        Deterministic byte representation for signing and hashing.

        We sort keys to ensure all nodes serialize the same logical transaction
        to the same bytes.
        """
        return json.dumps(self.payload(), sort_keys=True).encode("utf-8")

    @property
    def tx_id(self) -> str:
        """
        Deterministic transaction identifier: hex-encoded SHA-256.

        Computed as:
            hash( serialize(payload) || signature )

        If `signature` is None (e.g. pre-signing), this is still deterministic
        but not valid for a real system.
        """
        hasher = hashlib.sha256()
        hasher.update(self.serialize())
        if self.signature:
            hasher.update(self.signature)
        return hasher.hexdigest()

    def to_record(self) -> Dict:
        """
        Convert to a dictionary suitable for inclusion in a block record.

        Includes:
            - full payload
            - tx_id
            - signature (hex-encoded)
        """
        return {
            **self.payload(),
            "tx_id": self.tx_id,
            "signature": self.signature.hex() if self.signature else None,
        }


# ============================================================
# Block model
# ============================================================

@dataclass
class Block:
    """
    A simple block in the blockchain.

    Fields:
        index       : height of the block (0-based).
        prev_hash   : hash of the previous block (or '0' * 64 for genesis).
        transactions: list of Transaction objects included in this block.

    Design:
        - For simplicity, we do not implement PoW or PoS.
        - Block hash is SHA-256 over (index, prev_hash, serialized tx records).
        - In this demo, each block contains exactly one transaction, but
          the structure supports multiple.
    """
    index: int
    prev_hash: str
    transactions: List[Transaction]

    def serialize(self) -> bytes:
        """
        Deterministic serialization of the block header + body.

        We convert transactions to their dictionary records, then JSON-encode
        a structure with sorted keys.
        """
        block_dict = {
            "index": self.index,
            "prev_hash": self.prev_hash,
            "transactions": [tx.to_record() for tx in self.transactions],
        }
        return json.dumps(block_dict, sort_keys=True).encode("utf-8")

    @property
    def hash(self) -> str:
        """
        Deterministic block hash (hex-encoded SHA-256).
        """
        return hashlib.sha256(self.serialize()).hexdigest()

    def to_record(self) -> Dict:
        """
        Convert to a dictionary representation suitable for printing / exporting.

        Includes:
            - index
            - prev_hash
            - hash
            - list of transaction records
        """
        return {
            "index": self.index,
            "prev_hash": self.prev_hash,
            "hash": self.hash,
            "transactions": [tx.to_record() for tx in self.transactions],
        }


# ============================================================
# Node: local state, validation, and applying accepted blocks/tx
# ============================================================

class Node:
    """
    A node maintains:
        - its own keypair (SigningKey / VerifyingKey)
        - local balances (account -> amount)
        - a blockchain (list of Block objects)
        - consensus-related state:
            * next_nonce_per_sender
            * seen_tx_ids
        - a set of peers (other Node instances in the network)

    Responsibilities:
        - describe how to propose (create) a transaction
        - validate a transaction under its current local state
        - apply an accepted block and update state accordingly
    """

    def __init__(self, name: str, initial_balances: Dict[str, int]):
        self.name: str = name

        # Local view of the blockchain: list of Block objects.
        self.blockchain: List[Block] = []

        # Peers in the network (set to avoid duplicates).
        self.peers: Set["Node"] = set()

        # ECDSA keypair for signing and verifying this node's transactions.
        self.sk: SigningKey = SigningKey.generate(curve=NIST256p)
        self.vk: VerifyingKey = self.sk.get_verifying_key()

        # Local view of account balances.
        self.balances: Dict[str, int] = dict(initial_balances)

        # Next expected nonce per sender.
        # Start each account at nonce = 1.
        self.next_nonce_per_sender: Dict[str, int] = {
            account: 1 for account in initial_balances
        }

        # Set of transaction IDs already applied (replay protection).
        self.seen_tx_ids: Set[str] = set()

    # --------------------------------------------------------
    # Topology management
    # --------------------------------------------------------

    def connect(self, other: "Node") -> None:
        """
        Add another node as a peer.

        Notes:
            - We prevent self-connections.
            - Connections are not automatically symmetric; caller must handle that
              or use a helper in the setup code.
        """
        if other is self:
            return
        self.peers.add(other)

    # --------------------------------------------------------
    # Utility: blockchain / ledger helpers
    # --------------------------------------------------------

    def last_block_hash(self) -> str:
        """
        Return the hash of the last block in the local chain,
        or 64 zeros if there are no blocks (genesis anchor).
        """
        if not self.blockchain:
            return "0" * 64
        return self.blockchain[-1].hash

    def height(self) -> int:
        """
        Current blockchain height = number of blocks.
        """
        return len(self.blockchain)

    def full_ledger(self) -> List[Dict]:
        """
        Flatten the blockchain into a linear transaction ledger.

        Returns:
            A list of transaction records in chain order.
        """
        records: List[Dict] = []
        for block in self.blockchain:
            records.extend(tx.to_record() for tx in block.transactions)
        return records

    # --------------------------------------------------------
    # Transaction creation
    # --------------------------------------------------------

    def create_transaction(self, receiver: str, amount: int) -> Transaction | None:
        """
        Create and sign a new transaction from this node to `receiver`.

        - Checks this node's local balance.
        - Uses the current expected nonce for self.
        - Signs the serialized payload with this node's private key.

        Returns:
            Transaction if created successfully, or None if insufficient funds.

        Note:
            Nonce is only incremented when the transaction is actually applied
            (inside add_block/apply_accepted), not at proposal time.
        """
        sender_balance = self.balances.get(self.name, 0)
        if sender_balance < amount:
            return None

        current_nonce = self.next_nonce_per_sender.get(self.name, 1)

        tx = Transaction(
            sender=self.name,
            receiver=receiver,
            amount=amount,
            nonce=current_nonce,
        )
        tx.signature = self.sk.sign(tx.serialize())
        return tx

    # --------------------------------------------------------
    # Transaction validation
    # --------------------------------------------------------

    def validate(self, tx: Transaction, public_keys: Dict[str, VerifyingKey]) -> bool:
        """
        Validate a transaction under this node's current local state.

        Checks:
            1) Transaction has a signature.
            2) Sender is known in the public key registry.
            3) Signature matches the serialized payload.
            4) Nonce equals the expected nonce for the sender.
            5) tx_id has not been seen (no replay).
            6) Sender has enough balance.

        Returns:
            True if valid, False otherwise.
        """
        # 1) Signature present?
        if not tx.signature:
            return False

        # 2) Sender known?
        try:
            sender_vk = public_keys[tx.sender]
        except KeyError:
            return False

        # 3) Signature valid?
        try:
            sender_vk.verify(tx.signature, tx.serialize())
        except BadSignatureError:
            return False

        # 4) Nonce correct?
        expected_nonce = self.next_nonce_per_sender.get(tx.sender, 1)
        if tx.nonce != expected_nonce:
            return False

        # 5) Not a replay?
        if tx.tx_id in self.seen_tx_ids:
            return False

        # 6) Balance sufficient?
        sender_balance = self.balances.get(tx.sender, 0)
        if sender_balance < tx.amount:
            return False

        return True

    # --------------------------------------------------------
    # Apply accepted transaction and block
    # --------------------------------------------------------

    def apply_accepted_transaction(self, tx: Transaction) -> None:
        """
        Apply a single transaction that has already passed consensus.

        This updates:
            - balances
            - next_nonce_per_sender for the sender
            - seen_tx_ids
        """
        # Update balances.
        self.balances[tx.sender] = self.balances.get(tx.sender, 0) - tx.amount
        self.balances[tx.receiver] = self.balances.get(tx.receiver, 0) + tx.amount

        # Mark tx_id as seen.
        self.seen_tx_ids.add(tx.tx_id)

        # Increment expected nonce for this sender.
        current_expected = self.next_nonce_per_sender.get(tx.sender, 1)
        self.next_nonce_per_sender[tx.sender] = current_expected + 1

    def add_block(self, block: Block) -> None:
        """
        Add a block to this node's blockchain and apply its transactions.

        Steps:
            - Optionally verify linkage: block.prev_hash == last_block_hash().
            - For each transaction in the block, call apply_accepted_transaction.
            - Append the block to the blockchain.

        Assumptions in this toy setting:
            - Blocks are applied in the same order on all nodes.
            - Validation occurred before block creation via consensus.
        """
        # Simple linkage check (could be an assert in a toy setting).
        if block.prev_hash != self.last_block_hash():
            # In a more robust implementation, you'd handle forks or errors.
            raise ValueError(
                f"Unexpected prev_hash for block {block.index} at node {self.name}"
            )

        # Apply all transactions in the block.
        for tx in block.transactions:
            self.apply_accepted_transaction(tx)

        # Append block to local chain.
        self.blockchain.append(block)


# ============================================================
# Network + consensus orchestration
# ============================================================

class Network:
    """
    Simple synchronous network + majority-vote consensus.

    Responsibilities:
        - Maintain a list of nodes.
        - Maintain a public key registry.
        - For a proposed transaction from an origin node:
            * ask origin + its peers to validate (Node.validate)
            * if majority approves, wrap tx in a Block and apply it on validators.
    """

    def __init__(self, nodes: List[Node]):
        self.nodes: List[Node] = nodes

        # Public key registry: node name -> verifying key.
        self.public_keys: Dict[str, VerifyingKey] = {
            node.name: node.vk for node in nodes
        }

    def broadcast_transaction(self, tx: Transaction | None, origin: Node) -> bool:
        """
        Run consensus for a transaction proposed by `origin`.

        Process:
            1) If tx is None, treat as "no-op" (e.g. insufficient funds).
            2) Collect validators = origin + its peers.
            3) Ask each validator to validate the tx.
            4) If > 50% approve, create a Block with this tx and apply it to all
               validators' blockchains.
            5) Return True if accepted, False otherwise.
        """
        if tx is None:
            return False

        # Determine validators: origin + its peers.
        validators: Set[Node] = set(origin.peers)
        validators.add(origin)
        validators_list: List[Node] = list(validators)

        # Gather votes.
        votes = [node.validate(tx, self.public_keys) for node in validators_list]
        num_approvals = sum(votes)
        threshold = len(validators_list) // 2  # simple majority

        if num_approvals <= threshold:
            print("Transaction rejected by consensus:", tx.payload())
            return False

        # At this point, the tx is accepted by majority.
        # We now wrap it in a block and apply that block to each validator.
        # We assume all validators currently have identical chains.
        reference_node = validators_list[0]
        new_index = reference_node.height()  # next block height
        prev_hash = reference_node.last_block_hash()

        block = Block(
            index=new_index,
            prev_hash=prev_hash,
            transactions=[tx],  # single-tx block for simplicity
        )

        for node in validators_list:
            node.add_block(block)

        return True


# ============================================================
# Demo simulation
# ============================================================

if __name__ == "__main__":
    random.seed(42)  # reproducible randomness

    # --------------------------------------------------------
    # Initial balances (same starting state for all nodes)
    # --------------------------------------------------------
    initial_balances: Dict[str, int] = {
        "Alice": 50,
        "Bob": 50,
        "Carol": 50,
        "Dave": 50,
    }

    # --------------------------------------------------------
    # Create nodes
    # --------------------------------------------------------
    nodes: List[Node] = [
        Node("Alice", initial_balances),
        Node("Bob", initial_balances),
        Node("Carol", initial_balances),
        Node("Dave", initial_balances),
    ]

    # --------------------------------------------------------
    # Fully connect network (undirected graph)
    #
    # For each pair of distinct nodes (i, j), connect both ways.
    # Node.connect uses a set, so repeated calls are harmless.
    # --------------------------------------------------------
    for i, ni in enumerate(nodes):
        for j, nj in enumerate(nodes):
            if i != j:
                ni.connect(nj)

    # --------------------------------------------------------
    # Create network / consensus orchestrator
    # --------------------------------------------------------
    network = Network(nodes)

    # --------------------------------------------------------
    # Simulate random transactions
    # --------------------------------------------------------
    N_TX = 20
    for _ in range(N_TX):
        # Choose a random sender.
        sender = random.choice(nodes)

        # Choose a random receiver different from the sender.
        possible_receivers = [n for n in nodes if n is not sender]
        receiver = random.choice(possible_receivers)

        # Random positive amount.
        amount = random.randint(1, 30)

        # Sender creates a transaction.
        tx = sender.create_transaction(receiver.name, amount)

        # Network runs consensus and (if accepted) creates a block.
        network.broadcast_transaction(tx, origin=sender)

    # --------------------------------------------------------
    # Result summary: heights & balances
    # --------------------------------------------------------
    print("\nBlockchain height per node (number of blocks):")
    for n in nodes:
        print(f"{n.name}: {n.height()}")

    print("\nFinal balances per node:")
    for n in nodes:
        print(n.name, "->", n.balances)

    # --------------------------------------------------------
    # Optional: print full blockchain for each node
    # Set this flag to False if you don't want verbose output.
    # --------------------------------------------------------
    PRINT_BLOCKCHAIN = True

    if PRINT_BLOCKCHAIN:
        for n in nodes:
            print(f"\n=== Blockchain for {n.name} ===")
            for block in n.blockchain:
                print(json.dumps(block.to_record(), indent=2))

    # --------------------------------------------------------
    # Consistency checks
    # --------------------------------------------------------
    # 1) All blockchains identical (block by block, record by record).
    chain_records = [[b.to_record() for b in n.blockchain] for n in nodes]
    base_chain = chain_records[0]
    assert all(base_chain == cr for cr in chain_records[1:]), "Chains diverged!"

    # 2) All balances identical.
    balances_list = [n.balances for n in nodes]
    base_balances = balances_list[0]
    assert all(base_balances == b for b in balances_list[1:]), "Balances diverged!"

    # 3) Total supply conserved.
    total = sum(base_balances.values())
    print("\nTotal supply:", total)
    assert total == sum(initial_balances.values()), "Total supply changed!"


"""
proposal_crypto.py
~~~~~~~~~~~~~~~~~~
Pure cryptographic helpers for building, hashing, and proving Merkle trees
used by the SmartnodesCoordinator proposal workflow.

All functions accept a ``web3.Web3`` instance for keccak / checksum operations
so they can be used independently of ``ContractManager`` and tested against a
local in-process chain without any network connectivity.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, TypedDict

from eth_abi import encode
from hexbytes import HexBytes
from web3 import Web3


class Participant(TypedDict):
    """A single worker entry used as a Merkle leaf."""

    addr: str  # EIP-55 checksummed address
    capacity: int  # byte-hours contributed


def build_participants(
    workers: List[str],
    job_capacities: List[int],
    w3: Web3,
) -> List[Participant]:
    """
    Zip a list of worker addresses and capacities into ``Participant`` dicts,
    normalising addresses to EIP-55 checksum form.

    Args:
        workers:        Raw worker addresses (any case).
        job_capacities: Parallel capacity list; defaults to 0 for missing entries.
        w3:             Web3 instance used for checksum normalisation.

    Returns:
        List of Participant dicts, one per worker.
    """
    return [
        Participant(
            addr=w3.to_checksum_address(addr),
            capacity=job_capacities[i] if i < len(job_capacities) else 0,
        )
        for i, addr in enumerate(workers)
    ]


def _leaf_hash(participant: Participant, w3: Web3) -> bytes:
    """
    Hash a single participant into a Merkle leaf exactly as the Solidity
    contract does::

        keccak256(abi.encode(addr, capacity))
    """
    return bytes(
        w3.keccak(
            encode(
                ["address", "uint256"],
                [w3.to_checksum_address(participant["addr"]), participant["capacity"]],
            )
        )
    )


def build_merkle_tree(leaves: List[bytes], w3: Web3) -> bytes:
    """
    Build a binary Merkle tree root from a list of pre-hashed leaves using
    sorted-pair combination (``min || max``) to match the Solidity verifier.

    Args:
        leaves: Pre-hashed leaf values (32-byte sequences).
        w3:     Web3 instance for keccak.

    Returns:
        32-byte Merkle root, or ``bytes(32)`` for an empty list.
    """
    if not leaves:
        return bytes(32)
    if len(leaves) == 1:
        return bytes(leaves[0])

    current: List[bytes] = [bytes(leaf) for leaf in leaves]

    while len(current) > 1:
        next_level: List[bytes] = []
        for i in range(0, len(current), 2):
            left = current[i]
            right = current[i + 1] if i + 1 < len(current) else current[i]
            combined = bytes(
                w3.keccak(left + right) if left <= right else w3.keccak(right + left)
            )
            next_level.append(combined)
        current = next_level

    return current[0]


def merkle_root_from_participants(participants: List[Participant], w3: Web3) -> bytes:
    """
    Convenience wrapper: leaf-hash every participant then build the tree.

    Returns ``bytes(32)`` for an empty participant list.
    """
    if not participants:
        return bytes(32)
    leaves = [_leaf_hash(p, w3) for p in participants]
    return build_merkle_tree(leaves, w3)


def generate_merkle_proof(
    participants: List[Participant],
    target_address: str,
    w3: Web3,
) -> List[bytes]:
    """
    Generate an inclusion proof for ``target_address`` in ``participants``.

    The proof is a list of sibling hashes from leaf to root that a Solidity
    ``MerkleProof.verify`` call can validate.

    Args:
        participants:   Full participant list (same order used to build the tree).
        target_address: Address to prove membership for (any case).
        w3:             Web3 instance.

    Returns:
        List of 32-byte sibling hashes.

    Raises:
        ValueError: If ``target_address`` is not in ``participants``.
    """
    if not participants:
        return []

    target = w3.to_checksum_address(target_address)
    target_index: Optional[int] = next(
        (
            i
            for i, p in enumerate(participants)
            if w3.to_checksum_address(p["addr"]) == target
        ),
        None,
    )

    if target_index is None:
        raise ValueError(f"Participant {target_address} not found in participant list")

    leaves = [_leaf_hash(p, w3) for p in participants]

    if len(leaves) <= 1:
        return []

    proof: List[bytes] = []
    current = leaves[:]
    idx = target_index

    while len(current) > 1:
        sibling = idx + 1 if idx % 2 == 0 else idx - 1
        proof.append(
            bytes(current[sibling]) if sibling < len(current) else bytes(current[idx])
        )

        next_level: List[bytes] = []
        for i in range(0, len(current), 2):
            left = current[i]
            right = current[i + 1] if i + 1 < len(current) else current[i]
            combined = bytes(
                w3.keccak(left + right) if left <= right else w3.keccak(right + left)
            )
            next_level.append(combined)

        current = next_level
        idx //= 2

    return proof


def verify_merkle_proof(
    proof: List[bytes],
    leaf: bytes,
    root: bytes,
    w3: Web3,
) -> bool:
    """
    Verify a Merkle inclusion proof.

    Mirrors the sorted-pair combination used in ``build_merkle_tree`` and the
    Solidity ``MerkleProof`` library.

    Args:
        proof:  Sibling hashes from ``generate_merkle_proof``.
        leaf:   The leaf hash to verify (use ``_leaf_hash`` to compute it).
        root:   The expected Merkle root.
        w3:     Web3 instance for keccak.

    Returns:
        ``True`` if the proof is valid.
    """
    computed = bytes(leaf)
    for sibling in proof:
        sibling = bytes(sibling)
        combined = (
            w3.keccak(computed + sibling)
            if computed <= sibling
            else w3.keccak(sibling + computed)
        )
        computed = bytes(combined)
    return computed == bytes(root)


def hash_proposal_data(
    merkle_root: str,
    validators: List[str],
    job_hashes: List[str],
    workers_hash: str,
    capacities_hash: str,
    w3: Web3,
) -> bytes:
    """
    ABI-encode the five proposal fields and keccak-hash the result, exactly
    as the SmartnodesCoordinator contract does on-chain.

    All hash arguments are accepted as hex strings (the natural Python
    representation after ``bytes.hex()``); ``HexBytes`` handles the
    conversion transparently.

    Args:
        merkle_root:     Hex string of the participant Merkle root.
        validators:      Checksummed addresses of validators to remove.
        job_hashes:      Hex strings of completed job hashes (bytes32 each).
        workers_hash:    Hex string of ``keccak(abi.encode(address[]))``.
        capacities_hash: Hex string of ``keccak(abi.encode(uint256[]))``.
        w3:              Web3 instance.

    Returns:
        32-byte proposal hash.
    """
    encoded = encode(
        ["bytes32", "address[]", "bytes32[]", "bytes32", "bytes32"],
        [
            HexBytes(merkle_root),
            validators,
            [HexBytes(j) for j in job_hashes],
            HexBytes(workers_hash),
            HexBytes(capacities_hash),
        ],
    )
    return bytes(w3.keccak(encoded))

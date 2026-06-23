from typing import List, Tuple, Optional, Dict, Any, TypedDict
from web3.exceptions import ContractLogicError
from dotenv import get_key
from eth_abi import encode
from hexbytes import HexBytes
import logging
from collections import defaultdict
import threading
import time

from tensorlink.eth.rpc_backoff import RPCBackoff
from tensorlink.p2p.torch_node import Torchnode
from tensorlink.crypto.merkle import (
    build_participants,
    merkle_root_from_participants,
    generate_merkle_proof,
    hash_proposal_data,
)


class ProposalData(TypedDict):
    """
    Typed structure for a proposal dict built by _build_proposal and consumed
    by _hash_proposal_data / _execute_proposal / get_worker_claim_data.
    """

    validators: List[str]  # checksummed addresses
    job_hashes: List[str]  # hex strings (bytes32)
    job_capacities: List[int]
    workers: List[str]  # checksummed addresses
    total_capacity: List[int]
    total_workers: List[int]
    distribution_id: Optional[int]
    timestamp: float
    merkle_root: str  # hex string produced by bytes.hex()
    workers_hash: str  # hex string
    capacities_hash: str  # hex string


class ContractManager:
    """
    Manages blockchain contract interactions for validator proposals and job management.

    This class handles the creation, submission, voting, and execution of proposals
    for validator removal, job completion, and reward distribution on the blockchain.
    """

    # Maximum number of consecutive 429 failures before we stop retrying entirely
    # inside a single call-site (the outer loop will still wake up later).
    _MAX_RATE_LIMIT_FAILURES = 6

    def __init__(
        self,
        node: Torchnode,
        multi_sig_contract,
        chain,
        public_key: str,
    ):
        """
        Initialize the Contract Manager.

        Args:
            node: Parent node instance that contains DHT query methods
            multi_sig_contract: Web3 contract instance for multi-signature operations
            chain: Web3 chain connection
            public_key: Public key of the node
        """
        self.node = node
        self.coordinator_contract = multi_sig_contract
        self.chain = chain
        self.public_key = public_key

        # State tracking
        self.validators_to_clear: List[str] = []
        self.jobs_to_complete: List[str] = []
        self.current_proposal: Optional[int] = (
            self.coordinator_contract.functions.nextProposalId.call()
        )
        self.terminate_flag = node.terminate_flag

        self.proposals: Dict[str, threading.Thread] = {}

        # Shared back-off tracker so all threads respect the same cool-down window
        self._rpc_backoff = RPCBackoff()

        # Track whether we have already submitted a proposal this round so that
        # proposal_creator does not re-enter create_and_submit_proposal after a
        # successful submission.
        self._submitted_this_round: bool = False

    def proposal_validator(self) -> None:
        """Listen for new proposals created on SmartnodesCoordinator and validate them."""
        while not self.terminate_flag.is_set():
            try:
                self._rpc_backoff.wait()

                current_proposal_id = (
                    self.coordinator_contract.functions.nextProposalId().call()
                )
                self._rpc_backoff.success()

                # Update variables for new round of proposals
                if current_proposal_id != self.current_proposal:
                    self.current_proposal = current_proposal_id
                    self.proposals = {}

                expected_proposals = self._get_expected_proposal_count()

                for proposal_num in range(1, expected_proposals + 1):
                    if self.terminate_flag.is_set():
                        break
                    self._try_validate_proposal(proposal_num)

                self._wait_for_next_round()

            except Exception as e:
                if RPCBackoff.is_rate_limit(e):
                    delay = self._rpc_backoff.failure()
                    self.node.debug_print(
                        f"Rate limited while fetching proposals, backing off {delay:.0f}s",
                        colour="yellow",
                        level=logging.WARNING,
                        tag="ContractManager",
                    )
                else:
                    self._rpc_backoff.success()  # not a rate-limit; reset counter
                    self.node.debug_print(
                        f"Error while fetching created proposals: {e}",
                        colour="bright_red",
                        level=logging.ERROR,
                        tag="ContractManager",
                    )

            time.sleep(5)

    def proposal_creator(self) -> None:
        """Create proposals when this node is selected as a round validator."""
        while not self.terminate_flag.is_set():
            try:
                self._rpc_backoff.wait()

                if not self._is_in_current_round_validators():
                    self._wait_for_next_round()
                    continue

                (next_proposal_id, execution_time, round_validators) = (
                    self.coordinator_contract.functions.getState().call()
                )
                self._rpc_backoff.success()
                time.sleep(1)

                is_expired = self.coordinator_contract.functions.isRoundExpired().call()
                time.sleep(1)

                # Reset submission flag when a new round starts
                if next_proposal_id != self.current_proposal:
                    self.current_proposal = next_proposal_id
                    self._submitted_this_round = False

                if self._submitted_this_round:
                    # Already submitted this round, wait until the next one
                    self._wait_for_next_round()
                    self._submitted_this_round = False
                    continue

                if self.public_key in round_validators or is_expired:
                    self.create_and_submit_proposal()
                    self._wait_for_next_round()
                else:
                    time.sleep(30)

            except Exception as e:
                if RPCBackoff.is_rate_limit(e):
                    delay = self._rpc_backoff.failure()
                    self.node.debug_print(
                        f"Rate limited in proposal_creator, backing off {delay:.0f}s",
                        colour="yellow",
                        level=logging.WARNING,
                        tag="ContractManager",
                    )
                else:
                    self._rpc_backoff.success()
                    self.node.debug_print(
                        f"Error processing new entries: {e}",
                        colour="bright_red",
                        level=logging.ERROR,
                        tag="ContractManager",
                    )

            time.sleep(10)

    # -----------------------------------------------------------------------
    # Proposal creation / submission
    # -----------------------------------------------------------------------

    def create_and_submit_proposal(self) -> None:
        """Build, store, and submit a single proposal for the current round."""
        self.node.debug_print(
            "Creating proposal...",
            colour="bright_blue",
            level=logging.INFO,
            tag="ContractManager",
        )

        max_attempts = 3

        for attempt in range(1, max_attempts + 1):
            if self.terminate_flag.is_set():
                return

            proposal, proposal_hash = self._build_proposal(attempt)

            if proposal is None or proposal_hash is None:
                # _build_proposal already logged the reason; a None return
                # means an unrecoverable error for this attempt.
                return

            code = self._submit_proposal(proposal_hash)

            if code == 0:
                # Successfully submitted
                self._submitted_this_round = True
                break
            elif code == 1:
                # Permanent failure (e.g., already submitted, max retries hit)
                return
            elif code == 2:
                # Transient failure (e.g., too early); try again next iteration
                continue
        else:
            self.node.debug_print(
                "Max proposal creation attempts reached",
                colour="bright_red",
                level=logging.ERROR,
                tag="ContractManager",
            )
            return

        if not proposal_hash:
            return

        # Guard against shutdown occurring between submission and monitoring.
        if self.terminate_flag.is_set():
            self.node.debug_print(
                "Node is shutting down, skipping proposal monitoring",
                colour="yellow",
                level=logging.INFO,
                tag="ContractManager",
            )
            return

        self._wait_for_next_block()
        self._monitor_and_execute_proposal(proposal_hash.hex())

    def _build_proposal(
        self, attempt: int
    ) -> Tuple[Optional[ProposalData], Optional[bytes]]:
        """
        Gather on-chain and network state and build a proposal dict.

        Returns:
            (proposal, proposal_hash) on success, or (None, None) on failure.
        """
        try:
            self._rpc_backoff.wait()

            # Dry-run to confirm we are allowed to submit
            self.coordinator_contract.functions.createProposal(
                encode(["uint256"], [12345])
            ).call({"from": self.public_key})

            self._rpc_backoff.success()

            # Collect network data
            self.node.get_workers()
            validators_to_remove = self.verify_and_remove_validators()
            job_hashes, job_capacities, job_workers = self.process_jobs()

            # Compute the three hex-string digest fields separately so the
            # final ProposalData literal has fully explicit, typed values.
            participants_partial = build_participants(
                job_workers, job_capacities, self.chain
            )
            merkle_root: str = merkle_root_from_participants(
                participants_partial, self.chain
            ).hex()
            workers_hash: str = self.chain.keccak(
                encode(["address[]"], [job_workers])
            ).hex()
            capacities_hash: str = self.chain.keccak(
                encode(["uint256[]"], [job_capacities])
            ).hex()

            proposal: ProposalData = {
                "validators": validators_to_remove,
                "job_hashes": [j.hex() for j in job_hashes],
                "job_capacities": job_capacities,
                "workers": job_workers,
                "total_capacity": [
                    int(
                        sum(
                            w.get("total_gpu_memory", 0)
                            for w in self.node.all_workers.values()
                        )
                    )
                ],
                "total_workers": [len(self.node.all_workers)],
                "distribution_id": self.current_proposal,
                "timestamp": time.time(),
                "merkle_root": merkle_root,
                "workers_hash": workers_hash,
                "capacities_hash": capacities_hash,
            }

            proposal_hash = hash_proposal_data(
                proposal["merkle_root"],
                proposal["validators"],
                proposal["job_hashes"],
                proposal["workers_hash"],
                proposal["capacities_hash"],
                self.chain,
            )
            self.node.dht.store(proposal_hash.hex(), dict(proposal))

            return proposal, proposal_hash

        except Exception as e:
            msg = str(e).lower()

            if RPCBackoff.is_rate_limit(e):
                delay = self._rpc_backoff.failure()
                self.node.debug_print(
                    f"Rate limited while building proposal (attempt {attempt}), "
                    f"backing off {delay:.0f}s",
                    colour="yellow",
                    level=logging.WARNING,
                    tag="ContractManager",
                )
                if (
                    self._rpc_backoff.consecutive_failures
                    >= self._MAX_RATE_LIMIT_FAILURES
                ):
                    self.node.debug_print(
                        "Too many consecutive rate-limit errors, aborting proposal creation",
                        colour="bright_red",
                        level=logging.ERROR,
                        tag="ContractManager",
                    )
                    return None, None
                # Return a sentinel that tells the caller to retry
                return None, None

            if "updatetime - 2min" in msg:
                self.node.debug_print(
                    f"Waiting for next round (attempt {attempt})",
                    colour="yellow",
                    level=logging.DEBUG,
                    tag="ContractManager",
                )
                self._wait_for_next_round()
                return None, None

            if "0xde813857" in msg:
                self.node.debug_print(
                    "Already submitted proposal this round!",
                    colour="yellow",
                    level=logging.DEBUG,
                    tag="ContractManager",
                )
                self._submitted_this_round = True
                return None, None

            self.node.debug_print(
                f"Cannot create proposal (attempt {attempt}): {e}",
                colour="bright_red",
                level=logging.ERROR,
                tag="ContractManager",
            )
            return None, None

    def _submit_proposal(self, proposal_hash: bytes) -> int:
        """
        Submit the proposal to the blockchain.

        Returns:
            0: success
            1: permanent failure (abort)
            2: transient / timing failure (retry outer loop)
        """
        max_retries = 3

        for retry in range(1, max_retries + 1):
            if self.terminate_flag.is_set():
                return 1

            try:
                self._rpc_backoff.wait()

                # Verify proposal can be submitted (dry-run)
                self.coordinator_contract.functions.createProposal(proposal_hash).call(
                    {"from": self.public_key}
                )

                # Fetch a fresh nonce immediately before signing to avoid
                # "nonce too low" and "replacement transaction underpriced" errors.
                nonce = self.chain.eth.get_transaction_count(self.public_key, "pending")

                tx = self.coordinator_contract.functions.createProposal(
                    proposal_hash
                ).build_transaction(
                    {
                        "from": self.public_key,
                        "nonce": nonce,
                        "gas": 6_721_975,
                        "gasPrice": self.chain.eth.gas_price,
                    }
                )

                tx_hash = self._submit_transaction(tx)
                self._rpc_backoff.success()

                self.node.debug_print(
                    f"Proposal ({proposal_hash.hex()}) submitted! ({tx_hash.hex()})",
                    colour="green",
                    level=logging.INFO,
                    tag="ContractManager",
                )
                return 0

            except Exception as e:
                e_str = str(e)

                # Already submitted this round
                if "0xde813857" in e_str:
                    self.node.debug_print(
                        "Validator has already submitted a proposal this round!",
                        colour="bright_red",
                        level=logging.INFO,
                        tag="ContractManager",
                    )
                    self._submitted_this_round = True
                    return 0

                # Too early, wait and signal outer loop to retry
                if "updateTime - 2min" in e_str:
                    self.node.debug_print(
                        "Not enough time since last proposal! Waiting for next round...",
                        colour="green",
                        level=logging.DEBUG,
                        tag="ContractManager",
                    )
                    self._wait_for_next_round()
                    return 2

                # Nonce issues, refresh and retry immediately
                if RPCBackoff.is_nonce_error(e):
                    self.node.debug_print(
                        f"Nonce error on attempt {retry}, will refresh nonce and retry: {e_str}",
                        colour="yellow",
                        level=logging.WARNING,
                        tag="ContractManager",
                    )
                    time.sleep(2)
                    continue

                # Rate limit, back off and retry
                if RPCBackoff.is_rate_limit(e):
                    delay = self._rpc_backoff.failure()
                    self.node.debug_print(
                        f"Rate limited on submission attempt {retry}, "
                        f"backing off {delay:.0f}s",
                        colour="yellow",
                        level=logging.WARNING,
                        tag="ContractManager",
                    )
                    if (
                        self._rpc_backoff.consecutive_failures
                        >= self._MAX_RATE_LIMIT_FAILURES
                    ):
                        self.node.debug_print(
                            "Too many consecutive rate-limit errors, aborting submission",
                            colour="bright_red",
                            level=logging.ERROR,
                            tag="ContractManager",
                        )
                        return 1
                    continue

                # Unknown error
                self.node.debug_print(
                    f"Error creating proposal (attempt {retry}): {e_str}",
                    colour="bright_red",
                    level=logging.WARNING if retry < max_retries else logging.ERROR,
                    tag="ContractManager",
                )
                if retry < max_retries:
                    time.sleep(10)
                else:
                    return 1

        return 1

    # -----------------------------------------------------------------------
    # Proposal validation
    # -----------------------------------------------------------------------

    def _try_validate_proposal(self, proposal_num: int) -> None:
        """Fetch and start a validation thread for a single proposal number."""
        try:
            self._rpc_backoff.wait()
            proposal_data = self.coordinator_contract.functions.getProposal(
                proposal_num
            ).call()
            self._rpc_backoff.success()

            proposal_hash = proposal_data[-1].hex()
            author = proposal_data[0]
            time.sleep(1)

            if proposal_hash in self.proposals:
                return

            t = threading.Thread(
                target=self.validate_proposal,
                args=(author, proposal_hash, proposal_num),
                name=f"proposal_validator_{proposal_num}",
                daemon=True,
            )
            self.proposals[proposal_hash] = t
            t.start()

        except ContractLogicError:
            # Proposal not published yet (normal, skip silently)
            pass
        except Exception as e:
            if RPCBackoff.is_rate_limit(e):
                delay = self._rpc_backoff.failure()
                self.node.debug_print(
                    f"Rate limited fetching proposal {proposal_num}, "
                    f"backing off {delay:.0f}s",
                    colour="yellow",
                    level=logging.WARNING,
                    tag="ContractManager",
                )
            else:
                self._rpc_backoff.success()
                self.node.debug_print(
                    f"Error fetching proposal {proposal_num}: {e}",
                    colour="bright_red",
                    level=logging.ERROR,
                    tag="ContractManager",
                )

    def validate_proposal(
        self, author: str, proposal_hash: str, proposal_num: int
    ) -> None:
        """Validate a proposal found on-chain by checking the DHT data."""
        self.node.debug_print(
            f"Validation started for proposal: {proposal_hash}",
            colour="bright_blue",
            level=logging.INFO,
            tag="ContractManager",
        )

        try:
            proposal_data = self.node.dht.query(proposal_hash)
        except Exception as e:
            self.node.debug_print(
                f"DHT query failed for proposal {proposal_hash}: {e}",
                colour="bright_red",
                level=logging.ERROR,
                tag="ContractManager",
            )
            return

        if proposal_data is None:
            self.node.debug_print(
                f"Proposal {proposal_hash} not found in DHT!",
                tag="ContractManager",
            )
            return

        try:
            pd = ProposalData(**proposal_data)  # cast untyped DHT result
            proposal_data_hash = hash_proposal_data(
                pd["merkle_root"],
                pd["validators"],
                pd["job_hashes"],
                pd["workers_hash"],
                pd["capacities_hash"],
                self.chain,
            ).hex()
        except Exception as e:
            self.node.debug_print(
                f"Failed to hash proposal data for {proposal_hash}: {e}",
                colour="bright_red",
                level=logging.ERROR,
                tag="ContractManager",
            )
            return

        if proposal_data_hash != proposal_hash:
            self.node.debug_print(
                "Invalid proposal hash!", colour="red", tag="ContractManager"
            )
            return

        self._approve_transaction(proposal_num, proposal_hash)

    def _approve_transaction(self, proposal_num: int, proposal_hash: str) -> None:
        """Cast a vote for a proposal on-chain."""
        try:
            self._rpc_backoff.wait()

            nonce = self.chain.eth.get_transaction_count(self.public_key, "pending")
            tx = self.coordinator_contract.functions.voteForProposal(
                proposal_num
            ).build_transaction(
                {
                    "from": self.public_key,
                    "nonce": nonce,
                    "gas": 6_721_975,
                    "gasPrice": self.chain.eth.gas_price,
                }
            )
            signed_tx = self.chain.eth.account.sign_transaction(
                tx, get_key(".tensorlink.env", "PRIVATE_KEY")
            )
            tx_hash = self.chain.eth.send_raw_transaction(signed_tx.raw_transaction)
            self._rpc_backoff.success()

            self.node.debug_print(
                f"Proposal {proposal_num}: {proposal_hash} approved! ({tx_hash.hex()})",
                colour="green",
                level=logging.INFO,
                tag="ContractManager",
            )

        except Exception as e:
            e_str = str(e)
            if "Validator has already voted!" in e_str:
                self.node.debug_print(
                    f"Have already voted on proposal {proposal_num}, continuing...",
                    colour="green",
                    level=logging.DEBUG,
                    tag="ContractManager",
                )
            elif RPCBackoff.is_rate_limit(e):
                delay = self._rpc_backoff.failure()
                self.node.debug_print(
                    f"Rate limited while voting on proposal {proposal_num}, "
                    f"will retry later ({delay:.0f}s backoff)",
                    colour="yellow",
                    level=logging.WARNING,
                    tag="ContractManager",
                )
            elif RPCBackoff.is_nonce_error(e):
                self.node.debug_print(
                    f"Nonce error while voting on proposal {proposal_num}: {e_str}",
                    colour="yellow",
                    level=logging.WARNING,
                    tag="ContractManager",
                )
            else:
                self.node.debug_print(
                    f"Error voting on proposal {proposal_num}: {e_str}",
                    colour="bright_red",
                    level=logging.ERROR,
                    tag="ContractManager",
                )

    # -----------------------------------------------------------------------
    # Validator / job helpers
    # -----------------------------------------------------------------------

    def add_validator_to_clear(self, validator_id: str) -> None:
        """Add a validator to the list of validators to be cleared."""
        if validator_id not in self.validators_to_clear:
            self.validators_to_clear.append(validator_id)

    def add_job_to_complete(self, job_data: dict) -> None:
        """Add a job to the list of jobs to be completed."""
        if (
            job_data.get("end_time", 0) - job_data.get("timestamp", 0) > 1
            and job_data.get("id") not in self.jobs_to_complete
        ):
            self.jobs_to_complete.append(job_data["id"])

    def verify_and_remove_validators(self) -> List[str]:
        """
        Verify validator status and create list of offline validators to remove.

        Returns:
            List of checksummed validator addresses to be removed.
        """
        validators_to_remove = []

        for validator in self.validators_to_clear:
            try:
                node_info = self.node.dht.query(validator)
                if not node_info:
                    continue
                if not self._is_validator_online(node_info):
                    node_address = self._get_validator_address(validator)
                    if node_address:
                        validators_to_remove.append(node_address)
            except Exception as e:
                self.node.debug_print(
                    f"Error verifying validator {validator}: {e}",
                    colour="yellow",
                    level=logging.WARNING,
                    tag="ContractManager",
                )

        return [self.chain.to_checksum_address(v) for v in validators_to_remove]

    def process_jobs(self) -> Tuple[List[bytes], List[int], List[str]]:
        """
        Process jobs to be completed and collect necessary information.

        Returns:
            Tuple of (job_hashes, squished_capacities, unique_workers).
        """
        all_job_ids: List[bytes] = []
        all_capacities: List[int] = []
        all_workers: List[str] = []

        for job_id in self.jobs_to_complete:
            try:
                job = self.node.dht.query(job_id)
                if not job:
                    continue
                job_hash, capacities, workers = self._process_single_job(job, job_id)
                all_job_ids.append(job_hash)
                all_capacities.extend(capacities)
                all_workers.extend(workers)
            except Exception as e:
                self.node.debug_print(
                    f"Error processing job {job_id}: {e}",
                    colour="yellow",
                    level=logging.WARNING,
                    tag="ContractManager",
                )

        squished: Dict[str, int] = defaultdict(int)
        for worker, cap in zip(all_workers, all_capacities):
            squished[worker] += cap

        return all_job_ids, list(squished.values()), list(squished.keys())

    # -----------------------------------------------------------------------
    # Proposal monitoring / execution
    # -----------------------------------------------------------------------

    def _monitor_and_execute_proposal(self, proposal_hash: str) -> None:
        """Monitor proposal status and execute when ready."""
        try:
            proposal_time, _ = self._get_time_config()
        except Exception as e:
            self.node.debug_print(
                f"Could not fetch time config, skipping proposal monitoring: {e}",
                colour="bright_red",
                level=logging.ERROR,
                tag="ContractManager",
            )
            return

        max_wait_time = int(time.time()) + proposal_time

        while not self.terminate_flag.is_set():
            if int(time.time()) > max_wait_time:
                self.node.debug_print(
                    f"Proposal period expired for {proposal_hash}",
                    colour="yellow",
                    level=logging.INFO,
                    tag="ContractManager",
                )
                return

            try:
                self._rpc_backoff.wait()

                if not self._is_proposal_valid():
                    return

                proposal_number, is_ready = self._is_proposal_ready()
                self._rpc_backoff.success()

                if is_ready:
                    self._execute_proposal(proposal_number, proposal_hash)
                    try:
                        self.node.proposals.append(proposal_hash)
                    except AttributeError:
                        pass
                    return

            except Exception as e:
                if RPCBackoff.is_rate_limit(e):
                    delay = self._rpc_backoff.failure()
                    self.node.debug_print(
                        f"Rate limited while monitoring proposal, "
                        f"backing off {delay:.0f}s",
                        colour="yellow",
                        level=logging.WARNING,
                        tag="ContractManager",
                    )
                else:
                    self._rpc_backoff.success()
                    self.node.debug_print(
                        f"Error monitoring proposal {proposal_hash}: {e}",
                        colour="bright_red",
                        level=logging.ERROR,
                        tag="ContractManager",
                    )

            time.sleep(10)

    def _is_proposal_valid(self) -> bool:
        """Check if the current proposal round is still active."""
        try:
            proposal_id = self.coordinator_contract.functions.nextProposalId().call()
            time.sleep(1)
            return self.current_proposal == proposal_id
        except Exception as e:
            self.node.debug_print(
                f"Could not verify proposal validity: {e}",
                colour="yellow",
                level=logging.WARNING,
                tag="ContractManager",
            )
            return True  # optimistic, keep monitoring

    def _is_proposal_ready(self) -> Tuple[int, bool]:
        """Check if the proposal is ready for execution."""
        proposal_number = self.coordinator_contract.functions.hasSubmittedProposal(
            self.public_key
        ).call()
        time.sleep(1)
        is_ready = self.coordinator_contract.functions.isProposalReady(
            proposal_number
        ).call()
        return proposal_number, is_ready

    def _execute_proposal(self, proposal_number: int, proposal_hash: str) -> bool:
        """Execute the proposal with correct merkle proof generation."""
        try:
            _raw = self.node.dht.query(proposal_hash)
            if _raw is None:
                self.node.debug_print(
                    f"Cannot execute: proposal {proposal_hash} not found in DHT",
                    colour="bright_red",
                    level=logging.ERROR,
                    tag="ContractManager",
                )
                return False

            proposal = ProposalData(**_raw)  # cast untyped DHT result
            merkle_root = HexBytes(proposal["merkle_root"])
            total_capacity = proposal["total_capacity"][0]
            validators = proposal["validators"]
            job_hashes = [HexBytes(j) for j in proposal["job_hashes"]]
            workers_hash = HexBytes(proposal["workers_hash"])
            capacities_hash = HexBytes(proposal["capacities_hash"])

            self._rpc_backoff.wait()

            # Test execution first (dry-run)
            self.coordinator_contract.functions.executeProposal(
                proposal_number,
                merkle_root,
                total_capacity,
                validators,
                job_hashes,
                workers_hash,
                capacities_hash,
            ).call({"from": self.public_key})

            nonce = self.chain.eth.get_transaction_count(self.public_key, "pending")
            execute_tx = self.coordinator_contract.functions.executeProposal(
                proposal_number,
                merkle_root,
                total_capacity,
                validators,
                job_hashes,
                workers_hash,
                capacities_hash,
            ).build_transaction(
                {
                    "from": self.public_key,
                    "nonce": nonce,
                    "gas": 6_721_975,
                    "gasPrice": self.chain.eth.gas_price,
                }
            )

            execute_tx_hash = self._submit_transaction(execute_tx)
            self._rpc_backoff.success()

            self.node.debug_print(
                f"Proposal executed! ({execute_tx_hash.hex()})",
                colour="green",
                level=logging.INFO,
                tag="ContractManager",
            )

            self._clear_completed_items()
            return True

        except Exception as e:
            if RPCBackoff.is_rate_limit(e):
                self._rpc_backoff.failure()
            self._handle_execution_error(e)
            return False

    # -----------------------------------------------------------------------
    # Transaction helpers
    # -----------------------------------------------------------------------

    def _submit_transaction(self, tx: Dict[str, Any]) -> bytes:
        """Sign and broadcast a transaction; return the tx hash."""
        signed_tx = self.chain.eth.account.sign_transaction(
            tx, get_key(".tensorlink.env", "PRIVATE_KEY")
        )
        return self.chain.eth.send_raw_transaction(signed_tx.raw_transaction)

    def _wait_for_next_block(self, timeout: int = 120) -> None:
        """
        Wait for the next blockchain block.

        Args:
            timeout: Maximum seconds to wait before giving up (default 120).
        """
        try:
            current_block = self.chain.eth.block_number
        except Exception:
            time.sleep(15)
            return

        deadline = time.time() + timeout
        while not self.terminate_flag.is_set():
            if time.time() > deadline:
                self.node.debug_print(
                    "Timed out waiting for next block",
                    colour="yellow",
                    level=logging.WARNING,
                    tag="ContractManager",
                )
                return
            try:
                new_block = self.chain.eth.block_number
                if new_block > current_block:
                    return
            except Exception:
                pass
            time.sleep(5)

    # -----------------------------------------------------------------------
    # Job / validator helpers
    # -----------------------------------------------------------------------

    def _is_validator_online(self, node_info: Dict[str, Any]) -> bool:
        """Check if a validator is online and connected to the network."""
        try:
            node_host = node_info["host"]
            node_port = node_info["port"]
            return self.node.connect_node(node_host, node_port, node_info["id"])
        except Exception:
            return False

    def _get_validator_address(self, validator: str) -> Optional[str]:
        """Get the blockchain address for a validator."""
        try:
            return self.node.contract.functions.validatorAddressByHash(validator).call()
        except Exception as e:
            self.node.debug_print(
                f"Could not fetch validator address for {validator}: {e}",
                colour="yellow",
                level=logging.WARNING,
                tag="ContractManager",
            )
            return None

    def _process_single_job(
        self, job: Dict[str, Any], job_id: str
    ) -> Tuple[bytes, List[int], List[str]]:
        """Process a single job and return its data."""
        job_hash = bytes.fromhex(job_id)
        workers: List[str] = []
        capacities: List[int] = []

        distribution = job.get("distribution", {})
        for module_info in distribution.values():
            for worker_id in module_info.get("assigned_workers", []):
                try:
                    worker_info = self.node.dht.query(worker_id)
                    if not worker_info:
                        self.node.debug_print(
                            f"Worker {worker_id} not found in DHT, skipping",
                            colour="yellow",
                            level=logging.WARNING,
                            tag="ContractManager",
                        )
                        continue
                    worker_address = self.chain.to_checksum_address(
                        worker_info["address"]
                    )
                    capacity = round(
                        module_info.get("memory", 0)
                        * (job.get("end_time", 0) - job.get("timestamp", 0))
                        / 3600
                    )
                    workers.append(worker_address)
                    capacities.append(capacity)
                except Exception as e:
                    self.node.debug_print(
                        f"Error processing worker {worker_id} in job {job_id}: {e}",
                        colour="yellow",
                        level=logging.WARNING,
                        tag="ContractManager",
                    )

        return job_hash, capacities, workers

    # -----------------------------------------------------------------------
    # Worker claim data
    # -----------------------------------------------------------------------

    def get_worker_claim_data(self, worker_address: str) -> List[Dict[str, Any]]:
        """
        Get all available claim data for a worker across all proposals.

        Returns:
            List of dicts with keys: distribution_id, worker, capacity,
            merkle_proof, total_capacity, proposal_hash, merkle_root.
        """
        claims: List[Dict[str, Any]] = []
        worker_address = self.chain.to_checksum_address(worker_address)

        try:
            for proposal_hash in self.node.proposals:
                try:
                    proposal_data = self.node.dht.query(proposal_hash)
                    if proposal_data is None:
                        continue

                    if worker_address not in [
                        self.chain.to_checksum_address(w)
                        for w in proposal_data.get("workers", [])
                    ]:
                        continue

                    distribution_id = proposal_data.get("distribution_id")
                    if distribution_id is None:
                        continue

                    participants = build_participants(
                        proposal_data.get("workers", []),
                        proposal_data.get("job_capacities", []),
                        self.chain,
                    )

                    worker_participant = next(
                        (p for p in participants if p["addr"] == worker_address),
                        None,
                    )
                    if not worker_participant:
                        continue

                    try:
                        merkle_proof = generate_merkle_proof(
                            participants, worker_address, self.chain
                        )
                    except Exception as e:
                        self.node.debug_print(
                            f"Failed to generate merkle proof for {worker_address} "
                            f"in distribution {distribution_id}: {e}",
                            colour="yellow",
                            level=logging.WARNING,
                            tag="ContractManager",
                        )
                        continue

                    claims.append(
                        {
                            "distribution_id": distribution_id,
                            "worker": worker_address,
                            "capacity": worker_participant["capacity"],
                            "merkle_proof": [p.hex() for p in merkle_proof],
                            "total_capacity": sum(p["capacity"] for p in participants),
                            "proposal_hash": proposal_hash,
                            "merkle_root": proposal_data["merkle_root"],
                        }
                    )

                except Exception as e:
                    self.node.debug_print(
                        f"Error processing proposal {proposal_hash} for claim: {e}",
                        colour="yellow",
                        level=logging.WARNING,
                        tag="ContractManager",
                    )

        except Exception as e:
            self.node.debug_print(
                f"Error getting claim data for worker {worker_address}: {e}",
                colour="bright_red",
                level=logging.ERROR,
                tag="ContractManager",
            )

        return claims

    def _get_time_config(self) -> Tuple[int, int]:
        """Get proposal timing configuration from contract."""
        return self.coordinator_contract.functions.timeConfig().call()

    def _get_current_round_validators(self) -> List[str]:
        """Get the list of current round validators."""
        return self.coordinator_contract.functions.getCurrentRoundValidators().call()

    def _calculate_next_round_time(self) -> int:
        """Calculate when the next proposal round will start."""
        proposal_time, last_execution_time = self._get_time_config()
        return last_execution_time + proposal_time

    def _wait_for_next_round(self) -> None:
        """Sleep until the next proposal round begins (respects terminate_flag)."""
        try:
            next_round_time = self._calculate_next_round_time()
        except Exception as e:
            self.node.debug_print(
                f"Could not calculate next round time: {e}, sleeping 60s",
                colour="yellow",
                level=logging.WARNING,
                tag="ContractManager",
            )
            time.sleep(60)
            return

        sleep_duration = next_round_time - int(time.time())
        if sleep_duration > 0:
            self.node.debug_print(
                f"Waiting {sleep_duration}s for next proposal round",
                colour="yellow",
                level=logging.INFO,
                tag="ContractManager",
            )
            # Sleep in small increments so terminate_flag is checked regularly
            deadline = time.time() + sleep_duration
            while not self.terminate_flag.is_set() and time.time() < deadline:
                time.sleep(min(10, deadline - time.time()))

    def _is_in_current_round_validators(self) -> bool:
        """Check if this node is in the current round of validators."""
        try:
            current_validators = self._get_current_round_validators()
            return self.public_key in current_validators or not current_validators
        except Exception as e:
            self.node.debug_print(
                f"Could not fetch current round validators: {e}",
                colour="yellow",
                level=logging.WARNING,
                tag="ContractManager",
            )
            return False

    def _get_expected_proposal_count(self) -> int:
        """Get the expected number of proposals for this round."""
        try:
            return len(self._get_current_round_validators())
        except Exception as e:
            self.node.debug_print(
                f"Could not get expected proposal count: {e}",
                colour="yellow",
                level=logging.WARNING,
                tag="ContractManager",
            )
            return 0

    def _handle_execution_error(self, error: Exception) -> None:
        """Log errors during proposal execution."""
        e_str = str(error)
        if "Not enough proposal votes!" in e_str:
            self.node.debug_print(
                "Not enough proposal votes, sleeping...",
                colour="green",
                level=logging.DEBUG,
                tag="ContractManager",
            )
        elif RPCBackoff.is_rate_limit(error):
            self.node.debug_print(
                "Rate limited during proposal execution",
                colour="yellow",
                level=logging.WARNING,
                tag="ContractManager",
            )
        else:
            self.node.debug_print(
                f"Error executing proposal: {e_str}",
                colour="bright_red",
                level=logging.ERROR,
                tag="ContractManager",
            )

    def _clear_completed_items(self) -> None:
        """Clear lists of completed validators and jobs after a successful execution."""
        self.validators_to_clear = []
        self.jobs_to_complete = []

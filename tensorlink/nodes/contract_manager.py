from typing import List, Tuple, Optional, Dict, Any
from web3.exceptions import ContractLogicError
from dotenv import get_key
from eth_abi import encode
from hexbytes import HexBytes
import logging
from collections import defaultdict
import threading
import time


class ContractManager:
    """
    Manages blockchain contract interactions for validator proposals and job management.

    This class handles the creation, submission, voting, and execution of proposals for
    validator removal, job completion, and reward distribution on the blockchain.
    """

    def __init__(self, node, multi_sig_contract, chain, public_key: str):
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

        self.proposals = {}

    def proposal_validator(self):
        """Listen for new proposals created on SmartnodesCoordinator and validate them."""
        while not self.terminate_flag.is_set():
            try:
                # Check if a new round of proposals has started
                current_proposal_id = (
                    self.coordinator_contract.functions.nextProposalId().call()
                )
                time.sleep(1)

                # Update variables for new round of proposals
                if current_proposal_id != self.current_proposal:
                    self.current_proposal = current_proposal_id
                    self.proposals = {}

                # Get expected number of proposals for this round
                expected_proposals = self._get_expected_proposal_count()

                # Validate all proposals in the current round
                for proposal_num in range(1, expected_proposals + 1):
                    if self.terminate_flag.is_set():
                        break

                    try:
                        # Get proposal data
                        proposal_data = self.coordinator_contract.functions.getProposal(
                            proposal_num
                        ).call()
                        proposal_hash = proposal_data[-1].hex()  # hash is at index -1
                        author = proposal_data[0]  # author is at index 0
                        time.sleep(1)

                        # Skip if we've already started validating this proposal
                        if proposal_hash in self.proposals:
                            continue

                        # Start validation thread
                        t = threading.Thread(
                            target=self.validate_proposal,
                            args=(author, proposal_hash, proposal_num),
                            name=f"proposal_validator_{proposal_num}",
                            daemon=True,
                        )
                        self.proposals[proposal_hash] = t
                        t.start()

                    except ContractLogicError:
                        # Proposal has not been published yet, continue to next
                        continue

                # Wait for next round after processing all expected proposals
                self._wait_for_next_round()

            except Exception as e:
                self.node.debug_print(
                    f"Error while fetching created proposals: {e}",
                    colour="bright_red",
                    level=logging.ERROR,
                    tag="ContractManager",
                )

            time.sleep(5)  # Short sleep before checking again

    def validate_proposal(self, author, proposal_hash, proposal_num):
        # TODO if we are the proposal creator (ie a selected validator), automatically cast a vote.
        #  We should also use our proposal data to quickly verify matching data
        self.node.debug_print(
            f"Validation started for proposal: {proposal_hash}",
            colour="bright_blue",
            level=logging.INFO,
            tag="ContractManager",
        )

        # Query network for the detailed proposal info
        proposal_data = self.node.dht.query(proposal_hash)

        # If no proposal is found
        if proposal_data is None:
            # Perhaps some logic to directly query the node who published the proposal to ensure it is not found TODO
            self.node.debug_print(
                f"Validating proposal {proposal_hash} not found in DHT!",
                tag="ContractManager",
            )
            return

        proposal_data_hash = self._hash_proposal_data(proposal_data).hex()

        if proposal_data_hash != proposal_hash:
            # Proposal hash must match smart contract listed proposal
            self.node.debug_print(
                "Invalid proposal hash!", colour="red", tag="ContractManager"
            )
            return

        self._approve_transaction(proposal_num, proposal_hash)

        # for function_type, call_data in proposal_data:
        #     if self.proposal_flag.is_set():
        #         self.debug_print(f"Validator -> Validation interrupted for proposal {proposal_data_hash}!")
        #         return
        #
        #     # Deactivate validator call: attempt connection to determine if validator should be deactivated
        #     if function_type == 0:
        #         node_address = self.chain.to_checksum_address(
        #             decode(["address"], call_data)[0]
        #         )
        #         node_hash = self.contract.functions.getValidatorBytes(node_address).call().hex()
        #         node_info = self.dht.query(node_hash)
        #
        #         if node_info:
        #             node_host, node_port = node_info["host"], node_info["port"]
        #             connected = self.connect_node(node_host, node_port, node_info["id"])
        #
        #             # Verify the roles is online and in the network
        #             if connected:
        #                 flag = True
        #                 reason = f"Validator still online."
        #                 connection = self.nodes[node_info]
        #                 self.close_connection_socket(connection, "Heart beat complete.")
        #
        #         else:
        #             flag = True
        #             reason = f"Node not found: {node_hash}"
        #
        #     # Create job, ensure job is requested on p2p network
        #     elif function_type == 1:
        #         # TODO ensure the job type is for the tensorlink sub-network
        #         user_hash, job_hash, capacities = decode(
        #             ["bytes32", "bytes32", "uint256[]"], call_data
        #         )
        #
        #         user_hash = user_hash.hex()
        #         job_hash = job_hash.hex()
        #
        #         validated = self.validate_job(job_hash, user_id=user_hash, capacities=capacities)
        #
        #         if not validated:
        #             flag = True
        #             reason = f"Invalid job state! {job_hash}"
        #
        #     # Complete job, determine status on p2p network
        #     elif function_type == 2:
        #         job_hash, worker_addresses = decode(
        #             ["bytes32", "address[]"], call_data
        #         )
        #
        #         job_hash = job_hash.hex()
        #
        #         validated = self.validate_job(job_hash, active=False)
        #
        #         if not validated:
        #             flag = True
        #             reason = f"Invalid job state! {job_hash}"
        #
        #     else:
        #         flag = True
        #         reason = "Invalid function type."
        #
        #     if flag is True:
        #         self.debug_print(f"Validator -> Job validation failed: {reason}", colour="bright_red",
        #         level=logging.WARNING)
        #         break

    def _approve_transaction(self, proposal_num: int, proposal_hash: str):
        """Vote for a proposal instead of approving transaction."""
        try:
            tx = self.coordinator_contract.functions.voteForProposal(
                proposal_num
            ).build_transaction(
                {
                    "from": self.public_key,
                    "nonce": self.chain.eth.get_transaction_count(self.public_key),
                    "gas": 6721975,
                    "gasPrice": self.chain.eth.gas_price,
                }
            )
            signed_tx = self.chain.eth.account.sign_transaction(
                tx, get_key(".tensorlink.env", "PRIVATE_KEY")
            )
            tx_hash = self.chain.eth.send_raw_transaction(signed_tx.raw_transaction)
            self.node.debug_print(
                f"Proposal {proposal_num}: {proposal_hash} approved! ({tx_hash.hex()})",
                colour="green",
                level=logging.INFO,
                tag="ContractManager",
            )

        except Exception as e:
            if "Validator has already voted!" in str(e):
                self.node.debug_print(
                    f"Have already voted on proposal {proposal_num}, continuing...",
                    colour="green",
                    level=logging.DEBUG,
                    tag="ContractManager",
                )
                pass
            else:
                raise e

    def add_validator_to_clear(self, validator_id: str) -> None:
        """Add a validator to the list of validators to be cleared."""
        if validator_id not in self.validators_to_clear:
            self.validators_to_clear.append(validator_id)

    def add_job_to_complete(self, job_data: dict) -> None:
        """Add a job to the list of jobs to be completed."""
        # Update job to contract if a certain threshold of time passed
        if (
            job_data["end_time"] - job_data["timestamp"] > 1
            and job_data["id"] not in self.jobs_to_complete
        ):
            self.jobs_to_complete.append(job_data["id"])

    def verify_and_remove_validators(self) -> List[str]:
        """
        Verify validator status and create list of offline validators to remove.

        Returns:
            List[str]: List of validator addresses to be removed
        """
        validators_to_remove = []

        for validator in self.validators_to_clear:
            node_info = self.node.dht.query(validator)
            if not node_info:
                continue

            if not self._is_validator_online(node_info):
                node_address = self._get_validator_address(validator)
                if node_address:
                    validators_to_remove.append(node_address)

        formatted_validators = [
            self.chain.to_checksum_address(v) for v in validators_to_remove
        ]

        return formatted_validators

    def process_jobs(self) -> Tuple[List[bytes], List[int], List[str]]:
        """
        Process jobs to be completed and collect necessary information.

        Returns:
            Tuple containing:
            - List[bytes]: Job hashes
            - List[int]: Job capacities
            - List[str]: Worker addresses
        """
        all_job_ids = []
        all_capacities = []
        all_workers = []

        for job_id in self.jobs_to_complete:
            job = self.node.dht.query(job_id)
            if not job:
                continue

            job_hash, capacities, workers = self._process_single_job(job, job_id)

            all_job_ids.append(job_hash)
            all_capacities.extend(capacities)
            all_workers.extend(workers)

        squished = defaultdict(int)
        for worker, cap in zip(all_workers, all_capacities):
            squished[worker] += cap

        unique_workers = list(squished.keys())
        squished_capacities = list(squished.values())

        return all_job_ids, squished_capacities, unique_workers

    def proposal_creator(self):
        """Create proposals when this node is selected as a round validator."""
        while not self.terminate_flag.is_set():
            try:
                # Check if we're in the current round of validators
                if not self._is_in_current_round_validators():
                    # Wait for next round
                    self._wait_for_next_round()
                    continue

                # Fetch state from the contract
                (next_proposal_id, execution_time, round_validators) = (
                    self.coordinator_contract.functions.getState().call()
                )
                time.sleep(1)
                is_expired = self.coordinator_contract.functions.isRoundExpired().call()
                time.sleep(1)

                # Create proposal if we're a current validator or round is expired
                if self.public_key in round_validators or is_expired:
                    self.create_and_submit_proposal()

                    # Wait for next round after creating proposal
                    self._wait_for_next_round()
                else:
                    # Wait a bit before checking again
                    time.sleep(30)

            except Exception as e:
                self.node.debug_print(
                    f"Error processing new entries: {e}",
                    colour="bright_red",
                    level=logging.ERROR,
                    tag="ContractManager",
                )

            time.sleep(10)

    def create_and_submit_proposal(self) -> None:
        self.node.debug_print(
            "Creating proposal...",
            colour="bright_blue",
            level=logging.INFO,
            tag="ContractManager",
        )

        max_attempts = 3
        proposal = None
        proposal_hash = None

        for attempt in range(1, max_attempts + 1):
            try:
                # Dry run
                self.coordinator_contract.functions.createProposal(
                    encode(["uint256"], [12345])
                ).call({"from": self.public_key})

                # Gather data
                self.node.get_workers()
                validators_to_remove = self.verify_and_remove_validators()
                job_hashes, job_capacities, job_workers = self.process_jobs()

                proposal = {
                    "validators": validators_to_remove,
                    "job_hashes": [j.hex() for j in job_hashes],
                    "job_capacities": job_capacities,
                    "workers": job_workers,
                    "total_capacity": [
                        int(
                            sum(
                                w["total_gpu_memory"]
                                for w in self.node.all_workers.values()
                            )
                        )
                    ],
                    "total_workers": [len(self.node.all_workers)],
                    "distribution_id": self.current_proposal,
                    "timestamp": time.time(),
                }

                participants = self._build_participants(proposal)
                proposal["merkle_root"] = self._build_merkle_tree_from_participants(
                    participants
                ).hex()
                proposal["workers_hash"] = self.chain.keccak(
                    encode(["address[]"], [job_workers])
                ).hex()
                proposal["capacities_hash"] = self.chain.keccak(
                    encode(["uint256[]"], [job_capacities])
                ).hex()

                proposal_hash = self._hash_proposal_data(proposal)
                self.node.dht.store(proposal_hash.hex(), proposal.copy())

                code = self._submit_proposal(proposal_hash)
                if code == 0:
                    break
                elif code == 1:
                    return
                elif code == 2:
                    continue

            except Exception as e:
                msg = str(e).lower()
                if "updatetime - 2min" in msg:
                    self.node.debug_print(
                        f"Waiting for next round (attempt {attempt})",
                        colour="yellow",
                        level=logging.DEBUG,
                        tag="ContractManager",
                    )
                    self._wait_for_next_round()
                    continue
                elif "0xde813857" in msg:
                    self.node.debug_print(
                        "Already submitted proposal!",
                        colour="yellow",
                        level=logging.DEBUG,
                        tag="ContractManager",
                    )
                else:
                    self.node.debug_print(
                        f"Cannot create proposal: {e}",
                        colour="bright_red",
                        level=logging.ERROR,
                        tag="ContractManager",
                    )
                    return
        else:
            self.node.debug_print(
                "Max proposal creation attempts reached",
                colour="bright_red",
                level=logging.ERROR,
                tag="ContractManager",
            )
            return

        if not proposal_hash:
            self.node.debug_print(
                "No valid proposal hash to monitor",
                colour="bright_red",
                level=logging.ERROR,
                tag="ContractManager",
            )
            return

        self._wait_for_next_block()
        self._monitor_and_execute_proposal(proposal_hash.hex())

    def _is_validator_online(self, node_info: Dict[str, Any]) -> bool:
        """Check if a validator is online and connected to the network."""
        node_host, node_port = node_info["host"], node_info["port"]
        return self.node.connect_node(node_host, node_port, node_info["id"])

    def _get_validator_address(self, validator: str) -> Optional[str]:
        """Get the blockchain address for a validator."""
        return self.node.contract.functions.validatorAddressByHash(validator).call()

    def _process_single_job(
        self, job: Dict[str, Any], job_id: str
    ) -> Tuple[bytes, List[int], List[str]]:
        """Process a single job and return its data."""
        job_hash = bytes.fromhex(job_id)
        workers = []
        capacities = []

        for module_info in job["distribution"].values():
            for worker_id in module_info["assigned_workers"]:
                worker_info = self.node.dht.query(worker_id)
                worker_address = self.chain.to_checksum_address(worker_info["address"])
                capacity = round(
                    module_info["memory"] * (job["end_time"] - job["timestamp"]) / 3600
                )  # Capacity measured in byte hours
                workers.append(worker_address)
                capacities.append(capacity)

        return job_hash, capacities, workers

    def _get_worker_address(self, worker_info: dict) -> Optional[str]:
        """Get the blockchain address for a worker."""
        if not worker_info:
            return None

        if not self._is_validator_online(worker_info):
            return None

        # worker_node = self.node.nodes[worker_id]
        # return self.node.query_node(hashlib.sha256(b"ADDRESS").hexdigest(), worker_node)

    def _build_participants(self, proposal_data: dict):
        job_workers = [
            self.chain.to_checksum_address(w) for w in proposal_data.get("workers", [])
        ]
        job_capacities = proposal_data.get("job_capacities", [])

        participants = []
        for i, worker in enumerate(job_workers):
            capacity = job_capacities[i] if i < len(job_capacities) else 0
            participants.append({"addr": worker, "capacity": capacity})

        return participants

    def _hash_proposal_data(self, proposal_data: dict):
        """Hash proposal data according to coordinator contract structure."""
        encoded_data = encode(
            ["bytes32", "address[]", "bytes32[]", "bytes32", "bytes32"],
            [
                HexBytes(proposal_data["merkle_root"]),
                proposal_data["validators"],
                [HexBytes(j) for j in proposal_data["job_hashes"]],
                HexBytes(proposal_data["workers_hash"]),
                HexBytes(proposal_data["capacities_hash"]),
            ],
        )

        return self.chain.keccak(encoded_data)

    def _submit_proposal(self, proposal_hash: bytes) -> int:
        """Submit the proposal to the blockchain."""
        max_retries = 3
        retry_count = 0

        while not self.terminate_flag.is_set() and retry_count < max_retries:
            try:
                # Verify proposal can be submitted
                self.coordinator_contract.functions.createProposal(proposal_hash).call(
                    {"from": self.public_key}
                )

                # Build and submit transaction
                tx = self.coordinator_contract.functions.createProposal(
                    proposal_hash
                ).build_transaction(
                    {
                        "from": self.public_key,
                        "nonce": self.chain.eth.get_transaction_count(self.public_key),
                        "gas": 6_721_975,
                        "gasPrice": self.chain.eth.gas_price,
                    }
                )

                tx_hash = self._submit_transaction(tx)

                self.node.debug_print(
                    f"Proposal ({proposal_hash.hex()}) submitted! ({tx_hash.hex()})",
                    colour="green",
                    level=logging.INFO,
                    tag="ContractManager",
                )
                return 0

            except Exception as e:
                if "0xde813857" in str(e):
                    self.node.debug_print(
                        "Validator has already submitted a proposal this round!",
                        colour="bright_red",
                        level=logging.INFO,
                        tag="ContractManager",
                    )
                    return 0

                elif "updateTime - 2min" in str(e):
                    self.node.debug_print(
                        "Not enough time since last proposal! Waiting for next round...",
                        colour="green",
                        level=logging.DEBUG,
                        tag="ContractManager",
                    )
                    # Wait for next round instead of arbitrary sleep
                    self._wait_for_next_round()
                    return 2
                else:
                    retry_count += 1
                    self.node.debug_print(
                        f"Error creating proposal (attempt {retry_count}): {str(e)}",
                        colour="bright_red",
                        level=(
                            logging.WARNING
                            if retry_count < max_retries
                            else logging.ERROR
                        ),
                        tag="ContractManager",
                    )
                    if retry_count < max_retries:
                        time.sleep(10)  # Short retry delay
                    else:
                        return 1

        return 1

    def _build_proposal_transaction(self, proposal_hash: bytes) -> Dict[str, Any]:
        """Build the proposal transaction."""
        return self.coordinator_contract.functions.createProposal(
            proposal_hash
        ).build_transaction(
            {
                "from": self.public_key,
                "nonce": self.chain.eth.get_transaction_count(self.public_key),
                "gas": 6_721_975,
                "gasPrice": self.chain.eth.gas_price,
            }
        )

    def _submit_transaction(self, tx: Dict[str, Any]) -> str:
        """Submit a transaction to the blockchain."""
        signed_tx = self.chain.eth.account.sign_transaction(
            tx, get_key(".tensorlink.env", "PRIVATE_KEY")
        )
        return self.chain.eth.send_raw_transaction(signed_tx.raw_transaction)

    def _wait_for_next_block(self) -> None:
        """Wait for the next blockchain block."""
        current_block = self.chain.eth.block_number
        while True:
            new_block = self.chain.eth.block_number
            if new_block > current_block:
                break
            time.sleep(5)

    def _monitor_and_execute_proposal(self, proposal_hash: str) -> None:
        """Monitor proposal status and execute when ready."""
        # Calculate the maximum time to wait (proposal period)
        proposal_time, _ = self._get_time_config()
        max_wait_time = int(time.time()) + proposal_time

        while not self.terminate_flag.is_set():
            current_time = int(time.time())

            # Stop monitoring if we've exceeded the proposal period
            if current_time > max_wait_time:
                self.node.debug_print(
                    f"Proposal period expired for {proposal_hash}",
                    colour="yellow",
                    level=logging.INFO,
                    tag="ContractManager",
                )
                return

            if not self._is_proposal_valid():
                return

            proposal_number, is_ready = self._is_proposal_ready()
            if is_ready:
                self._execute_proposal(proposal_number, proposal_hash)
                self.node.proposals.append(proposal_hash)
                return

            time.sleep(10)

    def _is_proposal_valid(self) -> bool:
        """Check if the current proposal is still valid."""
        proposal_id = self.coordinator_contract.functions.nextProposalId.call()
        time.sleep(1)
        return self.current_proposal == proposal_id

    def _is_proposal_ready(self) -> (int, bool):
        """Check if the proposal is ready for execution."""
        proposal_number = self.coordinator_contract.functions.hasSubmittedProposal(
            self.public_key
        ).call()
        time.sleep(1)
        return (
            proposal_number,
            self.coordinator_contract.functions.isProposalReady(proposal_number).call(),
        )

    def _execute_proposal(self, proposal_number: int, proposal_hash: str) -> bool:
        """Execute the proposal with correct merkle proof generation."""
        try:
            proposal = self.node.dht.query(proposal_hash)
            merkle_root = HexBytes(proposal["merkle_root"])
            total_capacity = proposal["total_capacity"][0]
            validators = proposal["validators"]
            job_hashes = [HexBytes(j) for j in proposal["job_hashes"]]
            workers_hash = HexBytes(proposal["workers_hash"])
            capacities_hash = HexBytes(proposal["capacities_hash"])

            # Test execution first
            self.coordinator_contract.functions.executeProposal(
                proposal_number,
                merkle_root,
                total_capacity,
                validators,
                job_hashes,
                workers_hash,
                capacities_hash,
            ).call({"from": self.public_key})

            # Build and execute transaction
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
                    "nonce": self.chain.eth.get_transaction_count(self.public_key),
                    "gas": 6_721_975,
                    "gasPrice": self.chain.eth.gas_price,
                }
            )

            execute_tx_hash = self._submit_transaction(execute_tx)

            self.node.debug_print(
                f"Proposal executed! ({execute_tx_hash.hex()})",
                colour="green",
                level=logging.INFO,
                tag="ContractManager",
            )

            self._clear_completed_items()
            return True

        except Exception as e:
            self._handle_execution_error(e)
            return False

    def _build_execution_transaction(
        self,
        proposal_number: int,
        validators_to_remove: List[str],
        job_hashes: List[bytes],
        job_capacities: List[int],
        job_workers: List[str],
    ) -> Dict[str, Any]:
        """Build the execution transaction"""
        return self.node.multi_sig_contract.functions.executeProposal(
            proposal_number,
            validators_to_remove,
            job_hashes,
            job_workers,
            job_capacities,
        ).build_transaction(
            {
                "from": self.public_key,
                "nonce": self.chain.eth.get_transaction_count(self.public_key),
                "gas": 6_721_975,
                "gasPrice": self.chain.eth.gas_price,
            }
        )

    def _handle_execution_error(self, error: Exception) -> None:
        """Handle errors during proposal execution."""
        if "Not enough proposal votes!" in str(error):
            self.node.debug_print(
                "Not enough proposal votes, sleeping...",
                colour="green",
                level=logging.DEBUG,
                tag="ContractManager",
            )
        else:
            self.node.debug_print(
                f"Error executing proposal: {error}",
                colour="bright_red",
                level=logging.ERROR,
                tag="ContractManager",
            )

    def _clear_completed_items(self) -> None:
        """Clear lists of completed validators and jobs."""
        self.validators_to_clear = []
        self.jobs_to_complete = []

    def _build_merkle_tree_from_participants(self, participants):
        """Build merkle tree from participant data"""
        if not participants:
            return bytes(32)

        # Generate leaves exactly like Solidity: keccak256(abi.encode(addr, capacity))
        leaves = []
        for participant in participants:
            leaf = self.chain.keccak(
                encode(
                    ["address", "uint256"],
                    [
                        self.chain.to_checksum_address(participant['addr']),
                        participant['capacity'],
                    ],
                )
            )
            leaves.append(leaf)

        return self._build_merkle_tree(leaves)

    def _build_merkle_tree(self, leaves):
        """Build merkle tree from leaves"""
        if len(leaves) == 0:
            return bytes(32)
        if len(leaves) == 1:
            return leaves[0]

        current_level = leaves[:]

        while len(current_level) > 1:
            # next_level_size = (len(current_level) + 1) // 2
            next_level = []

            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = (
                    current_level[i + 1]
                    if i + 1 < len(current_level)
                    else current_level[i]
                )

                if left <= right:
                    combined = self.chain.keccak(left + right)
                else:
                    combined = self.chain.keccak(right + left)

                next_level.append(combined)

            current_level = next_level

        return current_level[0]

    def _generate_merkle_proof(self, participants, target_address):
        """Generate Merkle proof for a specific participant - matches Solidity test logic."""
        if not participants:
            return []

        # Find target index
        target_index = None
        for i, participant in enumerate(participants):
            if self.chain.to_checksum_address(
                participant['addr']
            ) == self.chain.to_checksum_address(target_address):
                target_index = i
                break

        if target_index is None:
            raise ValueError(f"Participant {target_address} not found")

        # Generate leaves
        leaves = []
        for participant in participants:
            leaf = self.chain.keccak(
                encode(
                    ["address", "uint256"],
                    [
                        self.chain.to_checksum_address(participant['addr']),
                        participant['capacity'],
                    ],
                )
            )
            leaves.append(leaf)

        if len(leaves) <= 1:
            return []

        proof = []
        current_level = leaves[:]
        current_index = target_index

        while len(current_level) > 1:
            # Determine sibling index
            if current_index % 2 == 0:
                sibling_index = current_index + 1
            else:
                sibling_index = current_index - 1

            # Add sibling to proof
            if sibling_index < len(current_level):
                proof.append(current_level[sibling_index])
            else:
                proof.append(current_level[current_index])

            # Build next level
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = (
                    current_level[i + 1]
                    if i + 1 < len(current_level)
                    else current_level[i]
                )

                if left <= right:
                    combined = self.chain.keccak(left + right)
                else:
                    combined = self.chain.keccak(right + left)

                next_level.append(combined)

            current_level = next_level
            current_index = current_index // 2

        return proof

    def get_worker_claim_data(self, worker_address: str) -> List[Dict[str, Any]]:
        """
        Get all available claim data for a worker across all proposals.
        Returns a list of claim objects, each containing the data needed to claim rewards.

        Args:
            worker_address: The blockchain address of the worker

        Returns:
            List of dictionaries containing claim data, each with:
            - distribution_id: The distribution/proposal ID
            - worker: Checksummed worker address
            - capacity: Worker's capacity for this distribution
            - merkle_proof: Proof array for Merkle tree verification
            - total_capacity: Total capacity for the distribution
            - is_claimed: Whether rewards have already been claimed
        """
        claims = []
        worker_address = self.chain.to_checksum_address(worker_address)

        try:
            # Iterate through all stored proposals
            for proposal_hash in self.node.proposals:
                proposal_data = self.node.dht.query(proposal_hash)
                # Check if worker participated in this proposal
                if worker_address not in proposal_data.get("workers", []):
                    continue

                # Get the distribution ID
                distribution_id = proposal_data.get("distribution_id")
                if distribution_id is None:
                    continue

                # Prepare proposal payload to get participants
                participants = self._build_participants(proposal_data)

                # Find the worker's participation data
                worker_participant = None
                for participant in participants:
                    if (
                        self.chain.to_checksum_address(participant["addr"])
                        == worker_address
                    ):
                        worker_participant = participant
                        break

                if not worker_participant:
                    continue

                # Generate Merkle proof for this worker
                try:
                    merkle_proof = self._generate_merkle_proof(
                        participants, worker_address
                    )
                except Exception as e:
                    self.node.debug_print(
                        f"Failed to generate merkle proof for {worker_address} in distribution {distribution_id}: {e}",
                        colour="yellow",
                        level=logging.WARNING,
                        tag="ContractManager",
                    )
                    continue

                # Calculate total capacity
                total_capacity = sum(p["capacity"] for p in participants)

                # Create claim data object
                claim_data = {
                    "distribution_id": distribution_id,
                    "worker": worker_address,
                    "capacity": worker_participant["capacity"],
                    "merkle_proof": [
                        proof.hex() for proof in merkle_proof
                    ],  # Convert bytes to hex strings
                    "total_capacity": total_capacity,
                    "proposal_hash": proposal_hash,
                    "merkle_root": proposal_data["merkle_root"],
                }

                claims.append(claim_data)

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
        """Wait until the next proposal round begins."""
        next_round_time = self._calculate_next_round_time()
        current_time = int(time.time())

        if next_round_time > current_time:
            sleep_duration = next_round_time - current_time
            self.node.debug_print(
                f"Waiting {sleep_duration} seconds for next proposal round",
                colour="yellow",
                level=logging.INFO,
                tag="ContractManager",
            )
            time.sleep(sleep_duration)

    def _is_in_current_round_validators(self) -> bool:
        """Check if this node is in the current round of validators."""
        current_validators = self._get_current_round_validators()
        return self.public_key in current_validators or not current_validators

    def _get_expected_proposal_count(self) -> int:
        """Get the expected number of proposals for this round."""
        return len(self._get_current_round_validators())

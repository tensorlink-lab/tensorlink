from tensorlink.p2p.connection import Connection
from tensorlink.p2p.torch_node import Torchnode
from tensorlink.nodes.contract_manager import ContractManager
from tensorlink.nodes.job_monitor import JobMonitor
from tensorlink.nodes.keeper import Keeper
from tensorlink.api.node import TensorlinkAPI

from dotenv import get_key
from typing import Dict
import threading
import hashlib
import logging
import queue
import json
import time
import os


FREE_JOB_MAX_TIME = 60 * 30  # 30 minutes in seconds for a free job


class ValidatorThread(Torchnode):
    def __init__(
        self,
        request_queue,
        response_queue,
        print_level=logging.DEBUG,
        max_connections: int = 0,
        upnp=True,
        on_chain=False,
        local_test=False,
        endpoint=True,
        endpoint_ip="0.0.0.0",
        load_previous_state=False,
        priority_nodes: list = None,
        seed_validators: list = None,
    ):
        super(ValidatorThread, self).__init__(
            request_queue,
            response_queue,
            "V",
            max_connections=max_connections,
            upnp=upnp,
            on_chain=on_chain,
            local_test=local_test,
            priority_nodes=priority_nodes,
            seed_validators=seed_validators,
        )

        # Additional attributes specific to the Validator class
        self.print_level = print_level

        self.debug_print(
            f"Launching Validator: {self.rsa_key_hash} ({self.host}:{self.port})",
            level=logging.INFO,
            tag="Validator",
        )

        self.worker_memories = {}
        self.all_workers = {}

        # Params for smart contract state aggregation
        self.proposal_flag = threading.Event()
        self.current_proposal = 0

        self.contract_manager = None
        self.proposal_listener = None
        self.execution_listener = None
        self.endpoint = None
        self.endpoint_requests = {"incoming": [], "outgoing": []}

        # For caching node state and network information
        self.keeper = Keeper(self)
        self.latest_network_status = {}
        self.latest_network_status_time = 0
        self.proposals = []

        if on_chain:
            # Ensure validator is activated on smartnodes first
            self.public_key = get_key(".tensorlink.env", "PUBLIC_KEY")
            if self.public_key is None:
                self.debug_print(
                    "Public key not found in .tensorlink.env file, terminating...",
                    tag="Validator",
                    level=logging.CRITICAL,
                )
                self.terminate_flag.set()

            self.contract_manager = ContractManager(
                self, self.multi_sig_contract, self.chain, self.public_key
            )
            self.dht.store(hashlib.sha256(b"ADDRESS").hexdigest(), self.public_key)

            time.sleep(0.1)
            (is_active, pub_key_hash) = self.contract.functions.getValidatorInfo(
                self.public_key
            ).call()

            if is_active and bytes.hex(pub_key_hash) == self.rsa_key_hash:
                self.current_proposal = (
                    self.multi_sig_contract.functions.nextProposalId.call()
                )

            else:
                self.debug_print(
                    "Validator is inactive on SmartnodesMultiSig or has a different RSA "
                    f"key [expected: {bytes.hex(pub_key_hash)}, received: {self.rsa_key_hash}).",
                    level=logging.CRITICAL,
                    tag="Validator",
                )
                self.terminate_flag.set()

        # Start up the API for handling public jobs
        if endpoint:
            self.endpoint = TensorlinkAPI(self, host=endpoint_ip)
            if not local_test:
                self.add_port_mapping(64747, 64747)

        # Finally, load up previous saved state if any
        if on_chain or load_previous_state:
            self.keeper.load_previous_state()

    def handle_data(self, data, node: Connection):
        """
        Callback function to receive streamed data from worker roles.
        """
        try:
            handled = super().handle_data(data, node)
            ghost = 0

            # Try worker-related tags if not found in parent class
            if not handled:
                # Job acceptance from worker
                if b"ACCEPT-JOB" == data[:10]:
                    return self._handle_accept_job(data, node)
                # Job decline from worker
                elif b"DECLINE-JOB" == data[:11]:
                    return self._handle_decline_job(data, node)
                # Job creation request from user
                elif b"JOB-REQ" == data[:7]:
                    return self._handle_job_req(data, node)

                elif b"REQUEST-WORKERS" == data[:15]:
                    return self._handle_request_workers(data, node)

                elif b"ALL-WORKER-STATS" == data[:16]:
                    return self._handle_worker_aggregation_response(data, node)

                elif b"STATS-RESPONSE" == data[:14]:
                    return self._handle_worker_stats_response(data, node)

                elif b"JOB-UPDATE" == data[:10]:
                    self.debug_print(
                        "User requested update to job structure", tag="Validator"
                    )
                    self.update_job(data[10:])

                elif b"USER-GET-WORKERS" == data[:16]:
                    self.debug_print(
                        "User requested workers.", colour="bright_blue", tag="Validator"
                    )
                    self.request_worker_stats()
                    time.sleep(0.5)
                    stats = {}

                    for worker in self.workers:
                        stats[worker] = self.nodes[worker].stats

                    stats = json.dumps(stats)
                    self.send_to_node(node, b"WORKERS" + stats.encode())

                else:
                    return False

            if ghost > 0:
                node.ghosts += ghost
                # TODO: potentially some form of reporting mechanism via ip and port

            return True

        except Exception as e:
            self.debug_print(
                f"Error handling data: stream_data: {e}",
                colour="bright_red",
                level=logging.ERROR,
                tag="Validator",
            )
            raise e

    def _handle_worker_stats_response(self, data: bytes, node: Connection):
        self.debug_print(
            f"Received stats from worker: {node.node_id}: {json.loads(data[14:])}",
            colour="bright_blue",
            tag="Validator",
        )

        if (
            node.node_id not in self.requests
            or b"STATS" not in self.requests[node.node_id]
        ):
            self.debug_print(
                f"Received unrequested stats from worker: {node.node_id}",
                colour="red",
                level=logging.WARNING,
                tag="Validator",
            )
            node.ghosts += 1

        else:
            stats = json.loads(data[14:])
            self.requests[node.node_id].remove(b"STATS")
            self.nodes[node.node_id].stats = stats
            self.worker_memories[node.node_id] = stats["gpu_memory"]

    def _handle_worker_aggregation_response(self, data: bytes, node: Connection):
        if (
            node.node_id in self.requests.keys()
            and b"ALL-WORKER-STATS" in self.requests[node.node_id]
        ):
            self.requests[node.node_id].remove(b"ALL-WORKER-STATS")
            workers = json.loads(data[16:])
            # TODO thread for aggregation and worker stats aggregation (ie average/most common values)
            for worker, stats in workers.items():
                self.all_workers[worker] = stats

    def _handle_request_workers(self, data: bytes, node: Connection):
        if node.role == "V":
            t = threading.Thread(
                target=self.request_worker_stats,
                args=(node.node_id,),
                daemon=True,
            )
            t.start()

    def handle_requests(self, request=None):
        """Handles interactions between model and node process"""
        try:
            if request is None:
                try:
                    request = self.request_queue.get(timeout=3)
                except queue.Empty:
                    return
            req_type = request.get("type")
            if not req_type:
                self.response_queue.put(
                    {"status": "FAILURE", "error": "Invalid request type"}
                )
                return

            handlers = {
                "get_jobs": self._handle_get_jobs,
                "check_job": self._handle_check_job,
                "send_job_request": self.create_base_job,
                "update_api_request": self._handle_update_api,
                "update_stream": self._handle_update_stream,
                "get_model_demand_stats": self._get_api_demand,
                "get_workers": self._get_workers,
            }

            handler = handlers.get(req_type)

            if handler:
                handler(request.get("args", None))
            else:
                super().handle_requests(request)

        except Exception as e:
            self.response_queue.put({"status": "FAILURE", "error": str(e)})

    def _get_api_demand(self):
        self.response_queue.put(
            {"status": "SUCCESS", "return": self.endpoint.model_name_to_request}
        )

    def _handle_get_jobs(self, request):
        """
        Check if we have received any job requests for huggingface models and fully hosted or api jobs,
        then relay that information back to the DistributedValidator process
        """
        try:
            # Check for job requests coming from API
            if self.rsa_key_hash in self.requests:
                job_req = next(
                    (
                        v
                        for v in self.requests[self.rsa_key_hash]
                        if v.startswith("HF-JOB-REQ")
                    ),
                    None,
                )

                if job_req:
                    # Parse the job data
                    job_data = json.loads(job_req[10:])

                    # Remove the request
                    self._remove_request(self.rsa_key_hash, job_req)

                    self.response_queue.put({"status": "SUCCESS", "return": job_data})
                else:
                    # No jobs found
                    self.response_queue.put({"status": "SUCCESS", "return": None})
            else:
                # No requests for this node
                self.response_queue.put({"status": "SUCCESS", "return": None})
        except Exception as e:
            print(e)
            # Handle any unexpected errors
            self.response_queue.put({"status": "FAILURE", "error": str(e)})

    def _handle_check_job(self, request):
        # Check if a job is still active
        model_name, job_id = request

        job_data = self.dht.query(job_id)
        return_val = job_data.get("active", False)

        self.response_queue.put({"status": "SUCCESS", "return": return_val})

    # def _handle_send_job(self, job_data: dict):
    #     distribution = job_data.get("distribution", {})
    #
    #     if distribution:
    #         worker = distribution

    # # Send the updated job data with worker info to the user
    # self.send_to_node(
    #     requesting_node,
    #     b"ACCEPT-JOB" + job_id.encode() + json.dumps(job_data).encode(),
    # )
    #
    # self.jobs.append(job_id)
    #
    # for module, module_info in job_data["distribution"].items():
    #     # Remove worker info and just replace with id
    #     worker_ids = list(a[0] for a in module_info["workers"])
    #     module_info["workers"] = worker_ids
    #
    # job_data["timestamp"] = time.time()
    # job_data["last_seen"] = time.time()
    #
    # self.dht.store(job_id, job_data)
    #
    # # Start monitor_job as a background task and store it in the list
    # job_monitor = JobMonitor(self)
    # t = threading.Thread(target=job_monitor.monitor_job, args=(job_id,))
    # t.start()

    def create_hf_job(self, job_info: dict, requesters_ip: str = None):
        """
        This can be invoked directly from the API endpoint for a hosted HF model, or via a User
        request for hosting on the user's device. This will trigger HF model inspection in the
        Validator ML process and will create a config of eligible workers and their assigned modules.
        """

        # Rate limitation checks for requested jobs
        if requesters_ip:
            if self.rate_limiter.is_blocked(requesters_ip):
                self.debug_print(
                    f"Job declined! Reason: UserIPBlocked ({requesters_ip})",
                    tag="Validator",
                )
                return False

            self.rate_limiter.record_attempt(requesters_ip)

        if job_info.get("payment", 0) == 0:
            _time = FREE_JOB_MAX_TIME
        else:
            _time = job_info.get("time")

        job_data = job_info
        job_data["time"] = _time

        if not job_data.get("id"):
            job_id = hashlib.sha256(json.dumps(job_data).encode()).hexdigest()
            job_data["id"] = job_id

        # Hand off model dissection and worker assignment to DistributedValidator process
        request_value = "HF-JOB-REQ" + json.dumps(job_data)
        self._store_request(self.rsa_key_hash, request_value)

    def _handle_update_api(self, request: tuple):
        """Checks for and handles any API requests received"""
        # Case 1: ML process is checking for incoming requests
        if len(request) == 2:
            model_name, model_id = request

            if self.endpoint_requests["incoming"]:
                for i, api_request in enumerate(self.endpoint_requests["incoming"]):
                    if not api_request.processing and api_request.hf_name == model_name:
                        api_request.processing = True
                        api_request = self.endpoint_requests["incoming"].pop(i)
                        self.response_queue.put(
                            {"status": "SUCCESS", "return": api_request}
                        )
                        return

            # No matching request found
            self.response_queue.put({"status": "SUCCESS", "return": None})
            return

        # Case 2: ML process is returning completed result
        elif len(request) == 1:
            response = request[0]
            if response.processing:
                self.endpoint_requests["outgoing"].append(response)
                self.response_queue.put({"status": "SUCCESS", "return": None})
            return

        # Invalid request format
        self.response_queue.put(
            {"status": "FAILURE", "error": "Invalid request format"}
        )

    def _handle_update_stream(self, request: tuple):
        """
        Forward streaming tokens from ML process to API endpoint.

        Args:
            request: Tuple of (request_id, token_data)
                token_data = {
                    "token": str,
                    "done": bool,
                    "full_text": str (optional, only when done),
                    "total_tokens": int (optional),
                    "error": str (optional),
                    "timestamp": float
                }
        """
        try:
            if len(request) != 2:
                self.response_queue.put(
                    {"status": "FAILURE", "error": "Invalid stream update format"}
                )
                return

            request_id, token_data = request

            # Forward to API endpoint if it exists
            if self.endpoint and hasattr(self.endpoint, 'send_token_to_stream'):
                self.endpoint.send_token_to_stream(request_id, **token_data)
                self.response_queue.put({"status": "SUCCESS", "return": None})
            else:
                self.response_queue.put(
                    {"status": "FAILURE", "error": "API endpoint not available"}
                )

        except Exception as e:
            self.response_queue.put({"status": "FAILURE", "error": str(e)})

    def _handle_job_req(self, data: bytes, node: Connection):
        """
        This method is invoked by a job request directly from a User. If a model name
        was provided, we call create_hf_job, otherwise we create_base_job
        """
        job_req = json.loads(data[7:])

        # Get author of job listed on SC and confirm job and roles id TODO to be implemented post-alpha
        node_info = self.dht.query(node.node_id)

        self.debug_print(
            f"User: {node.node_id} requested job -> JobRequest({job_req})",
            colour="bright_blue",
            level=logging.INFO,
            tag="Validator",
        )

        if (
            node.role != "U" or not node_info or node_info["reputation"] < 50
        ):  # TODO reputation
            node.ghosts += 1
        elif job_req.get("model_name"):
            threading.Thread(
                target=self.create_hf_job, args=(job_req, node.host)
            ).start()
        else:
            threading.Thread(target=self.create_base_job, args=(job_req,)).start()

    def _handle_decline_job(self, data: bytes, node: Connection):
        self.debug_print(
            f"Worker: {node.node_id} has declined job!",
            colour="red",
            level=logging.INFO,
            tag="Validator",
        )
        # Ensure the request has come in a valid from
        if node.node_id in self.requests and b"JOB-REQ" in self.requests[node.node_id]:
            self.requests[node.node_id].remove(b"JOB-REQ")
        else:
            node.ghosts += 1

    def _handle_accept_job(self, data: bytes, node: Connection):
        job_id = data[10:74].decode()
        module_id = data[74:148].decode()

        # Check that we have requested worker for a job
        if (
            node.node_id in self.requests
            and job_id + module_id in self.requests[node.node_id]
        ):
            self.debug_print(
                f"Worker: {node.node_id} has accepted job!",
                colour="bright_blue",
                level=logging.INFO,
                tag="Validator",
            )
            self.requests[node.node_id].remove(job_id + module_id)

        else:
            node.ghosts += 1

    def check_job_availability(self, job_data: dict):
        """Asserts that the specified user does not have an active job."""
        user_id = job_data.get("author")

        if user_id and user_id != self.rsa_key_hash:
            # Check that user doesn't have an active job already
            user_info = self.dht.query(user_id, keys_to_exclude=[self.rsa_key_hash])

            # Check for active job
            if user_info:
                current_user_job_id = user_info.get("job")

                if current_user_job_id:
                    current_user_job = self.dht.query(current_user_job_id)

                    if current_user_job and current_user_job["active"]:
                        return False

    def create_base_job(self, job_data: dict):
        modules, job_id, author, n_pipelines = self._prepare_job(job_data)
        requesting_node = self._get_requesting_node(job_data, author)
        distribution = job_data.get("distribution", {})

        if not distribution.get("success"):
            self._decline_job(
                job_data, requesting_node, "Could not find enough workers."
            )
            return

        job_data["distribution"] = distribution["config"]

        self.check_job_availability(job_data)

        # Store job info in DHT
        self.dht.store(job_id, job_data)

        # Recruit the workers
        worker_connection_info = self._assign_workers_to_modules(
            modules, author, job_id, job_data
        )

        if worker_connection_info is None:
            return  # job declined inside _assign_workers_to_modules

        self._update_job_distribution(job_data, n_pipelines, requesting_node)

        if requesting_node:
            if any(
                "offloaded" in module_info.get("type", "")
                and (
                    module_info.get("assigned_workers") is None
                    or len(module_info.get("assigned_workers", [])) == 0
                    or module_info.get("assigned_workers")[-1] is None
                )
                for module_info in job_data["distribution"].values()
            ):
                # Decline job workers not present
                self._decline_job(
                    job_data, requesting_node, "Could not find enough workers."
                )
                return

            self._send_acceptance(requesting_node, job_id, job_data)
        else:
            self._setup_hosted_job(job_id, job_data)

        self._finalize_job(job_id, job_data)

    def _prepare_job(self, job_data):
        modules = job_data.get("distribution").copy()
        job_id = job_data.get("id")
        author = job_data.get("author")
        n_pipelines = job_data.get("n_pipelines")

        # If a user has not paid for the job, allocate the default time
        if job_data.get("payment", 0) == 0:
            _time = FREE_JOB_MAX_TIME
            public = True
        else:
            _time = job_data.get("time")
            public = False

        job_data["time"] = _time
        job_data["public"] = public
        job_data["timestamp"] = time.time()

        if modules.get("success"):
            modules = modules.get("config")

        if job_id is None:
            job_id = hashlib.sha256(json.dumps(job_data).encode()).hexdigest()
            job_data["id"] = job_id

        return modules, job_id, author, n_pipelines

    def _get_requesting_node(self, job_data, author):
        if job_data.get("hosted"):
            job_data["author"] = self.rsa_key_hash
            return None
        return self.nodes[author]

    def _decline_job(self, job_data, requesting_node, reason):
        self.debug_print(f"Declining job '{job_data['id']}': {reason}", tag="Validator")
        self.response_queue.put({"status": "SUCCESS", "return": False})
        if requesting_node:
            self.decline_job(requesting_node, reason)

    def _assign_workers_to_modules(self, modules, author, job_id, job_data):
        worker_connection_info = {}
        groups = {}
        job_data["worker_modules"] = {}
        for module_name, module_info in modules.items():
            module_id = hashlib.sha256(json.dumps(module_info).encode()).hexdigest()

            if "offloaded" in module_info.get("type", ""):
                worker_id = module_info.get("assigned_workers", [])[-1]
                groups[module_id] = module_info
                self.recruit_worker(worker_id, author, job_id, module_info, module_id)
                job_data["worker_modules"][worker_id] = module_id
                worker_connection_info[module_id] = worker_id
            else:
                # Hosted modules on our device
                groups[module_id] = module_info

        job_data["distribution"] = groups

        return worker_connection_info

    def _update_job_distribution(self, job_data, n_pipelines, requesting_node):
        pass
        # for module_id, module_info in job_data["distribution"].items():
        #     if len(module_info["workers"]) != n_pipelines:
        #         self._decline_job(
        #             job_data,
        #             requesting_node,
        #             f"Pipeline initialization error! Expected {n_pipelines}, Received {len(module_info['workers'])}.",
        #         )
        #         return None

    def _send_acceptance(self, requesting_node, job_id, job_data):
        self.send_to_node(
            requesting_node,
            b"ACCEPT-JOB" + job_id.encode() + json.dumps(job_data).encode(),
        )

    def _setup_hosted_job(self, job_id, job_data):
        self.debug_print(
            f"Creating public inference job with model {job_data.get('model_name')}",
            tag="Validator",
        )

        for module_id, module_info in job_data["distribution"].items():
            if "offloaded" in module_info.get("type", ""):
                worker_id = module_info["assigned_workers"][0]

                self.modules[module_id] = {
                    "job_id": job_id,
                    "mem_info": module_id,
                    "host": self.rsa_key_hash,
                    "model_name": job_data.get("model_name", ""),
                    "forward_queue": {},
                    "backward_queue": {},
                    "optimizer": None,
                    "training": False,
                    "assigned_workers": [worker_id],
                    "distribution": module_info,
                    "public": job_data.get("public", True),
                }
                self.state_updates[module_id] = []

                if len(self.modules[module_id]["assigned_workers"]) < 1:
                    self.debug_print(
                        f"Network could not find workers for job '{job_id}' module {module_id}.",
                        level=logging.INFO,
                        colour="red",
                        tag="Validator",
                    )
                    self.response_queue.put({"status": "SUCCESS", "return": False})
                    return

    def _finalize_job(self, job_id, job_data):
        self.response_queue.put({"status": "SUCCESS", "return": job_data})

        self.jobs.append(job_id)

        job_data["timestamp"] = time.time()
        job_data["last_seen"] = time.time()

        self.dht.store(job_id, job_data)

        job_monitor = JobMonitor(self)
        t = threading.Thread(target=job_monitor.monitor_job, args=(job_id,))
        t.start()

    def decline_job(self, node, reason: str = None):
        if node:
            if reason is None:
                reason = b""

            self.send_to_node(node, b"DECLINE-JOB" + reason.encode())

    def recruit_worker(
        self,
        worker_id: str,
        user_id: str,
        job_id: str,
        module_info: dict,
        module_id: str,
    ):
        if user_id is None:
            user_id = self.rsa_key_hash

        data = json.dumps([user_id, job_id, module_id, module_info])
        module_size = module_info.get("memory", 0)

        data = b"JOB-REQ" + data.encode()
        node = self.nodes[worker_id]
        self.debug_print(
            f"Attempting to recruit worker: '{worker_id}' for job: '{job_id}'",
            tag="Validator",
        )

        # Send a job request to the worker
        self._store_request(node.node_id, job_id + module_id)
        self.send_to_node(node, data)

        # Await 3 seconds for the job request
        timeout = 3
        start_time = time.time()
        while module_id in self.requests[node.node_id]:
            if time.time() - start_time > timeout:
                self.debug_print(
                    f"Worker: '{worker_id}' timed out during recruitment request.",
                    tag="Validator",
                )
                self.requests[node.node_id].remove(module_id)
                return False

        # Worker accepted the job, update stats
        node.stats["gpu_memory"] -= module_size
        self.debug_print(
            f"Worker: '{worker_id}' recruited for job '{job_id}'", tag="Validator"
        )
        return True

    def get_workers(self):
        """Request information of all workers on the network. Sends request to all current worker connections,
        as well as a request for other validators to do the same."""
        self.all_workers = {}
        self.request_worker_stats()

        for node_id, node in self.nodes.items():
            if node.role == "V":
                self.send_to_node(node, b"REQUEST-WORKERS")
                self._store_request(node_id, "ALL-WORKER-STATS")

        for worker in self.workers:
            if hasattr(self.nodes[worker], "stats"):
                self.all_workers[worker] = self.nodes[worker].stats

    def _get_workers(self, request):
        """Get worker statistics"""
        self.get_workers()

        self.response_queue.put({"status": "SUCCESS", "return": self.all_workers})

    def request_worker_stats(self, send_to=None):
        for worker_id in self.workers:
            connection = self.nodes[worker_id]
            message = b"STATS-REQUEST"
            self.send_to_node(connection, message)
            self._store_request(connection.node_id, b"STATS")

        time.sleep(1)

        if send_to is not None:
            node = self.nodes[send_to]
            workers = {}

            for worker in self.workers:
                stats = self.nodes[worker].stats
                if stats:
                    workers[worker] = stats

            self.send_to_node(node, b"ALL-WORKER-STATS" + json.dumps(workers).encode())

    def distribute_job(self):
        """Distribute job to a few other non-seed validators"""
        for validator in self.validators:
            pass

    #         # Query job information from seed validators
    #         job_responses = [
    #             self.query_node(job_id, validator)
    #             for validator in job_info["seed_validators"]
    #         ]
    #
    #         # Include the user's response if it exists
    #         if user_response:
    #             job_responses.append(user_response)
    #
    #         if len(job_responses) > 0:
    #             # Sort workers and seed_validators for comparison
    #             for response in job_responses:
    #                 if isinstance(response, dict):
    #                     response["workers"] = sorted(response["workers"])
    #                     response["seed_validators"] = sorted(
    #                         response["seed_validators"]
    #                     )
    #
    #             # Create hashable tuples for counting most common responses
    #             normalized_responses = [
    #                 (tuple(response["workers"]), tuple(response["seed_validators"]))
    #                 for response in job_responses
    #             ]
    #
    #             # Count the most common response
    #             response_count = Counter(normalized_responses)
    #             most_common, count = response_count.most_common(1)[0]
    #
    #             percent_match = count / len(job_responses)
    #
    #             if percent_match >= 0.66:
    #                 # Gather the full original responses that match the most common normalized response
    #                 matching_responses = [
    #                     response
    #                     for response in job_responses
    #                     if (
    #                         tuple(response["workers"]),
    #                         tuple(response["seed_validators"]),
    #                     )
    #                     == most_common
    #                 ]

    def run(self):
        super().run()

        if self.on_chain:
            time.sleep(15)
            self.execution_listener = threading.Thread(
                target=self.contract_manager.proposal_creator, daemon=True
            )
            self.execution_listener.start()
            self.proposal_listener = threading.Thread(
                target=self.contract_manager.proposal_validator, daemon=True
            )
            self.proposal_listener.start()

        counter = 0
        # Loop for active job and network moderation
        while not self.terminate_flag.is_set():
            if counter % 300 == 0:
                self.keeper.write_state()
            if counter % 120 == 0:
                self.keeper.clean_node()
                self.clean_port_mappings()
                self.get_workers()
            if counter % 180 == 0:
                self.print_status()

            time.sleep(1)
            counter += 1

    def stop(self):
        self.keeper.write_state()
        super().stop()

    def print_status(self):
        self.print_base_status()
        print(f" Current Proposal: {self.current_proposal}")
        print("=============================================\n")

    def get_tensorlink_status(self):
        # Path to package root (where this file lives)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_path = os.path.join(base_dir, "..", "config", "models.json")

        with open(models_path, "rb") as f:
            models = json.load(f)
            free_models = models["DEFAULT_MODELS"]

        summary = self.keeper.get_network_summary()["current"]

        return {
            "validators": summary["validators"] + 1,
            "workers": summary["workers"],
            "users": summary["users"],
            "jobs": summary["jobs"],
            "proposal": self.current_proposal,
            "available_capacity": summary["available_capacity"],
            "used_capacity": summary["used_capacity"],
            "models": free_models,
        }

    def get_network_status(
        self, days: int = 30, include_weekly: bool = False, include_summary: bool = True
    ) -> Dict:
        return self.keeper.get_network_status(days, include_weekly, include_summary)

import logging
import signal
import sys
import threading
import time
import torch.multiprocessing as mp
from dataclasses import dataclass
from typing import Optional, List

from tensorlink.ml.worker import DistributedWorker
from tensorlink.nodes.user_thread import UserThread
from tensorlink.nodes.validator_thread import ValidatorThread
from tensorlink.nodes.worker_thread import WorkerThread


@dataclass
class BaseNodeConfig:
    upnp: bool = True
    max_connections: int = 0
    on_chain: bool = False
    local_test: bool = False
    print_level: int = logging.INFO
    priority_nodes: Optional[List[str]] = None
    seed_validators: Optional[List[str]] = None


@dataclass
class WorkerConfig(BaseNodeConfig):
    duplicate: str = ""
    load_previous_state: bool = False


@dataclass
class ValidatorConfig(BaseNodeConfig):
    endpoint: bool = True
    endpoint_ip: str = "0.0.0.0"
    load_previous_state: bool = False


@dataclass
class UserConfig(BaseNodeConfig):
    pass


def spinning_cursor():
    """Generator for a spinning cursor animation."""
    for cursor in "|/-\\":
        yield cursor


def show_spinner(stop_event, message="Processing"):
    """
    Displays a spinner in the console.

    Args:
        stop_event (threading.Event): Event to signal when to stop the spinner.
        message (str): The message to display alongside the spinner.
    """
    spinner = spinning_cursor()
    while not stop_event.is_set():
        sys.stdout.write(f"\r{message} {next(spinner)}")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * (len(message) + 2) + "\r")  # Clear the line
    sys.stdout.flush()


mp.set_start_method("spawn", force=True)


class BaseNode:
    def __init__(
        self,
        config: BaseNodeConfig,
        trusted: bool = False,
        utilization: bool = True,
    ):
        self.config = config
        self.trusted = trusted
        self.utilization = utilization

        self.node_requests = mp.Queue()
        self.node_responses = mp.Queue()
        self.mpc_lock = mp.Lock()

        self.node_process = None
        self._stop_event = mp.Event()

        self._setup_signal_handlers()
        self.start()

    def _setup_signal_handlers(self):
        """
        Set up signal handlers for graceful shutdown.
        Uses a multiprocessing Event to signal across processes.
        """

        def handler(signum, frame):
            print(f"Received signal {signum}. Initiating shutdown...")
            self._stop_event.set()
            self.cleanup()
            sys.exit(0)

        # Register handlers for common termination signals
        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT]:
            signal.signal(sig, handler)

    def start(self):
        self.node_process = mp.Process(target=self.run_role, daemon=True)
        self.node_process.start()

    def cleanup(self):
        # Process cleanup
        if self.node_process is not None and self.node_process.exitcode is None:
            # Send a stop request to the role instance
            response = self.send_request("stop", (None,), timeout=15)
            if response:
                self.node_process.join(timeout=15)

            # If the process is still alive, terminate it
            if self.node_process.is_alive():
                print("Forcing termination for node process.")
                self.node_process.terminate()

            # Final join to ensure it's completely shut down
            self.node_process.join()
            self.node_process = None  # Reset to None after cleanup

    def send_request(self, request_type, args, timeout=5):
        """
        Sends a request to the roles and waits for the response.
        """
        request = {"type": request_type, "args": args}

        try:
            self.mpc_lock.acquire(timeout=timeout)
            self.node_requests.put(request)
            response = self.node_responses.get(
                timeout=timeout
            )  # Blocking call, waits for response

        except Exception as e:
            print(f"Error sending '{request_type}' request: {e}")
            response = {"return": str(e)}

        finally:
            self.mpc_lock.release()

        return response["return"]

    def run_role(self):
        raise NotImplementedError("Subclasses must implement this method")

    def connect_node(self, host: str, port: int, node_id: str = None, timeout: int = 5):
        if node_id is None:
            node_id = ""

        self.send_request("connect_node", (host, port, node_id), timeout=timeout)


class Worker(BaseNode):
    def __init__(self, config: WorkerConfig, **kwargs):
        self.mining_active = mp.Value('b', False)
        self.reserved_memory = mp.Value('d', 0.0)
        super().__init__(config, **kwargs)

    def run_role(self):
        node = WorkerThread(
            self.node_requests,
            self.node_responses,
            **vars(self.config),
            mining_active=self.mining_active,
            reserved_memory=self.reserved_memory,
        )

        node.activate()
        node.run()

        while node.is_alive():
            time.sleep(1)

    def start(self):
        super().start()
        distributed_worker = DistributedWorker(self, trusted=self.trusted)
        if self.utilization:
            t = threading.Thread(target=distributed_worker.run, daemon=True)
            t.start()
            time.sleep(1)
        else:
            distributed_worker.run()


class Validator(BaseNode):
    def __init__(self, config: ValidatorConfig, **kwargs):
        super().__init__(config, **kwargs)

    def run_role(self):
        node = ValidatorThread(
            self.node_requests,
            self.node_responses,
            **vars(self.config),
        )

        node.run()

        while node.is_alive():
            time.sleep(1)

    def start(self):
        from tensorlink.ml.validator import DistributedValidator

        super().start()
        distributed_validator = DistributedValidator(self, trusted=self.trusted)
        if self.utilization:
            t = threading.Thread(target=distributed_validator.run, daemon=True)
            t.start()
            time.sleep(3)
        else:
            distributed_validator.run()


class User(BaseNode):
    def __init__(self, config: UserConfig, **kwargs):
        super().__init__(config, **kwargs)

    def run_role(self):
        node = UserThread(
            self.node_requests,
            self.node_responses,
            **vars(self.config),
        )

        node.run()

        while node.is_alive():
            time.sleep(1)

    def cleanup(self):
        """Downloads parameters from workers before shutting down"""
        if hasattr(self, "distributed_model"):
            if self.distributed_model.training:
                self.distributed_model.parameters(distributed=True, load=False)

        super().cleanup()

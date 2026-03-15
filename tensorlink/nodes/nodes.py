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
    """
    Base configuration shared across all Tensorlink node roles.

    Attributes
    ----------
    upnp : bool
        Whether to attempt UPnP port forwarding.
    max_connections : int
        Maximum number of peer connections allowed.
    on_chain : bool
        Whether to interact with the on-chain Smartnodes layer.
    local_test : bool
        Enables local-only networking for testing.
    print_level : int
        Logging verbosity level.
    priority_nodes : Optional[List[List[str]]]
        Preferred peers to connect to first.
    seed_validators : Optional[List[List[str]]]
        Bootstrap validators for network discovery.
    """

    upnp: bool = True
    max_connections: int = 0
    on_chain: bool = True
    local_test: bool = False
    print_level: int = logging.INFO
    priority_nodes: Optional[List[List[str]]] = None
    seed_validators: Optional[List[List[str]]] = None


@dataclass
class WorkerConfig(BaseNodeConfig):
    """
    Configuration specific to Worker nodes.
    """

    duplicate: str = ""
    load_previous_state: bool = False
    max_memory_gb: float = 0


@dataclass
class ValidatorConfig(BaseNodeConfig):
    """
    Configuration specific to Validator nodes.
    """

    endpoint: bool = True
    endpoint_url: str = "0.0.0.0"
    endpoint_port: int = 64747
    load_previous_state: bool = False


@dataclass
class UserConfig(BaseNodeConfig):
    """
    Configuration specific to User nodes.
    """

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
    """
    Base node runner that handles the startup of the P2P node thread
    alongside a distributed ML process (i.e. DistributedWorker,
    DistributedValidator, or DistributedModel).
    """

    def __init__(
        self,
        config: BaseNodeConfig,
        trusted: bool = False,
        utilization: bool = True,
    ):
        """
        Initialize a BaseNode instance.

        Parameters
        ----------
        config : BaseNodeConfig
            Configuration object for the node role.

        trusted : bool, optional
            Whether this node is trusted within the network (bypasses some
            verification or security checks). Default is False.

        utilization : bool, optional
            If True, runs distributed ML logic in a background thread to allow
            concurrent network operation. If False, runs synchronously.
        """
        self.config = config
        self.trusted = trusted
        self.utilization = utilization

        # IPC primitives for communicating with the role process
        self.node_requests = mp.Queue()
        self.node_responses = mp.Queue()
        self.mpc_lock = mp.Lock()

        # Multiprocessing lifecycle handles
        self.node_process = None
        self._stop_event = mp.Event()

        # Install signal handlers and immediately start the node
        self._setup_signal_handlers()
        self.start()

    def _setup_signal_handlers(self):
        """
        Set up OS signal handlers for graceful shutdown.

        Uses a multiprocessing Event to propagate stop signals across processes.
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
        """
        Spawn the multiprocessing role process.
        """
        self.node_process = mp.Process(target=self.run_role, daemon=True)
        self.node_process.start()

    def cleanup(self):
        """
        Gracefully shut down the role process and release resources.
        """
        if self.node_process is not None:
            # Signal process to stop
            self._stop_event.set()

            # Ask the role to stop cleanly first via IPC
            try:
                _ = self.send_request("stop", (None,), timeout=10)
            except Exception as e:
                print(f"Error sending stop request: {e}")

            # Wait for graceful shutdown
            self.node_process.join(timeout=10)

            # Force terminate if still alive
            if self.node_process.is_alive():
                print("Forcing termination for node process.")
                self.node_process.terminate()
                self.node_process.join()

            self.node_process = None

    def send_request(self, request_type, args, timeout=5):
        """
        Send a request to the role process and wait for a response.

        Parameters
        ----------
        request_type : str
            Type of request to send.
        args : tuple
            Arguments to forward to the role.
        timeout : int, optional
            Timeout in seconds for request/response.

        Returns
        -------
        Any
            Value returned by the role handler.
        """
        request = {"type": request_type, "args": args}

        try:
            self.mpc_lock.acquire(timeout=timeout)
            self.node_requests.put(request)

            # Blocking wait for response
            response = self.node_responses.get(timeout=timeout)

        except Exception as e:
            print(f"Error sending '{request_type}' request: {e}")
            response = {"return": str(e)}

        finally:
            self.mpc_lock.release()

        return response["return"]

    def run_role(self):
        """
        Entry point for the multiprocessing role process.

        Subclasses must override this to construct and run the appropriate
        node thread (WorkerThread, ValidatorThread, UserThread).
        """
        raise NotImplementedError("Subclasses must implement this method")

    def connect_node(self, host: str, port: int, node_id: str = None, timeout: int = 5):
        """
        Request a connection to another node in the network.
        """
        if node_id is None:
            node_id = ""

        self.send_request("connect_node", (host, port, node_id), timeout=timeout)


class Worker(BaseNode):
    """
    Tensorlink Worker node runner.

    Workers perform distributed ML execution and communicate with validators
    to run offloaded modules.
    """

    def __init__(self, config: WorkerConfig, **kwargs):
        # Shared state for mining / memory tracking
        self.mining_active = mp.Value("b", False)
        self.max_memory_gb = config.max_memory_gb

        super().__init__(config, **kwargs)

    def run_role(self):
        """
        Launch the WorkerThread inside the role process.
        """
        node = WorkerThread(
            self.node_requests,
            self.node_responses,
            **vars(self.config),
            mining_active=self.mining_active,
        )

        node.run()

        # Keep process alive while the node thread is running
        while node.is_alive():
            time.sleep(1)

    def start(self):
        """
        Start the worker role and the DistributedWorker controller.
        """
        super().start()

        distributed_worker = DistributedWorker(self, trusted=self.trusted)

        if self.utilization:
            t = threading.Thread(target=distributed_worker.run, daemon=True)
            t.start()
            time.sleep(1)
        else:
            distributed_worker.run()


class Validator(BaseNode):
    """
    Tensorlink Validator node runner.

    Validators coordinate jobs, verify execution, and optionally host
    distributed modules.
    """

    def __init__(
        self,
        config: ValidatorConfig,
        enable_hosting: bool = False,
        max_memory_gb: float = 0,
        max_module_bytes: int = 0,
        **kwargs,
    ):
        """
        Initialize a Validator node.

        Parameters
        ----------
        enable_hosting : bool
            Whether this validator may host modules locally.
        max_memory_gb : float
            Maximum VRAM budget for hosted execution.
        max_module_bytes : int
            Maximum module size allowed for hosting.
        """
        self._enable_hosting = enable_hosting
        self._max_vram_gb = max_memory_gb
        self._max_module_bytes = max_module_bytes

        super().__init__(config, **kwargs)

        self.config = config

    def run_role(self):
        """
        Launch the ValidatorThread inside the role process.
        """
        node = ValidatorThread(
            self.node_requests,
            self.node_responses,
            **vars(self.config),
        )

        node.run()

        while node.is_alive():
            time.sleep(1)

    def start(self):
        """
        Start the validator role and DistributedValidator controller.
        """
        from tensorlink.ml.validator import DistributedValidator

        super().start()

        distributed_validator = DistributedValidator(
            self,
            trusted=self.trusted,
            endpoint=self.config.endpoint,
            enable_hosting=self._enable_hosting,
            max_vram_gb=self._max_vram_gb,
            max_module_bytes=self._max_module_bytes,
        )

        if self.utilization:
            t = threading.Thread(target=distributed_validator.run, daemon=True)
            t.start()
            time.sleep(3)
        else:
            distributed_validator.run()


class User(BaseNode):
    """
    Tensorlink User node runner.

    Users submit jobs and interact with distributed models but do not
    perform validation or heavy execution themselves.
    """

    def __init__(self, config: UserConfig, **kwargs):
        super().__init__(config, **kwargs)

    def run_role(self):
        """
        Launch the UserThread inside the role process.
        """
        node = UserThread(
            self.node_requests,
            self.node_responses,
            **vars(self.config),
        )

        node.run()

        while node.is_alive():
            time.sleep(1)

    def cleanup(self):
        """
        Download parameters from workers before shutting down.
        """
        if hasattr(self, "distributed_model"):
            if self.distributed_model.training:
                self.distributed_model.parameters(distributed=True, load=False)

        super().cleanup()

from tensorlink.crypto.rsa import (
    decrypt,
    encrypt,
    authenticate_public_key,
    get_rsa_pub_key,
)
from tensorlink.p2p.connection import Connection
from tensorlink.p2p.monitor import ConnectionMonitor
from tensorlink.p2p.dht import DHT

from logging.handlers import TimedRotatingFileHandler
from dotenv import get_key, set_key
from typing import Tuple, Union, Optional, List
from miniupnpc import UPnP
from web3 import Web3
import hashlib
import ipaddress
import json
import logging
import os
import random
import requests
import socket
import threading
import time


COLOURS = {
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bright_black": "\033[90m",
    "bright_red": "\033[91m",
    "bright_green": "\033[92m",
    "bright_yellow": "\033[93m",
    "bright_blue": "\033[94m",
    "bright_magenta": "\033[95m",
    "bright_cyan": "\033[96m",
    "bright_white": "\033[97m",
}

VERBOSE = 5

# Map logging levels to colors
LEVEL_COLOURS = {
    VERBOSE: "gray",
    logging.DEBUG: "blue",
    logging.INFO: "green",
    logging.WARNING: "yellow",
    logging.ERROR: "red",
    logging.CRITICAL: "bright_red",
}


BACKGROUND_COLOURS = {
    "Validator": "\033[40m",
    "Torchnode": "\033[41m",
    "Smartnode": "\033[44m",
    "User": "\033[45m",
    "Worker": "\033[47m",
    "DHT": "\033[100m",
    "ContractManager": "\033[105m",
    "JobMonitor": "\033[102m",
    "Keeper": "\033[103m",
    "bright_blue": "\033[104m",
    "bright_magenta": "\033[105m",
    "bright_cyan": "\033[106m",
    "bright_white": "\033[107m",
}


# Grab smart contract information
base_dir = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(base_dir, "../config")
SM_CONFIG_PATH = os.path.join(CONFIG_PATH, "SmartnodesCore.json")
MS_CONFIG_PATH = os.path.join(CONFIG_PATH, "SmartnodesCoordinator.json")
TOKEN_CONFIG_PATH = os.path.join(CONFIG_PATH, "SmartnodesERC20.json")

API = get_key(".tensorlink.env", "API")

with open(os.path.join(CONFIG_PATH, "config.json"), "r") as f:
    config = json.load(f)
    CHAIN_URL = API if API else config["api"]["chain-url"]

    CONTRACT = config["api"]["core"]
    MULTI_SIG_CONTRACT = config["api"]["multi-sig"]
    TOKEN = config["api"]["token"]

with open(SM_CONFIG_PATH, "r") as f:
    METADATA = json.load(f)
ABI = METADATA["abi"]

with open(MS_CONFIG_PATH, "r") as f:
    MS_METADATA = json.load(f)
MULTI_SIG_ABI = MS_METADATA["abi"]

with open(TOKEN_CONFIG_PATH, "r") as f:
    TOKEN_METADATA = json.load(f)
TOKEN_ABI = TOKEN_METADATA["abi"]

SNO_EVENT_SIGNATURES = {
    "JobRequest": "JobRequested(uint256,uint256,address[])",
    "JobComplete": "JobCompleted(uint256,uint256)",
    "JobDispute": "JobDisputed(uint256,uint256)",
    "ProposalCreated": "ProposalCreated(uint256,bytes)",
    "ProposalExecuted": "ProposalExecuted(uint256)",
}


# Configure logging with TimedRotatingFileHandler
os.makedirs("logs", exist_ok=True)
os.makedirs("tmp", exist_ok=True)

log_handler = TimedRotatingFileHandler(
    "logs/runtime.log", when="midnight", interval=1, backupCount=7
)
log_handler.setFormatter(logging.Formatter("[%(asctime)s] - %(message)s"))
log_handler.suffix = "%Y%m%d"
logging.getLogger().addHandler(log_handler)
logging.getLogger().setLevel(logging.DEBUG)
BASE_PORT = 38751


def calculate_xor(key_hash, node_id):
    """
    Calculate the XOR distance between a key and a nodes ID.
    """
    return int(key_hash, 16) ^ int(node_id, 16)


def get_public_ip():
    """Get the public IP address of the local machine."""
    try:
        response = requests.get("https://api.ipify.org")
        return response.text
    except requests.RequestException as e:
        print(f"Error retrieving public IP: {e}")
        return None


def is_private_ip(ip):
    """Check if the IP address is private."""
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return False


def get_connection_info(node, main_port=None, upnp=True):
    """Connection info for routing table storage"""
    node_host = node.host

    if is_private_ip(node_host) and upnp:
        node_host = get_public_ip()

    info = {
        "host": node_host,
        "port": node.port if not main_port else main_port,
        "role": node.role,
        "id": node.node_id,
        "reputation": node.reputation,
        "address": node.node_address,
        "last_seen": time.time(),
    }

    return info


def log_entry(node, metadata):
    entry = {"timestamp": time.ctime(), "node": node.node_id, "metadata": metadata}
    logging.info(json.dumps(entry))


def clean():
    for file in os.listdir(os.path.join(os.getcwd(), "tmp")):
        if "streamed_data" in file.title():
            path = os.path.join(os.path.join(os.getcwd(), "tmp"), file)
            os.remove(path)


def _parse_initial_connection(connection: socket.socket) -> dict:
    """Parse and extract initial connection information"""
    id_info = connection.recv(1024)
    _, role, node_address, connected_node_id = json.loads(id_info)
    connected_node_id = connected_node_id.encode()
    node_id_hash = hashlib.sha256(connected_node_id).hexdigest()

    return {
        'role': role,
        'node_address': node_address,
        'node_id': connected_node_id,
        'node_id_hash': node_id_hash,
        'random': _,
    }


class Smartnode(threading.Thread):
    """
    A P2P nodes secured by RSA encryption and smart contract validation for the Smartnodes ecosystem.
    Combines smart contract queries and a kademlia-like DHT implementation for data storage and access.
    """

    def __init__(
        self,
        role,
        max_connections: int = 0,
        upnp: bool = True,
        on_chain: bool = False,
        local_test: bool = False,
        debug_colour=None,
        priority_nodes: list = None,
        seed_validators: list = None,
    ):
        super(Smartnode, self).__init__()

        # Node info
        self.terminate_flag = threading.Event()
        self.connection_listener = None

        # Connection info
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        clean()

        # Get private ip
        if not local_test:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            self.host = s.getsockname()[0]
            s.close()
        else:
            self.host = "127.0.0.1"

        self.port = BASE_PORT
        self.used_ports = set()
        self.max_connections = max_connections
        self.print_level = logging.INFO

        # Connection Settings
        self.upnp = upnp
        self.nodes = {}  # node hash: Connection
        self.max_attempts_per_minute = 5
        self.block_duration = 600
        self.rate_limiter = ConnectionMonitor(
            self.max_attempts_per_minute, self.block_duration
        )

        self.debug_colour = None
        if debug_colour:
            self.debug_colour = debug_colour

        # More parameters for smart contract / p2p info
        self.role = role
        self.rsa_pub_key = get_rsa_pub_key(self.role, True)
        self.rsa_key_hash = hashlib.sha256(self.rsa_pub_key).hexdigest()

        # Stores key of stored values
        self.validators = []
        self.workers = []
        self.users = []
        self.jobs = []
        self.requests = {}

        self.sno_events = {
            name: Web3.keccak(text=sig).hex()
            for name, sig in SNO_EVENT_SIGNATURES.items()
        }
        self.on_chain = on_chain
        self.local_test = local_test

        if local_test:
            self.upnp = False

        self.public_key = None
        self._start_time = time.time()

        # DHT Storage
        self.dht = DHT(self)

        if self.upnp:
            self._init_upnp()

        self.VERBOSE = VERBOSE

        self._init_sock()
        self._priority_nodes = priority_nodes or []
        self._seed_validators = seed_validators or []
        if self.on_chain:
            # Smart nodes parameters for additional security and contract connectivity
            self.url = CHAIN_URL
            self.chain = Web3(Web3.HTTPProvider(CHAIN_URL))
            self.contract_address = Web3.to_checksum_address(CONTRACT)

            # Grab the Smartnode contract
            try:
                self.contract = self.chain.eth.contract(
                    address=self.contract_address, abi=ABI
                )
                self.multi_sig_contract = self.chain.eth.contract(
                    address=MULTI_SIG_CONTRACT, abi=MULTI_SIG_ABI
                )
                self.token = self.chain.eth.contract(address=TOKEN, abi=TOKEN_ABI)

            except Exception as e:
                self.debug_print(
                    f"Could not retrieve contract: {e}",
                    colour="bright_red",
                    level=logging.CRITICAL,
                    tag="Smartnode",
                )
                self.stop()

    def handle_data(self, data: bytes, node: Connection) -> bool:
        """
        Process incoming data from node connections based on message type.

        Args:
            data (bytes): Incoming data packet
            node (Connection): Source node connection

        Returns:
            bool: True if data was processed, False if message type unknown
        """
        try:
            # Handle streamed data completion
            if data.startswith(b"DONE STREAM"):
                return self._handle_stream_completion(data, node)

            # Handle ping/pong communication
            if data.startswith(b"PING"):
                self.send_to_node(node, b"PONG")
                return True

            if data.startswith(b"PONG"):
                return self._handle_pong_response(node)

            # Handle distributed hash table (DHT) value requests and responses
            if data.startswith(b"REQUEST-VALUE-RESPONSE"):
                return self._handle_value_response(data, node)

            if data.startswith(b"REQUEST-VALUE"):
                return self._handle_value_request(data, node)

            # Unknown message type
            return False

        except Exception as e:
            self._log_error(f"Error handling data: {e}")
            return False

    def _handle_stream_completion(self, data: bytes, node: Connection) -> bool:
        """
        Process completed data streams from nodes.

        Args:
            data (bytes): Completed stream data
            node (Connection): Source node connection

        Returns:
            bool: True if stream processed successfully
        """
        file_name = f"tmp/streamed_data_{node.host}_{node.port}_{self.host}_{self.port}"

        # Handle module or parameter stream
        if b"MODULE" in data[11:]:
            os.rename(file_name, data[17:81].decode() + self.rsa_key_hash)
            streamed_bytes = data[11:]
        elif b"PARAMETERS" in data[11:]:
            os.rename(file_name, f"tmp/{data[21:].decode()}_parameters")
            streamed_bytes = data[11:]
        else:
            # Read and remove temporary file for generic streams
            with open(file_name, "rb") as file:
                streamed_bytes = file.read()

            os.remove(file_name)

        # Recursively process streamed data
        self.handle_data(streamed_bytes, node)

        return True

    def _handle_pong_response(self, node: Connection) -> bool:
        """
        Update node latency based on ping-pong timing.

        Args:
            node (Connection): Node that sent pong response

        Returns:
            bool: True if pong processed successfully
        """
        if node.pinged > 0:
            node.ping = time.time() - node.pinged
            node.pinged = -1
        else:
            self._log_warning("Received pong with no corresponding ping")
            node.ghosts += 1

        return True

    def _handle_value_response(self, data: bytes, node: Connection) -> bool:
        """
        Process value response from a DHT node.

        Args:
            data (bytes): Response data packet
            node (Connection): Source node connection

        Returns:
            bool: True if response processed successfully
        """
        # Validate response packet size
        if len(data) < 86:
            self._log_warning(
                f"Received incomplete value response from: {node.node_id}"
            )
            return False

        # Check if we have an active request for this node
        if node.node_id not in self.requests:
            self._log_warning(f"Received unsolicited data from: {node.node_id}")
            return False

        value_id = data[22:86].decode()
        request_key = f"REQUEST-VALUE{value_id}"

        # Process only if this is a known request
        if request_key in self.requests[node.node_id]:
            value = json.loads(data[86:])
            self.requests[node.node_id].remove(request_key)
            self._store_request(value_id, value)
            return True

        self._log_warning(f"Received ghost data from node: {node.node_id}")
        node.ghosts += 1
        return False

    def _handle_value_request(self, data: bytes, node: Connection) -> bool:
        """
        Respond to a value request from another node.

        Args:
            data (bytes): Request data packet
            node (Connection): Requesting node connection

        Returns:
            bool: True if request processed successfully
        """
        # Validate request packet size
        if len(data) < 141:
            return False

        value_hash = data[13:77].decode()
        requester = data[77:141].decode()

        # Query local DHT for requested value
        value = self.dht.query(value_hash, requester)

        # Send response back to requesting node
        response = (
            b"REQUEST-VALUE-RESPONSE" + value_hash.encode() + json.dumps(value).encode()
        )
        self.send_to_node(node, response)

        return True

    def _log_error(self, message: str, tag="Smartnode") -> None:
        """
        Log error messages with appropriate severity.

        Args:
            message (str): Error message to log
        """
        self.debug_print(f"{message}", level=logging.ERROR, tag=tag)

    def _log_warning(self, message: str, tag="Smartnode") -> None:
        """
        Log warning messages with appropriate severity.

        Args:
            message (str): Error message to log
        """
        self.debug_print(f"{message}", level=logging.ERROR, tag=tag)

    def _log_debug(self, message: str, tag="Smartnode") -> None:
        """
        Log debug messages with appropriate severity.

        Args:
            message (str): Error message to log
        """
        self.debug_print(f"{message}", level=logging.DEBUG, tag=tag)

    def debug_print(self, message, level=logging.DEBUG, colour=None, tag=None) -> None:
        """Print to console if debug is enabled"""
        logging.log(level, message)

        if level >= self.print_level:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            role_colour = "\033[37m"
            if self.role == "U":
                role_colour = COLOURS["magenta"]
            elif self.role.startswith("W"):
                role_colour = COLOURS["cyan"]
            elif self.role == "V":
                role_colour = COLOURS["bright_black"]

            if colour is None:
                colour = LEVEL_COLOURS.get(level, "white")

            colour_code = COLOURS.get(colour, "\033[37m")
            reset_colour = "\033[0m"

            tag_width = 15  # Adjust as needed
            if tag:
                centered_tag = tag.center(tag_width)
                background_colour = BACKGROUND_COLOURS.get(tag.strip(), "\033[40m")
                tag = f" {background_colour}{centered_tag}{reset_colour}"
            else:
                tag = " " * (tag_width + 1)

            print(
                f"[{role_colour}{timestamp}{reset_colour}]{tag} -> {colour_code}{message}{reset_colour}"
            )

    """Methods for DHT Query and Storage"""

    def query_node(
        self,
        key_hash: Union[str, bytes],
        node: Connection,
        requester: Union[str, bytes] = None,
        ids_to_exclude: Optional[List] = None,
    ):
        """Query a specific nodes for a value"""
        if requester is None:
            requester = self.rsa_key_hash
        if isinstance(key_hash, bytes):
            key_hash = key_hash.decode()
        if isinstance(requester, bytes):
            requester = requester.decode()

        start_time = time.time()
        message = b"REQUEST-VALUE" + key_hash.encode() + requester.encode()
        self.send_to_node(node, message)

        # Logs what value were requesting and from what nodes
        self._store_request(node.node_id, "REQUEST-VALUE" + key_hash)

        # Blocking wait to receive the data
        while "REQUEST-VALUE" + key_hash in self.requests[node.node_id]:
            # TODO some better timeout management
            # Wait for 3 seconds and then find a new nodes to query
            if time.time() - start_time > 3:
                if ids_to_exclude is not None:
                    if len(ids_to_exclude) > 1:
                        return None
                    ids_to_exclude.append(node.node_id)
                else:
                    ids_to_exclude = [node.node_id]

                # Re route request to the next closest nodes
                self.requests[node.node_id].remove(key_hash)
                return self.dht.query(key_hash, requester, ids_to_exclude)

            if ids_to_exclude and len(ids_to_exclude) > 1:
                return None

        return_val = self.requests[key_hash][-1]

        return return_val

    def _store_request(self, node_id: str, key: str):
        """Stores a log of the request we have made to a nodes and for what value"""
        if node_id in self.requests:
            self.requests[node_id].append(key)
        else:
            self.requests[node_id] = [key]

    def _remove_request(self, node_id: str, key: str):
        if node_id in self.requests:
            self.requests[node_id].remove(key)

    """Peer-to-peer methods"""

    def _listen(self):
        """Listen for incoming connections and initialize custom handshake"""
        while not self.terminate_flag.is_set():
            try:
                if self.sock.fileno() == -1:
                    return

                self.sock.settimeout(2.0)

                try:
                    # Unpack nodes info
                    connection, node_address = self.sock.accept()
                    ip_address = node_address[0]
                except socket.timeout:
                    continue

                # Check rate limiting
                if self.rate_limiter.is_blocked(ip_address):
                    self.close_connection(
                        connection,
                        f"listen: connection refused, rate limit exceeded for {ip_address}",
                    )
                    continue

                self.rate_limiter.record_attempt(ip_address)

                # Attempt custom handshake
                if self.max_connections == 0 or len(self.nodes) < self.max_connections:
                    self._handshake(connection, node_address)

                else:
                    self.close_connection(
                        connection,
                        "listen: connection refused, max connections reached",
                    )

            except socket.timeout:
                pass

            except Exception as e:
                self._log_error(
                    f"Listen connection error {e}",
                    colour="bright_red",
                    level=logging.CRITICAL,
                    tag="Smartnode",
                )

            # self.reconnect_nodes()

    def _handshake(
        self, connection: socket.socket, node_address, instigator=False
    ) -> bool:
        """
        Orchestrate the handshake process with a new node. Validates incoming node's keys via a random
        number swap, along with SC verification of connecting user.

        This handshake method is called by both the instigator (connecting node) and receiving node. Both first
        verify on-chain credentials of the other node and determine if they should connect. After this, the
        connection instigator decrypts a random number generated by the remote node with their private key to
        confirm their identity. This is then reciprocated on the other node.
        """
        try:
            connection.settimeout(10)

            # Receive and parse initial node information
            node_info = _parse_initial_connection(connection)

            # Perform reputation check and role-specific checks
            if not self._validate_node_credentials(node_info, connection):
                return False

            # Verify node's identity via cryptographic proofs
            verified, our_port, new_port, main_port = self._verify_node_identity(
                connection, node_info, instigator, node_address
            )

            if not verified:
                return False

            # Establish the connection
            return self._finalize_connection(
                our_port, new_port, main_port, instigator, node_info, node_address
            )

        except Exception as e:
            self.close_connection(connection, f"Handshake failed: {e}")
            return False

    def _validate_node_credentials(
        self, node_info: dict, connection: socket.socket
    ) -> bool:
        """Perform comprehensive credential validation"""

        # Check node reputation from validator nodes
        if len(self.nodes) > 0 and self.on_chain:
            dht_info = self.dht.query(
                node_info['node_id_hash'], keys_to_exclude=[node_info['node_id_hash']]
            )

            if dht_info and dht_info.get("reputation", 0) < 40:
                self._log_warning(f"Poor reputation: {node_info['node_id_hash']}")
                connection.close()
                return False

        # If we are a worker, only allow connections from users which we have been coordinated to work with
        # by a validator (i.e. node_id_hash is in self.requests)
        if node_info['role'] == "U" and self.role.startswith("W"):
            if node_info['node_id_hash'] not in self.requests:
                self.close_connection(
                    connection, f"Not assigned to user: {node_info['node_id_hash']}"
                )
                return False

        # Public key validation
        if not authenticate_public_key(node_info['node_id']):
            self.close_connection(connection, "Invalid RSA key")
            return False

        # On-chain validation
        return self._validate_on_chain_credentials(node_info, connection)

    def _validate_on_chain_credentials(
        self, node_info: dict, connection: socket.socket
    ) -> bool:
        """Validate node credentials against on-chain information"""
        if not self.on_chain:
            return True

        try:
            if node_info['role'] == "V":
                is_active, pub_key_hash = self.contract.functions.getValidatorInfo(
                    node_info['node_address']
                ).call()

                if not is_active or node_info['node_id_hash'] != bytes.hex(
                    pub_key_hash
                ):
                    self.close_connection(
                        connection, f"Validator not listed: {node_info['node_id']}"
                    )
                    return False

            elif not node_info['role'].startswith("W") and not node_info[
                "role"
            ].startswith("W"):
                self.close_connection(connection, f"Invalid role: {node_info['role']}")
                return False

        except Exception as e:
            self.close_connection(connection, f"Contract query error: {e}")
            return False

        return True

    def _verify_node_identity(
        self,
        connection: socket.socket,
        node_info: dict,
        instigator: bool,
        node_address: list,
    ) -> Tuple[bool, int, int, int]:
        """Perform cryptographic identity verification"""
        rand_n = random.random()
        encrypted_number = encrypt(
            str(rand_n).encode(), self.role, node_info['node_id']
        )

        # Prepare verification message
        if instigator:
            proof = self._handle_instigator_proof(node_info)
            if proof is None:
                return False

            message = json.dumps((self.port, proof, encrypted_number.decode()))

        else:
            message = json.dumps(
                (
                    encrypted_number.decode(),
                    self.role,
                    self.public_key,
                    self.rsa_pub_key.decode(),
                )
            )

        connection.send(message.encode())

        # Verify response
        return self._validate_response(connection, rand_n, instigator, node_address)

    def _handle_instigator_proof(self, node_info: dict) -> float:
        """Handle proof verification for connection instigator"""
        try:
            proof = decrypt(node_info['random'].encode(), self.role)
            return float(proof)
        except Exception as e:
            self._log_warning(f"Proof validation failed: {e}")
            return 0

    def _validate_response(
        self,
        connection: socket.socket,
        expected_rand_n: float,
        instigator: bool,
        node_address: list,
    ) -> Tuple[bool, int, int, int]:
        """
        Validate the response from the remote node. Perform a random number swap, and once confirmed swap
        non-instigator node to a new port.
        """
        try:
            response = connection.recv(1024)

            if instigator:
                main_port = node_address[1]
                new_port, rand_n_proof = json.loads(response)
                rand_n_proof = float(rand_n_proof)
                our_port = connection.getsockname()[1]
                connection.close()
            else:
                # Unpack data from node
                response = json.loads(response)
                main_port, rand_n_proof, verification = response
                verification = decrypt(verification.encode(), self.role)

                # Select a new port for the node to use (since we accepted connection from the listening/main socket)
                our_port = self._get_next_port()
                self._log_debug(f"Selected next port: {our_port} for new connection.")
                self.add_port_mapping(our_port, our_port)

                # Send the new port and proof of random number
                response = json.dumps((our_port, verification.decode()))
                new_port = node_address[1]
                connection.send(response.encode())
                connection.close()

            return rand_n_proof == expected_rand_n, our_port, new_port, main_port

        except Exception as e:
            self._log_warning(f"Response validation failed: {e}")
            return False, 0, 0, 0

    def _finalize_connection(
        self,
        our_port: int,
        new_port: int,
        main_port: int,
        instigator: bool,
        node_info: dict,
        node_address: tuple,
    ) -> bool:
        """Finalize the node connection process"""
        connection = None

        try:
            # Handle the connection differently based on role
            if instigator:
                connection = self._establish_instigator_connection(
                    node_address[0], new_port
                )
            else:
                connection = self._establish_receiver_connection(our_port)
                if not connection:
                    return False

            # Check for existing connection
            if self._is_duplicate_connection(node_address):
                if connection:
                    connection.close()
                return False

            # Create connection thread
            thread_client = self._create_connection(
                connection,
                host=node_address[0],
                port=new_port,
                main_port=main_port,
                node_id=node_info['node_id'],
                role=node_info['role'],
                node_address=node_info['node_address'],
            )
            thread_client.start()

            # Store and categorize node
            return self._store_node_connection(thread_client, node_info)

        except Exception as e:
            self._log_error(f"Connection finalization failed: {e}")
            if connection:
                try:
                    connection.close()
                except Exception as close_error:
                    self._log_error(f"Error closing connection: {close_error}")
            return False

    def _establish_instigator_connection(self, host: str, port: int) -> socket.socket:
        """Establish connection as the instigator"""
        self._log_debug(f"Switching connection to new port: {host}:{port}")

        # Increase wait time to allow receiver to fully set up socket
        time.sleep(2.5)  # Increased from 1 to 2.5 seconds

        new_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        new_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            # Set connect timeout to avoid freezing
            new_sock.settimeout(5)  # Set timeout before connect operation
            new_sock.connect((host, port))
            new_sock.settimeout(10)  # Standard operation timeout after connection
            return new_sock

        except socket.timeout:
            self._log_warning(f"Port swap connection timeout: {host}:{port}")
            new_sock.close()
            raise ConnectionError(f"Connection timeout to {host}:{port}")

        except ConnectionRefusedError:
            self._log_warning(f"Port swap connection refused: {host}:{port}")
            new_sock.close()
            raise ConnectionError(f"Connection refused to {host}:{port}")

        except Exception as e:
            self._log_error(f"Port swap failed ({self.host}:{port}): {e}")
            new_sock.close()
            raise

    def _establish_receiver_connection(self, port: int) -> Optional[socket.socket]:
        """Establish connection as the receiver"""
        self._log_debug(f"Listening for the instigator on the new port: {port}")

        # Create socket earlier to reduce race condition
        new_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        new_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            new_sock.bind((self.host, port))
            new_sock.listen(5)

            # Signal we're ready by sleeping AFTER socket setup
            time.sleep(1)

            # Accept with reasonable timeout
            new_sock.settimeout(10)  # Increased from 6 to 10 seconds
            connection, _ = new_sock.accept()
            return connection

        except socket.timeout:
            self._log_error(f"Timeout waiting for instigator on port {port}")
            new_sock.close()
            return None

        except Exception as e:
            self._log_error(f"Error accepting instigator connection: {e}")
            new_sock.close()
            return None

        finally:
            # Close the listening socket after accepting connection
            new_sock.close()

    def _is_duplicate_connection(self, node_address: tuple) -> bool:
        """Check if we already have a connection to this node"""
        for node in self.nodes.values():
            if node.host == node_address[0] and node.port == node_address[1]:
                self._log_debug(f"Already connected to node: {node.node_id}")
                return True
        return False

    def _store_node_connection(self, thread_client, node_info: dict) -> bool:
        """Store node connection details and categorize"""
        if thread_client.terminate_flag.is_set():
            return False

        # Retrieve or generate node information
        stored_info = self.dht.query(node_info['node_id_hash']) or get_connection_info(
            thread_client,
            main_port=thread_client.main_port,
            upnp=self.upnp,
        )

        # Store node details
        self.nodes[node_info['node_id_hash']] = thread_client
        self.dht.store(node_info['node_id_hash'], stored_info)

        # Categorize node by role
        if node_info["role"] == "V":
            self.validators.append(node_info["node_id_hash"])

        elif node_info["role"].startswith("W"):
            self.workers.append(node_info["node_id_hash"])

            # If we are connecting to a worker (for a job), boost connection speed
            if self.role == "U":
                thread_client.adjust_chunk_size("large")

        elif node_info["role"] == "U":
            self.users.append(node_info["node_id_hash"])

            # If we are connecting to a user (for a job), boost connection speed
            if self.role.startswith("W"):
                thread_client.adjust_chunk_size("large")

        self.debug_print(
            f"Connected to node: {thread_client.host}:{thread_client.port}",
            level=logging.INFO,
            colour="bright_green",
            tag="Smartnode",
        )

        return True

    def _handle_user_connection(self, thread_client, node_info: dict):
        """Special handling for user connections"""
        self.users.append(node_info['node_id_hash'])
        if self.role.startswith("W"):
            thread_client.adjust_chunk_size("large")

    def connect_node(
        self,
        host: str,
        port: int,
        id_hash: Union[bytes, str] = None,
        reconnect: bool = False,
    ) -> bool:
        """
        Attempt to connect to another node in the Smartnodes network.

        This method will:
        - Check if the node is already connected (via `id_hash`).
        - Validate if the given host:port can accept a connection.
        - Attempt up to `max_attempts` (default 3) to establish a TCP connection
          using exponential backoff for retries.
        - Bind a local socket to a free port chosen by `_get_next_port()`.
        - Exchange basic identity/role information with the peer node
          (role, public keys) and perform a handshake.

        Args:
            host (str): Host address of the target node.
            port (int): Port of the target node.
            id_hash (Union[bytes, str], optional): Unique identifier of the target node.
            reconnect (bool, optional): Whether to attempt reconnecting if already connected. Defaults to False.

        Returns:
            bool: True if the handshake and connection succeed, False otherwise.
        """
        # Validate id_hash requirement based on on_chain setting
        if self.on_chain and id_hash is None:
            raise ValueError("id_hash is required when on_chain is True")

        if id_hash is not None and isinstance(id_hash, bytes):
            id_hash = id_hash.decode()

        # Override host for local testing BEFORE any connection checks
        if self.local_test:
            host = "127.0.0.1"

        # Verify if the host/port is reachable before attempting
        _can_connect = self._can_connect(host, port)

        # Avoid duplicate connections
        if id_hash is not None and id_hash in self.nodes and not reconnect:
            self._log_debug(f"connect_node: Already connected to {id_hash}")
            return True

        if _can_connect:
            backoff = 1  # Initial wait time in seconds
            max_attempts = 3  # Number of retry attempts

            for attempt in range(max_attempts):
                try:
                    # Initialize TCP socket
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

                    # Select a free local port for outbound connection
                    our_port = self._get_next_port()
                    self.add_port_mapping(our_port, our_port)
                    self._log_debug(
                        f"Selected next port: {our_port} for new connection"
                    )

                    # Bind locally and connect to target node
                    sock.bind((self.host, our_port))
                    sock.connect((host, port))
                    sock.settimeout(10)  # Prevent hanging connections

                    self._log_debug(f"connect_node: connecting to {host}:{port}")

                    # Send initial identity message (role + keys)
                    message = json.dumps(
                        (None, self.role, self.public_key, self.rsa_pub_key.decode())
                    )
                    sock.send(message.encode())

                    # Perform handshake; closes socket afterwards
                    success = self._handshake(sock, (host, port), instigator=True)
                    sock.close()
                    return success

                except Exception as e:
                    # Retry with exponential backoff
                    wait_time = backoff * (2**attempt)
                    self._log_warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed: {e}. Retrying in {wait_time}s"
                    )
                    time.sleep(wait_time)

                    # Clean up port mapping on failure
                    self.remove_port_mapping(our_port)

        else:
            return False

    def bootstrap(self):
        """
        Bootstrap node to priority nodes. These are specified in the user's config.json file
        or as direct kwargs to the node.
        """
        # Connect with some seed nodes from config file
        if self.on_chain and not self.local_test:
            self.debug_print(
                "Bootstrapping to public network...",
                tag="Smartnode",
                level=logging.INFO,
                colour="cyan",
            )
            for seed_validator in self._seed_validators:
                host, port, id_hash = seed_validator
                connected = self.connect_node(host, port, id_hash)
                if not connected:
                    self.dht.delete(id_hash)

            # # Connect to additional randomly selected validators from the network
            # n_validators = 1
            # sample_size = min(n_validators, 0)
            # for i in [random.randint(1, n_validators) for _ in range(sample_size)]:
            #     # Random validator id
            #     validator_id = random.randrange(1, n_validators + 1)
            #
            #     # Get key validator information from smart contract
            #     validator_contract_info = self.get_validator_info(validator_id)
            #
            #     if validator_contract_info is not None:
            #         is_active, id_hash = validator_contract_info
            #         id_hash = id_hash.hex()
            #         validator_p2p_info = self.dht.query(id_hash)
            #
            #         if validator_p2p_info is None:
            #             self.dht.delete(id_hash)
            #             continue
            #
            #         # Connect to the validator's node and exchange information
            #         # TODO what if we receive false connection info from validator: how to report?
            #         connected = self.connect_node(
            #             validator_p2p_info["host"], validator_p2p_info["port"], id_hash
            #         )
            #
            #         if not connected:
            #             self.dht.delete(id_hash)
            #             continue
            #
            #         candidates.append(validator_id)

        if self._priority_nodes:
            self.debug_print(
                "Connecting priority nodes...",
                tag="Smartnode",
                level=logging.INFO,
                colour="cyan",
            )
            for seed_node in self._priority_nodes:
                host, port = seed_node
                self.connect_node(host, port)

    def _init_sock(self) -> None:
        """Initializes the main socket for handling incoming connections."""
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Check for prior port usage (best to maintain same port for simple bootstrapping)
        port = get_key(".tensorlink.env", self.rsa_key_hash)
        if port is None or not port:
            # If no port is found, get the next available one
            port = self._get_next_port()

        port = int(port)
        self.port = int(port)

        if self.upnp:
            while True:
                result = self.add_port_mapping(
                    port, port
                )  # Forward the port using UPnP
                if result is False:
                    self.port += 1
                    port += 1
                elif result is True:
                    break
                else:
                    raise "Error binding port."

        try:
            self.sock.bind((self.host, port))
        except OSError:
            self.port += 1
            port += 1
            self.sock.bind((self.host, port))

        self.sock.settimeout(3)
        self.sock.listen(5)

        # Update the port in the config if necessary
        set_key(".tensorlink.env", self.rsa_key_hash, str(port))

    def _init_upnp(self) -> None:
        """Enables UPnP on main socket to allow connections"""
        self.upnp = UPnP()
        self.upnp.discoverdelay = 2_000
        devices_found = self.upnp.discover()
        self.upnp.selectigd()

        # Clean up mappings previously created by this application.
        try:
            if devices_found == 0:
                self._log_error("No UPnP devices found.")
                return

            self.clean_port_mappings()

        except Exception as e:
            self._log_error(f"Error during UPnP cleanup: {e}")

        self.add_port_mapping(self.port, self.port)

    def _get_port_identifier(self):
        """Generate a unique identifier for this node's UPnP mappings"""
        # Use first 8 chars of hash to generate a unique identifier
        short_hash = self.rsa_key_hash[:8]
        return f"Smartnode-{short_hash}-{self.role}"

    def add_port_mapping(self, external_port, internal_port):
        """Open up a port via UPnP for a connection"""
        if self.upnp:
            try:
                port_identifier = self._get_port_identifier()
                result = self.upnp.addportmapping(
                    external_port,
                    "TCP",
                    self.upnp.lanaddr,
                    internal_port,
                    port_identifier,
                    "",
                )

                if result:
                    self._log_debug(
                        f"UPnP port forward successful on port {self.port}",
                        tag="Smartnode",
                    )
                    return True
                else:
                    self.debug_print(
                        f"Failed to initialize UPnP. (internal port: {internal_port},"
                        f" external port: {external_port})",
                        level=logging.CRITICAL,
                        tag="Smartnode",
                    )
                    return False

            except Exception as e:
                if "ConflictInMapping" in str(e):
                    self._log_debug(f"Port {external_port} is already mapped.")
                    return False
                else:
                    raise e

    def remove_port_mapping(self, external_port):
        """Close a port via UPnP"""
        if self.upnp:
            try:
                # Attempt to delete the port mapping
                result = self.upnp.deleteportmapping(external_port, "TCP")

                if result is True:
                    self._log_debug(
                        f"Successfully removed UPnP port mapping for external port {external_port}"
                    )
                else:
                    self._log_warning(f"Could not remove port mapping: {result}")
            except Exception as e:
                self._log_error(
                    f"Error removing UPnP port mapping for port {external_port}: {e}"
                )

    def clean_port_mappings(self):
        """
        Lists all port mappings on the UPnP-enabled router.
        """
        mappings = []
        index = 38751

        if not self.upnp:
            self._log_warning("UPnP is not initialized.")
            return mappings

        while True:
            try:
                # Retrieve port mapping at the current index
                mapping = self.upnp.getspecificportmapping(index, "TCP")
                if mapping:
                    _, port, description, _, _ = mapping
                    if description != self._get_port_identifier():
                        self.remove_port_mapping(port)
                index += 1

            except Exception as e:
                # Stop when there are no more entries
                if "SpecifiedArrayIndexInvalid" in str(e):
                    break

                self._log_error(f"Error retrieving port mapping at index {index}: {e}")
                break

            if index > 39_000:
                break

        return mappings

    def get_external_ip(self):
        """Get public IP address"""
        return self.upnp.externalipaddress()

    def _create_connection(
        self,
        connection: socket.socket,
        host: str,
        port: int,
        main_port: int,
        node_id: bytes,
        role: int,
        node_address: str,
    ) -> Connection:
        """Creates a connection thread object from connection.py for individual connections"""
        return Connection(
            self, connection, host, port, main_port, node_id, role, node_address
        )

    def _can_connect(self, host: str, port: int):
        """Makes sure we are not trying to connect to ourselves or a connected nodes"""
        # Check if trying to connect to self
        if host == self.host and port == self.port:
            self._log_warning("connect_with_node: cannot connect with yourself!")
            return False

        # Check if already connected
        for node in self.nodes.values():
            if node.host == host and (node.port == port or node.main_port == port):
                self._log_warning(
                    f"connect_with_node: already connected with node: {node.node_id}"
                )
                return False

        return True

    def send_to_node(self, n: Connection, data: bytes) -> None:
        """Send data to a connected nodes"""
        if n in self.nodes.values():
            self.debug_print(
                f"send_to_node: Sending {len(data)} to node: {n.host}:{n.port}",
                level=self.VERBOSE,
            )
            n.send(data)
        else:
            self._log_warning("send_to_node: node not found!")

    def send_to_node_from_file(self, n: Connection, file, tag):
        if n in self.nodes.values():
            n.send_from_file(file, tag)
        else:
            self._log_warning("send_to_node: node not found!")

    def handle_message(self, node: Connection, data) -> None:
        """Callback method to handles incoming data from connections"""
        self._log_debug(
            f"handle_message from {node.host}:{node.port} -> {data.__sizeof__() / 1e6}MB"
        )

        # Update last seen value
        node_info = self.dht.query(node.node_id)
        if isinstance(node_info, dict):
            node_info["last_seen"] = time.time()

        self.handle_data(data, node)

    def ping_node(self, n: Connection):
        """Measure latency nodes latency"""
        n.pinged = time.time()
        self.send_to_node(n, b"PING")

    def run(self):
        self.connection_listener = threading.Thread(target=self._listen, daemon=True)
        self.connection_listener.start()

    def disconnect_node(self, node_id: str):
        if node_id in self.nodes:
            node = self.nodes[node_id]

            if node_id in self.validators:
                self.validators.remove(node_id)
            elif node_id in self.users:
                self.users.remove(node_id)
            elif node_id in self.workers:
                self.workers.remove(node_id)

            node.terminate_flag.set()
            self.remove_port_mapping(node.port)
            del self.nodes[node_id]

    def close_connection(self, n: socket.socket, additional_info: str = None) -> None:
        message = "closing connection"
        if additional_info:
            message += f": {additional_info}"

        self.debug_print(message, colour="red", tag="Smartnode")
        self.remove_port_mapping(n.getsockname()[1])
        n.close()

    def _stop_upnp(self) -> None:
        """Shuts down UPnP on port"""
        if self.upnp:
            self.clean_port_mappings()
            self._log_debug("_stop_upnp: UPnP cleaned.")

    def stop(self) -> None:
        """Shut down nodes and all associated connections/threads"""
        self.debug_print(
            "Node stopping...",
            colour="bright_yellow",
            level=logging.INFO,
            tag="Smartnode",
        )
        self.terminate_flag.set()

        try:
            self.sock.close()
        except Exception as e:
            self._log_error(f"Error closing socket: {e}")

        for node in list(self.nodes.values()):
            node.stop()

        for node in list(self.nodes.values()):
            node.join()

        self._stop_upnp()
        clean()

        self.debug_print(
            "Node stopped.", colour="bright_yellow", level=logging.INFO, tag="Smartnode"
        )

    """Methods to Interact with Flask Endpoints"""

    def get_self_info(self):
        data = {
            "id": self.rsa_key_hash,
            "validators": [k.decode() for k in self.validators],
            "workers": [k.decode() for k in self.workers],
            "users": [k.decode() for k in self.users],
        }
        return data

    def _get_next_port(self):
        port = self.port

        while True:
            if port not in self.used_ports:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.bind((self.host, port))
                    sock.close()
                    self.used_ports.add(port)
                    return port

                except OSError:
                    port += random.randint(1, 50)

                except Exception as e:
                    raise e
            else:
                port += random.randint(1, 50)

    """Methods for Smart Contract Interactions"""

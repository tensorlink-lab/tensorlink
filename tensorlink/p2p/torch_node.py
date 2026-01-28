from tensorlink.ml.utils import get_gpu_memory
from tensorlink.nodes.shared_memory import get_from_shared_memory
from tensorlink.p2p.connection import Connection
from tensorlink.p2p.smart_node import Smartnode

from multiprocessing import shared_memory
import logging
import queue
import threading
import json
import os
import time
import psutil


MSG_TOKEN = b"TOKEN"
MSG_STREAM_END = b"END__"


def _bar(current, total, width=20):
    if total <= 0:
        return "?" * width
    ratio = min(max(current / total, 0), 1)
    filled = int(width * ratio)
    return "█" * filled + "░" * (width - filled)


def _fmt_gb(x):
    return f"{x / 1e9:.2f}"


def _uptime(start_time):
    s = int(time.time() - start_time)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02}:{m:02}:{s:02}"


class ANSI:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    GRAY = "\033[90m"


def format_size(size_bytes):
    """
    Format the size to display in GB, MB, or KB with one decimal place.
    """
    if size_bytes >= 1e9:
        return f"{round(size_bytes / 1e9, 1)} GB"
    elif size_bytes >= 1e6:
        return f"{round(size_bytes / 1e6, 1)} MB"
    elif size_bytes >= 1e3:
        return f"{round(size_bytes / 1e3, 1)} KB"
    else:
        return f"{size_bytes} bytes"


class Torchnode(Smartnode):
    """"""

    def __init__(
        self,
        request_queue,
        response_queue,
        role,
        max_connections: int = 0,
        upnp=True,
        on_chain=False,
        local_test=False,
        priority_nodes: list = None,
        seed_validators: list = None,
        max_vram_gb: float = 0,
    ):
        super(Torchnode, self).__init__(
            role=role,
            max_connections=max_connections,
            upnp=upnp,
            on_chain=on_chain,
            local_test=local_test,
            priority_nodes=priority_nodes,
            seed_validators=seed_validators,
        )

        # Available GPU mpc estimation
        self._max_vram_gb = max_vram_gb
        self.available_gpu_memory = get_gpu_memory(self._max_vram_gb)
        self.total_gpu_memory = self.available_gpu_memory
        self.available_ram = psutil.virtual_memory().available

        self._mpc_comms = None
        self.memory_manager = {}
        self.request_queue = request_queue
        self.response_queue = response_queue

        # Pointers to model parameters in DistributedModels
        self.modules = {}
        self.state_updates = {}

        # Master flag for handling different types of storage as master
        self.master = False
        self.mpc_terminate_flag = threading.Event()

    def handle_data(self, data: bytes, node: Connection):
        try:
            # Call parent class's handle_data, if applicable
            handled = super().handle_data(data, node)

            if not handled:
                # Define a dictionary mapping prefixes to handler methods
                handlers = {
                    b"LOADED": self._handle_module_loaded,
                    b"FORWARD": self._handle_forward,
                    b"BACKWARD": self._handle_backward,
                    b"GENERATE": self._handle_generate,
                    b"OPTIMIZER-RESPONSE": self._handle_optimizer_response,
                    b"OPTIMIZER": self._handle_optimizer_request,
                    b"PARAMS-REQ": self._handle_parameters_request,
                    b"PARAMETERS": self._handle_parameters,
                    b"MODULE": self._handle_module,
                    b"UPDATE-TRAIN": self._update_train,
                    b"TRAIN-UPDATED": self._train_updated,
                }

                # Iterate through handlers to find the matching prefix
                for prefix, handler in handlers.items():
                    if data.startswith(prefix):
                        return (
                            handler(data, node)
                            if prefix
                            in (
                                b"LOADED",
                                b"FORWARD",
                                b"GENERATE",
                                b"BACKWARD",
                                b"OPTIMIZER-RESPONSE",
                                b"OPTIMIZER",
                                b"MODULE",
                                b"UPDATE-TRAIN",
                            )
                            else handler(data)
                        )

                return False

        except Exception as e:
            self._log_error(f"Error handling data: {e}", tag="Torchnode")

    def _train_updated(self, data: bytes):
        mode = False if data[13:14] == b"0" else True
        module_id = data[14:78].decode()
        if module_id in self.modules:
            self.modules[module_id]["training"] = mode

    def _update_train(self, data: bytes, node: Connection):
        mode = False if data[12:13] == b"0" else True
        module_id = data[13:77].decode()
        self.modules[module_id]["training"] = mode
        self.send_train_updated(node, mode, module_id)

    def _handle_parameters(self, data: bytes):
        module_id = data[10:74].decode()
        self._log_debug(f"Received Parameters for: {module_id}", tag="Torchnode")

        file_name = f"tmp/{module_id}_parameters"
        key = "PREQPREQPREQ" + module_id
        self.memory_manager[key] = file_name

        return True

    def _handle_parameters_request(self, data: bytes):
        self._log_debug("RECEIVED PARAMS REQUEST", tag="Torchnode")

        # TODO Must ensure requesting node is indeed the master or an overseeing validator
        module_id = data[10:74].decode()
        self.memory_manager["PREQPREQPREQ" + module_id] = True
        return True

    def _handle_optimizer_request(self, data: bytes, node: Connection):
        if self.role == "V" or node.node_id not in self.nodes:
            node.ghosts += 1
            return False
        else:
            module_id, optimizer_fn, optimizer_kwargs = json.loads(data[9:])
            self.state_updates[module_id].append((optimizer_fn, optimizer_kwargs))
            return True

    def _handle_optimizer_response(self, data: bytes, node: Connection):
        if self.role == "V" or node.node_id not in self.nodes:
            node.ghosts += 1
            return False
        else:
            module_id, response_type = json.loads(data[18:])

            if response_type == "loaded":
                self.debug_print(
                    f"Optimizer for module: {module_id} loaded on worker {node.node_id}",
                    colour="bright_cyan",
                    tag="Torchnode",
                )
            elif response_type == "stepped":
                self.debug_print(
                    f"Optimizer for module: {module_id} stepped on worker {node.node_id}",
                    colour="bright_cyan",
                    tag="Torchnode",
                )
            elif response_type == "zeroed":
                self.debug_print(
                    f"Optimizer for module: {module_id} zeroed on worker {node.node_id}",
                    colour="bright_cyan",
                    tag="Torchnode",
                )

            self.state_updates[module_id].append(response_type + node.node_id)
            return True

    def _handle_backward(self, data: bytes, node: Connection):
        # Basic check, must be upgraded to check if we are expecting the request
        if self.role == "V" or node.node_id not in self.nodes:
            node.ghosts += 1
            return False
        else:
            # Find size parameter within bytes
            eos = data.find(b"::")
            size = int(data[8:eos])

            formatted_size = format_size(size)
            self.debug_print(f"RECEIVED BACKWARD: {formatted_size}", tag="Torchnode")

            # TODO we must check that the forward received corresponds to a sent pass/specific module
            # must also do with backwards
            tensor = data[eos + 2 : eos + 2 + size]
            key = tuple(json.loads(data[eos + 2 + size :]))

            # Create shared mpc block and store tensor
            self._store_tensor_in_shared_memory(key, tensor, backward=True)
            return True

    def _handle_forward(self, data: bytes, node: Connection):
        """Handle a received forward pass from a node"""
        # Basic check, must be upgraded to check if we are expecting the request
        if node.node_id not in self.nodes:
            node.ghosts += 1
            return False

        # Received a forward pass
        eos = data.find(b"::")
        size = int(data[7:eos])
        formatted_size = format_size(size)
        self.debug_print(f"RECEIVED FORWARD: {formatted_size}", tag="Torchnode")

        # TODO we must check that the forward received corresponds to a sent pass/specific module
        # must also do with backwards
        tensor = data[eos + 2 : eos + 2 + size]
        payload = json.loads(data[eos + 2 + size :])

        if isinstance(payload, dict):
            module_id = payload.get("module_id")
            key = payload.get("key")
        else:
            module_id = None
            key = payload

        if not isinstance(key, str):
            key = tuple(key)
            # Create shared mpc block and store tensor
            self._store_tensor_in_shared_memory(key, tensor)
            return True

        if module_id not in self.modules:
            self.debug_print(
                f"Unknown module_id in forward: {module_id}", tag="Torchnode"
            )
            return False

        shm = shared_memory.SharedMemory(create=True, size=size)
        buffer = shm.buf[:size]
        buffer[:] = tensor

        self.modules[module_id]["forward_queue"][key] = (size, shm.name)
        self.memory_manager[key] = shm.name

        del buffer
        shm.close()
        return True

    def _handle_generate(self, data: bytes, node: Connection):
        # Received a forward pass
        self.debug_print("RECEIVED GENERATE", tag="Torchnode")

        # Unpack data
        module_id_size = 64
        header_size = 8
        stream_flag_size = 1
        module_id = data[
            header_size
            + stream_flag_size : module_id_size
            + header_size
            + stream_flag_size
        ]
        stream_flag = data[header_size]
        payload = data[module_id_size + header_size + stream_flag_size :]

        stream = bool(stream_flag)
        key = module_id.decode()
        size = len(payload)

        shm = shared_memory.SharedMemory(create=True, size=size)
        buffer = shm.buf[:size]
        buffer[:size] = payload

        self.modules[key]["forward_queue"][key] = (size, shm.name, stream)
        self.memory_manager[key] = shm.name
        del buffer
        shm.close()
        return True

    def _handle_module(self, data: bytes, node: Connection):
        """
        Load a module sent by a validator node
        """
        module_id = data[6:70].decode()
        file_name = module_id + self.rsa_key_hash
        if os.path.exists(file_name):
            try:
                with open(file_name, "rb") as f:
                    module_info = json.load(f)
            except json.JSONDecodeError:
                module_info = json.loads(data[70:])

            os.remove(file_name)

        else:
            module_info = json.loads(data[70:])

        request_to_remove = []

        if node.node_id in self.requests:
            for req in self.requests[node.node_id]:
                if module_id in req or (
                    isinstance(req, dict) and module_id == req["id"]
                ):
                    request_to_remove.append(req)

                if "OPTIMIZER" in req:
                    request_to_remove.append(req)

            for req in request_to_remove:
                self._remove_request(node.node_id, req)

            self.debug_print(
                f"Loading distributed module: {module_info}",
                colour="bright_cyan",
                tag="Torchnode",
            )

            module_info["mem_info"] = module_id
            module_info["host"] = node.node_id
            module_info["forward_queue"] = {}
            module_info["backward_queue"] = {}
            module_info["status"] = "loading"

            self.modules[module_id] = module_info

            self.state_updates[module_id] = []
            return True

        else:
            node.ghosts += 1

        return False

    def _handle_module_loaded(self, data: bytes, node: Connection):
        """Remove load module request to signal to distributed process"""
        self.debug_print(
            f"Successfully offloaded submodule to: {node.node_id}",
            level=logging.INFO,
            colour="bright_cyan",
            tag="Torchnode",
        )
        module_id = data[6:70].decode()
        self._remove_request(node.node_id, "MODULE" + module_id)
        return True

    def handle_requests(self, request=None):
        """Handles interactions between model and node processes."""
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
                "get_connection": self._handle_get_connection,
                "send_model": self._handle_send_model,
                "check_loaded": self._handle_check_module_loaded,
                "module_loaded": self._handle_module_loaded_request,
                "optimizer_response": self._handle_optimizer_response_request,
                "send_forward": self._handle_send_forward,
                "send_backward": self._handle_send_backward,
                "send_parameters": self._handle_send_parameters,
                "check_module": self._handle_check_module,
                "check_module_request": self._handle_check_module_request,
                "check_forward": self._handle_check_forward,
                "check_generate": self._handle_check_generate,
                "check_backward": self._handle_check_backward,
                "send_optimizer_request": self._handle_send_optimizer_request,
                "check_state_update": self._handle_check_state_update,
                "check_validators": self._handle_check_validators,
                "check_parameters_request": self._handle_check_parameters_request,
                "check_parameters": self._handle_check_parameters,
                "request_parameters": self._handle_request_parameters,
                "update_train": self._handle_update_train,
                "check_train": self._handle_check_train,
                "release_memory": self._handle_release_memory,
                "check_shutdown": self._handle_check_shutdown,
                "stop": self._handle_stop,
                "connect_node": self._handle_connect_node,
                "info": self._handle_get_info,
                "debug_print": self._handle_debug_print,
                "generate": self._handle_send_generate,
                "send_token": self._handle_send_token,
                "send_stream_end": self._handle_send_stream_end,
            }

            handler = handlers.get(req_type)

            if handler:
                handler(request)
            else:
                self.response_queue.put(
                    {"status": "FAILURE", "error": f"Unknown request type: {req_type}"}
                )
        except Exception as e:
            self.response_queue.put({"status": "FAILURE", "error": str(e)})

    def _handle_get_connection(self, request):
        # Get connection info from a node id
        node_id = request["args"]
        node = self.nodes[node_id]
        self.response_queue.put({"status": "SUCCESS", "return": node})

    def _handle_send_model(self, request):
        # Send module that is stored in shared mpc to another node
        name, worker_id, module_id, module_info = request["args"]
        node = self.nodes[worker_id]
        node.adjust_chunk_size("large")
        self.send_module(name, module_id, module_info, node)
        self.response_queue.put({"status": "SUCCESS", "return": None})

    def _handle_check_module_loaded(self, request):
        # Check if sent module has been received and loaded on the other nodes
        worker_id, module_id = request["args"]
        return_val = False

        if "MODULE" + module_id not in self.requests[worker_id]:
            return_val = True

        self.response_queue.put({"status": "SUCCESS", "return": return_val})

    def _handle_module_loaded_request(self, request):
        """
        Send module loaded message from worker back to a validator
        """
        module_id = request["args"]
        module = self.modules[module_id]
        node_id = module["host"]
        module["status"] = "loaded"
        node = self.nodes[node_id]
        self.send_to_node(node, b"LOADED" + module_id.encode())
        self.response_queue.put({"status": "SUCCESS", "return": None})

    def _handle_optimizer_response_request(self, request):
        """
        Send response after an update to the distributed optimizer was called
        """
        module_id, response_type = request["args"]
        node_id = self.modules[module_id]["host"]
        node = self.nodes[node_id]

        self.send_to_node(
            node,
            b"OPTIMIZER-RESPONSE" + json.dumps((module_id, response_type)).encode(),
        )

        self.response_queue.put({"status": "SUCCESS", "return": None})

    def _handle_send_forward(self, request):
        # Send forward pass tensor from shared mpc to a node
        worker_id, module_id, size, shm_name, tag = request["args"]
        node = self.nodes[worker_id]
        forward_bytes = get_from_shared_memory(size, shm_name, encoded=True)
        self.send_forward(node, forward_bytes, tag, module_id)
        self.response_queue.put({"status": "SUCCESS", "return": None})

    def _handle_send_generate(self, request):
        node_id, size, shm_name, stream = request["args"]
        node = self.nodes[node_id]
        generate_bytes = get_from_shared_memory(size, shm_name, encoded=True)
        stream_flag = b"\x01" if stream else b"\x00"

        packet = b"GENERATE" + stream_flag + generate_bytes
        self.send_to_node(node, packet)

        self.response_queue.put({"status": "SUCCESS", "return": None})

    def _handle_send_token(self, request):
        module_id, token, host_id = request["args"]
        node = self.nodes[host_id]

        token_bytes = token.to_bytes(4, byteorder="big", signed=True)
        bytes_to_send = MSG_TOKEN + module_id.encode() + b"|" + token_bytes
        self.send_to_node(node, bytes_to_send)

        self.response_queue.put({"status": "SUCCESS", "return": None})

    def _handle_send_stream_end(self, request):
        module_id, host_id = request["args"]
        node = self.nodes[host_id]

        end_bytes = MSG_STREAM_END + module_id.encode()
        self.send_to_node(node, end_bytes)

        self.response_queue.put({"status": "SUCCESS", "return": None})

    def _handle_send_backward(self, request):
        # Send backwards pass from shared mpc to a node
        worker_id, size, shm_name, tag = request["args"]
        node = self.nodes[worker_id]
        backward_bytes = get_from_shared_memory(size, shm_name, encoded=True)
        self.send_backward(node, backward_bytes, tag)
        self.response_queue.put({"status": "SUCCESS", "return": None})

    def _handle_send_parameters(self, request):
        node_id, module_id = request["args"]
        node = self.nodes[node_id]
        self.send_to_node_from_file(
            node, f"parameters_{module_id}", b"PARAMETERS" + module_id.encode()
        )
        self.response_queue.put({"status": "SUCCESS", "return": None})

    def _handle_check_module(self, request):
        """
        Invoked by a worker or validator ML process to see if there are any significant state
        changes to any modules (ie loading or termination).
        """
        if self.role == "V":
            return_val = {
                "job_id": request["args"],
                "distribution": {},
                "model_name": None,
                "optimizer": None,
                "training": False,
            }
        else:
            return_val = None

        try:
            for module_id, module in self.modules.items():
                # "mem_info" is added to module info upon initially receiving it
                if "mem_info" in module:
                    # Return the module info to the ML process
                    if self.role == "V":
                        if return_val.get("job_id") == module.get("job_id"):
                            return_val["distribution"][module_id] = module[
                                "distribution"
                            ]
                            return_val["model_name"] = module.get("model_name", "")
                            return_val["optimizer"] = module["optimizer"]
                            return_val["training"] = module["training"]
                    else:
                        return_val = module
                        return_val["module_id"] = module_id

                    del module["mem_info"]

                # "termination" is added to module info when the job is closing
                elif "termination" in module:
                    return_val = module_id
                    del self.modules[module_id]
                    break

            self.response_queue.put({"status": "SUCCESS", "return": return_val})

        except Exception as e:
            self._log_error(f"Error handling module: {e}")
            self.response_queue.put({"status": "FAILURE", "return": None})

    def _handle_check_module_request(self, request):
        request_type, worker_id, module_id = request["args"]
        return_val = False
        key = None

        if module_id in self.state_updates.keys():
            key = request_type + worker_id

        if key in self.state_updates[module_id]:
            self.state_updates[module_id].remove(key)
            return_val = True

        self.response_queue.put({"status": "SUCCESS", "return": return_val})

    def _handle_check_generate(self, request):
        return_val = None
        module_id = request["args"]
        if module_id in self.modules:
            if "generate" in self.modules[module_id]["forward_queue"]:
                return_val = self.modules[module_id]["forward_queue"]["generate"]
                del self.modules[module_id]["forward_queue"]["generate"]

        self.response_queue.put({"status": "SUCCESS", "return": return_val})

    def _handle_check_forward(self, request):
        # Check if forward pass has been received and is loaded in shared mpc
        return_val = None

        if self.role.startswith("W"):
            module_id = request["args"]

            if module_id in self.modules:
                module = self.modules[module_id]
                if module_id in module["forward_queue"].keys():
                    return_val = (module_id, module["forward_queue"][module_id])
                    del module["forward_queue"][module_id]

                else:
                    min_iter, min_micro = -1, -1
                    for n_iter, n_micro, module_id in module["forward_queue"].keys():
                        if n_iter <= min_iter or min_iter == -1:
                            min_iter = n_iter
                        if n_micro <= min_micro or min_micro == -1:
                            min_micro = n_micro

                    key = (min_iter, min_micro, module_id)

                    if key in module["forward_queue"]:
                        return_val = (key, module["forward_queue"][key])
                        del module["forward_queue"][key]

        else:
            n_iter, n_micro, module_id = request["args"]

            if module_id in self.modules:
                if request["args"] in self.modules[module_id]["forward_queue"]:
                    return_val = self.modules[module_id]["forward_queue"][
                        request["args"]
                    ]
                    del self.modules[module_id]["forward_queue"][request["args"]]

        self.response_queue.put({"status": "SUCCESS", "return": return_val})

    def _handle_check_backward(self, request):
        # Check if backward pass has been received and is loaded in shared mpc
        args = request["args"]
        return_val = None

        if self.role.startswith("W"):
            module_hash = args
            module = self.modules[module_hash]
            min_iter, min_micro = -1, -1
            for n_iter, n_micro, module_id in module["backward_queue"].keys():
                if n_iter <= min_iter or min_iter == -1:
                    min_iter = n_iter
                if n_micro <= min_micro or min_micro == -1:
                    min_micro = n_micro

            key = (min_iter, min_micro, module_hash)

            if key in module["backward_queue"]:
                return_val = (key, module["backward_queue"][key])
                del module["backward_queue"][key]

        else:
            n_iter, n_micro, module_hash, module_id = args
            key = (n_iter, n_micro, module_id)
            if module_hash in self.modules:
                if key in self.modules[module_hash]["backward_queue"]:
                    return_val = self.modules[module_hash]["backward_queue"][key]
                    del self.modules[module_id]["backward_queue"][key]

        self.response_queue.put({"status": "SUCCESS", "return": return_val})

    def _handle_send_optimizer_request(self, request):
        worker_id, module_id, optimizer_fn, optimizer_kwargs = request["args"]
        node = self.nodes[worker_id]
        data = json.dumps((module_id, optimizer_fn, optimizer_kwargs)).encode()
        self.send_to_node(node, b"OPTIMIZER" + data)
        self.response_queue.put({"status": "SUCCESS", "return": None})

    def _handle_check_state_update(self, request):
        module_id = request["args"]
        return_val = None
        if self.state_updates.get(module_id):
            return_val = self.state_updates[module_id].pop()
        self.response_queue.put({"status": "SUCCESS", "return": return_val})

    def _handle_check_validators(self, request):
        return_val = len(self.validators)
        self.response_queue.put({"status": "SUCCESS", "return": return_val})

    def _handle_check_parameters_request(self, request):
        key = "PREQPREQPREQ" + request["args"]
        return_val = False

        if key in self.memory_manager:
            del self.memory_manager[key]
            return_val = True
        else:
            return_val = False

        self.response_queue.put({"status": "SUCCESS", "return": return_val})

    def _handle_check_parameters(self, request):
        module_id = request["args"]
        key = "PREQPREQPREQ" + module_id
        if key in self.memory_manager:
            file_name = self.memory_manager[key]
            return_val = file_name
        else:
            return_val = None

        self.response_queue.put({"stats": "SUCCESS", "return": return_val})

    def _handle_request_parameters(self, request):
        worker_id, module_id = request["args"]
        node = self.nodes[worker_id]
        self.send_parameters_req(node, module_id)
        self.response_queue.put({"status": "SUCCESS", "return": None})

    def _handle_update_train(self, request):
        worker_id, mode, module_id = request["args"]
        mode = b"0" if mode is False else b"1"
        node = self.nodes[worker_id]
        self.send_to_node(node, b"UPDATE-TRAIN" + mode + module_id.encode())
        self.response_queue.put({"status": "SUCCESS", "return": None})

    def _handle_check_train(self, request):
        module_id = request["args"]
        return_val = None

        if module_id in self.modules:
            if "training" in self.modules[module_id].keys():
                return_val = self.modules[module_id]["training"]

        self.response_queue.put({"status": "SUCCESS", "return": return_val})

    def _handle_release_memory(self, request):
        data_type, module_id, key = tuple(request["args"])
        del self.memory_manager[key]
        if key in self.modules[module_id][data_type]:
            del self.modules[module_id][data_type][key]

        self.response_queue.put({"status": "SUCCESS", "return": None})

    def _handle_connect_node(self, request):
        host, port, node_id = request["args"]
        connected = self.connect_node(host, port, node_id)
        self.response_queue.put({"status": "SUCCESS", "return": connected})

    def _handle_get_info(self, request):
        self.response_queue.put(
            {
                "status": "SUCCESS",
                "return": (self.rsa_key_hash, self.host, self.port),
            }
        )

    def _handle_stop(self, request):
        self.terminate_flag.set()
        self.response_queue.put({"status": "SUCCESS", "return": True})

    def _handle_debug_print(self, request):
        if len(request["args"]) == 1:
            message = request["args"][0]
            colour = None
            level = logging.DEBUG
        else:
            message, colour, level = request["args"]

        if " -> " not in message:
            tag = "Torchnode"
        else:
            tag, message = message.split(" -> ", 1)

        self.debug_print(message, colour=colour, level=level, tag=tag)
        self.response_queue.put({"status": "SUCCESS", "return": False})

    def send_forward(self, node: Connection, forward_bytes, context, module_id):
        """Send forward pass to node, must contain args (module args) and context (module + epoch id)"""

        # Inject module_id into context
        payload = {
            "module_id": module_id,
            "key": context,
        }

        size = str(len(forward_bytes)).encode() + b"::"
        json_data = b"FORWARD" + size + forward_bytes + json.dumps(payload).encode()
        self.send_to_node(node, json_data)

    def _store_tensor_in_shared_memory(self, key, tensor: bytes, backward=False):
        id_hash = key[2]
        size = len(tensor)

        shm = shared_memory.SharedMemory(create=True, size=size)
        buffer = shm.buf[:size]
        buffer[:] = tensor

        queue = "forward_queue" if not backward else "backward_queue"

        self.modules[id_hash][queue][key] = (size, shm.name)
        self.memory_manager[key] = shm.name
        del buffer
        shm.close()

    def store_parameters_in_shared_memory(self, key, parameters):
        module_id = key[1:]
        parameters = json.dumps(parameters).encode()
        size = len(parameters)

        shm = shared_memory.SharedMemory(create=True, size=size)
        buffer = shm.buf[:size]
        buffer[:] = parameters

        self.modules[module_id]["parameters"][key] = (size, shm.name)
        self.memory_manager[key] = shm.name

    def send_backward(self, node: Connection, backward_bytes, context):
        """Send backward pass to node, must contain args (module args) and context (module + epoch id)"""
        size = str(len(backward_bytes)).encode() + b"::"
        json_data = b"BACKWARD" + size + backward_bytes + json.dumps(context).encode()
        self.send_to_node(node, json_data)

    def send_parameters_req(self, node: Connection, module_id: str):
        """Request parameters from a specific worker"""
        self.send_to_node(node, b"PARAMS-REQ" + module_id.encode())

    def send_train_updated(self, node: Connection, mode: bool, module_id: str):
        mode = b"0" if mode is False else b"1"
        self.send_to_node(node, b"TRAIN-UPDATED" + mode + module_id.encode())

    def send_module(
        self, file_name: bytes, module_id: str, module_info: dict, node: Connection
    ):
        self.debug_print(
            f"Sending module: {module_id} to worker: {node.node_id}",
            level=logging.INFO,
            colour="bright_blue",
            tag="Torchnode",
        )
        self._store_request(node.node_id, "MODULE" + module_id)
        self.state_updates[module_id] = []
        module_info_bytes = json.dumps(module_info).encode()
        self.send_to_node_from_file(
            node, file_name, b"MODULE" + module_id.encode() + module_info_bytes
        )

    def _store_request(self, node_id: str, key: str):
        super()._store_request(node_id, key)

    def _remove_request(self, node_id: str, key: str):
        super()._remove_request(node_id, key)

    def _listen_requests(self):
        while not self.mpc_terminate_flag.is_set():
            self.handle_requests()
            time.sleep(0.001)

    def get_module_hash_from_id(self, mod_id: bytes):
        for mod_hash in self.modules:
            if str(self.modules[mod_hash]["mod_id"]).encode() == mod_id:
                return mod_hash
        return None

    def run(self):
        super().run()
        self._mpc_comms = threading.Thread(target=self._listen_requests, daemon=True)
        self._mpc_comms.start()

    def stop(self):
        super().stop()
        self._stop_mpc_comms()

    def _handle_check_shutdown(self, request):
        if self.terminate_flag.is_set():
            self.response_queue.put({"status": "SUCCESS", "return": True})
        else:
            self.response_queue.put({"status": "SUCCESS", "return": False})

    def _stop_mpc_comms(self):
        self.mpc_terminate_flag.set()
        self.debug_print("Shutting down distributed ML processes...", tag="Torchnode")
        self._mpc_comms.join()

    def print_ui_status(self):
        total_vram = self.total_gpu_memory
        used_vram = total_vram - get_gpu_memory()

        ram = psutil.virtual_memory()
        used_ram = ram.total - ram.available

        streams = len(getattr(self, "stream_buffers", {}))
        modules = len(self.modules)

        in_q = len(getattr(self, "endpoint_requests", {}).get("incoming", []))
        out_q = len(getattr(self, "endpoint_requests", {}).get("outgoing", []))

        def c(label, colour):
            return f"{colour}{label}{ANSI.RESET}"

        def line(label, value, colour=ANSI.CYAN):
            return f"{c(label + ':', ANSI.DIM):<16} {colour}{value}{ANSI.RESET}"

        width = 80
        sep = f"{ANSI.DIM}{'─' * width}{ANSI.RESET}"

        # --- Header ---
        role_name = "Validator" if self.role.startswith("V") else "Worker"
        title = f" Tensorlink {role_name} Node "

        print()
        print(sep)
        print(f"{ANSI.BOLD}{ANSI.MAGENTA}{title.center(width)}{ANSI.RESET}")
        print(sep)

        # --- Identity ---
        print(line("Node ID", self.rsa_key_hash, ANSI.YELLOW))
        print(line("Address", f"{self.host}:{self.port}", ANSI.GREEN))
        print(line("Uptime", _uptime(self._start_time), ANSI.BLUE))

        # --- Network ---
        print(sep)
        print(line("Connections", len(self.nodes), ANSI.CYAN))
        print(line("    Workers", len(self.workers), ANSI.CYAN))
        print(line("    Validators", len(self.validators), ANSI.CYAN))
        print(line("    Users", len(self.users), ANSI.CYAN))

        # --- Resources ---
        print(sep)

        vram_bar = _bar(used_vram, total_vram)
        ram_bar = _bar(used_ram, ram.total)

        print(
            f"{ANSI.DIM}{'VRAM':<14}:{ANSI.RESET} "
            f"{ANSI.MAGENTA}[{vram_bar}]{ANSI.RESET} "
            f"{ANSI.YELLOW}{_fmt_gb(used_vram)} / {_fmt_gb(total_vram)} GB{ANSI.RESET}"
        )
        print(
            f"{ANSI.DIM}{'RAM':<14}:{ANSI.RESET} "
            f"{ANSI.GREEN}[{ram_bar}]{ANSI.RESET} "
            f"{ANSI.YELLOW}{_fmt_gb(used_ram)} / {_fmt_gb(ram.total)} GB{ANSI.RESET}"
        )

        print(line("Modules", modules, ANSI.MAGENTA))

        if self.print_level == logging.DEBUG:
            print(line("Module Info", len(self.modules), ANSI.MAGENTA))
            for k in list(self.modules)[:10]:
                print(f"{ANSI.DIM}  └─ {k}{ANSI.RESET}")
            if len(self.modules) > 10:
                print(f"{ANSI.DIM}  ... {len(self.modules) - 10} more{ANSI.RESET}")

        # Jobs (if present)
        jobs = getattr(self, "jobs", {})
        print(line("Jobs", len(jobs), ANSI.CYAN))

        if self.print_level == logging.DEBUG:
            for jid in list(jobs)[:10]:
                print(f"{ANSI.DIM}  └─ {jid}{ANSI.RESET}")

        # --- Validator ---
        if self.role.startswith("V"):
            print(sep)
            print(line("Proposal ID", self.current_proposal, ANSI.YELLOW))
            print(line("Streams", streams, ANSI.BLUE))
            print(line("API Jobs", f"in={in_q} out={out_q}", ANSI.CYAN))
            print(line("Queues", f"in={in_q} out={out_q}", ANSI.CYAN))

        print(sep)
        print()

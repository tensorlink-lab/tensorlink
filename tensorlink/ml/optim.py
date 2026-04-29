import threading
import time

import torch


class DistributedParameter(torch.nn.Parameter):
    """
    DistributedParameter:
        A wrapper around a tensor that represents a parameter offloaded to another worker device.
        This acts as a proxy to access the remote parameter when needed.
    """

    def __new__(cls, parent_model, module_id, worker_id, param_name, *args, **kwargs):
        data = torch.empty(0)
        instance = super(DistributedParameter, cls).__new__(
            cls, data, requires_grad=True
        )
        return instance

    def __init__(self, parent_model, module_id, worker_id, param_name):
        # Initialize with an empty tensor, as the actual data is managed remotely
        self.parent_model = parent_model
        self.module_id = module_id
        self.worker_id = worker_id
        self.param_name = param_name

    def _update_from_worker(self):
        """
        Update the parameter data from the worker.
        This would involve sending a request to the worker to get the parameter value.
        """
        response = self.parent_model.send_request(
            "get_param", (self.worker_id, self.module_id, self.param_name)
        )
        if "error" not in response:
            self.data = response["return"].data  # Update the data with received tensor

    def _send_update_to_worker(self):
        """
        Send parameter updates back to the worker.
        """
        self.parent_model.send_request(
            "set_param", (self.worker_id, self.module_id, self.param_name, self.data)
        )

    def fetch_parameter(self):
        """
        Fetch the current value of the parameter from the worker.
        """
        return self.model.send_request(
            "get_parameter_value", (self.worker_id, self.param_id)
        )

    def send_gradients(self, gradients):
        """
        Send gradients to the worker node for parameter update.

        Args:
            gradients (torch.Tensor): The gradient tensor to be applied to the parameter.
        """
        self.model.send_request(
            "apply_gradients", (self.worker_id, self.param_id, gradients)
        )

    def step(self):
        """
        Request the worker to perform an optimization step with the current gradients.
        """
        self.model.send_request("step_optimizer", (self.worker_id, self.param_id))

    def zero_grad(self):
        """
        Request the worker to zero out gradients for this parameter.
        """
        self.model.send_request("zero_grad", (self.worker_id, self.param_id))


def create_distributed_optimizer(model, base_optimizer_class, **optimizer_kwargs):
    if base_optimizer_class is None:
        base_optimizer_class = torch.optim.Adam

    class DistributedOptimizer(base_optimizer_class):
        def __init__(self, distributed_model, **_optimizer_kwargs):
            """
            Distributed Optimizer that initializes and synchronizes optimizers across worker nodes.

            Args:
                model (DistributedModel): The model containing potentially distributed modules.
                base_optimizer_class (type): The base optimizer class (e.g., torch.optim.AdamW).
                _optimizer_kwargs: Additional arguments for the optimizer (e.g., learning rate).
            """
            # Extract parameters for local optimizer (parameters not offloaded)
            params = distributed_model.parameters(distributed=False)
            if len(list(params)) > 0:
                super().__init__(params, **_optimizer_kwargs)
            # TODO pass the references to offloaded module optimizers when updating state

            self.model = distributed_model
            self.modules = distributed_model.distributed_graph
            self.optimizer_kwargs = _optimizer_kwargs

            # Initialize the base optimizer with the parameter proxies (local)
            self.base_optimizer = None
            if any(params):
                self.base_optimizer = base_optimizer_class(params, **_optimizer_kwargs)

            worker_requests = []

            # Initialize optimizer on offloaded devices
            for module_id, module_info in self.modules.items():
                if module_info["type"] == "offloaded":
                    for worker_id in module_info["assigned_workers"]:
                        self.model.send_request(
                            "send_optimizer_request",
                            (worker_id, module_id, "init", _optimizer_kwargs),
                        )
                        t = threading.Thread(
                            target=self._wait_for_worker,
                            args=("loaded", worker_id, module_id),
                        )

                        t.start()
                        worker_requests.append(t)

            for r in worker_requests:
                r.join()

        def step(self, closure=None):
            """
            Perform an optimization step for both the local and remote optimizers.
            Args:
                closure (callable, optional): A closure that reevaluates the model and returns the loss.
            """
            # Perform a local step for parameters directly on the master
            if self.base_optimizer is not None:
                self.base_optimizer.step(closure)

            # Send step commands to all workers asynchronously
            threads = []
            for module_id, module_info in self.modules.items():
                if module_info["type"] == "offloaded":
                    for worker_id in module_info["assigned_workers"]:
                        self.model.send_request(
                            "send_optimizer_request",
                            (worker_id, module_id, "step", closure),
                        )
                        t = threading.Thread(
                            target=self._wait_for_worker,
                            args=("stepped", worker_id, module_id),
                        )
                        t.start()
                        threads.append(t)

            # Wait for all threads to finish to ensure consistency
            for thread in threads:
                thread.join()

        def zero_grad(self):
            """
            Zero out the gradients for both local and remote parameters.
            """
            # Zero local gradients
            if self.base_optimizer is not None:
                self.base_optimizer.zero_grad()

            # Send zero_grad commands to all workers asynchronously
            threads = []
            for module_id, module_info in self.modules.items():
                if module_info["type"] == "offloaded":
                    for worker_id in module_info["assigned_workers"]:
                        self.model.send_request(
                            "send_optimizer_request",
                            (worker_id, module_id, "zero_grad", None),
                        )
                        thread = threading.Thread(
                            target=self._wait_for_worker,
                            args=("zeroed", worker_id, module_id),
                        )
                        thread.start()
                        threads.append(thread)

            # Wait for all threads to finish
            for thread in threads:
                thread.join()

        def _wait_for_worker(self, request_type, worker_id, module_id):
            waiting = True
            start_time = time.time()
            while waiting:
                time.sleep(1)
                args = self.model.send_request(
                    "check_module_request", (request_type, worker_id, module_id)
                )

                if args is True:
                    waiting = False

                if time.time() - start_time >= 300:
                    # Logic here to request another worker take his place
                    waiting = False

    return DistributedOptimizer(model, **optimizer_kwargs)

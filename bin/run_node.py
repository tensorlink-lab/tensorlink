"""
Tensorlink node runner.
Supports both Worker and Validator node types based on config.json
"""

from tensorlink.nodes import Worker, WorkerConfig, Validator, ValidatorConfig
from tensorlink.config import load, log_level as config_log_level, network_flags
import logging
import os
import subprocess
import sys
import time

import torch.cuda as cuda


def get_root_dir():
    """Get the root directory of the application."""
    if getattr(sys, "frozen", False):  # Check if running as an executable
        return os.path.dirname(sys.executable)
    else:  # Running as a Python script
        return os.path.dirname(os.path.abspath(__file__))


def create_env_file(env_path, cfg):
    """Create a default .tensorlink.env file if it doesn't exist."""
    if not os.path.exists(env_path):
        with open(env_path, "w") as env_file:
            env_file.write(f"PUBLIC_KEY={cfg.chain.address}\n")


def check_env_file(env_path):
    """Check if .env file exists, raise error if not."""
    if not os.path.exists(env_path):
        raise FileNotFoundError(
            ".tensorlink.env does not exist! Create a .env file with PUBLIC_KEY "
            "and PRIVATE_KEY as per the documentation."
        )


def setup_logging(cfg):
    """Setup logging based on configuration."""
    log_level = config_log_level(cfg)
    logging.basicConfig(level=log_level)
    return log_level


def confirm_trusted_mode():
    """Prompts the user with a confirmation message before proceeding in trusted mode."""
    while True:
        response = (
            input(
                "Trusted mode is enabled. Are you sure you want to proceed? (yes/no, y/n): "
            )
            .strip()
            .lower()
        )
        if response in {"yes", "y"}:
            print("Proceeding with trusted mode.")
            break
        elif response in {"no", "n"}:
            print("Aborting initialization in trusted mode.")
            exit(1)
        else:
            print("Invalid input. Please type 'yes'/'y' or 'no'/'n'.")


# ===== Worker-specific functions =====


def is_gpu_available(worker_node):
    """Check if GPU is available for mining."""
    try:
        is_loaded = worker_node.send_request("is_loaded", "", timeout=10)
    except Exception as e:
        logging.error(f"Error checking worker node status: {e}")
        is_loaded = False

    if not is_loaded and cuda.is_available():
        return True
    return False


def start_mining(mining_script, use_sudo=False):
    """Start the mining process using the specified script."""
    if not os.path.isabs(mining_script):
        mining_script = os.path.abspath(mining_script)

    if not os.path.exists(mining_script):
        raise FileNotFoundError(f"Mining script not found: {mining_script}")

    if use_sudo:
        return subprocess.Popen(["sudo", mining_script])
    else:
        return subprocess.Popen([mining_script])


def stop_mining(mining_process):
    """Stop the mining process if it is running."""
    if mining_process and mining_process.poll() is None:
        mining_process.terminate()
        mining_process.wait()


def run_worker_loop(worker, cfg):
    """Main loop for worker nodes with mining management."""
    mining_enabled = cfg.chain.mining
    mining_script = cfg.chain.mining_script
    use_sudo = os.geteuid() == 0
    mining_process = None

    try:
        while True:
            if mining_enabled and mining_script:
                if is_gpu_available(worker):
                    if not mining_process or mining_process.poll() is not None:
                        logging.info("Starting mining...")
                        mining_process = start_mining(mining_script, use_sudo)
                        worker.mining_active.value = True
                        time.sleep(2)

                        total_mem = cuda.get_device_properties(0).total_memory
                        reserved = cuda.memory_reserved(0)
                        worker.reserved_memory.value = total_mem - reserved
                else:
                    if mining_process and mining_process.poll() is None:
                        logging.info("Stopping mining...")
                        stop_mining(mining_process)
                        worker.mining_active.value = False
                        worker.reserved_memory.value = 0.0

            time.sleep(5)
            if not worker.node_process.is_alive():
                break

    except KeyboardInterrupt:
        logging.info("Exiting...")
        print("Exiting...")

    finally:
        worker.cleanup()
        if mining_process:
            stop_mining(mining_process)


def run_validator_loop(validator):
    """Main loop for validator nodes."""
    try:
        while True:
            time.sleep(5)
            if not validator.node_process.is_alive():
                break

    except KeyboardInterrupt:
        logging.info("Exiting...")
        print("Exiting...")

    finally:
        validator.cleanup()


def main():
    root_dir = get_root_dir()
    env_path = os.path.join(root_dir, ".tensorlink.env")
    cfg = load(search_dirs=[root_dir])
    if cfg.path:
        # Child processes (e.g. the DistributedValidator) read the same file
        os.environ.setdefault("TENSORLINK_CONFIG", str(cfg.path))

    # Setup logging
    log_level = setup_logging(cfg)

    # Get node type from config
    node_type = cfg.node.type.lower()

    if node_type not in ["worker", "validator"]:
        raise ValueError(
            f"Invalid node type: {node_type}. Must be 'worker' or 'validator'."
        )

    max_memory_gb = cfg.ml.max_memory_gb
    max_module_gb = cfg.ml.max_module_gb
    benchmark = cfg.ml.benchmark
    enable_hosting = True

    # Parse common config
    trusted = cfg.ml.trusted
    local, upnp, on_chain = network_flags(cfg.node.mode)

    # Handle env file
    if node_type == "worker":
        create_env_file(env_path, cfg)
    else:  # validator
        check_env_file(env_path)

    # Confirm trusted mode if enabled
    if trusted:
        confirm_trusted_mode()

    # Create and run the appropriate node
    if node_type == "worker":
        logging.info("Starting Worker node...")
        worker = Worker(
            config=WorkerConfig(
                upnp=upnp,
                local_test=local,
                on_chain=on_chain,
                print_level=log_level,
                priority_nodes=cfg.node.priority_nodes,
                seed_validators=cfg.chain.seed_validators,
                max_memory_gb=max_memory_gb,
            ),
            trusted=trusted,
            utilization=True,
            benchmark=benchmark,
        )
        run_worker_loop(worker, cfg)

    else:
        logging.info("Starting Validator node...")
        validator = Validator(
            config=ValidatorConfig(
                upnp=upnp,
                local_test=local,
                on_chain=on_chain,
                endpoint=cfg.node.endpoint,
                endpoint_url=cfg.node.endpoint_url,
                endpoint_port=cfg.node.endpoint_port,
                print_level=log_level,
                priority_nodes=cfg.node.priority_nodes,
                seed_validators=cfg.chain.seed_validators,
                max_memory_gb=max_memory_gb,
            ),
            trusted=trusted,
            enable_hosting=enable_hosting,
            max_module_gb=max_module_gb,
            benchmark=benchmark,
        )
        run_validator_loop(validator)


if __name__ == "__main__":
    main()

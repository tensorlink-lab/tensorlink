import json
import logging
import os
import subprocess
import sys
import time

import dotenv
import torch.cuda as cuda

from tensorlink.nodes import Validator, ValidatorConfig


def get_root_dir():
    if getattr(sys, "frozen", False):  # Check if running as an executable
        return os.path.dirname(sys.executable)
    else:  # Running as a Python script
        return os.path.dirname(os.path.abspath(__file__))


def check_env_file(_env_path, _config):
    """
    Create a default .env file at the specified path if it doesn't exist.
    """
    if not os.path.exists(_env_path):
        raise FileNotFoundError(
            ".tensorlink.env does not exist! Create a .env file with PUBLIC_KEY and PRIVATE_KEY as per the documentation."
        )


def load_config(config_path="config.json"):
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"Config file {config_path} not found, using defaults")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in config file: {e}")
        return {}


def main():
    root_dir = get_root_dir()
    env_path = os.path.join(root_dir, ".tensorlink.env")

    # Load config if needed
    config = load_config(os.path.join(root_dir, "config.json"))
    network_config = config["network"]
    crypto_config = config["crypto"]
    ml_config = config["ml"]

    check_env_file(env_path, config)

    trusted = ml_config.get("trusted", False)
    local = network_config.get("local", False)
    upnp = True
    if local == "true":
        upnp = False

    priority_nodes = network_config["priority_nodes"]
    seed_validators = crypto_config["seed_validators"]

    validator = Validator(
        config=ValidatorConfig(
            upnp=upnp,
            local_test=local,
            off_chain_test=local,
            print_level=logging.DEBUG,
            priority_nodes=priority_nodes,
            seed_validators=seed_validators,
        ),
        trusted=trusted,
    )

    try:
        while True:
            time.sleep(5)

            if not validator.node_process.is_alive():
                break

    except KeyboardInterrupt:
        logging.info("Exiting...")


if __name__ == "__main__":
    main()

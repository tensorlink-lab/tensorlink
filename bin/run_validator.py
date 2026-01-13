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
            config = json.load(f)
            if config.get("config"):
                return config.get("config")
            return config

    except FileNotFoundError:
        logging.warning(f"Config file {config_path} not found, using defaults")
        return {}

    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in config file: {e}")
        return {}


def main():
    root_dir = get_root_dir()
    env_path = os.path.join(root_dir, ".tensorlink.env")

    config = load_config(os.path.join(root_dir, "config.json"))
    network_config = config["network"]
    crypto_config = config["crypto"]
    ml_config = config["ml"]

    check_env_file(env_path, config)

    trusted = ml_config.get("trusted", False)
    mode = network_config.get("mode", "private")

    # Defaults
    local = False
    upnp = True
    on_chain = False

    if mode == "local":
        local = True
        upnp = False
    elif mode == "public":
        on_chain = True
    elif mode == "private":
        pass
    else:
        raise ValueError(f"Unknown network mode: {mode}")

    validator = Validator(
        config=ValidatorConfig(
            upnp=upnp,
            local_test=local,
            on_chain=on_chain,
            print_level=logging.DEBUG,
            priority_nodes=network_config.get("priority_nodes", []),
            seed_validators=crypto_config.get("seed_validators", []),
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

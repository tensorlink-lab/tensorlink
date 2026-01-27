"""
Test the distribution of models across workers to visually analyze distribution methods
and ensure different distribution techniques are working accordingly
"""

from tensorlink.ml.graphing import ModelParser
from tensorlink.ml.utils import estimate_memory, format_memory_size
import pandas as pd

MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
]

# Two workers with a 24GB and 16GB GPU capacity
WORKERS = {
    '509d89bf56704c67873c328e4f706a705b2fdc1671ebacab1083c9c6d2df650f': {
        'id': '509d89bf56704c67873c328e4f706a705b2fdc1671ebacab1083c9c6d2df650f',
        'gpu_memory': 24e9,
        'total_gpu_memory': 24e9,
        'role': 'W',
        'training': False,
    },
    '209d89bf56704c67873c328e4f706a705b2fdc1671ebacab1083c9c6d2df650f': {
        'id': '209d89bf56704c67873c328e4f706a705b2fdc1671ebacab1083c9c6d2df650f',
        'gpu_memory': 16e9,
        'total_gpu_memory': 16e9,
        'role': 'W',
        'training': False,
    },
}


def test_model_distributions():
    """
    Print out model distributions for a variety of different models to inspect their memory
    estimates and how they are distributed among the default workers.

    No actual assertion statements are made in this function.
    """
    batch_sizes = [1]
    seq_lengths = [1024, 4096, 8196]

    # Collect rows for the final DataFrame
    rows = []

    for model_name in MODELS:
        for bs in batch_sizes:
            for seqlen in seq_lengths:
                parser = ModelParser(verbose=False)

                try:
                    config = parser.create_distributed_config(
                        model_name,
                        WORKERS,
                        training=False,
                        trusted=False,
                        input_obfuscation=False,
                        host_max_memory_bytes=50e6,
                        max_seq_len=seqlen,
                        batch_size=bs,
                        optimizer_type="adam",
                    )

                    success = config.get("success", False)
                    model_memory = (
                        format_memory_size(config["model_memory"]) if success else 0
                    )
                    components_memory = {
                        k: {n['type']: n['memory']} for k, n in config["config"].items()
                    }
                    total_components_memory = sum(
                        [list(v.values())[-1] for v in components_memory.values()]
                    )
                    memory_breakdown = {
                        k: {k2: format_memory_size(v2) for k2, v2 in v.items()}
                        for k, v in components_memory.items()
                    }
                    components_memory = {
                        k: {n['type']: format_memory_size(n['memory'])}
                        for k, n in config["config"].items()
                    }

                    # Append to rows
                    rows.append(
                        {
                            "model": model_name,
                            "batch_size": bs,
                            "seq_length": seqlen,
                            "model_memory": model_memory,
                            "components_sum": format_memory_size(
                                total_components_memory
                            ),
                            "success": success,
                            "error": None if success else config.get("error", None),
                            "memory_breakdown": memory_breakdown,
                        }
                    )

                except Exception as e:
                    rows.append(
                        {
                            "model": model_name,
                            "batch_size": bs,
                            "seq_length": seqlen,
                            "success": False,
                            "error": str(e),
                        }
                    )

    # Create a pandas DataFrame
    df = pd.DataFrame(rows)

    # Configure pandas to display multi-line cells
    pd.set_option('display.max_colwidth', None)  # No truncation
    pd.set_option('display.width', None)  # Auto-detect terminal width
    pd.set_option('display.max_rows', None)  # Show all rows

    print("\n\n========== FINAL RESULTS TABLE ==========\n")
    print(df.to_string(index=False))
    print("\nTable stored in variable: df")


def test_config_combinations():
    """
    Test various combinations of configuration parameters to see how they affect
    model distribution and memory allocation. Results shown as a simple DataFrame.
    """
    # Base test parameters
    test_model = "Qwen/Qwen3-14B"
    batch_size = 1
    seq_length = 4096

    # Define test configurations to explore
    test_workers = {
        '509d89bf56704c67873c328e4f706a705b2fdc1671ebacab1083c9c6d2df650f': {
            'id': '509d89bf56704c67873c328e4f706a705b2fdc1671ebacab1083c9c6d2df650f',
            'gpu_memory': 4e8,
            'total_gpu_memory': 4e8,
            'role': 'W',
            'training': False,
        },
        '209d89bf56704c67873c328e4f706a705b2fdc1671ebacab1083c9c6d2df650f': {
            'id': '209d89bf56704c67873c328e4f706a705b2fdc1671ebacab1083c9c6d2df650f',
            'gpu_memory': 4e8,
            'total_gpu_memory': 4e8,
            'role': 'W',
            'training': False,
        },
    }

    test_configs = [
        {
            "input_obfuscation": False,
            "host_max_memory_bytes": 0,
            "host_max_module_bytes": 1e8,
            "host_max_depth": 1,
        },
        # {
        #     "input_obfuscation": False,
        #     "host_max_memory_bytes": 0,
        #     "host_max_module_bytes": 1e8,
        #     "host_max_depth": 1,
        # },
        # {
        #     "input_obfuscation": False,
        #     "host_max_memory_bytes": 4e8,
        #     "host_max_module_bytes": 1e8,
        #     "host_max_depth": 1,
        # },
    ]

    results = []

    for test_config in test_configs:
        parser = ModelParser(verbose=False)

        try:
            config = parser.create_distributed_config(
                test_model,
                WORKERS,
                training=False,
                trusted=False,
                optimizer_type="adam",
                max_seq_len=seq_length,
                batch_size=batch_size,
                **test_config,
            )

            success = config.get("success", False)

            if success:
                components_memory = {
                    k: {n['type']: n['memory']} for k, n in config["config"].items()
                }
                total_components_memory = sum(
                    [list(v.values())[-1] for v in components_memory.values()]
                )
                components_memory = {
                    k: {n['type']: format_memory_size(n['memory'])}
                    for k, n in config["config"].items()
                }
            else:
                components_memory = {}
                total_components_memory = 0

            results.append(
                {
                    "input_obfuscation": test_config["input_obfuscation"],
                    "success": success,
                    "model_memory": (
                        format_memory_size(config["model_memory"]) if success else "N/A"
                    ),
                    "components_sum": (
                        format_memory_size(total_components_memory)
                        if success
                        else "N/A"
                    ),
                    "component_breakdown": components_memory,
                    "error": None if success else config.get("error", "Unknown error"),
                }
            )

        except Exception as e:
            results.append(
                {
                    "input_obfuscation": test_config["input_obfuscation"],
                    "success": False,
                    "model_memory": "N/A",
                    "components_sum": "N/A",
                    "num_components": 0,
                    "component_breakdown": "N/A",
                    "error": str(e),
                }
            )

    # Create DataFrame
    df = pd.DataFrame(results)

    # Set display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print("\n\n========== CONFIGURATION COMPARISON ==========\n")
    print(f"Test Model: {test_model}")
    print(f"Batch Size: {batch_size}, Seq Length: {seq_length}\n")
    print(df.to_string(index=False))
    print("\n\nResults stored in variable: df")

    return df

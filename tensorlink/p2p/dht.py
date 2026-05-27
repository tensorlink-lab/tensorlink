from __future__ import annotations
from typing import Union, Optional, List, TYPE_CHECKING

# import time

if TYPE_CHECKING:
    from tensorlink.p2p.smart_node import Smartnode


class Bucket:
    """A bucket for storing local values in the Kademlia-inspired DHT"""

    def __init__(self, distance_from_key, max_values):
        self.values = []
        self.distance_from_key = distance_from_key
        self.max_values = max_values * (2**distance_from_key)

    def is_full(self):
        return len(self.values) >= self.max_values

    def add_node(self, value):
        if not self.is_full():
            self.values.append(value)

    def remove_node(self, value):
        if value in self.values:
            self.values.remove(value)


class DHT:
    """
    A Kademlia-inspired distributed hash table (DHT) where key-value pairs
    are stored based on XOR distances between their keys. When queried values
    are not found, the dht routes the request to the nearest node. Nodes store
    values in k-buckets, where nodes store values more close to them.
    """

    def __init__(
        self,
        node: Smartnode,
        replication_factor=2,
        bucket_size=2,
    ):
        self.node = node
        self.node_key = node.rsa_key_hash
        self.replication_factor = replication_factor
        self.bucket_size = bucket_size

        self.routing_table = {}
        self.buckets = [Bucket(d, bucket_size) for d in range(256)]

    def query(
        self,
        key: Union[str, bytes],
        requester: Optional[str] = None,
        keys_to_exclude: Optional[List[str]] = None,
    ) -> Optional[Union[str | dict]]:
        """
        Retrieve stored value from DHT or query the closest nodes to a given key.

        Args:
            key (str | bytes): Key (hash) associated with the queried value.
            requester (Optional[str]): The ID of the requesting node to respond to (used in chained calls).
            keys_to_exclude (Optional[List[str]]): Node IDs to avoid routing the request to.

        Returns:
            Optional[str]: The value if found, otherwise None.

        TODO should be run in its own thread due to blocking RPC
        """
        if isinstance(key, bytes):
            key = key.decode()

        if keys_to_exclude is None:
            keys_to_exclude = []

        if self.node.rsa_key_hash not in keys_to_exclude:
            keys_to_exclude.append(self.node.rsa_key_hash)

        self.node.debug_print(f"Querying for {key}", tag="DHT", level=self.node.VERBOSE)

        closest_node = None
        closest_distance = float("inf")

        # Find nearest node in routing table to key based on XOR
        for node_key, node in self.routing_table.items():
            if node_key in keys_to_exclude:
                continue

            # Get XOR distance between keys
            distance = int(key, 16) ^ int(node_key, 16)
            if distance < closest_distance:
                closest_node = (node_key, node)
                closest_distance = distance

        if requester is None:
            requester = self.node_key

        if closest_node is not None:
            node_hash, node_value = closest_node
            # If we get None, try one more query
            if node_value is None:
                self.delete(node_hash)
                return self.query(key, requester, keys_to_exclude + [node_hash])

            # Found the quested value
            elif node_hash == key:
                return node_value

            # We don't have the stored value, and must route the request to the nearest nodes
            else:
                try:
                    # Only query values from other validators
                    if node_hash in self.node.validators:
                        closest_node_id = node_value["id"]
                        return self.node.query_node(
                            key,
                            self.node.nodes[closest_node_id],
                            requester,
                            keys_to_exclude,
                        )  # Send query to validator

                except Exception as e:
                    self.node.debug_print(f"Failed to query node {node_hash}: {e}")
                    return self.query(key, requester, keys_to_exclude + [node_hash])

    def store(self, key: str, value: object, replicate: int = 0):
        bucket_index = self.calculate_bucket_index(key)
        bucket = self.buckets[bucket_index]

        if not bucket.is_full() or key in bucket:
            bucket.add_node(key)
            self.routing_table[key] = value

        # TODO Duplication method across validators
        if 5 > replicate > 0:
            pass

    def delete(self, key: Union[str, bytes]):
        """
        Delete a key-value pair from the DHT.
        """
        if isinstance(key, bytes):
            key = key.decode()

        if key in self.routing_table:
            bucket_index = self.calculate_bucket_index(key)
            bucket = self.buckets[bucket_index]
            bucket.remove_node(key)

            if key in self.node.nodes or self.node.jobs:
                # Do not delete information related to active connections or jobs
                return

            del self.routing_table[key]
            self.node.debug_print(
                f"Key {key} deleted from DHT.", colour="blue", tag="DHT"
            )

    def calculate_bucket_index(self, key: str):
        """Find the index of a bucket in the DHT given the key"""
        key_int = int(key.encode(), 16)
        bucket_index = key_int % len(self.buckets)

        return bucket_index

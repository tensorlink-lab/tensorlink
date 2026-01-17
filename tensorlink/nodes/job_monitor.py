from __future__ import annotations

from tensorlink.ml.proofs import MLProofOfWork
from typing import TYPE_CHECKING, Dict, Optional, List, Tuple
from cryptography.hazmat.primitives import hashes
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import numpy as np
import logging
import time

if TYPE_CHECKING:
    from validator_thread import Validator


class JobStatus(Enum):
    ACTIVE = "active"
    PENDING_OFFLINE = "pending_offline"
    OFFLINE = "offline"
    FAILED = "failed"
    COMPLETED = "completed"


class ResourceUsage(Enum):
    FREE = "free"
    PAID = "paid"
    SUSPENDED = "suspended"


@dataclass
class WorkerHealth:
    last_seen: float
    epochs_completed: int
    status: str
    metrics: Dict
    last_proof: Dict
    proof_history: List[Dict]


def calculate_gradient_hash(gradients: List[np.ndarray]) -> bytes:
    """Calculate deterministic hash of model gradients."""
    hasher = hashes.Hash(hashes.SHA256())

    for grad in gradients:
        # Convert gradient values to bytes in a deterministic way
        grad_bytes = grad.tobytes()
        hasher.update(grad_bytes)

    return hasher.finalize()


def _get_max_theoretical_flops(hardware_info: Dict) -> float:
    """Calculate theoretical maximum FLOPS based on hardware info."""
    # This is a simplified calculation - should be adjusted based on actual hardware specs
    gpu_flops = hardware_info.get("gpu_tflops", 0) * 1e12
    cpu_flops = (
        hardware_info.get("cpu_cores", 1) * hardware_info.get("cpu_freq_ghz", 1) * 1e9
    )
    return max(gpu_flops, cpu_flops)


def _verify_computation_metrics(proof: Dict, metrics: Dict) -> bool:
    """Verify computation-related metrics."""
    # Verify that computational metrics make sense
    expected_compute_time = proof.get("compute_time", 0)
    reported_flops = proof.get("flops", 0)
    batch_size = metrics.get("batch_size", 0)

    if batch_size <= 0 or reported_flops <= 0:
        return False

    # Check if reported FLOPS are within reasonable range for the hardware
    max_theoretical_flops = _get_max_theoretical_flops(proof.get("hardware_info", {}))
    if reported_flops > max_theoretical_flops:
        return False

    # Verify computation time makes sense for the reported operations
    expected_min_time = (
        reported_flops * 0.7
    ) / max_theoretical_flops  # Allow 30% efficiency loss
    if expected_compute_time < expected_min_time:
        return False

    return True


class JobMonitor:
    def __init__(self, node: Validator | None):
        self.node = node
        self.terminate_flag = node.terminate_flag
        self.worker_health_checks: Dict[str, WorkerHealth] = {}
        self.ml_pow = MLProofOfWork()

        # Configuration parameters
        self.WORKER_TIMEOUT_SECONDS = 60
        self.JOB_CREATION_TIMEOUT_SECONDS = 180
        self.JOB_OFFLINE_THRESHOLD_SECONDS = 90
        self.HEALTH_CHECK_INTERVAL_SECONDS = 30
        self.PROOF_OF_WORK_INTERVAL = 5
        self.RESOURCE_UPDATE_INTERVAL_SECONDS = 60
        self.BILLING_CYCLE_SECONDS = 360

    def monitor_job(self, job_id: str):
        """Monitor job progress and workers asynchronously."""
        self.node.debug_print(
            f"Job monitor beginning for job: {job_id}",
            colour="blue",
            level=logging.INFO,
            tag="JobMonitor",
        )

        job_data = self._get_job_data(job_id)
        if not job_data:
            self._handle_job_failure(job_id, "Failed to retrieve job data")
            return

        job_status = JobStatus.ACTIVE

        try:
            while not self.terminate_flag.is_set():
                time.sleep(self.HEALTH_CHECK_INTERVAL_SECONDS)
                job_data = self._get_job_data(job_id)
                if not job_data:
                    self._handle_job_failure(job_id, "Failed to retrieve job data")
                    return

                try:
                    job_status = self._check_job_health(job_id, job_data)

                    if job_status != JobStatus.ACTIVE:
                        if self._should_terminate_job(job_data, job_status):
                            break
                    else:
                        self.node.debug_print(
                            f"Job inspection complete for job: {job_id}",
                            colour="blue",
                            tag="JobMonitor",
                        )
                        job_data["last_seen"] = time.time()
                        self.node.dht.routing_table[job_id] = job_data

                except Exception as e:
                    self.node.debug_print(
                        f"Error in health check cycle: {str(e)}",
                        colour="bright_red",
                        level=logging.ERROR,
                        tag="JobMonitor",
                    )
                    break

        finally:
            self._cleanup_job(job_data, job_status)

    def _check_job_health(self, job_id: str, job_data: Dict) -> JobStatus:
        """Comprehensive health check of the job and its components."""
        # Check if job is expired
        if job_data.get("last_seen", 15) - job_data.get("timestamp", 0) > job_data.get(
            "time", 0
        ):
            return JobStatus.COMPLETED

        # Check user connection
        user_status = self._check_user_status(job_data)
        if not user_status:
            return JobStatus.PENDING_OFFLINE

        # Check worker health
        worker_status = self._check_workers_health(job_data)
        if not worker_status:
            return JobStatus.PENDING_OFFLINE

        # TODO Proof of learning

        return JobStatus.ACTIVE

    def _check_single_worker(self, worker: str, module_id: str) -> bool:
        """Enhanced worker check with ML proof of work verification."""
        try:
            worker_info = self.node.dht.query(worker)
            connected = self.node.connect_node(
                worker_info["host"], worker_info["port"], worker.encode()
            )

            if not connected:
                self._handle_worker_failure(worker, module_id)
                return False

            # Get worker metrics and proof of work
            # metrics, proof = self._get_worker_metrics_and_proof(worker)

            # Verify proof of work
            # if not self._verify_worker_proof(worker, proof, metrics):
            #     self._handle_invalid_proof(worker, module_id)
            #     return False

            # Update worker health data
            # self.worker_health_checks[worker] = WorkerHealth(
            #     last_seen=time.time(),
            #     epochs_completed=metrics.get("epochs_completed", 0),
            #     status="healthy",
            #     metrics=metrics,
            #     last_proof=proof,
            #     proof_history=self.worker_health_checks.get(worker,
            #                                                WorkerHealth(0, 0, "", {}, {}, [])).proof_history + [proof]
            # )

            return True

        except Exception as e:
            self.node.debug_print(
                f"Worker health check failed for {worker}: {str(e)}",
                colour="yellow",
                level=logging.WARNING,
                tag="JobMonitor",
            )
            return False

    def _get_worker_metrics_and_proof(self, worker: str) -> Tuple[Dict, Dict]:
        """Get worker metrics and proof of work data."""
        try:
            response = self.node.query_node(
                "GET_METRICS_AND_PROOF", self.node.nodes[worker]
            )

            metrics = response.get("metrics", {})
            proof = response.get("proof", {})

            return metrics, proof

        except Exception as e:
            raise Exception(f"Failed to get worker metrics: {str(e)}")

    def _verify_worker_proof(self, worker: str, proof: Dict, metrics: Dict) -> bool:
        """Verify worker's proof of work."""
        try:
            # Skip verification if not enough epochs have passed
            current_epoch = metrics.get("epochs_completed", 0)
            if current_epoch % self.PROOF_OF_WORK_INTERVAL != 0:
                return True

            # Get previous proof for comparison
            prev_health = self.worker_health_checks.get(worker)
            if prev_health:
                prev_proof = prev_health.last_proof
            else:
                prev_proof = None

            # Verify various aspects of the proof TODO
            validations = (
                self._verify_gradient_proof(proof, prev_proof),
                self._verify_loss_proof(proof),
                _verify_computation_metrics(proof, metrics),
            )

            return all(validations)

        except Exception as e:
            self.node.debug_print(
                f"Error verifying proof of work: {str(e)}",
                colour="bright_red",
                level=logging.ERROR,
                tag="JobMonitor",
            )
            return False

    def _verify_gradient_proof(self, proof: Dict, prev_proof: Optional[Dict]) -> bool:
        """Verify gradient-based proofs."""
        if not proof.get("gradients"):
            return False

        current_gradients = np.array(proof["gradients"])

        if prev_proof and prev_proof.get("gradients"):
            prev_gradients = np.array(prev_proof["gradients"])
            if not self.ml_pow.verify_gradient_continuity(
                prev_gradients, current_gradients
            ):
                return False

        # Verify gradient hash
        gradient_hash = calculate_gradient_hash([current_gradients])
        return gradient_hash == proof.get("gradient_hash")

    def _verify_loss_proof(self, proof: Dict) -> bool:
        """Verify loss-based proofs."""
        loss_history = proof.get("loss_history", [])
        if not loss_history:
            return False

        return self.ml_pow.verify_loss_trajectory(loss_history)

    def _handle_invalid_proof(self, worker: str, module_id: str):
        """Handle case where worker provides invalid proof of work."""
        self.node.debug_print(
            f"Invalid proof of work from worker {worker} for module {module_id}",
            colour="bright_red",
            level=logging.WARNING,
            tag="JobMonitor",
        )

        # Record violation
        if worker in self.worker_health_checks:
            self.worker_health_checks[worker].status = "suspicious"

        # Implement penalty mechanism
        self._penalize_worker(worker, module_id)

    def _penalize_worker(self, worker: str, module_id: str):
        """Implement penalties for workers that fail verification."""
        try:
            # Record violation in worker's reputation
            self.node.update_worker_reputation(
                worker, -0.2
            )  # Decrease reputation by 20%

            # If multiple violations, remove worker
            reputation = self.node.get_worker_reputation(worker)
            if reputation < 0.5:  # Threshold for removal
                self._handle_worker_failure(worker, module_id)

        except Exception as e:
            self.node.debug_print(
                f"Error applying worker penalty: {str(e)}",
                colour="bright_red",
                level=logging.ERROR,
                tag="JobMonitor",
            )

    def _get_job_data(self, job_id: str) -> Optional[Dict]:
        """Retrieve and validate job data."""
        try:
            return self.node.dht.query(job_id)
        except Exception as e:
            self.node.debug_print(
                f"Failed to retrieve job data: {str(e)}",
                colour="bright_red",
                level=logging.ERROR,
                tag="JobMonitor",
            )
            return None

    def _check_user_status(self, job_data: Dict) -> bool:
        """Verify user connection and job activity status."""
        try:
            if job_data["author"] != self.node.rsa_key_hash:
                user_data = self.node.dht.query(job_data["author"])
                connected = self.node.connect_node(
                    user_data["host"], user_data["port"], job_data["author"]
                )

                if connected:
                    job_status = self.node.query_node(
                        job_data["id"], self.node.nodes[job_data["author"]]
                    )

                    return job_status is not None and job_status.get("active", False)

                return False

            return True

        except Exception as e:
            self.node.debug_print(
                f"Error checking user status: {str(e)}",
                colour="yellow",
                level=logging.WARNING,
                tag="JobMonitor",
            )
            return False

    def _check_workers_health(self, job_data: Dict) -> bool:
        """Monitor worker health and collect performance metrics."""
        all_workers_healthy = True

        for module_id, module_info in job_data["distribution"].items():
            if module_info["type"] != "offloaded":
                continue

            for worker_id in module_info["assigned_workers"]:
                worker_healthy = self._check_single_worker(worker_id, module_id)
                all_workers_healthy = all_workers_healthy and worker_healthy

        return all_workers_healthy

    def _handle_worker_failure(self, worker: str, module_id: str):
        """Handle worker failure and initiate recovery process."""
        self.node.debug_print(
            f"Worker {worker} failed for module {module_id}. Initiating recovery...",
            colour="yellow",
            level=logging.WARNING,
            tag="JobMonitor",
        )
        # Implement worker recovery strategy
        # TODO: Add worker replacement logic
        pass

    def _cleanup_job(self, job_data: Dict, final_status: JobStatus):
        """Perform comprehensive job cleanup."""
        try:
            # Update job status
            job_data["active"] = False
            job_data["end_time"] = time.time()
            job_data["final_status"] = final_status.value

            # Clean up worker resources
            self._cleanup_workers(job_data)

            self.node.debug_print(
                f"Job {job_data['id']} cleaned up successfully with status: {final_status.value}",
                colour="green",
                level=logging.INFO,
                tag="JobMonitor",
            )

            # Pass over job to contract manager
            if self.node.contract_manager:
                self.node.contract_manager.add_job_to_complete(job_data)

            # Remove module from node
            for module_id in job_data["distribution"]:
                del self.node.modules[module_id]

        except Exception as e:
            self.node.debug_print(
                f"Error during job cleanup: {str(e)}",
                colour="bright_red",
                level=logging.ERROR,
                tag="JobMonitor",
            )

    def _cleanup_workers(self, job_data: Dict):
        """Clean up worker resources and send shutdown signals."""
        for module_id, module_info in job_data["distribution"].items():
            if module_info["type"] == "offloaded":
                for worker_id in module_info["assigned_workers"]:
                    try:
                        node = self.node.nodes[worker_id]
                        self.node.send_to_node(
                            node, b"SHUTDOWN-JOB" + module_id.encode()
                        )
                        # Clear worker health data
                        self.worker_health_checks.pop(worker_id, None)
                    except Exception as e:
                        self.node.debug_print(
                            f"Error shutting down worker {worker_id}: {str(e)}",
                            colour="yellow",
                            level=logging.WARNING,
                            tag="JobMonitor",
                        )

    def _should_terminate_job(self, job_data: Dict, current_status: JobStatus) -> bool:
        """
        Determine if a job should be terminated based on its current status and conditions.

        Args:
            job_data: Dictionary containing job information
            current_status: Current JobStatus enum value

        Returns:
            bool: True if job should be terminated, False otherwise
        """
        # Check if job has been running too long
        if "start_time" in job_data:
            job_duration = time.time() - job_data["start_time"]
            if job_duration > job_data.get("max_duration", float("inf")):
                self._handle_job_failure(
                    job_data["id"], "Job exceeded maximum duration"
                )
                return True

        # Handle different job statuses
        if current_status == JobStatus.FAILED:
            return True

        elif current_status == JobStatus.COMPLETED:
            return True

        elif current_status == JobStatus.PENDING_OFFLINE:
            # Check if job has been offline too long
            last_seen = job_data.get("last_seen", 0)
            offline_duration = time.time() - last_seen

            if offline_duration > self.JOB_OFFLINE_THRESHOLD_SECONDS:
                self._handle_job_failure(
                    job_data["id"], "Job offline threshold exceeded"
                )
                return True

        return False

    def _handle_job_failure(self, job_id: str, reason: str):
        """
        Handle job failure by updating status and notifying relevant parties.

        Args:
            job_id: Identifier of the failed job
            reason: Reason for job failure
        """
        try:
            # Get job data
            job_data = self._get_job_data(job_id)
            if not job_data:
                self.node.debug_print(
                    f"Could not retrieve data for failed job {job_id}",
                    colour="bright_red",
                    level=logging.ERROR,
                    tag="JobMonitor",
                )
                return

            # Update job status
            job_data["active"] = False
            job_data["status"] = JobStatus.FAILED.value
            job_data["failure_reason"] = reason
            job_data["failure_time"] = datetime.now().isoformat()

            # Notify job owner
            # try:
            #     user_data = self.node.dht.query(job_data["author"])
            #     if user_data:
            #         connected = self.node.connect_node(
            #             user_data["host"],
            #             user_data["port"],
            #             job_data["author"].encode()
            #         )
            #         if connected:
            #             # Send alert (decline or cancel job)
            #             pass
            #
            # except Exception as e:
            #     self.node.debug_print(
            #         f"Failed to notify job owner of failure: {str(e)}",
            #         colour="yellow",
            #         level=logging.WARNING,
            #     )

            # Clean up worker resources
            self._cleanup_workers(job_data)

            # Log failure
            self.node.debug_print(
                f"Job {job_id} failed: {reason}. Cleaning Up",
                colour="bright_red",
                level=logging.ERROR,
                tag="JobMonitor",
            )

        except Exception as e:
            self.node.debug_print(
                f"Error handling job failure: {str(e)}",
                colour="bright_red",
                level=logging.ERROR,
                tag="JobMonitor",
            )

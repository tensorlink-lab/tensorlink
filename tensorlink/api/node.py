from tensorlink.ml.utils.utils import get_popular_model_stats
from tensorlink.api.models import (
    JobRequest,
    ChatCompletionRequest,
    AnyResponseRequest,
    ModelStatusResponse,
    ModelStatusDistributionResponse,
    ModelDistributionEntry,
    EmbeddingCreateRequest,
)
from tensorlink.api.functions import build_hf_job_data, build_generation_request
from tensorlink.ml.utils.model_types import ModelType
from tensorlink.api.generation import GenerationMixin
from tensorlink.api.embeddings import EmbeddingMixin
from fastapi import FastAPI, HTTPException, APIRouter, Request, Query
from threading import Thread
import logging
import uvicorn
import asyncio


class TensorlinkAPI(GenerationMixin, EmbeddingMixin):
    """
    Supports API requests to request and interact with models, along with
    probing node & job information.
    """

    def __init__(self, smart_node, host="0.0.0.0", port=64747):
        self.smart_node = smart_node
        self.host = host
        self.port = port
        self.app = FastAPI()
        self.router = APIRouter()
        self.server_loop = None

        self._init_generation_state()
        self._init_embedding_state()
        self._define_routes()
        self._start_server()

    def _define_routes(self):
        """Register all API routes by delegating to specialized methods"""
        self._register_generate_routes()
        self._register_model_routes()
        self._register_stats_routes()
        self._register_network_routes()
        self.app.include_router(self.router)

    def _register_generate_routes(self):
        """Register text, embedding and multi-modal generation endpoints"""
        routes = (
            ("/v1/chat/completions", self._route_chat_completions, "chat_completions"),
            ("/v1/embeddings", self._route_embeddings, "embeddings"),
            ("/v1/responses", self._route_responses, "responses"),
        )
        for path, endpoint, name in routes:
            self.router.add_api_route(path, endpoint, methods=["POST"], name=name)

    @staticmethod
    async def _guarded(call):
        """Await `call()`; let HTTPExceptions through, map anything else to a 500."""
        try:
            return await call()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def _route_chat_completions(self, request: ChatCompletionRequest):
        return await self._guarded(
            lambda: self._dispatch_text(build_generation_request(request))
        )

    async def _route_embeddings(self, request: EmbeddingCreateRequest):
        return await self._guarded(
            lambda: self._dispatch_embedding(
                request.model,
                request.input,
                request.encoding_format,
                request.dimensions,
            )
        )

    async def _route_responses(self, request: AnyResponseRequest):
        handlers = {
            "text": self._handle_text_response,
            "image": self._handle_image_response,
            "embedding": self._handle_embedding_response,
        }
        handler = handlers.get(request.type)
        if not handler:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported response type: '{request.type}'",
            )
        return await self._guarded(lambda: handler(request))

    def _register_model_routes(self):
        """Register model management endpoints"""
        self.router.add_api_route(
            "/v1/models/request",
            self._route_request_model,
            methods=["POST"],
            response_model=ModelStatusResponse,
            name="request_model_v1",
        )
        self.router.add_api_route(
            "/v1/models/status",
            self._route_model_status,
            methods=["GET"],
            response_model=ModelStatusDistributionResponse,
            name="get_model_status_v1",
        )
        self.router.add_api_route(
            "/v1/models/demand",
            self._route_model_demand,
            methods=["GET"],
            name="get_api_demand_stats_v1",
        )
        self.router.add_api_route(
            "/v1/models/available",
            self._route_available_models,
            methods=["GET"],
            name="list_available_models_v1",
        )

    def _route_request_model(self, job_request: JobRequest, request: Request):
        """
        Request a model to be loaded on the network.

        Status values in response:
          - active: validator and workers have fully loaded the model
          - initializing: worker(s) are currently loading their assigned modules
          - inactive: model not found / request rejected
        """
        model_name = job_request.hf_name
        try:
            # Mark this model as API-requested for prioritization (also when a job
            # already exists, so repeat requests keep it prioritized)
            self.api_requested_models.add(model_name)

            # Check if a job already exists for this model
            _job_id, job_data = self._find_job_for_model(model_name)
            if job_data is not None:
                status_info = self._derive_status_from_job(job_data)
                return ModelStatusResponse(
                    model=model_name,
                    status=status_info["status"],
                    message=status_info["message"],
                )

            # No existing job, create one
            if not self._submit_model_job(
                model_name,
                job_request.model_type,
                payment=job_request.payment,
                duration=job_request.time,
                client_ip=request.client.host,
            ):
                # Job creation was immediately rejected (e.g. rate-limited)
                return ModelStatusResponse(
                    model=model_name,
                    status="inactive",
                    message=f"Model {model_name} request was rejected.",
                )

            return ModelStatusResponse(
                model=model_name,
                status="initializing",
                message=f"Model {model_name} is initializing.",
            )

        except Exception as e:
            logging.error(f"Error in /v1/models/request: {e}")
            return ModelStatusResponse(
                model=model_name,
                status="inactive",
                message=f"Error requesting model: {str(e)}",
            )

    def _submit_model_job(
        self,
        model_name: str,
        model_type: ModelType = ModelType.CHAT,
        *,
        payment: int = 0,
        duration: int = 0,
        client_ip: str = None,
    ) -> bool:
        """
        The single way the API asks the network to host a model.

        Marks the model as API-requested (for prioritisation), builds the hosted
        job and submits it. Returns False if the job was rejected outright (e.g.
        rate-limited), in which case the prioritisation mark is undone. Callers
        decide how to report that: an HTTP status, a 503, or just a log line.
        """
        self.api_requested_models.add(model_name)
        job_data = build_hf_job_data(
            model_name=model_name,
            author=self.smart_node.rsa_key_hash,
            model_type=model_type,
            payment=payment,
            duration=duration,
        )
        args = (job_data,) if client_ip is None else (job_data, client_ip)
        if self.smart_node.create_hf_job(*args) is False:
            self.api_requested_models.discard(model_name)
            return False
        return True

    def _route_model_status(
        self, model: str = Query(..., description="HuggingFace model name")
    ):
        """
        Check the loading status of a specific model, including per-module
        distribution across workers.

        Status values:
          - active: all offloaded modules are fully loaded
          - initializing: at least one offloaded module is still loading
          - inactive: no job found for this model
        """
        try:
            _job_id, job_data = self._find_job_for_model(model)

            if job_data is None:
                return ModelStatusDistributionResponse(
                    model_name=model,
                    status="inactive",
                    message="Model not found.",
                    distribution=None,
                )

            distribution = self._build_distribution_info(job_data)
            status_info = self._derive_status_from_distribution(distribution)

            return ModelStatusDistributionResponse(
                model_name=model,
                status=status_info["status"],
                message=status_info["message"],
                distribution=distribution,
            )

        except Exception as e:
            logging.error(f"Error in /v1/models/status: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _route_model_demand(
        self,
        days: int = Query(30, ge=1, le=90),
        limit: int = Query(10, ge=1, le=50),
    ):
        """Return current API demand statistics"""
        return get_popular_model_stats(days=days, limit=limit)

    def _route_available_models(self):
        """List currently active models and whether each supports reasoning"""
        try:
            models = self._collect_available_models()
            return {
                "active_models": list(models),  # unchanged, for existing clients
                "models": list(models.values()),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def _collect_available_models(self) -> dict:
        """Map model name -> {name, type, supports_reasoning} (public, active jobs)."""
        models = {}
        for job_id in self.smart_node.jobs:
            job = self.smart_node.dht.query(job_id)
            if not self._is_public_active_job(job):
                continue
            name = job.get("model_name")
            entry = models.setdefault(
                name, {"name": name, "type": "text", "supports_reasoning": None}
            )
            self._apply_job_capabilities(entry, job)
        return models

    @staticmethod
    def _is_public_active_job(job) -> bool:
        return bool(isinstance(job, dict) and job.get("public") and job.get("active"))

    @staticmethod
    def _apply_job_capabilities(entry: dict, job: dict):
        if job.get("model_type") == "embedding":
            entry["type"] = "embedding"
            entry["supports_reasoning"] = False
        elif job.get("supports_reasoning") is not None:  # null = unknown
            entry["supports_reasoning"] = job["supports_reasoning"]

    def _register_stats_routes(self):
        """Register statistics and monitoring endpoints"""

        @self.app.get("/stats")
        async def get_network_stats():
            return self.smart_node.get_tensorlink_status()

        @self.app.get("/network-history")
        async def get_network_history(
            days: int = Query(30, ge=1, le=1093),
            include_weekly: bool = False,
            include_summary: bool = True,
            include_device: bool = False,
        ):
            return self.smart_node.get_network_status(
                days=days,
                include_weekly=include_weekly,
                include_summary=include_summary,
                include_device=include_device,
            )

        @self.app.get("/proposal-history")
        async def get_proposals(limit: int = Query(30, ge=1, le=180)):
            """
            Retrieve historical proposals from the node's archive cache.
            """
            return self.smart_node.keeper.get_proposals(limit=limit)

    def _register_network_routes(self):
        """Register network and node information endpoints"""

        @self.app.get("/node-info")
        async def get_node_info(node_id: str):
            """
            Get information about a specific node in the network.
            Returns node type, last seen, and relevant data based on role.
            """
            node_info = self.smart_node.dht.query(node_id)
            if node_info:
                return_package = {
                    "pubKeyHash": node_id,
                    "type": node_info["role"],
                    "lastSeen": node_info["last_seen"],
                    "data": {},
                }

                if node_info["role"] == "V":
                    # Validator-specific data
                    pass
                elif node_info["role"] == "W":
                    # Worker-specific data
                    node_info["rewards"] = (
                        self.smart_node.contract_manager.get_worker_claim_data(
                            node_info["address"]
                        )
                    )
                return return_package
            else:
                return {}

        @self.app.get("/claim-info")
        async def get_worker_claims(node_address: str):
            """Get claim information for a specific worker node"""
            return self.smart_node.contract_manager.get_worker_claim_data(node_address)

    def _find_job_for_model(self, model_name: str):
        """
        Search for an active, API-hosted job matching model_name.

        Returns (job_id, job_data) if found, else (None, None).
        """
        try:
            for job_id in self.smart_node.jobs:
                job_data = self.smart_node.dht.query(job_id)
                if (
                    isinstance(job_data, dict)
                    and job_data.get("model_name", "") == model_name
                    and job_data.get("hosted")
                    and job_data.get("api")
                    and job_data.get("active")
                ):
                    return job_id, job_data
        except Exception as e:
            logging.error(f"Error finding job for model '{model_name}': {e}")
        return None, None

    def _build_distribution_info(self, job_data: dict) -> dict:
        """
        Build the per-module distribution map for a job.

        For each module in the job's distribution config, returns a
        ModelDistributionEntry with:
          - worker_id: assigned worker ID (null for validator-hosted modules)
          - loaded: True only if the worker has confirmed the module is loaded
          - type: module type string from the distribution config
          - memory: memory footprint in bytes from the distribution config

        A module is considered loaded when the ValidatorThread has received
        the worker's LOADED confirmation and set modules[module_id]["loaded"] = True
        (see the _handle_module_loaded override in validator_thread.py).
        """
        distribution = {}
        job_distribution = job_data.get("distribution", {})

        for module_id, module_info in job_distribution.items():
            if not isinstance(module_info, dict):
                continue

            module_type = module_info.get("type", "")
            memory = int(module_info.get("memory", 0))

            if "offloaded" in module_type:
                assigned_workers = module_info.get("assigned_workers", [])
                worker_id = assigned_workers[-1] if assigned_workers else None

                # Check the node thread's module tracker for the loaded flag.
                module_state = self.smart_node.modules.get(module_id, {})
                loaded = bool(module_state.get("status", "inactive") == "loaded")

                distribution[module_id] = ModelDistributionEntry(
                    worker_id=worker_id,
                    loaded=loaded,
                    type=module_type,
                    memory=memory,
                )
            else:
                # Validator-hosted module
                distribution[module_id] = ModelDistributionEntry(
                    worker_id=None,
                    loaded=True,
                    type=module_type,
                    memory=memory,
                )

        return distribution

    def _derive_status_from_distribution(self, distribution: dict) -> dict:
        """
        Derive model status from the built distribution map.

        active       : all offloaded modules have loaded=True
        initializing : at least one offloaded module has loaded=False
        """
        offloaded = [
            entry for entry in distribution.values() if "offloaded" in entry.type
        ]

        if not offloaded:
            # No offloaded modules validator hosts everything; treat as active
            return {
                "status": "active",
                "message": "Model is active and ready to use.",
            }

        if all(entry.loaded for entry in offloaded):
            return {
                "status": "active",
                "message": "Model is active and ready to use.",
            }
        else:
            return {
                "status": "initializing",
                "message": "Model is initializing, workers are loading their assigned modules.",
            }

    def _derive_status_from_job(self, job_data: dict) -> dict:
        """Convenience wrapper: build distribution then derive status."""
        distribution = self._build_distribution_info(job_data)
        return self._derive_status_from_distribution(distribution)

    def _check_model_status(self, model_name: str) -> dict:
        """
        [Legacy] Check if a model is loaded, loading, or not loaded.
        Returns a dict with 'status' and 'message'.
        """
        status = "not_loaded"
        message = "Model is not currently loaded"

        try:
            # Check if there is a public job with this module
            for job_id in self.smart_node.jobs:
                job_data = self.smart_node.dht.query(job_id)
                if (
                    isinstance(job_data, dict)
                    and job_data.get("model_name", "") == model_name
                    and job_data.get("hosted")
                    and job_data.get("api")
                    and job_data.get("active")
                ):
                    status = "loaded"
                    message = f"Model {model_name} is loaded and ready"
                    break

        except Exception as e:
            logging.error(f"Error checking model status: {e}")
            status = "error"
            message = f"Error checking model status: {str(e)}"

        return {"status": status, "message": message}

    def _trigger_model_load(self, model_name: str):
        """Trigger the ML validator to load a specific model"""
        try:
            self._submit_model_job(model_name, ModelType.CHAT)

        except Exception as e:
            logging.error(f"Error triggering model load: {e}")

    def _start_server(self):
        """Start the FastAPI server in a separate thread"""

        def run_server():
            async def app_startup():
                self.server_loop = asyncio.get_running_loop()

            self.app.add_event_handler("startup", app_startup)

            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                timeout_keep_alive=20,
                limit_concurrency=100,
                lifespan="on",
            )

        Thread(target=run_server, daemon=True).start()

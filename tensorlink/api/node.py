from tensorlink.ml.utils.utils import get_popular_model_stats
from tensorlink.api.models import (
    JobRequest,
    GenerationRequest,
    ModelStatusResponse,
    ChatCompletionRequest,
    AnyResponseRequest,
    TextResponseRequest,
    ImageResponseRequest,
    EmbeddingResponseRequest,
)
from tensorlink.ml.utils.formatter import ResponseFormatter
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException, APIRouter, Request, Query
from collections import defaultdict
from threading import Thread
import logging
import uvicorn
import asyncio
import random
import time


def build_hf_job_data(
    *,
    model_name: str,
    author: str,
    model_type: str = "hf",
    payment: int = 0,
    duration: int = 0,
    hosted: bool = True,
    training: bool = False,
    seed_validators=None,
):
    if seed_validators is None:
        seed_validators = [author]

    return {
        "author": author,
        "api": True,
        "active": True,
        "hosted": hosted,
        "training": training,
        "payment": payment,
        "time": duration,
        "capacity": 0,
        "n_pipelines": 1,
        "dp_factor": 1,
        "distribution": {"model_name": model_name},
        "n_workers": 0,
        "model_name": model_name,
        "seed_validators": seed_validators,
        "model_type": model_type,
    }


def _parse_chat_messages(messages):
    """
    Parse chat messages into system messages, history, and last user message.
    Returns: (system_messages, history, last_user_message)
    """
    system_messages = []
    conversation = []

    for msg in messages:
        if msg.role not in ("system", "user", "assistant"):
            continue

        if msg.role == "system":
            system_messages.append(msg.content)
        else:
            conversation.append({"role": msg.role, "content": msg.content})

    # Find last user message
    last_user_message = None
    last_user_idx = None

    for idx in range(len(conversation) - 1, -1, -1):
        if conversation[idx]["role"] == "user":
            last_user_message = conversation[idx]["content"]
            last_user_idx = idx
            break

    if last_user_message is None:
        raise HTTPException(status_code=400, detail="No user message found")

    # Build history (everything before the last user message)
    history = conversation[:last_user_idx]

    # Prepend system message to history if present
    if system_messages:
        combined_system = "\n".join(system_messages)
        history.insert(0, {"role": "system", "content": combined_system})

    return system_messages, history, last_user_message


def _build_generation_request(request) -> GenerationRequest:
    """Shared factory: ChatCompletionRequest or TextResponseRequest → GenerationRequest."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")
    _, history, last_user_message = _parse_chat_messages(request.messages)
    return GenerationRequest(
        hf_name=request.model,
        message=last_user_message,
        history=history,
        temperature=request.temperature,
        top_p=request.top_p,
        max_new_tokens=request.max_tokens,
        stream=request.stream,
        input_format="chat",
        output_format="openai",
        do_sample=(request.temperature or 0) > 0,
        is_chat_completion=True,
    )


class TensorlinkAPI:
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

        self.model_name_to_request = {}
        self.model_request_timestamps = defaultdict(list)

        # Track models requested via API for prioritization
        self.api_requested_models = set()
        self.streaming_responses = {}
        self.pending_requests: dict[int, asyncio.Future] = {}
        self.api_loop: asyncio.AbstractEventLoop = None
        self._cancelled_requests: set = set()
        self.server_loop = None

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
        @self.router.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            try:
                return await self._dispatch_text(_build_generation_request(request))
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/v1/responses")
        async def responses(request: AnyResponseRequest):
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
            try:
                return await handler(request)
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    def _register_model_routes(self):
        """Register model management endpoints"""

        @self.router.post("/v1/models/request", response_model=ModelStatusResponse)
        def request_model(job_request: JobRequest, request: Request):
            """
            Explicitly request a model to be loaded on the network. Currently, models
            are only publicly accessible. Paid jobs for private use are unavailable at
            this time.

            Returns a ModelStatusResponse with status:
              - "active"   : validator and all workers have fully loaded the model
              - "loading"  : model job exists but worker(s) are still loading modules
              - "inactive" : model not found; loading has been initiated
            """
            try:
                client_ip = request.client.host
                model_name = job_request.hf_name

                # Mark this model as API-requested for prioritization
                self.api_requested_models.add(model_name)

                # Check current status
                status = self._check_model_status(model_name)

                if status["status"] == "active":
                    return ModelStatusResponse(
                        model_name=model_name,
                        status="active",
                        message="Model is already loaded and ready to use.",
                    )
                elif status["status"] == "loading":
                    return ModelStatusResponse(
                        model_name=model_name,
                        status="loading",
                        message="Model is currently being loaded by worker(s).",
                    )

                # Model not present, trigger the loading process
                job_data = build_hf_job_data(
                    model_name=model_name,
                    author=self.smart_node.rsa_key_hash,
                    payment=job_request.payment,
                    duration=job_request.time,
                    model_type=job_request.model_type,
                )

                self.smart_node.create_hf_job(job_data, client_ip)

                return ModelStatusResponse(
                    model_name=model_name,
                    status="inactive",
                    message=f"Model {model_name} not found. Loading has been initiated.",
                )

            except Exception as e:
                return ModelStatusResponse(
                    model_name=job_request.hf_name,
                    status="inactive",
                    message=f"Error requesting model: {str(e)}",
                )

        @self.router.get("/v1/models/status", response_model=ModelStatusResponse)
        def get_model_status(
            model_name: str = Query(
                ..., description="HuggingFace model name, e.g. Qwen/Qwen3-8B"
            )
        ):
            """
            Check the loading status of a specific model.

            Pass the model name as a query parameter to avoid path-routing issues
            with model names that contain forward slashes (e.g. Qwen/Qwen3-8B).

            Example: GET /v1/models/status?model_name=Qwen/Qwen3-8B

            Returns status:
              - "active"   : validator and all workers have fully loaded the model
              - "loading"  : job exists but worker(s) are still loading their modules
              - "inactive" : model not found on the network
            """
            status = self._check_model_status(model_name)
            return ModelStatusResponse(
                model_name=model_name,
                status=status["status"],
                message=status["message"],
            )

        @self.router.get("/v1/models/demand")
        async def get_api_demand_stats(
            days: int = Query(30, ge=1, le=90),
            limit: int = Query(10, ge=1, le=50),
        ):
            """Return current API demand statistics"""
            return get_popular_model_stats(days=days, limit=limit)

        @self.router.get("/v1/models/available")
        def list_available_models():
            """List all currently active (fully loaded) models"""
            try:
                jobs = [self.smart_node.dht.query(a) for a in self.smart_node.jobs]
                active_models = set(
                    j.get("model_name")
                    for j in jobs
                    if isinstance(j, dict)
                    and j.get("active")
                    and j.get("api")
                    and j.get("hosted")
                )

                return {
                    "active_models": list(active_models),
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    def _register_stats_routes(self):
        """Register statistics and monitoring endpoints"""

        @self.app.get("/stats")
        async def get_network_stats():
            return self.smart_node.get_tensorlink_status()

        @self.app.get("/network-history")
        async def get_network_history(
            days: int = Query(30, ge=1, le=90),
            include_weekly: bool = False,
            include_summary: bool = True,
        ):
            return self.smart_node.get_network_status(
                days=days,
                include_weekly=include_weekly,
                include_summary=include_summary,
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

    async def _dispatch_text(self, gen_request: GenerationRequest):
        start_time = time.time()
        self._log_model_request(gen_request.hf_name)
        gen_request.output = None
        gen_request.id = hash(f"req_{random.random()}")

        model_status = self._check_model_status(gen_request.hf_name)
        if model_status["status"] == "inactive":
            self._trigger_model_load(gen_request.hf_name)
            raise HTTPException(
                status_code=503,
                detail=f"Model '{gen_request.hf_name}' is not available. "
                f"Loading has been requested, try again shortly.",
            )
        if model_status["status"] == "loading":
            raise HTTPException(
                status_code=503,
                detail=f"Model '{gen_request.hf_name}' is still loading. Try again shortly.",
            )

        if gen_request.stream:
            return StreamingResponse(
                self._generate_stream(gen_request, str(gen_request.id), start_time),
                media_type="text/event-stream",
            )
        gen_request = await self._wait_for_result(gen_request)
        if getattr(gen_request, "formatted_response", None):
            return gen_request.formatted_response
        return {"text": gen_request.output}

    async def _handle_text_response(self, request: TextResponseRequest):
        return await self._dispatch_text(_build_generation_request(request))

    async def _handle_image_response(self, request: ImageResponseRequest):
        raise HTTPException(
            status_code=501, detail="Image generation is not yet implemented."
        )

    async def _handle_embedding_response(self, request: EmbeddingResponseRequest):
        raise HTTPException(
            status_code=501, detail="Embeddings are not yet implemented."
        )

    def _log_model_request(self, model_name: str):
        """Log and track model requests for prioritization"""
        current_time = time.time()
        self.model_request_timestamps[model_name].append(current_time)

        # Keep only requests from last 5 minutes
        cutoff = current_time - 300
        self.model_request_timestamps[model_name] = [
            ts for ts in self.model_request_timestamps[model_name] if ts > cutoff
        ]

        if model_name not in self.model_name_to_request:
            self.model_name_to_request[model_name] = 1
        self.model_name_to_request[model_name] += 1

    async def _generate_stream(self, request, request_id, start_time):
        """Generator function for streaming tokens"""
        loop = asyncio.get_running_loop()
        self.api_loop = loop

        token_queue = asyncio.Queue()
        self.streaming_responses[request.id] = token_queue

        request.stream = True
        request.start_time = start_time
        self.smart_node.endpoint_requests["incoming"].append(request)

        try:
            while True:
                try:
                    token_data = await asyncio.wait_for(token_queue.get(), timeout=30.0)

                    if token_data.get("done"):
                        sse_chunk = token_data.get("token", "data: [DONE]\n\n")
                        yield sse_chunk
                        break

                    sse_chunk = token_data.get("token")
                    if sse_chunk:
                        yield sse_chunk

                except asyncio.TimeoutError:
                    yield ResponseFormatter.format_stream_error(
                        error_message="Generation timed out", error_type="timeout_error"
                    )
                    break

        except asyncio.CancelledError:
            # Client disconnected
            request.cancelled = True
            raise

        except Exception as e:
            yield ResponseFormatter.format_stream_error(
                error_message=str(e), error_type="internal_error"
            )

        finally:
            self.streaming_responses.pop(request.id, None)

    def send_token_to_stream(self, request_id, token=None, done=False, **kwargs):
        """Push pre-formatted streaming chunks to the SSE queue"""
        # Drop tokens for cancelled/disconnected requests
        if getattr(self, '_cancelled_requests', set()).__contains__(request_id):
            return

        if not self.server_loop:
            return

        queue = self.streaming_responses.get(request_id)
        if not queue:
            return

        data = {"token": token, "done": done, **kwargs}
        asyncio.run_coroutine_threadsafe(queue.put(data), self.server_loop)

    def resolve_pending_request(self, response):
        """Resolve a non-streaming Future from the ML thread"""
        if not self.api_loop:
            return

        fut = self.pending_requests.get(response.id)
        if fut and not fut.done():
            self.api_loop.call_soon_threadsafe(fut.set_result, response)

    def _has_pending_module_request(self, worker_id: str, module_id: str) -> bool:
        pending = self.smart_node.requests.get(worker_id, [])
        return any(isinstance(r, str) and r == f"MODULE{module_id}" for r in pending)

    def _distribution_still_loading(self, distribution: dict) -> bool:
        for module_id, module_info in distribution.items():
            if "offloaded" not in module_info.get("type", ""):
                continue

            assigned_workers = module_info.get("assigned_workers") or []

            for worker_id in assigned_workers:
                if worker_id is None:
                    continue

                if self._has_pending_module_request(worker_id, module_id):
                    return True

        return False

    def _worker_modules_still_loading(self, worker_modules: dict) -> bool:
        for worker_id, module_id in worker_modules.items():
            if self._has_pending_module_request(worker_id, module_id):
                return True

        return False

    def _check_model_status(self, model_name: str) -> dict:
        """
        Check whether a model is active, loading, or inactive.
        This intentionally doesn't rely on the ML-process-side model_state dict
        (which lives in DistributedValidator and is not directly accessible from the
        node process).
        """
        try:
            for job_id in self.smart_node.jobs:
                job_data = self.smart_node.dht.query(job_id)

                if not isinstance(job_data, dict):
                    continue

                matches_model = (
                    job_data.get("model_name") == model_name
                    and job_data.get("hosted")
                    and job_data.get("api")
                    and job_data.get("active")
                )

                if not matches_model:
                    continue

                distribution = job_data.get("distribution", {})
                worker_modules = job_data.get("worker_modules", {})

                still_loading = self._worker_modules_still_loading(
                    worker_modules
                ) or self._distribution_still_loading(distribution)

                if still_loading:
                    return {
                        "status": "loading",
                        "message": (
                            f"Model {model_name} is being loaded by worker(s)."
                        ),
                    }

                return {
                    "status": "active",
                    "message": f"Model {model_name} is loaded and ready.",
                }

        except Exception as e:
            logging.error(f"Error checking model status: {e}")

            return {
                "status": "inactive",
                "message": f"Error checking model status: {str(e)}",
            }

        return {
            "status": "inactive",
            "message": f"Model {model_name} not found on the network.",
        }

    def _trigger_model_load(self, model_name: str):
        """Trigger the ML validator to load a specific model"""
        try:
            # Mark as API requested
            self.api_requested_models.add(model_name)
            job_data = build_hf_job_data(
                model_name=model_name,
                author=self.smart_node.rsa_key_hash,
            )
            self.smart_node.create_hf_job(job_data)

        except Exception as e:
            logging.error(f"Error triggering model load: {e}")

    async def _wait_for_result(self, request: GenerationRequest, timeout: int = 300):
        """Wait for the generation result using a Future instead of polling outgoing list"""
        loop = asyncio.get_running_loop()
        self.api_loop = loop

        fut = loop.create_future()
        self.pending_requests[request.id] = fut
        self.smart_node.endpoint_requests["incoming"].append(request)

        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            request.cancelled = True
            raise HTTPException(status_code=504, detail="Request timed out")
        finally:
            self.pending_requests.pop(request.id, None)

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

from tensorlink.ml.utils import get_popular_model_stats
from tensorlink.api.models import (
    JobRequest,
    GenerationRequest,
    ModelStatusResponse,
    ChatCompletionRequest,
)
from tensorlink.ml.formatter import ResponseFormatter
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
    time: int = 0,
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
        "time": time,
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


class TensorlinkAPI:
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
        """Register generation and chat completion endpoints"""

        @self.router.post("/v1/generate")
        async def generate(request: GenerationRequest):
            """Updated /v1/generate endpoint"""
            try:
                start_time = time.time()
                request.input_format = getattr(request, "input_format", "raw")
                request.output_format = getattr(request, "output_format", "simple")

                # Log model request
                self._log_model_request(request.hf_name)

                request.output = None
                request_id = f"req_{hash(random.random())}"
                request.id = hash(request_id)

                # Model status checks
                model_status = self._check_model_status(request.hf_name)
                if model_status["status"] == "not_loaded":
                    self._trigger_model_load(request.hf_name)
                    raise HTTPException(
                        status_code=503,
                        detail=f"Model '{request.hf_name}' has been requested on the network. "
                        f"Please try again in a few moments.",
                    )
                elif model_status["status"] == "loading":
                    raise HTTPException(
                        status_code=503,
                        detail=f"Model {request.hf_name} is still loading. Please try again.",
                    )

                stream = getattr(request, 'stream', False)

                if stream:
                    return StreamingResponse(
                        self._generate_stream(request, request_id, start_time),
                        media_type="text/event-stream",
                    )
                else:
                    # Non-streaming
                    request = await self._wait_for_result(request)

                    # Return formatted response (not just output text)
                    if hasattr(request, 'formatted_response'):
                        return request.formatted_response
                    else:
                        # Fallback for legacy compatibility
                        return {"text": request.output}

            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """
            OpenAI-compatible chat completions endpoint.
            Maps OpenAI request into Tensorlink GenerationRequest.
            """
            try:
                if not request.messages:
                    raise HTTPException(
                        status_code=400, detail="messages cannot be empty"
                    )

                # Parse messages into system messages, history, and last user message
                system_messages, history, last_user_message = _parse_chat_messages(
                    request.messages
                )

                # Create GenerationRequest
                gen_request = GenerationRequest(
                    hf_name=request.model,
                    message=last_user_message,
                    history=history,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_new_tokens=request.max_tokens,
                    stream=request.stream,
                    input_format="chat",
                    output_format="openai",
                    do_sample=request.temperature > 0,
                    is_chat_completion=True,
                )

                # Call generate endpoint
                return await generate(gen_request)

            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    def _register_model_routes(self):
        """Register model management endpoints"""

        @self.router.post("/request-model", response_model=ModelStatusResponse)
        def request_model(job_request: JobRequest, request: Request):
            """
            Explicitly request a model to be loaded on the network. Currently, models
            are only publicly accessible. Paid jobs for private use are unavailable at
            this time.
            """
            try:
                client_ip = request.client.host
                model_name = job_request.hf_name

                # Mark this model as API-requested for prioritization
                self.api_requested_models.add(model_name)

                # Check current status
                status = self._check_model_status(model_name)

                if status["status"] == "loaded":
                    return ModelStatusResponse(
                        model_name=model_name,
                        status="loaded",
                        message="Model is already loaded and ready to use.",
                    )
                elif status["status"] == "loading":
                    return ModelStatusResponse(
                        model_name=model_name,
                        status="loading",
                        message="Model is currently being loaded.",
                    )

                # Trigger the loading process
                job_data = build_hf_job_data(
                    model_name=model_name,
                    author=self.smart_node.rsa_key_hash,
                    payment=job_request.payment,
                    time=job_request.time,
                    model_type=job_request.model_type,
                )

                self.smart_node.create_hf_job(job_data, client_ip)

                return ModelStatusResponse(
                    model_name=model_name,
                    status="loading",
                    message=f"Model {model_name} loading has been initiated",
                )

            except Exception as e:
                return ModelStatusResponse(
                    model_name=job_request.hf_name,
                    status="error",
                    message=f"Error requesting model: {str(e)}",
                )

        @self.router.get(
            "/model-status/{model_name}", response_model=ModelStatusResponse
        )
        def get_model_status(model_name: str):
            """Check the loading status of a specific model"""
            status = self._check_model_status(model_name)
            return ModelStatusResponse(
                model_name=model_name,
                status=status["status"],
                message=status["message"],
            )

        @self.router.get("/model-demand")
        async def get_api_demand_stats(
            days: int = Query(30, ge=1, le=90),
            limit: int = Query(10, ge=1, le=50),
        ):
            """Return current API demand statistics"""
            return get_popular_model_stats(days=days, limit=limit)

        @self.router.get("/models")
        def list_available_models():
            """List all currently loaded models"""
            try:
                public_models = set(
                    (
                        a.get("model_name", "")
                        if a.get("public", False) and a.get("", "")
                        else None
                    )
                    for a in self.smart_node.modules.values()
                )

                return {
                    "active_models": list(public_models),
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

    def _check_model_status(self, model_name: str) -> dict:
        """Check if a model is loaded, loading, or not loaded"""
        status = "not_loaded"
        message = "Model is not currently loaded"

        try:
            # Check if there is a public job with this module
            for job_id in self.smart_node.jobs:
                job_data = self.smart_node.dht.query(job_id)
                if (
                    job_data.get("model_name", "") == model_name
                    and job_data.get("hosted")
                    and job_data.get("api")
                    and job_data.get("public")
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

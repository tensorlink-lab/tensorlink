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
import queue
import time
import json


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

        self.server_loop = None

        self._define_routes()
        self._start_server()

    def _define_routes(self):
        @self.router.post("/v1/generate")
        async def generate(request: GenerationRequest):
            """Updated /v1/generate endpoint"""
            try:
                start_time = time.time()
                request.input_format = getattr(request, "input_format", "raw")
                request.output_format = getattr(request, "output_format", "simple")

                # Log model request
                current_time = time.time()
                self.model_request_timestamps[request.hf_name].append(current_time)
                cutoff = current_time - 300
                self.model_request_timestamps[request.hf_name] = [
                    ts
                    for ts in self.model_request_timestamps[request.hf_name]
                    if ts > cutoff
                ]

                if request.hf_name not in self.model_name_to_request:
                    self.model_name_to_request[request.hf_name] = 1
                self.model_name_to_request[request.hf_name] += 1

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
                    self.smart_node.endpoint_requests["incoming"].append(request)
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

                # Separate system messages from conversation
                system_messages = []
                conversation = []

                for msg in request.messages:
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

        @self.router.get("/available-models")
        def list_available_models():
            """List all currently loaded models"""
            try:
                loaded_models = []
                loading_models = []

                # Query the node's worker for model status
                response = self.smart_node.request_queue.put(
                    {"type": "get_loaded_models", "args": None}
                )

                # Wait for response
                try:
                    result = self.smart_node.response_queue.get(timeout=5)
                    if result.get("status") == "SUCCESS":
                        model_info = result.get("return", {})
                        loaded_models = model_info.get("loaded", [])
                        loading_models = model_info.get("loading", [])
                except queue.Empty:
                    pass

                return {
                    "loaded_models": loaded_models,
                    "loading_models": loading_models,
                    "api_requested_models": list(self.api_requested_models),
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

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

        self.app.include_router(self.router)

    async def _generate_stream(self, request, request_id, start_time):
        """Generator function for streaming tokens"""
        try:
            # Create queue for this request to receive tokens
            token_queue = asyncio.Queue()
            self.streaming_responses[request.id] = token_queue

            # Mark request as streaming
            request.stream = True
            request.start_time = start_time

            # Add to processing queue
            self.smart_node.endpoint_requests["incoming"].append(request)

            # Stream tokens as they arrive
            while True:
                try:
                    # Wait for next token with timeout
                    token_data = await asyncio.wait_for(token_queue.get(), timeout=30.0)

                    # Check if generation is complete
                    if token_data.get("done"):
                        # Get the SSE-formatted string (could be final chunk or error)
                        sse_chunk = token_data.get("token", "data: [DONE]\n\n")
                        yield sse_chunk
                        break

                    # Get the SSE-formatted chunk string
                    sse_chunk = token_data.get("token")
                    if sse_chunk:
                        yield sse_chunk
                    else:
                        # Skip empty chunks
                        continue

                except asyncio.TimeoutError:
                    error_chunk = ResponseFormatter.format_stream_error(
                        error_message="Generation timed out", error_type="timeout_error"
                    )
                    yield error_chunk
                    break

        except Exception as e:
            error_chunk = ResponseFormatter.format_stream_error(
                error_message=str(e), error_type="internal_error"
            )
            yield error_chunk
        finally:
            # Clean up
            if request.id in self.streaming_responses:
                del self.streaming_responses[request.id]

    def send_token_to_stream(self, request_id, token=None, done=False, **kwargs):
        """
        Push pre-formatted streaming chunks to the SSE queue.
        The 'token' here is the full SSE chunk string prepared by the validator.
        """
        if request_id not in self.streaming_responses:
            return

        if not self.server_loop:
            return

        response_queue = self.streaming_responses[request_id]

        # Build data dictionary to put into the asyncio queue
        data = {"token": token, "done": done, **kwargs}

        # Safely enqueue for StreamingResponse
        asyncio.run_coroutine_threadsafe(response_queue.put(data), self.server_loop)

    def _check_model_status(self, model_name: str) -> dict:
        """Check if a model is loaded, loading, or not loaded"""
        status = "not_loaded"
        message = "Model is not currently loaded"

        try:
            # Check if there is a public job with this module
            for module_id, module in self.smart_node.modules.items():
                if module.get("model_name", "") == model_name:
                    if module.get("public", False):
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
        """Wait for the generation result with timeout"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if result is ready
            for idx, req in enumerate(self.smart_node.endpoint_requests["outgoing"]):
                if req.id == request.id:
                    return self.smart_node.endpoint_requests["outgoing"].pop(idx)

            await asyncio.sleep(0.1)

        raise HTTPException(status_code=504, detail="Request timed out")

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

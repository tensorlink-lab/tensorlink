from tensorlink.ml.utils import get_popular_model_stats, extract_assistant_response
from tensorlink.api.models import (
    JobRequest,
    GenerationRequest,
    ModelStatusResponse,
)
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


def _format_response(
    request: GenerationRequest,
    processing_time: float,
    request_id: str,
):
    """
    Format the response based on the requested format type.

    Args:
        request: The original generation request with output
        processing_time: Time taken to process the request
        request_id: Unique identifier for this request

    Returns:
        Dictionary formatted according to response_format
    """
    timestamp = int(time.time())

    # Extract clean text from output
    clean_output = extract_assistant_response(request.output, request.hf_name)

    if request.response_format == "simple":
        # Minimal response - just the text
        return {"response": clean_output}

    elif request.response_format == "openai":
        # OpenAI-compatible format
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": timestamp,
            "model": request.hf_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": clean_output},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": -1,  # Not tracked in current implementation
                "completion_tokens": -1,
                "total_tokens": -1,
            },
        }

    else:  # "full" format (default, comprehensive response with all metadata)
        return {
            "id": request_id,
            "model": request.hf_name,
            "response": clean_output,
            "raw_output": request.output,
            "created": timestamp,
            "processing_time": round(processing_time, 3),
            "generation_params": {
                "max_length": request.max_length,
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "do_sample": request.do_sample,
                "num_beams": request.num_beams,
            },
            "metadata": {
                "has_history": bool(request.history),
                "history_length": len(request.history) if request.history else 0,
                "prompt_used": request.prompt is not None,
            },
        }


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
            try:
                start_time = time.time()

                # Log model request
                current_time = time.time()
                self.model_request_timestamps[request.hf_name].append(current_time)

                cutoff = current_time - 300
                self.model_request_timestamps[request.hf_name] = [
                    ts
                    for ts in self.model_request_timestamps[request.hf_name]
                    if ts > cutoff
                ]

                # Update request counter
                if request.hf_name not in self.model_name_to_request:
                    self.model_name_to_request[request.hf_name] = 1
                self.model_name_to_request[request.hf_name] += 1

                request.output = None
                request_id = f"req_{hash(random.random())}"
                request.id = hash(request_id)

                # Check if model is loaded, if not trigger loading
                model_status = self._check_model_status(request.hf_name)
                if model_status["status"] == "not_loaded":
                    # Trigger model loading
                    self._trigger_model_load(request.hf_name)
                    raise HTTPException(
                        status_code=503,
                        detail=f"Model '{request.hf_name}' has been requested on the network. Please try again in a few "
                        f"moments, or view available models at https://smartnodes.ca/app.",
                    )
                elif model_status["status"] == "loading":
                    raise HTTPException(
                        status_code=503,
                        detail=f"Model {request.hf_name} is still loading. Please try again in a few moments.",
                    )

                # Check if streaming is requested
                stream = getattr(request, 'stream', False)

                if stream:
                    # Return streaming response
                    return StreamingResponse(
                        self._generate_stream(request, request_id, start_time),
                        media_type="text/event-stream",
                    )
                else:
                    # Original non-streaming logic
                    self.smart_node.endpoint_requests["incoming"].append(request)
                    request = await self._wait_for_result(request)
                    processing_time = time.time() - start_time
                    formatted_response = _format_response(
                        request, processing_time, request_id
                    )
                    return formatted_response

            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            """
            OpenAI-compatible chat completions endpoint.
            Accepts OpenAI format and returns OpenAI format.
            """
            try:
                body = await request.json()

                # Extract OpenAI-style parameters
                model = body.get("model")
                messages = body.get("messages", [])
                temperature = body.get("temperature", 0.7)
                max_tokens = body.get("max_tokens", 2048)

                # Convert to our internal format
                history = []
                current_message = ""

                for msg in messages:
                    role = msg.get("role")
                    content = msg.get("content", "")

                    if role == "system":
                        # System messages added to history
                        history.append({"role": "system", "content": content})
                    elif role == "user":
                        # Last user message becomes current_message
                        if (
                            current_message
                        ):  # If there was a previous user message, add to history
                            history.append({"role": "user", "content": current_message})
                        current_message = content
                    elif role == "assistant":
                        history.append({"role": "assistant", "content": content})

                if not current_message:
                    raise HTTPException(
                        status_code=400,
                        detail="No user message found in messages array",
                    )

                # Create our internal request
                gen_request = GenerationRequest(
                    hf_name=model,
                    message=current_message,
                    history=history if history else None,
                    temperature=temperature,
                    max_new_tokens=max_tokens,
                    response_format="openai",
                )

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

            # Add to processing queue
            self.smart_node.endpoint_requests["incoming"].append(request)

            # Stream tokens as they arrive
            tokens_generated = 0
            full_text = ""

            while True:
                try:
                    # Wait for next token with timeout
                    token_data = await asyncio.wait_for(token_queue.get(), timeout=30.0)

                    if token_data.get("done"):
                        # Generation complete
                        processing_time = time.time() - start_time

                        # Send final message
                        final_data = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": int(start_time),
                            "model": request.hf_name,
                            "choices": [
                                {"index": 0, "delta": {}, "finish_reason": "stop"}
                            ],
                            "usage": {
                                "prompt_tokens": token_data.get("prompt_tokens", 0),
                                "completion_tokens": tokens_generated,
                                "total_tokens": token_data.get("prompt_tokens", 0)
                                + tokens_generated,
                            },
                        }
                        yield f"data: {json.dumps(final_data)}\n\n"
                        yield "data: [DONE]\n\n"
                        break

                    # Stream the token
                    token = token_data.get("token", "")
                    full_text += token
                    tokens_generated += 1

                    # Format as SSE (Server-Sent Events)
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(start_time),
                        "model": request.hf_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": token},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                except asyncio.TimeoutError:
                    # Generation timed out
                    error_chunk = {
                        "error": {
                            "message": "Generation timed out",
                            "type": "timeout_error",
                        }
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    break

        except Exception as e:
            error_chunk = {"error": {"message": str(e), "type": "internal_error"}}
            yield f"data: {json.dumps(error_chunk)}\n\n"
        finally:
            # Clean up
            if request.id in self.streaming_responses:
                del self.streaming_responses[request.id]

    def send_token_to_stream(self, request_id, token=None, done=False, **kwargs):
        """
        Called by the node to send tokens to the streaming response.

        Args:
            request_id: The request ID
            token: The generated token (if not done)
            done: Whether generation is complete
            **kwargs: Additional data (e.g., prompt_tokens, error, full_text)
        """
        if request_id not in self.streaming_responses:
            return

        if not self.server_loop:
            return

        response_queue = self.streaming_responses[request_id]
        data = {"token": token, "done": done, **kwargs}

        # Safely add to queue from potentially different thread
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

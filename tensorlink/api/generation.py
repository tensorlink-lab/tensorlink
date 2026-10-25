"""
Generation machinery for TensorlinkAPI: request dispatch, SSE streaming, and the
bridge between the FastAPI event loop and the ML thread.
"""

import logging

from tensorlink.api.models import (
    GenerationRequest,
    TextResponseRequest,
    ImageResponseRequest,
    EmbeddingResponseRequest,
)
from tensorlink.api.functions import build_generation_request
from tensorlink.ml.utils.formatter import ResponseFormatter
from fastapi.responses import StreamingResponse
from fastapi import HTTPException
from collections import defaultdict
import asyncio
import random
import time


class GenerationMixin:
    def _init_generation_state(self):
        self.model_name_to_request = {}
        self.model_request_timestamps = defaultdict(list)

        # Track models requested via API for prioritization
        self.api_requested_models = set()
        self.streaming_responses = {}
        self.pending_requests: dict[int, asyncio.Future] = {}
        self.api_loop: asyncio.AbstractEventLoop = None
        self._cancelled_requests: set = set()

    async def _dispatch_text(self, gen_request: GenerationRequest):
        start_time = time.time()
        self._log_model_request(gen_request.hf_name)
        gen_request.output = None
        gen_request.id = hash(f"req_{random.random()}")

        model_status = self._check_model_status(gen_request.hf_name)
        if model_status["status"] == "not_loaded":
            self._trigger_model_load(gen_request.hf_name)
            raise HTTPException(
                status_code=503,
                detail=f"Model '{gen_request.hf_name}' requested. Try again shortly.",
            )
        if model_status["status"] == "loading":
            raise HTTPException(
                status_code=503, detail=f"Model {gen_request.hf_name} is still loading."
            )

        self.smart_node.debug_print(
            f"Dispatching request: {gen_request}",
            colour="bright_red",
            level=logging.DEBUG,
            tag="API",
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
        return await self._dispatch_text(build_generation_request(request))

    async def _handle_image_response(self, request: ImageResponseRequest):
        raise HTTPException(
            status_code=501, detail="Image generation is not yet implemented."
        )

    async def _handle_embedding_response(self, request: EmbeddingResponseRequest):
        raise HTTPException(
            status_code=501,
            detail="Embeddings via /v1/responses are not implemented; use POST /v1/embeddings.",
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

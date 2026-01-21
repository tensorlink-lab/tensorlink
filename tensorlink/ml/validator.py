import inspect
import re

from tensorlink.ml.graphing import ModelParser
from tensorlink.ml.worker import DistributedWorker
from tensorlink.ml.module import DistributedModel, OffloadedModule
from tensorlink.ml.formatter import (
    ResponseFormatter,
    normalize_generate_args,
    format_chat_prompt,
    format_stream_chunk,
    format_stream_final,
    extract_reasoning_and_answer,
)
from tensorlink.ml.utils import load_models_cache, save_models_cache
from tensorlink.api.models import GenerationRequest

from transformers import AutoTokenizer, TextIteratorStreamer
from collections import defaultdict
from threading import Thread
import torch
import logging
import json
import time
import gc
import os


# Path to package root
base_dir = os.path.dirname(os.path.abspath(__file__))

# Path to config/models.json relative to this script
SUPPORTED_MODELS_PATH = os.path.join(base_dir, "..", "config", "models.json")


with open(SUPPORTED_MODELS_PATH, "rb") as f:
    MODELS = json.load(f)
    DEFAULT_MODELS = MODELS["DEFAULT_MODELS"]


def _format_response(
    request,
    clean_output: str,
    raw_output: str,
    processing_time: float,
):
    """
    Format the response based on the requested format type.
    This runs in the validator process after generation completes.

    Args:
        request: The original generation request
        clean_output: Cleaned/extracted output text
        raw_output: Raw model output
        processing_time: Time taken to process the request

    Returns:
        Dictionary formatted according to output_format
    """
    timestamp = int(time.time())
    request_id = getattr(request, 'id')

    if request.output_format == "simple":
        # Minimal response - just the text (no cleaning for simple)
        return {"response": raw_output}

    elif request.output_format == "openai":
        # OpenAI-compatible format (always cleaned)
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
                "prompt_tokens": getattr(request, 'prompt_tokens', -1),
                "completion_tokens": getattr(request, 'completion_tokens', -1),
                "total_tokens": getattr(request, 'total_tokens', -1),
            },
        }

    else:  # "full" format (default, comprehensive response with all metadata)
        # For full format, don't clean unless it's openai-style request
        output_text = raw_output
        return {
            "id": request_id,
            "model": request.hf_name,
            "response": output_text,
            "raw_output": raw_output,
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
                "formatted_as_chat": request.output_format == "openai",
            },
        }


class RemoteStreamer:
    def __init__(self, poll_fn, sleep=0.01):
        self.poll_fn = poll_fn
        self.sleep = sleep
        self.done = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.done:
            raise StopIteration

        while True:
            try:
                token = self.poll_fn()
            except Exception as e:
                self.done = True
                raise StopIteration from e

            if token is None:
                time.sleep(self.sleep)
                continue

            if token == "__END__":
                self.done = True
                raise StopIteration

            return token


class DistributedValidator(DistributedWorker):
    def __init__(self, node, trusted=False, endpoint=True):
        super().__init__(node, trusted)
        self.endpoint = endpoint
        self.model_cache = load_models_cache()
        self.models = {}  # job_id -> model instance
        self.model_state = (
            {}
        )  # job_id -> state ("initializing" | "distributing" | "ready")
        self.public_models = defaultdict(list)  # Model name -> list(job_id)

        self.tokenizers = {}

        # Track models that are in the process of being initialized (job_id)
        self.models_initializing = set()

        # Configuration
        self.TRACKING_DAYS = 7  # Track requests for past 7 days
        self.MIN_REQUESTS_THRESHOLD = 10  # Minimum requests to consider auto-loading
        self.MAX_AUTO_MODELS = 10  # Maximum models to auto-load

    def _ensure_model_entry(self, model_name: str):
        """Ensure a model has an entry in the cache with proper structure"""
        if model_name not in self.model_cache:
            self.model_cache[model_name] = {
                "distribution": None,
                "demand_metrics": {
                    "request_timestamps": [],
                    "total_requests": 0,
                    "last_accessed": None,
                },
            }

    def _record_request(self, model_name: str):
        """Record a request timestamp for a model in the JSON cache"""
        self._ensure_model_entry(model_name)

        current_time = time.time()

        # Add timestamp to the list
        self.model_cache[model_name]["demand_metrics"]["request_timestamps"].append(
            current_time
        )
        self.model_cache[model_name]["demand_metrics"]["total_requests"] += 1
        self.model_cache[model_name]["demand_metrics"]["last_accessed"] = current_time

        # Keep only recent timestamps to prevent unlimited growth
        cutoff_time = current_time - (self.TRACKING_DAYS * 24 * 3600)
        timestamps = self.model_cache[model_name]["demand_metrics"][
            "request_timestamps"
        ]
        self.model_cache[model_name]["demand_metrics"]["request_timestamps"] = [
            ts for ts in timestamps if ts >= cutoff_time
        ]

        # Save updated metrics
        save_models_cache(self.model_cache)

    def _get_recent_request_count(self, model_name: str, days: int = None) -> int:
        """Get number of requests for a model in the past X days from JSON cache"""
        if days is None:
            days = self.TRACKING_DAYS

        if model_name not in self.model_cache:
            return 0

        cutoff_time = time.time() - (days * 24 * 3600)
        timestamps = (
            self.model_cache[model_name]
            .get("demand_metrics", {})
            .get("request_timestamps", [])
        )

        return sum(1 for timestamp in timestamps if timestamp >= cutoff_time)

    def _cleanup_old_requests(self):
        """Remove request timestamps older than tracking period from all models in cache"""
        cutoff_time = time.time() - (self.TRACKING_DAYS * 24 * 3600)
        updated = False

        for model_name in list(self.model_cache.keys()):
            if "demand_metrics" in self.model_cache[model_name]:
                old_count = len(
                    self.model_cache[model_name]["demand_metrics"]["request_timestamps"]
                )

                # Filter out old timestamps
                self.model_cache[model_name]["demand_metrics"]["request_timestamps"] = [
                    ts
                    for ts in self.model_cache[model_name]["demand_metrics"][
                        "request_timestamps"
                    ]
                    if ts >= cutoff_time
                ]

                new_count = len(
                    self.model_cache[model_name]["demand_metrics"]["request_timestamps"]
                )

                if old_count != new_count:
                    updated = True

                # Remove entries with no recent activity
                if (
                    new_count == 0
                    and self.model_cache[model_name].get("distribution") is None
                ):
                    del self.model_cache[model_name]
                    updated = True

        if updated:
            save_models_cache(self.model_cache)

    def _get_popular_models(self) -> list:
        """Get list of models sorted by popularity from JSON cache"""
        model_popularity = []

        for model_name, model_data in self.model_cache.items():
            request_count = self._get_recent_request_count(model_name)
            if request_count >= self.MIN_REQUESTS_THRESHOLD:
                model_popularity.append((model_name, request_count))

        # Sort by request count (descending)
        model_popularity.sort(key=lambda x: x[1], reverse=True)
        return [model_name for model_name, _ in model_popularity]

    def _is_model_ready(self, job_id: str) -> bool:
        """Check if a model is ready for inference"""
        return self.model_state.get(job_id) == "ready"

    def _manage_auto_loaded_models(self):
        """Manage auto-loaded models based on popularity from JSON cache, falling back to DEFAULT_MODELS"""
        # If the API endpoint is not active, skip auto loading models
        if not self.endpoint:
            return

        # Get popular models based on their request counts
        model_demands = {}
        for model_name in self.model_cache.keys():
            model_demands[model_name] = self._get_recent_request_count(model_name)

        # Add DEFAULT_MODELS with minimum demand to keep them warm
        for m in DEFAULT_MODELS:
            model_demands[m] = max(model_demands.get(m, 0), self.MIN_REQUESTS_THRESHOLD)

        if not model_demands:
            return

        total_requests = sum(max(v, 0) for v in model_demands.values())
        if total_requests == 0:
            return

        # Get number of desired model instances based on demand
        desired_instances = {}
        for model_name, count in model_demands.items():
            share = count / total_requests
            desired_instances[model_name] = round(share * self.MAX_AUTO_MODELS)

        can_allocate = True
        # Ensure each model has at least one instance
        for model_name, desired in desired_instances.items():
            if not can_allocate:
                break

            current_total = len(self.public_models.get(model_name, []))
            current_total += sum(
                1 if job_id in self.models_initializing else 0
                for job_id in self.public_models[model_name]
            )

            if current_total == 0 and desired > 0:
                self.send_request(
                    "debug_print",
                    (
                        f"Initializing first instance of {model_name}",
                        "cyan",
                        logging.INFO,
                    ),
                )
                can_allocate = self._initialize_hosted_job(model_name)

        # Finalize any first-load initializations
        if self.models_initializing:
            self._try_finalize_initializing_models()

        # Allocate duplicates based on proportional demand
        for model_name, target_count in desired_instances.items():
            if not can_allocate:
                break

            current_total = len(self.public_models.get(model_name, []))
            current_total += sum(
                1 if job_id in self.models_initializing else 0
                for job_id in self.public_models[model_name]
            )

            if current_total < target_count:
                to_launch = target_count - current_total
                for _ in range(to_launch):
                    self.send_request(
                        "debug_print",
                        (
                            f"Scaling UP (duplicate) {model_name}: +1 instance",
                            "green",
                            logging.INFO,
                        ),
                    )
                    can_allocate = self._initialize_hosted_job(model_name)

                    if not can_allocate:
                        break

        # Finalize any duplicate initializations
        if self.models_initializing:
            self._try_finalize_initializing_models()

    def inspect_model(self, model_name: str, job_data: dict, hosted=False) -> dict:
        """Inspect a model to determine network requirements and store distribution in JSON cache"""
        parser = ModelParser()
        model_name: str = job_data.get("model_name", model_name)

        # Get network worker information to assign modules
        workers = self.send_request("get_workers", None)

        batch_size = job_data.get("batch_size", None)

        if batch_size is None:
            if job_data.get("training", False):
                batch_size = 256
            else:
                batch_size = 1

        if job_data.get("optimizer") is None:
            optimizer_type = "adam"
            optimizer_spec = {}
        else:
            optimizer_type = job_data["optimizer"]["type"]
            optimizer_spec = job_data.get("optimizer")

        # Load HF model, create and save distribution
        distribution = parser.create_distributed_config(
            model_name,
            workers=workers,
            training=job_data.get("training", False),
            trusted=False,
            handle_layers=False,
            input_obfuscation=False,
            optimizer_type=optimizer_type,
            optimizer_spec=optimizer_spec,
            host_load_small=hosted,
            host_max_depth=1,
            host_threshold_mb=75,
            max_offload_depth=3,
            batch_size=job_data.get("batch_size", batch_size),
            max_seq_len=job_data.get("max_seq_len", 4096),
            model_type=job_data.get("model_type", "chat"),
        )

        job_data["distribution"] = distribution

        offloaded_count = sum(
            1
            for v in distribution["config"].values()
            if "offloaded" in v.get("type", "")
        )

        if (
            len(distribution["config"]) == 0
            or offloaded_count
            > 4  # TODO This limit on number of distributions is not ideal
            or not distribution["success"]
        ):
            return {}

        # Store distribution in JSON cache
        self._ensure_model_entry(model_name)
        self.model_cache[model_name]["distribution"] = distribution
        save_models_cache(self.model_cache)

        self.send_request(
            "debug_print",
            (
                f"DistributedValidator -> Retrieved HF model: {job_data}",
                "bright_blue",
                logging.DEBUG,
            ),
        )

        gc.collect()  # Force garbage collection

        # Send out job request
        try:
            new_job_data = self.send_request("send_job_request", job_data)
            return new_job_data

        except Exception as e:
            print(str(e))

    def check_node(self):
        """Check for node requests/updates"""
        try:
            # When running on the public network, manage models automatically
            if self.node.config.endpoint and self.node.config.on_chain:
                # Periodic cleanup and model management
                if self.CHECK_COUNTER % self.GC_CHECK_INTERVAL == 0:
                    # Clean up old request data
                    self._cleanup_old_requests()

                    # Manage autoloaded models based on popularity (or DEFAULT_MODELS fallback)
                    self._manage_auto_loaded_models()

                    # Check if jobs are still active
                    for job_id, model in self.models.items():
                        model_name = model.model_name
                        if self._is_model_ready(job_id):
                            is_active = self.send_request(
                                "check_job", (model_name, job_id)
                            )
                            if not is_active:
                                self._remove_hosted_job(job_id)

                    self.CHECK_COUNTER = 1

                if self.models_initializing:
                    # Only call model management if we have models actively initializing
                    self._try_finalize_initializing_models()

            if self.CHECK_COUNTER % self.GC_CHECK_INTERVAL // 5 == 0:
                # Get job data for inspection to see if we can accommodate the model
                job_data = self.send_request("get_jobs", None)
                if isinstance(job_data, dict):
                    model_name: str = job_data.get("model_name", "")

                    if job_data.get("api"):
                        payment = job_data.get("payment", 0)
                        time_limit = job_data.get("time", 1800)
                        job_id = job_data.get("id")

                        # Check if this is a public job and there are already models of this type
                        can_allocate = self._initialize_hosted_job(
                            model_name,
                            job_data=job_data,
                            payment=payment,
                            time_limit=time_limit,
                        )

                        # Try to finalize if already initializing
                        if can_allocate and job_id in self.models_initializing:
                            self._finalize_hosted_job(job_id)

                    else:
                        # If request via user node, begin the model reqs inspection for the job request
                        self.inspect_model(model_name, job_data, hosted=False)

            # Check for inference generate calls
            for job_id, distributed_model in self.models.items():
                if self._is_model_ready(job_id):
                    model_name = distributed_model.model_name
                    # TODO Distinguish private generate requests from public ones so we dont use the same model

                    generate_request = self.send_request(
                        "update_api_request", (model_name, job_id)
                    )
                    if generate_request:
                        self._handle_generate_request(generate_request, job_id)

        except Exception as e:
            logging.error(f"Error checking for jobs: {str(e)}")

        self.CHECK_COUNTER += 1

    # def _handle_check_model_status(self, model_name: str):
    #     """Check the loading status of a model"""
    #     if model_name in self.models:
    #         if self._is_model_ready(model_name):
    #             # Model is fully loaded
    #             return {
    #                 "status": "loaded",
    #                 "message": f"Model {model_name} is loaded and ready",
    #             }
    #         else:
    #             # Model is in the process of loading
    #             return {
    #                 "status": "loading",
    #                 "message": f"Model {model_name} is currently loading",
    #             }
    #
    #     elif model_name in self.models_initializing:
    #         return {
    #             "status": "loading",
    #             "message": f"Model {model_name} initialization in progress",
    #         }
    #     else:
    #         return {
    #             "status": "not_loaded",
    #             "message": f"Model {model_name} is not loaded",
    #         }

    def _handle_generate_request(self, request: GenerationRequest, job_id: str):
        """Main entry point for generate requests"""
        self._record_request(request.hf_name)

        if not self._is_model_ready(job_id):
            error_response = ResponseFormatter.format_error_response(
                error_message="Model is currently not available through the Tensorlink API.",
                error_type="model_unavailable",
                status_code=503,
                request_id=str(request.id),
            )
            request.output = error_response["error"]["message"]
            request.formatted_response = error_response
            self.send_request("update_api_request", (request,))
            return

        start_time = getattr(request, 'start_time', time.time())

        if hasattr(request, "stream") and request.stream:
            # Streaming generation
            self._generate_streaming(request, job_id)
        else:
            # Generate
            self._generate(request, job_id, start_time)

        self.send_request("update_api_request", (request,))

    def _generate(self, request, job_id, start_time):
        """
        Fetches tokenizer, ensures generate arguments are not problematic with
        normalize_generate_args, and calls DistributedModel.generate.
        """
        distributed_model = self.models[job_id]
        tokenizer = self.tokenizers[request.hf_name]

        # FORMAT PROMPT
        if request.input_format == "chat":
            formatted_prompt = format_chat_prompt(
                request.hf_name,
                request.message,
                request.history,
                enable_thinking=request.reasoning,  # Use consistent field name
            )
        else:
            formatted_prompt = request.message

        # TOKENIZE
        model_max_length = getattr(tokenizer, 'model_max_length', 2048)
        if model_max_length > 100000:
            model_max_length = 2048

        max_length = min(
            getattr(request, 'max_length', 512),
            model_max_length - 10,
        )

        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        prompt_tokens = inputs.input_ids.shape[1]
        input_ids = inputs.input_ids.to(self.device)

        # NORMALIZE ARGS WITH PROMPT TOKEN COUNT
        try:
            args = normalize_generate_args(
                request,
                tokenizer,
                prompt_tokens=prompt_tokens,
                model_max_length=model_max_length,
                allowed_generate_args=distributed_model._generate_args,
            )

        except ValueError as e:
            request.output = f"Error: {str(e)}"
            request.formatted_response = ResponseFormatter.format_error_response(
                error_message=str(e),
                error_type="prompt_too_long",
                status_code=400,
                request_id=str(request.id),
            )
            return

        # GENERATE
        with torch.no_grad():
            try:
                outputs = distributed_model.generate(input_ids)
            except RuntimeError as e:
                error_msg = f"Generation failed: {str(e)}"
                request.output = error_msg
                request.formatted_response = ResponseFormatter.format_error_response(
                    error_message=error_msg,
                    error_type="generation_error",
                    status_code=500,
                    request_id=str(request.id),
                )
                return

        # DECODE
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt echo
        if generated_text.startswith(formatted_prompt):
            text = generated_text[len(formatted_prompt) :].strip()
        else:
            text = generated_text.strip()

        reasoning_text = None
        if request.input_format == "chat":
            reasoning_text, text = extract_reasoning_and_answer(text)

            # Respect reasoning flag - only include reasoning if explicitly enabled
            if not request.reasoning:
                reasoning_text = None

        request.output = text

        # COUNT TOKENS & FORMAT RESPONSE
        completion_tokens = len(tokenizer.encode(text, add_special_tokens=False))

        request.formatted_response = ResponseFormatter.format_non_streaming_response(
            request=request,
            output_text=text,
            reasoning_text=reasoning_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            start_time=start_time,
        )

    def _generate_streaming(self, request: GenerationRequest, job_id: str):
        """
        Fetches tokenizer, ensures generate arguments are not problematic with
        normalize_generate_args, and calls DistributedModel.generate with stream.
        """
        try:
            start_time = getattr(request, 'start_time', time.time())
            distributed_model = self.models[job_id]
            tokenizer = self.tokenizers[request.hf_name]

            # Format input
            if request.input_format == "chat":
                formatted_prompt = format_chat_prompt(
                    request.hf_name,
                    request.message,
                    request.history,
                    enable_thinking=request.reasoning,  # Use consistent field name
                )
            else:
                formatted_prompt = request.message

            # Tokenize
            model_max_length = getattr(tokenizer, 'model_max_length', 2048)
            if model_max_length > 100000:
                model_max_length = 2048

            max_length = min(getattr(request, 'max_length', 512), model_max_length - 10)

            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )

            input_ids = inputs.input_ids.to(self.device)
            prompt_tokens = input_ids.shape[1]

            # Normalize args
            try:
                args = normalize_generate_args(
                    request,
                    tokenizer,
                    prompt_tokens=prompt_tokens,
                    model_max_length=model_max_length,
                    allowed_generate_args=distributed_model._generate_args,
                )
            except ValueError as e:
                error_chunk = ResponseFormatter.format_stream_error(
                    error_message=str(e), error_type="prompt_too_long"
                )
                self.send_request(
                    "update_stream",
                    (request.id, {"done": True, "final_chunk": error_chunk}),
                )
                request.output = f"Error: {str(e)}"
                return

            # Build generation kwargs
            generation_kwargs = {"input_ids": input_ids, "stream": True}

            # Setup streamer
            if isinstance(distributed_model.model, OffloadedModule):
                generation_kwargs.update(**args)
                module_id = distributed_model.model.module_id
                streamer = RemoteStreamer(
                    poll_fn=lambda: self._poll_remote_token(module_id, tokenizer)
                )
                generation_thread = Thread(
                    target=distributed_model.generate, kwargs=generation_kwargs
                )
                generation_thread.start()
            else:
                streamer = TextIteratorStreamer(
                    tokenizer, skip_prompt=True, skip_special_tokens=True
                )
                generation_kwargs.pop("stream")
                generation_kwargs["streamer"] = streamer
                generation_thread = Thread(
                    target=distributed_model.generate, kwargs=generation_kwargs
                )
                generation_thread.start()

            # Stream tokens
            full_text = ""
            token_count = 0
            in_reasoning_block = False
            reasoning_buffer = ""

            for token_text in streamer:
                full_text += token_text

                # Track if we're inside a reasoning block (simple detection)
                if request.input_format == "chat" and not request.reasoning:
                    # Check for start of reasoning tags
                    if re.search(
                        r'<\s*(think|reflection|thought|internal|analysis)\s*>',
                        reasoning_buffer + token_text,
                        re.IGNORECASE,
                    ):
                        in_reasoning_block = True
                        reasoning_buffer += token_text
                        continue

                    # Check for end of reasoning tags
                    if in_reasoning_block:
                        reasoning_buffer += token_text
                        if re.search(
                            r'<\s*/\s*(think|reflection|thought|internal|analysis)\s*>',
                            reasoning_buffer,
                            re.IGNORECASE,
                        ):
                            in_reasoning_block = False
                            reasoning_buffer = ""
                        continue

                # Only send non-reasoning tokens when reasoning is disabled
                if not in_reasoning_block:
                    token_count += 1
                    formatted_chunk = ResponseFormatter.format_stream_chunk(
                        request=request,
                        token_text=token_text,
                        index=token_count,
                        start_time=start_time,
                    )

                    self.send_request(
                        "update_stream",
                        (request.id, {"chunk": formatted_chunk, "done": False}),
                    )

            reasoning_text = None
            cleaned_text = full_text

            # Extract reasoning and clean output
            if request.input_format == "chat":
                reasoning_text, cleaned_text = extract_reasoning_and_answer(full_text)

                # Only include reasoning if explicitly enabled
                if not request.reasoning:
                    reasoning_text = None

            request.output = cleaned_text

            # Send final chunk
            final_chunk = ResponseFormatter.format_stream_final(
                request=request,
                start_time=start_time,
                prompt_tokens=prompt_tokens,
                completion_tokens=token_count,
                full_text=cleaned_text,
                reasoning_text=reasoning_text,
            )

            self.send_request(
                "update_stream",
                (request.id, {"done": True, "final_chunk": final_chunk}),
            )

        except Exception as e:
            error_chunk = ResponseFormatter.format_stream_error(
                error_message=str(e), error_type="generation_error"
            )
            self.send_request(
                "update_stream",
                (request.id, {"done": True, "final_chunk": error_chunk}),
            )
            request.output = f"Error during generation: {str(e)}"

    def _poll_remote_token(self, module_id: str, tokenizer):
        item = self.send_request("check_token", (module_id,))

        if item is None:
            return None

        if item["type"] == "end":
            return "__END__"

        if item["type"] == "token":
            token_id = item["token"]

            # Decode exactly one token
            text = tokenizer.decode(
                [token_id],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            return text

        return None

    def _try_finalize_initializing_models(self):
        """Attempt to finalize all models that are currently initializing."""
        for job_id in list(self.models_initializing):
            if self._finalize_hosted_job(job_id):
                self.send_request(
                    "debug_print",
                    (
                        f"Successfully finalized model: {job_id}",
                        "green",
                        logging.INFO,
                    ),
                )

    def _initialize_hosted_job(
        self,
        model_name: str,
        payment: int = 0,
        time_limit: int = None,
        job_data: dict = None,
    ):
        """Initialize a hosted job by creating the distributed model and submitting inspection request."""
        if not job_data:
            job_data = {}

        try:
            # Prepare job data for inspection
            defaults = {
                "author": None,
                "active": True,
                "hosted": True,
                "training": False,
                "payment": payment,
                "time": time_limit,
                "capacity": 0,
                "n_pipelines": 1,
                "dp_factor": 1,
                "distribution": {"model_name": model_name},
                "model_type": "chat",
                "n_workers": 0,
                "model_name": model_name,
                "seed_validators": [],
            }

            for k, v in defaults.items():
                job_data.setdefault(k, v)

            # Inspect model to determine network requirements
            job_data = self.inspect_model(model_name, job_data, hosted=True)

            if not job_data:
                return False

            job_id = job_data.get("id")

            # Create distributed model instance
            distributed_model = DistributedModel(
                model_name,
                node=self.node,
                training=False,
            )
            distributed_model.config = job_data.get("distribution")

            self.models[job_id] = distributed_model

            if job_data.get("public"):
                self.public_models[model_name].append(job_id)

            self.model_state[job_id] = "initializing"
            self.models_initializing.add(job_id)
            return True

        except Exception as e:
            logging.error(f"Error initializing hosted job for {model_name}: {str(e)}")
            job_id = job_data.get("id")
            self.models_initializing.discard(job_id)
            del self.models[job_id]
            if job_id in self.model_state:
                del self.model_state[job_id]

            return False

    def _finalize_hosted_job(self, job_id: str):
        """Finalize a hosted job by setting up the distributed model with workers."""
        try:
            # Check if we have module info ready
            args = self.send_request("check_module", job_id)

            if not args or not isinstance(args, dict):
                # Module not ready yet
                return False

            distribution = args["distribution"]
            optimizer_name = args["optimizer"]
            training = args["training"]

            # Check if model is in initialization state
            if job_id not in self.models:
                return False

            # Get the DistributedModel instance
            distributed_model = self.models[job_id]

            # Update state
            self.model_state[job_id] = "distributing"

            # Register the distributed model's modules
            for module_id, module_info in distribution.items():
                if "offloaded" in module_info.get("type", ""):
                    module_info["job_id"] = job_id
                    self.modules[module_id] = module_info

            # Distribute the model across workers
            distributed_model.distribute_model(distribution)
            distributed_model.job_id = job_id

            model_name = distributed_model.model_name

            # Load tokenizer
            if model_name not in self.tokenizers:
                self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)

            setattr(distributed_model, 'tokenizer', self.tokenizers[model_name])

            # Mark as ready
            self.model_state[job_id] = "ready"
            self.models_initializing.discard(job_id)

            self.send_request(
                "debug_print",
                (
                    f"DistributedValidator -> Finalized hosted job for {model_name} with job_id {job_id}",
                    "green",
                    logging.INFO,
                ),
            )

            return True

        except Exception as e:
            logging.error(f"Error finalizing hosted job for {model_name}: {str(e)}")
            self.models_initializing.discard(job_id)
            if job_id in self.models:
                del self.models[job_id]
            return False

    def _remove_hosted_job(self, job_id: str):
        """Remove a hosted job and clean up all associated resources"""
        try:
            # Remove from initializing set if present
            self.models_initializing.discard(job_id)

            distributed_model = self.models[job_id]
            model_name = distributed_model.model_name

            # Clean up tokenizer if no other models require it
            if (
                model_name in self.tokenizers
                and len(self.public_models[model_name]) <= 1
            ):
                del self.tokenizers[model_name]
                self.send_request(
                    "debug_print",
                    (f"Removed tokenizer for {model_name}", "yellow", logging.INFO),
                )

            if model_name in self.public_models:
                if job_id in self.public_models[model_name]:
                    self.public_models[model_name].remove(job_id)

            # Clean up state tracking
            if job_id in self.model_state:
                del self.model_state[job_id]

            # Clean up model reference
            del self.models[job_id]

            # Find and remove any module entries that reference this model
            modules_to_remove = []
            for module_id, module_data in self.modules.items():
                # Check if this module belongs to the model we're removing
                if module_data.get("name") == model_name:
                    if module_data.get("job_id") == job_id:
                        modules_to_remove.append(module_id)

            for module_id in modules_to_remove:
                del self.modules[module_id]
                self.send_request(
                    "debug_print",
                    (
                        f"Removed module reference {module_id} for {model_name}",
                        "yellow",
                        logging.INFO,
                    ),
                )

            # Only remove model cache if it has no distribution data and no recent requests
            if (
                model_name in self.model_cache
                and self.model_cache[model_name].get("distribution") is not None
                and self._get_recent_request_count(model_name, days=1) == 0
            ):

                # Keep demand metrics but clear distribution if no recent activity
                self.model_cache[model_name]["distribution"] = None
                save_models_cache(self.model_cache)

                self.send_request(
                    "debug_print",
                    (
                        f"Cleared distribution cache for {model_name}",
                        "yellow",
                        logging.INFO,
                    ),
                )

            # Send cleanup request to node
            try:
                self.send_request("remove_job", {"model_name": model_name})
            except Exception as e:
                logging.warning(
                    f"Error sending job removal request for {model_name}: {str(e)}"
                )

            # Force garbage collection to free memory
            gc.collect()

            self.send_request(
                "debug_print",
                (
                    f"Successfully removed hosted job: {model_name}",
                    "green",
                    logging.INFO,
                ),
            )

        except Exception as e:
            logging.error(f"Error removing hosted job {model_name}: {str(e)}")
            self.send_request(
                "debug_print",
                (
                    f"Failed to remove hosted job {model_name}: {str(e)}",
                    "red",
                    logging.ERROR,
                ),
            )

    def main_loop(self):
        self.check_node()

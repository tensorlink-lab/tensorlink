"""
tensorlink/ml/utils/reasoning.py

Model-agnostic helpers for "thinking" models.

There is no single switch for reasoning across Hugging Face models:

  * Qwen3-style templates take a kwarg (``enable_thinking``); DeepSeek-V3.1
    style templates take ``thinking``.
  * R1/QwQ-style templates force-open ``<think>`` at the end of the
    generation prompt, so the model always starts inside a reasoning block.
  * Llama/Mistral/etc. have no reasoning mode at all.

Instead of hard-coding families we *probe* the tokenizer's chat template once
(render with the knob on vs. off and compare) and cache the result on the
tokenizer. Jinja ignores unknown kwargs, so a model without the knob renders
identically both ways and is classified as unsupported.

Not handled yet: gpt-oss style ``reasoning_effort`` / harmony channels, which
don't use ``<think>`` tags.

This module only depends on the standard library; tokenizers are passed in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

# (kwarg name, value that enables thinking, value that disables it)
_TEMPLATE_KNOBS = (
    ("enable_thinking", True, False),
    ("thinking", True, False),
)

THINK_TAGS = ("think", "thinking", "reflection", "thought", "internal", "analysis")
_TAG_ALT = "|".join(THINK_TAGS)

_OPEN_RE = re.compile(rf"<\s*(?:{_TAG_ALT})\s*>", re.IGNORECASE)
_CLOSE_RE = re.compile(rf"<\s*/\s*(?:{_TAG_ALT})\s*>", re.IGNORECASE)
_ENDS_OPEN_RE = re.compile(rf"<\s*(?:{_TAG_ALT})\s*>\s*$", re.IGNORECASE)

_PROBE_MESSAGES = [{"role": "user", "content": "hi"}]

# Name of the field that carries thinking in API responses / SSE deltas.
# DeepSeek/Qwen docs use "reasoning_content"; some newer servers use "reasoning".
# Change it here and everything follows.
REASONING_FIELD = "reasoning_content"


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReasoningControl:
    """How (and whether) a tokenizer's chat template lets us toggle thinking."""

    mode: str  # "kwarg" | "prefill" | "none"
    kwarg: Optional[str] = None
    on: Any = None
    off: Any = None

    @property
    def supported(self) -> bool:
        return self.mode != "none"


def detect_reasoning_control(tokenizer) -> ReasoningControl:
    """Probe the chat template; never raises."""
    if not getattr(tokenizer, "chat_template", None):
        return ReasoningControl("none")

    def render(**kw) -> str:
        return tokenizer.apply_chat_template(
            _PROBE_MESSAGES, tokenize=False, add_generation_prompt=True, **kw
        )

    for name, on, off in _TEMPLATE_KNOBS:
        try:
            if render(**{name: on}) != render(**{name: off}):
                return ReasoningControl("kwarg", name, on, off)
        except Exception:
            continue  # this knob broke the template; try the next one
    try:
        if _ENDS_OPEN_RE.search(render()):
            return ReasoningControl("prefill")
    except Exception:
        pass
    return ReasoningControl("none")


def get_reasoning_control(tokenizer) -> ReasoningControl:
    """Cached wrapper around :func:`detect_reasoning_control`."""
    cached = getattr(tokenizer, "_tl_reasoning_control", None)
    if isinstance(cached, ReasoningControl):
        return cached
    ctl = detect_reasoning_control(tokenizer)
    try:
        setattr(tokenizer, "_tl_reasoning_control", ctl)
    except Exception:
        pass
    return ctl


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
@dataclass
class PromptBuild:
    prompt: str
    control: ReasoningControl
    # True when the prompt itself already opened a <think> block, so the
    # model's first generated token is *inside* the reasoning section.
    starts_in_reasoning: bool = False
    dropped_messages: int = 0
    # True if the chat template failed/was missing and we used plain ChatML.
    used_fallback: bool = False


def _chatml_prompt(messages: List[dict]) -> str:
    """Last-resort prompt when the model has no (working) chat template."""
    msgs = list(messages)
    if not any(m.get("role") == "system" for m in msgs):
        msgs.insert(0, {"role": "system", "content": "You are a helpful assistant."})
    parts = [
        f"<|im_start|>{m.get('role', 'user')}\n{m.get('content', '')}<|im_end|>"
        for m in msgs
    ]
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _merge_system(messages: List[dict]) -> List[dict]:
    """Fold system text into the first user turn (for templates that reject it)."""
    sys_txt = "\n\n".join(
        m.get("content", "") for m in messages if m.get("role") == "system"
    )
    rest = [dict(m) for m in messages if m.get("role") != "system"]
    if sys_txt:
        for m in rest:
            if m.get("role") == "user":
                m["content"] = f"{sys_txt}\n\n{m.get('content', '')}"
                break
    return rest


def _render(tokenizer, messages, ctl: ReasoningControl, reasoning: Optional[bool]):
    """Returns (prompt, starts_in_reasoning, used_fallback)."""
    kw = {}
    if ctl.mode == "kwarg" and reasoning is not None:
        kw[ctl.kwarg] = ctl.on if reasoning else ctl.off

    prompt = None
    if getattr(tokenizer, "chat_template", None):
        attempts = (
            (messages, kw),  # normal
            (messages, {}),  # knob rejected at render time
            (_merge_system(messages), {}),  # template rejects system role / ordering
        )
        for msgs, extra in attempts:
            try:
                prompt = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, **extra
                )
                break
            except Exception:
                continue

    if prompt is None:
        return _chatml_prompt(messages), False, True

    # Works for every mode: if the template itself left a <think> block open
    # (R1 prefill, DeepSeek-V3.1 thinking=True, Qwen3-2507-thinking...), the
    # model starts *inside* reasoning.
    m = _ENDS_OPEN_RE.search(prompt)
    starts = m is not None

    if starts and reasoning is False:
        # Thinking was requested off but the template force-opened a block:
        # close it ourselves so the model skips straight to the answer.
        tag = re.sub(r"[<>\s]", "", m.group(0))
        prompt = prompt.rstrip() + f"\n\n</{tag}>\n\n"
        starts = False
    return prompt, starts, False


def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _drop_oldest_turn(messages: List[dict]) -> Optional[List[dict]]:
    """Drop the oldest non-system turn (keeping user/assistant alternation)."""
    system = [m for m in messages if m.get("role") == "system"]
    convo = [m for m in messages if m.get("role") != "system"]
    if len(convo) <= 1:
        return None
    convo.pop(0)
    while len(convo) > 1 and convo[0].get("role") == "assistant":
        convo.pop(0)
    return system + convo


def build_chat_prompt(
    tokenizer,
    messages: List[dict],
    reasoning: Optional[bool] = False,
    max_tokens: Optional[int] = None,
) -> PromptBuild:
    """
    Render ``messages`` with the model's chat template. Never raises on
    template problems; falls back to ChatML.

    reasoning: True/False force thinking on/off where the model supports it;
               None leaves the model's default. Unsupported models ignore it.
    max_tokens: if given, drop the *oldest* turns until the rendered prompt
                fits, so the newest question is never the thing that is cut.
    """
    ctl = get_reasoning_control(tokenizer)
    msgs = list(messages)
    dropped = 0

    while True:
        prompt, starts, fallback = _render(tokenizer, msgs, ctl, reasoning)
        if max_tokens is None or _count_tokens(tokenizer, prompt) <= max_tokens:
            break
        trimmed = _drop_oldest_turn(msgs)
        if trimmed is None:
            break
        dropped += len(msgs) - len(trimmed)
        msgs = trimmed

    if fallback:
        ctl = ReasoningControl("none")
    return PromptBuild(prompt, ctl, starts, dropped, fallback)


def think_tags_visible(tokenizer) -> bool:
    """
    False if the tokenizer marks <think>/</think> as *special* tokens, in which
    case ``skip_special_tokens=True`` (decode or TextIteratorStreamer) silently
    deletes the tags and the splitter can't see them. Cached on the tokenizer.
    """
    cached = getattr(tokenizer, "_tl_think_visible", None)
    if cached is not None:
        return cached
    visible = True
    try:
        for tag in ("<think>", "</think>"):
            tid = tokenizer.convert_tokens_to_ids(tag)
            if tid is None or tid == tokenizer.unk_token_id:
                continue  # not a single token -> plain text -> survives decode
            if tag not in tokenizer.decode([tid], skip_special_tokens=True):
                visible = False
    except Exception:
        pass
    try:
        setattr(tokenizer, "_tl_think_visible", visible)
    except Exception:
        pass
    return visible


# ---------------------------------------------------------------------------
# Decoding policy (only deviates from the default for the special-token case)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DecodePolicy:
    """
    How to turn token ids into text. The default is exactly what the pipeline
    did before (``skip_special_tokens=True``, no extra cleanup). It only
    changes for a thinking-capable model whose <think> tags are *special*
    tokens: those must be kept through decode so the splitter can see them,
    and every *other* special token (eos, im_end, ...) is stripped by hand.
    """

    skip_special_tokens: bool = True
    strip_re: Optional["re.Pattern"] = None

    def clean(self, text: str) -> str:
        return self.strip_re.sub("", text) if self.strip_re else text


_DEFAULT_DECODE = DecodePolicy()
_TAG_ANY_RE = re.compile(rf"<\s*/?\s*(?:{_TAG_ALT})\s*>", re.IGNORECASE)


def get_decode_policy(tokenizer, chat: bool = True) -> DecodePolicy:
    """
    Non-default only if: chat request AND the template has a thinking mode AND
    the think tags would be deleted by skip_special_tokens. Everything else
    (raw requests, non-thinking models, normal think tokens) is untouched.
    """
    if not chat:
        return _DEFAULT_DECODE
    cached = getattr(tokenizer, "_tl_decode_policy", None)
    if isinstance(cached, DecodePolicy):
        return cached

    policy = _DEFAULT_DECODE
    try:
        if get_reasoning_control(tokenizer).supported and not think_tags_visible(
            tokenizer
        ):
            specials = set(getattr(tokenizer, "all_special_tokens", []) or [])
            for tok in (getattr(tokenizer, "added_tokens_decoder", {}) or {}).values():
                if getattr(tok, "special", False):
                    specials.add(getattr(tok, "content", str(tok)))
            specials = {s for s in specials if s and not _TAG_ANY_RE.fullmatch(s)}
            strip_re = (
                re.compile(
                    "|".join(
                        re.escape(s) for s in sorted(specials, key=len, reverse=True)
                    )
                )
                if specials
                else None
            )
            policy = DecodePolicy(skip_special_tokens=False, strip_re=strip_re)
    except Exception:
        policy = _DEFAULT_DECODE  # on any doubt, behave exactly as before

    try:
        setattr(tokenizer, "_tl_decode_policy", policy)
    except Exception:
        pass
    return policy


def infer_finish_reason(
    num_new_tokens: int,
    max_new_tokens: Optional[int],
    last_token: Optional[int] = None,
    eos_ids: Tuple[Optional[int], ...] = (),
) -> str:
    """
    "length" if generation stopped because it hit max_new_tokens rather than
    emitting EOS; "stop" otherwise (including when no limit was set).
    Pass last_token=None when it isn't known (streaming): then it's "length"
    whenever the limit was reached.
    """
    if max_new_tokens is None or num_new_tokens < max_new_tokens:
        return "stop"
    eos = {i for i in eos_ids if i is not None}
    return "stop" if last_token is not None and last_token in eos else "length"


# ---------------------------------------------------------------------------
# Streaming: split generated text into reasoning vs. answer
# ---------------------------------------------------------------------------
class ThinkStreamSplitter:
    """
    Incrementally splits streamed text into ("reasoning", text) and
    ("content", text) segments.

    Unlike matching tags per token, this buffers a possible partial tag
    (``<thi`` + ``nk>``), handles answer text arriving in the same token as
    ``</think>``, and can start inside a reasoning block (R1-style prompts).
    """

    _PARTIAL_RE = re.compile(r"<\s*/?\s*[A-Za-z]*\s*$")
    _MAX_HOLD = 16

    def __init__(self, start_in_reasoning: bool = False):
        self.in_reasoning = start_in_reasoning
        self._buf = ""
        self._strip_leading = False

    @staticmethod
    def _emit(out, kind, text):
        if text:
            out.append((kind, text))

    def _partial_tail_len(self) -> int:
        m = self._PARTIAL_RE.search(self._buf)
        if m and len(self._buf) - m.start() <= self._MAX_HOLD:
            return len(self._buf) - m.start()
        return 0

    def feed(self, text: str) -> List[Tuple[str, str]]:
        self._buf += text
        out: List[Tuple[str, str]] = []

        while self._buf:
            if self.in_reasoning:
                m = _CLOSE_RE.search(self._buf)
                if m:
                    self._emit(out, "reasoning", self._buf[: m.start()])
                    self._buf = self._buf[m.end() :]
                    self.in_reasoning = False
                    self._strip_leading = True
                    continue
                kind = "reasoning"
            else:
                if self._strip_leading:
                    stripped = self._buf.lstrip()
                    if not stripped:
                        self._buf = ""
                        return out
                    self._buf = stripped
                    self._strip_leading = False
                m = _OPEN_RE.search(self._buf)
                if m:
                    self._emit(out, "content", self._buf[: m.start()])
                    self._buf = self._buf[m.end() :]
                    self.in_reasoning = True
                    continue
                kind = "content"

            keep = self._partial_tail_len()
            cut = len(self._buf) - keep
            self._emit(out, kind, self._buf[:cut])
            self._buf = self._buf[cut:]
            break

        return out

    def flush(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        kind = "reasoning" if self.in_reasoning else "content"
        self._emit(out, kind, self._buf)
        self._buf = ""
        return out


def split_reasoning(text: str, start_in_reasoning: bool = False) -> Tuple[str, str]:
    """
    Non-streaming counterpart of ThinkStreamSplitter (same code path, so both
    behave identically). Returns (reasoning, answer). If generation was cut off
    mid-thought, everything lands in ``reasoning`` and ``answer`` is "".
    """
    sp = ThinkStreamSplitter(start_in_reasoning)
    segs = sp.feed(text) + sp.flush()
    reasoning = "".join(t for k, t in segs if k == "reasoning").strip()
    answer = "".join(t for k, t in segs if k == "content").strip()
    return reasoning, answer

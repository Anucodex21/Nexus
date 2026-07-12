"""
LLM Client - calls whichever real AI provider API you've configured a key
for in .env, with automatic fallback if one fails. If no keys are set at
all, falls back to a plain offline message instead of a fake echo.

Add any of these to your .env to enable a provider:
  GROQ_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY, HUGGINGFACE_API_KEY,
  ANTHROPIC_API_KEY, OPENAI_API_KEY

NVIDIA NIM (build.nvidia.com / integrate.api.nvidia.com) keys are
account-scoped, not model-scoped, so ONE key can call any model on the
catalog just by changing the "model" field. Set NVIDIA_API_KEY once and
every "nvidia:<name>" provider below becomes available automatically.
If you actually hold separate NVIDIA accounts/keys for specific models,
you can optionally override any of them with its own env var (see
NVIDIA_MODELS below) - that's opt-in, not required.

Set LLM_PROVIDER_ORDER (comma-separated) to control priority, e.g.
"claude,groq,gemini,nvidia:glm". Free tiers are tried first by default.
"""

import os
import json
import requests

SYSTEM_PROMPT = (
    "You are Nexus, a helpful, direct, and knowledgeable AI assistant. "
    "Keep answers concise and useful."
)

# Every NVIDIA NIM model available as its own selectable provider, named
# "nvidia:<name>". Each entry: (model slug, optional dedicated key env var).
# The dedicated key env var is optional - if unset, the shared
# NVIDIA_API_KEY is used instead. All of these are OpenAI-compatible chat
# completion models served from https://integrate.api.nvidia.com/v1.
NVIDIA_MODELS = {
    "glm":       ("z-ai/glm-5.2",                                  "NVIDIA_GLM_API_KEY"),
    "minimax":   ("minimaxai/minimax-m3",                          "NVIDIA_MINIMAX_API_KEY"),
    "minimax2":  ("minimaxai/minimax-m2.7",                        "NVIDIA_MINIMAX2_API_KEY"),
    "nemotron":  ("nvidia/nemotron-3-ultra-550b-a55b",             "NVIDIA_NEMOTRON_API_KEY"),
    "nemotron-r":("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", "NVIDIA_NEMOTRON_R_API_KEY"),
    "stepflash": ("stepfun-ai/step-3.7-flash",                     "NVIDIA_STEPFLASH_API_KEY"),
    "mistral":   ("mistralai/mistral-medium-3.5-128b",             "NVIDIA_MISTRAL_API_KEY"),
    "mixtral":   ("mistralai/mixtral-8x7b-instruct-v0.1",          "NVIDIA_MIXTRAL_API_KEY"),
    "deepseek":  ("deepseek-ai/deepseek-v4-flash",                 "NVIDIA_DEEPSEEK_API_KEY"),
    "deepseek-pro": ("deepseek-ai/deepseek-v4-pro",                "NVIDIA_DEEPSEEK_PRO_API_KEY"),
    "llama33":   ("meta/llama-3.3-70b-instruct",                   "NVIDIA_LLAMA33_API_KEY"),
    "llama31-70b": ("meta/llama-3.1-70b-instruct",                 "NVIDIA_LLAMA31_70B_API_KEY"),
    "llama31-8b":  ("meta/llama-3.1-8b-instruct",                  "NVIDIA_LLAMA31_8B_API_KEY"),
    "llama32-3b":  ("meta/llama-3.2-3b-instruct",                  "NVIDIA_LLAMA32_3B_API_KEY"),
    "llama32-1b":  ("meta/llama-3.2-1b-instruct",                  "NVIDIA_LLAMA32_1B_API_KEY"),
    "gemma2":    ("google/gemma-2-2b-it",                          "NVIDIA_GEMMA2_API_KEY"),
    "diffgemma": ("google/diffusiongemma-26b-a4b-it",              "NVIDIA_DIFFGEMMA_API_KEY"),
}

# groq/openrouter/nvidia small models are fast token-streaming APIs.
# huggingface's free inference API is prone to slow "cold start" delays
# (10-20s+) when a model hasn't been called recently, so it's pushed to
# the back of the auto fallback chain instead of tried early.
DEFAULT_ORDER = (
    ["groq", "gemini", "openrouter", "claude", "openai", "huggingface"]
    + [f"nvidia:{name}" for name in NVIDIA_MODELS]
    + ["local"]
)

# Small instruct models that actually run on a CPU-only laptop in
# reasonable time. Override with LOCAL_MODEL_NAME in .env if you have a
# GPU and want something bigger.
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class LLMClient:
    def __init__(self):
        shared_nvidia_key = os.getenv("NVIDIA_API_KEY") or os.getenv("TEXT_CODING_PRO")

        self.keys = {
            "groq": os.getenv("GROQ_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY"),
            "openrouter": os.getenv("OPENROUTER_API_KEY"),
            "huggingface": os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN"),
            "claude": os.getenv("ANTHROPIC_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
            # Not a real API key - just an opt-in switch. A multi-hundred-MB
            # model download shouldn't happen silently just because cloud
            # providers failed, so this only activates if explicitly enabled.
            "local": "enabled" if os.getenv("ENABLE_LOCAL_MODEL", "").lower() in ("1", "true", "yes") else None,
        }
        self._local_engine = None  # lazy-loaded on first use, shared across requests
        self._local_model_name = os.getenv("LOCAL_MODEL_NAME", DEFAULT_LOCAL_MODEL)
        # Register one provider per NVIDIA model, each falling back to the
        # shared key if it doesn't have its own dedicated one set.
        self.nvidia_model_slug = {}
        for name, (model_slug, key_env) in NVIDIA_MODELS.items():
            provider_name = f"nvidia:{name}"
            self.keys[provider_name] = os.getenv(key_env) or shared_nvidia_key
            self.nvidia_model_slug[provider_name] = model_slug

        order_env = os.getenv("LLM_PROVIDER_ORDER")
        order = [p.strip() for p in order_env.split(",")] if order_env else DEFAULT_ORDER

        handler_map = {
            "groq": self._call_groq,
            "gemini": self._call_gemini,
            "openrouter": self._call_openrouter,
            "huggingface": self._call_huggingface,
            "claude": self._call_claude,
            "openai": self._call_openai,
            "local": self._call_local,
        }
        for provider_name in self.nvidia_model_slug:
            handler_map[provider_name] = self._make_nvidia_handler(provider_name)

        # Streaming handlers - only providers with a real token-streaming
        # API get one. Gemini/HuggingFace fall back to "whole response as
        # one chunk" so they still work through the same streaming code
        # path, just without the progressive token-by-token effect.
        stream_handler_map = {
            "groq": self._stream_groq,
            "gemini": self._make_whole_stream(self._call_gemini),
            "openrouter": self._stream_openrouter,
            "huggingface": self._make_whole_stream(self._call_huggingface),
            "claude": self._stream_claude,
            "openai": self._stream_openai,
            # Local generation isn't wired for token-by-token streaming yet
            # (needs a TextIteratorStreamer thread) - whole reply as one
            # chunk for now, same pattern used for Gemini/HuggingFace.
            "local": self._make_whole_stream(self._call_local),
        }
        for provider_name in self.nvidia_model_slug:
            stream_handler_map[provider_name] = self._make_nvidia_stream_handler(provider_name)

        self.provider_names = []
        self.providers = []
        self.stream_handlers = {}
        for name in order:
            if self.keys.get(name) and name in handler_map:
                self.provider_names.append(name)
                self.providers.append(handler_map[name])
                if name in stream_handler_map:
                    self.stream_handlers[name] = stream_handler_map[name]

    def available_providers(self):
        return list(self.provider_names)

    def chat(self, messages, preferred=None) -> tuple[str, str]:
        """Returns (reply_text, provider_name_used)."""
        if not self.providers:
            return self._offline_fallback(messages), "offline"

        order = list(zip(self.provider_names, self.providers))
        if preferred and preferred in self.provider_names:
            order.sort(key=lambda pair: 0 if pair[0] == preferred else 1)

        last_error = None
        for name, provider in order:
            try:
                return provider(messages), name
            except Exception as e:
                last_error = e
                continue
        return (
            f"All configured providers failed (last error: {last_error}).\n"
            + self._offline_fallback(messages),
            "offline",
        )

    def chat_stream(self, messages, preferred=None):
        """Generator yielding dicts as a reply is generated, so a caller
        can forward text to the client as soon as it exists instead of
        waiting for the full response:
          {"type": "start", "provider": <name>}   - once, when a provider
                                                      starts responding
          {"type": "delta", "text": <chunk>}       - zero or more times
        Providers are tried in the same fallback order as chat(). A
        provider only "counts" as started once it successfully produces
        its first chunk - if it errors before that (bad key, network
        issue), the next provider in the chain is tried instead, with no
        partial output shown to the client.
        """
        if not self.stream_handlers:
            yield {"type": "start", "provider": "offline"}
            yield {"type": "delta", "text": self._offline_fallback(messages)}
            return

        order = [(n, self.stream_handlers[n]) for n in self.provider_names if n in self.stream_handlers]
        if preferred and preferred in self.stream_handlers:
            order.sort(key=lambda pair: 0 if pair[0] == preferred else 1)

        last_error = None
        for name, stream_fn in order:
            try:
                gen = stream_fn(messages)
                first_chunk = next(gen)
            except Exception as e:
                last_error = e
                continue

            yield {"type": "start", "provider": name}
            if first_chunk:
                yield {"type": "delta", "text": first_chunk}
            try:
                for chunk in gen:
                    if chunk:
                        yield {"type": "delta", "text": chunk}
            except Exception as e:
                yield {"type": "delta", "text": f"\n\n[connection interrupted: {e}]"}
            return

        yield {"type": "start", "provider": "offline"}
        yield {
            "type": "delta",
            "text": f"All configured providers failed (last error: {last_error}).\n"
            + self._offline_fallback(messages),
        }

    # ---------------- Providers ----------------

    def _call_groq(self, messages):
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.keys['groq']}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                "temperature": 0.7,
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_gemini(self, messages):
        contents = [
            {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
            for m in messages
        ]
        resp = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-1.5-flash:generateContent?key={self.keys['gemini']}",
            json={"contents": contents, "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]}},
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

    def _call_openrouter(self, messages):
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.keys['openrouter']}"},
            json={
                "model": "meta-llama/llama-3.1-8b-instruct:free",
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_huggingface(self, messages):
        prompt = SYSTEM_PROMPT + "\n" + "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        resp = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
            headers={"Authorization": f"Bearer {self.keys['huggingface']}"},
            json={"inputs": prompt, "parameters": {"max_new_tokens": 300}},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            return data[0].get("generated_text", "").replace(prompt, "").strip()
        return str(data)

    def _call_claude(self, messages):
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.keys["claude"],
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 1000,
                "system": SYSTEM_PROMPT,
                "messages": messages,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return "".join(block.get("text", "") for block in data.get("content", []))

    def _call_openai(self, messages):
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.keys['openai']}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_local(self, messages):
        """Run generation on a real pretrained model, fully offline once
        downloaded. Loaded lazily on first call (not at startup) since it
        can be a multi-hundred-MB download + real load time, and cached on
        self so later requests reuse the same instance instead of
        reloading the model from disk every time."""
        if self._local_engine is None:
            # Imported here, not at module level, so the whole backend
            # doesn't require torch/transformers to be installed just to
            # start up - only when someone actually enables local mode.
            from llm.inference import LLMInference
            self._local_engine = LLMInference(self._local_model_name)
        return self._local_engine.chat(messages, system_prompt=SYSTEM_PROMPT)

    def _make_nvidia_handler(self, provider_name):
        """Build a closure that calls one specific NVIDIA NIM model, using
        that provider's resolved key (dedicated env var if set, else the
        shared NVIDIA_API_KEY)."""
        model = self.nvidia_model_slug[provider_name]

        def handler(messages):
            resp = requests.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.keys[provider_name]}"},
                json={
                    "model": model,
                    "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                    "temperature": 0.7,
                    "max_tokens": 1024,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

        return handler

    # ---------------- Streaming providers ----------------
    # Groq / OpenRouter / OpenAI / NVIDIA NIM all speak the same
    # OpenAI-compatible SSE streaming format, so one shared helper covers
    # all of them. Each still has its own thin wrapper below so it plugs
    # into stream_handler_map the same way the non-streaming _call_*
    # methods plug into handler_map.

    def _stream_openai_compatible(self, url, headers, model, messages, extra=None):
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            "temperature": 0.7,
            "stream": True,
        }
        if extra:
            payload.update(extra)
        resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=30)
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or [{}]
            text = choices[0].get("delta", {}).get("content")
            if text:
                yield text

    def _stream_groq(self, messages):
        yield from self._stream_openai_compatible(
            "https://api.groq.com/openai/v1/chat/completions",
            {"Authorization": f"Bearer {self.keys['groq']}"},
            "llama-3.3-70b-versatile",
            messages,
        )

    def _stream_openrouter(self, messages):
        yield from self._stream_openai_compatible(
            "https://openrouter.ai/api/v1/chat/completions",
            {"Authorization": f"Bearer {self.keys['openrouter']}"},
            "meta-llama/llama-3.1-8b-instruct:free",
            messages,
        )

    def _stream_openai(self, messages):
        yield from self._stream_openai_compatible(
            "https://api.openai.com/v1/chat/completions",
            {"Authorization": f"Bearer {self.keys['openai']}"},
            "gpt-4o-mini",
            messages,
        )

    def _stream_claude(self, messages):
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.keys["claude"],
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 1000,
                "system": SYSTEM_PROMPT,
                "messages": messages,
                "stream": True,
            },
            stream=True,
            timeout=30,
        )
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "content_block_delta":
                text = obj.get("delta", {}).get("text")
                if text:
                    yield text

    def _make_nvidia_stream_handler(self, provider_name):
        """Streaming counterpart to _make_nvidia_handler - same
        OpenAI-compatible endpoint, just with stream=True."""
        model = self.nvidia_model_slug[provider_name]

        def handler(messages):
            yield from self._stream_openai_compatible(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                {"Authorization": f"Bearer {self.keys[provider_name]}"},
                model,
                messages,
                extra={"max_tokens": 1024},
            )

        return handler

    def _make_whole_stream(self, call_fn):
        """Wrap a non-streaming _call_* method so it fits the same
        generator interface, just yielding its entire reply as one chunk
        instead of token-by-token. Used for providers (Gemini, the free
        HuggingFace inference API) that don't offer easy SSE streaming
        through a plain requests call."""

        def gen(messages):
            yield call_fn(messages)

        return gen

    # ---------------- Offline fallback ----------------

    def _offline_fallback(self, messages):
        last_user_msg = messages[-1]["content"] if messages else ""
        return (
            "No AI provider is configured yet, so I can't have a full open-ended "
            f"conversation about '{last_user_msg}'. Add a free API key (Groq or "
            "Gemini are easiest to get) to your .env to unlock real responses."
        )

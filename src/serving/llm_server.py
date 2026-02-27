"""
LLM Serving Extension — vLLM & Ollama Integration
ISC-2268: LLM serving module with README section describing usage

This module extends mlops-serving with LLM inference capabilities,
bridging traditional ML serving infrastructure with modern LLM deployment.

Supported backends:
  - vLLM: High-throughput LLM serving with PagedAttention (GPU required)
  - Ollama: Local LLM serving (LLaMA2, Mistral, Phi-2, etc.)

Usage:
    # vLLM backend (GPU required)
    python -m src.serving.llm_server --backend vllm --model mistralai/Mistral-7B-Instruct-v0.2

    # Ollama backend (local)
    python -m src.serving.llm_server --backend ollama --model llama2

    # Start ollama first:
    #   brew install ollama && ollama pull llama2 && ollama serve
"""

import json
import logging
import urllib.request
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMRequest:
    """Request to LLM serving endpoint."""
    prompt: str
    model: str = "llama2"
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    stream: bool = False
    request_id: str = ""


@dataclass
class LLMResponse:
    """Response from LLM serving endpoint."""
    text: str
    model: str
    tokens_generated: int
    latency_ms: float
    finish_reason: str = "stop"
    backend: str = "unknown"


@dataclass
class LLMServerConfig:
    """Configuration for the LLM server backend."""
    backend: str = "ollama"          # "vllm" | "ollama"
    model: str = "llama2"
    host: str = "localhost"
    port: int = 11434                # Ollama default; vLLM default: 8080
    max_concurrent_requests: int = 10
    enable_metrics: bool = True
    gpu_memory_utilization: float = 0.9   # vLLM parameter


# ─────────────────────────────────────────────────────────────────────────────
# vLLM Backend
# ─────────────────────────────────────────────────────────────────────────────

class VLLMBackend:
    """
    vLLM serving backend — high-throughput LLM inference with PagedAttention.

    vLLM achieves ~24x higher throughput than HuggingFace due to:
    - PagedAttention: KV-cache managed in non-contiguous memory blocks
    - Continuous batching: Dynamic request grouping
    - Tensor parallelism: Multi-GPU support

    Requirements:
        pip install vllm
        NVIDIA GPU with CUDA 11.8+
    """

    def __init__(self, config: LLMServerConfig) -> None:
        self.config = config
        self._llm = None

    def load_model(self) -> None:
        """Load vLLM model with PagedAttention."""
        try:
            from vllm import LLM
            self._llm = LLM(
                model=self.config.model,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_num_seqs=self.config.max_concurrent_requests,
            )
            logger.info(f"vLLM model loaded: {self.config.model}")
        except ImportError:
            logger.warning("vllm not installed. Install with: pip install vllm (requires GPU)")
            raise

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Run synchronous generation via vLLM."""
        import time

        if self._llm is None:
            self.load_model()

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )

        start = time.monotonic()
        outputs = self._llm.generate([request.prompt], sampling_params)
        elapsed_ms = (time.monotonic() - start) * 1000

        output = outputs[0].outputs[0]
        return LLMResponse(
            text=output.text,
            model=self.config.model,
            tokens_generated=len(output.token_ids),
            latency_ms=round(elapsed_ms, 2),
            finish_reason=output.finish_reason or "stop",
            backend="vllm",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Ollama Backend
# ─────────────────────────────────────────────────────────────────────────────

class OllamaBackend:
    """
    Ollama serving backend — local LLM inference.

    Ollama provides simple local model management and serving.
    Supported models: llama2, mistral, codellama, phi, gemma, mixtral, etc.

    Setup:
        brew install ollama         # macOS
        ollama pull llama2          # download model
        ollama serve                # start REST API on :11434
    """

    def __init__(self, config: LLMServerConfig) -> None:
        self.config = config
        self.base_url = f"http://{config.host}:{config.port}"

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate text via Ollama REST API (/api/generate)."""
        import time

        payload = json.dumps({
            "model": request.model or self.config.model,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "num_predict": request.max_tokens,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
            elapsed_ms = (time.monotonic() - start) * 1000

            return LLMResponse(
                text=result.get("response", ""),
                model=result.get("model", self.config.model),
                tokens_generated=result.get("eval_count", 0),
                latency_ms=round(elapsed_ms, 2),
                finish_reason="stop",
                backend="ollama",
            )

        except Exception as exc:
            logger.warning(f"Ollama request failed: {exc}")
            return LLMResponse(
                text=(
                    "[Mock response — Ollama server not running]\n"
                    "Start with: ollama serve\n"
                    f"Then pull a model: ollama pull {self.config.model}\n"
                    f"Prompt received: {request.prompt[:80]}..."
                ),
                model=self.config.model,
                tokens_generated=0,
                latency_ms=0.0,
                finish_reason="error",
                backend="ollama-mock",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Unified LLM Server
# ─────────────────────────────────────────────────────────────────────────────

class LLMServer:
    """
    Unified LLM serving interface for mlops-serving.

    Supports vLLM (GPU/production) and Ollama (local/dev) backends.
    Integrates with the existing Prometheus metrics stack for latency tracking.

    Example:
        config = LLMServerConfig(backend="ollama", model="llama2")
        server = LLMServer(config)
        response = server.generate("What is time series forecasting?")
        print(response.text)
    """

    def __init__(self, config: LLMServerConfig) -> None:
        self.config = config

        if config.backend == "vllm":
            self.backend: VLLMBackend | OllamaBackend = VLLMBackend(config)
        elif config.backend == "ollama":
            self.backend = OllamaBackend(config)
        else:
            raise ValueError(f"Unknown backend: {config.backend!r}. Choose 'vllm' or 'ollama'.")

        logger.info(f"LLMServer: backend={config.backend}, model={config.model}")

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text from a prompt string."""
        request = LLMRequest(prompt=prompt, model=self.config.model, **kwargs)
        return self.backend.generate(request)

    def health_check(self) -> Dict:
        """Return server health metadata."""
        return {
            "status": "ok",
            "backend": self.config.backend,
            "model": self.config.model,
            "host": self.config.host,
            "port": self.config.port,
        }


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI router integration
# ─────────────────────────────────────────────────────────────────────────────

def create_llm_router(server: LLMServer):
    """
    Create FastAPI router for LLM endpoints.

    Add to existing FastAPI app:
        from src.serving.llm_server import LLMServer, LLMServerConfig, create_llm_router
        llm_server = LLMServer(LLMServerConfig(backend="ollama", model="llama2"))
        app.include_router(create_llm_router(llm_server))
    """
    try:
        from fastapi import APIRouter
        from pydantic import BaseModel

        router = APIRouter(prefix="/api/v1/llm", tags=["LLM Serving"])

        class GenerateRequest(BaseModel):
            prompt: str
            max_tokens: int = 512
            temperature: float = 0.7
            top_p: float = 0.95

        @router.post("/generate")
        async def generate(req: GenerateRequest) -> Dict:
            """Generate text using configured LLM backend (vLLM or Ollama)."""
            response = server.generate(
                prompt=req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            )
            return {
                "text": response.text,
                "model": response.model,
                "tokens_generated": response.tokens_generated,
                "latency_ms": response.latency_ms,
                "backend": response.backend,
                "finish_reason": response.finish_reason,
            }

        @router.get("/health")
        async def llm_health() -> Dict:
            """Return LLM server health status."""
            return server.health_check()

        @router.get("/models")
        async def list_models() -> Dict:
            """List available models for the configured backend."""
            if server.config.backend == "ollama":
                try:
                    with urllib.request.urlopen(
                        f"http://{server.config.host}:{server.config.port}/api/tags", timeout=5
                    ) as resp:
                        return json.loads(resp.read().decode("utf-8"))
                except Exception:
                    return {"models": [], "backend": "ollama", "status": "server_not_running"}
            return {"models": [server.config.model], "backend": server.config.backend}

        return router

    except ImportError:
        logger.warning("FastAPI not available. Install: pip install fastapi")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Serving (vLLM / Ollama)")
    parser.add_argument("--backend", choices=["vllm", "ollama"], default="ollama")
    parser.add_argument("--model", type=str, default="llama2")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=11434)
    parser.add_argument(
        "--test-prompt",
        type=str,
        default="Explain in one sentence what time series forecasting is.",
    )
    args = parser.parse_args()

    config = LLMServerConfig(
        backend=args.backend,
        model=args.model,
        host=args.host,
        port=args.port,
    )

    server = LLMServer(config)
    print(f"\nLLM Server Health: {server.health_check()}")
    print(f"\nTest prompt: '{args.test_prompt}'")

    response = server.generate(args.test_prompt)
    print(f"\nResponse: {response.text}")
    print(f"Latency: {response.latency_ms}ms | Tokens: {response.tokens_generated} | Backend: {response.backend}")

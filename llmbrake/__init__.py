"""llm-brake: real deadlines, a Stop that stops, and proof the GPU went idle, for local LLM servers."""

from .aio import pump
from .budget import CANCELLED, DEADLINE, DISCONNECTED, Budget
from .clients import GenerationStopped, OllamaClient, OpenAICompatClient, ServerError
from .gate import Gate, GateBusy, GateHeld
from .transport import HTTPError, Interrupted, open_stream
from .verify import NullVerifier, OllamaRunnerVerifier, SlotsVerifier, StopReport

__version__ = "0.1.0"

__all__ = [
    "Budget", "CANCELLED", "DEADLINE", "DISCONNECTED",
    "OllamaClient", "OpenAICompatClient", "GenerationStopped", "ServerError", "HTTPError",
    "Gate", "GateBusy", "GateHeld", "Interrupted", "open_stream",
    "SlotsVerifier", "OllamaRunnerVerifier", "NullVerifier", "StopReport", "pump",
]

"""LLM adapters for the tool-calling chat loop.

* :class:`HeuristicLLM` — offline, no model. Parses intent with regex/keywords
  and emits tool calls. Always available; the fallback when Ollama is absent.
* :class:`OllamaClient` — talks to a local Ollama server (default
  http://localhost:11434) via stdlib urllib (no extra deps). Uses native tool
  calling; falls back to a heuristic parse if the model returns no tool calls.
"""

from .heuristic import HeuristicLLM
from .ollama import OllamaClient

__all__ = ["HeuristicLLM", "OllamaClient"]

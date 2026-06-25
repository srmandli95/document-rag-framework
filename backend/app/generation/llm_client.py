import importlib
from abc import ABC, abstractmethod
from typing import Any

from app.config.settings import settings

try:
    OpenAI: Any | None = importlib.import_module("openai").OpenAI
except ImportError:
    OpenAI = None


class LLMClient(ABC):
    """
    Base LLM client interface.

    Future providers like Gemini, local models, or Bedrock can implement
    this same generate() method.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate an answer from a prompt.
        """
        raise NotImplementedError


class OpenAILLMClient(LLMClient):
    """
    OpenAI LLM client using chat completions.
    """

    def __init__(self, api_key: str, model_name: str):
        if not api_key or not api_key.strip():
            raise ValueError(
                "OPENAI_API_KEY is required when LLM_PROVIDER is set to openai"
            )

        if not model_name or not model_name.strip():
            raise ValueError("OPENAI_MODEL_NAME is required")

        if OpenAI is None:
            raise ImportError(
                "openai package is not installed. Run: pipenv install openai"
            )

        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("prompt is required")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a grounded document assistant. "
                        "Answer only from the supplied evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        answer = response.choices[0].message.content

        if not answer:
            return ""

        return answer.strip()


def get_llm_client() -> LLMClient:
    """
    Return the configured LLM client.

    Day 11 supports OpenAI first.
    Gemini/local can be added later without changing answer generation logic.
    """
    provider = getattr(settings, "LLM_PROVIDER", "openai").lower().strip()

    if provider == "openai":
        return OpenAILLMClient(
            api_key=getattr(settings, "OPENAI_API_KEY", ""),
            model_name=getattr(settings, "OPENAI_MODEL_NAME", "gpt-4o-mini"),
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{provider}'. Supported providers: openai"
    )

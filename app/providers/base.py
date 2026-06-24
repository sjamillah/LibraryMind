from abc import ABC, abstractmethod


class AIProvider(ABC):
    """Contract every AI provider must satisfy."""
    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @abstractmethod
    def generate(self, prompt: str, system: str = "", temperature: float = 0.7, max_tokens: int = 1000) -> str:
        ...

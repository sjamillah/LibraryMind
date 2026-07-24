"""Domain-specific exceptions.

Each one subclasses the builtin exception type the lab spec and existing test
suite already require — RuntimeError for provider/config failures, ValueError
for malformed AI output — so every `pytest.raises(RuntimeError)` /
`pytest.raises(ValueError)` assertion and the spec's own acceptance criteria
("a RuntimeError is raised with a helpful message") still hold.

The point isn't a new type for its own sake: it's giving each router something
specific to catch. `except RuntimeError` in an API route can't tell "all AI
providers failed" apart from any other RuntimeError a dependency happens to
raise — it would misreport an unrelated bug as a 503 provider outage. Catching
the specific subclass instead means the generic `except Exception -> 500`
fallback only ever catches what nobody anticipated.
"""


class ConfigurationError(RuntimeError):
    """Required configuration is missing or invalid."""


class AllProvidersFailedError(RuntimeError):
    """Every configured AI provider failed to generate a response."""


class GatewayRequestError(RuntimeError):
    """The AI gateway request failed after exhausting all retries."""


class InvalidAIResponseError(ValueError):
    """An AI response could not be parsed into the expected structured JSON."""


class EmbeddingModelError(RuntimeError):
    """The embedding model could not be loaded or failed to encode text."""

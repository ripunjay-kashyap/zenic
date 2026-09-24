"""Exception hierarchy for Zenic.

Every failure that crosses a module boundary is raised as a ZenicError subclass
so callers (nodes, the UI, scripts) can distinguish an operational failure —
a missing key, a dead upstream — from a genuine bug.
"""


class ZenicError(Exception):
    """Base class for every error Zenic raises deliberately."""


class ConfigError(ZenicError):
    """Required configuration is missing or invalid."""


class LLMError(ZenicError):
    """A call to the language model provider failed or returned unusable output."""


class LLMContextLimitError(LLMError):
    """The provider rejected the request size; less context may fit."""


class LLMOutputError(LLMError):
    """The provider could not produce valid structured output."""


class VectorStoreError(ZenicError):
    """The vector store is unreachable, misconfigured, or returned bad data."""


class RetrievalError(ZenicError):
    """The retrieval pipeline could not produce candidates."""


class ExternalAPIError(ZenicError):
    """A third-party HTTP API (USDA, wger, OpenFDA) failed."""

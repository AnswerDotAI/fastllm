"fastllm_v2 errors."

class FastLLMError(Exception):
    "Base fastllm_v2 error."

class UnsupportedCapabilityError(FastLLMError):
    "Raised when a requested feature is unsupported."

class ProtocolError(FastLLMError):
    "Raised when provider payloads do not match expected protocol shape."

class SpecError(FastLLMError):
    "Raised when an OpenAPI spec cannot be parsed as expected."

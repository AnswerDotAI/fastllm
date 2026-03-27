"fastllm_v2 package."

__version__ = "0.1.0"

from .clients import AnthropicClient, GeminiClient, OpenAIClient
from .config import ClientConfig
from .costs import CostBreakdown, ModelPrice, estimate_cost
from .errors import APIError, FastLLMError, ProtocolError, SpecError, UnsupportedCapabilityError
from .factory import mk_client
from .files import FileRef, afile_content, afile_create, afile_delete, afile_get, afile_list, to_input_file_part
from .highlevel import acompletion, infer_provider, mk_auto_client
from .oapi import OpenAPIClient, client_from_spec
from .spec import OpSpec, load_spec_file, load_spec_json, load_spec_url, merge_ops, spec_to_ops
from .streaming import StreamSummary, acollect_stream
from .types import Caps, Completion, Delta, Msg, Part, RequestOptions, ToolCall, Usage

__all__ = "__version__ FastLLMError APIError UnsupportedCapabilityError ProtocolError SpecError ClientConfig OpSpec OpenAPIClient spec_to_ops load_spec_json load_spec_file load_spec_url merge_ops client_from_spec Part Msg ToolCall Usage Caps RequestOptions Completion Delta OpenAIClient AnthropicClient GeminiClient ModelPrice CostBreakdown estimate_cost mk_client infer_provider mk_auto_client acompletion StreamSummary acollect_stream FileRef to_input_file_part afile_create afile_get afile_list afile_delete afile_content".split()

"fastllm package."

__version__ = "0.1.0"

from .clients import AnthropicClient, GeminiClient, OpenAIClient
from .costs import CostBreakdown, ModelPrice, estimate_cost
from .errors import APIError, FastLLMError, ProtocolError, SpecError, UnsupportedCapabilityError
from .files import FileRef, afile_content, afile_create, afile_delete, afile_get, afile_list, to_input_file_part
from .highlevel import acompletion, infer_provider, mk_auto_client
from .oapi import OpenAPIClient, client_from_spec
from .spec import OpSpec, anthropic_ops, gemini_ops, load_spec_file, load_spec_json, load_spec_url, load_spec_yaml
from .spec import merge_ops, openai_ops, spec_to_ops
from .streaming import StreamSummary, acollect_stream
from .types import Caps, Completion, Delta, Msg, Part, RequestOptions, ToolCall, Usage

__all__ = "__version__ FastLLMError APIError UnsupportedCapabilityError ProtocolError SpecError OpSpec OpenAPIClient spec_to_ops load_spec_json load_spec_yaml load_spec_file load_spec_url merge_ops openai_ops anthropic_ops gemini_ops client_from_spec Part Msg ToolCall Usage Caps RequestOptions Completion Delta OpenAIClient AnthropicClient GeminiClient ModelPrice CostBreakdown estimate_cost infer_provider mk_auto_client acompletion StreamSummary acollect_stream FileRef to_input_file_part afile_create afile_get afile_list afile_delete afile_content".split()

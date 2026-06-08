# Release notes

<!-- do not remove -->

## 0.0.14

### New Features

- Resize images returned from tool call (Anthropic) ([#41](https://github.com/AnswerDotAI/fastllm/issues/41))

### Bugs Squashed

- Anthropic `messages: text content blocks must be non-empty` ([#44](https://github.com/AnswerDotAI/fastllm/issues/44))

- Invalid 'input[95].call_id': string too long. Expected a string with maximum length 64, but got a string with length 116 instead. ([#45](https://github.com/AnswerDotAI/fastllm/issues/45))

- Fireworks generates tool_call.ids that result in Anthropic API calls failing ([#42](https://github.com/AnswerDotAI/fastllm/issues/42))

- Claude server `tool_result` yield bug ([#40](https://github.com/AnswerDotAI/fastllm/issues/40))
  - <img width="1776" height="370" alt="Image" src="https://github.com/user-attachments/assets/d59d4e84-9d65-437b-8094-ace1295dcf00" />

- fix truncation ([#39](https://github.com/AnswerDotAI/fastllm/pull/39)), thanks to [@RensDimmendaal](https://github.com/RensDimmendaal)


## 0.0.14

### New Features

- Anthropic `messages: text content blocks must be non-empty` ([#44](https://github.com/AnswerDotAI/fastllm/issues/44))

- Resize images returned from tool call (Anthropic) ([#41](https://github.com/AnswerDotAI/fastllm/issues/41))

### Bugs Squashed

- Invalid 'input[95].call_id': string too long. Expected a string with maximum length 64, but got a string with length 116 instead. ([#45](https://github.com/AnswerDotAI/fastllm/issues/45))
  - ```py
fastspec.errors.APIError: APIError(message="Invalid 'input[95].call_id': string too long. Expected a string with maximum length 64, but got a string with length 116 instead.", endpoint='POST /backend-api/codex/responses', status_code=400, error_type='invalid_request_error', code='string_above_max_length')
```

- Fireworks generates tool_call.ids that result in Anthropic API calls failing ([#42](https://github.com/AnswerDotAI/fastllm/issues/42))
  - Fireworks generates tool call IDs that have `.` and `:` in it and that is not allowed with the Anthropic API resulting in an error when we have such tool calls in the history and then send a request to Anthropic. 

I explained the bug and reproduced it in the following dialog: https://share.solveit.pub/d/fe0d51a8b7cac878aa551978210f07a2

- Claude server `tool_result` yield bug ([#40](https://github.com/AnswerDotAI/fastllm/issues/40))
  - <img width="1776" height="370" alt="Image" src="https://github.com/user-attachments/assets/d59d4e84-9d65-437b-8094-ace1295dcf00" />

- fix truncation ([#39](https://github.com/AnswerDotAI/fastllm/pull/39)), thanks to [@RensDimmendaal](https://github.com/RensDimmendaal)
  - This PR fixes response truncation. The issue was that `s.rstrip()` always returned a `str` even though `s` could be a subclass of `str` such as `FullResponse` or `Safe`.  I also added some tests.

I've made an identical PR to lisette.


## 0.0.12

### New Features

- Add `CODEX_AUTH_PATH` ([#38](https://github.com/AnswerDotAI/fastllm/issues/38))


## 0.0.11

### New Features

- rename tools to be as safepyrun/safecmd expects ([#36](https://github.com/AnswerDotAI/fastllm/pull/36)), thanks to [@RensDimmendaal](https://github.com/RensDimmendaal)
- Add Mimo vendor, Opus 4-8, `modern_llm` preset, and fix Anthropic cache cost fallback ([#35](https://github.com/AnswerDotAI/fastllm/issues/35))

### Bugs Squashed

- Wrong roles in `mk_msgs` when `fmt2hist` ends with `tool` ([#37](https://github.com/AnswerDotAI/fastllm/issues/37))
- Update 'gpt-5.3-codex-spark' meta to support tools ([#34](https://github.com/AnswerDotAI/fastllm/issues/34))
- Model name changes from the server response invalidates model meta patches ([#33](https://github.com/AnswerDotAI/fastllm/issues/33))


## 0.0.10

### New Features

- `MediaUrl` for direct url handling without byte reading ([#31](https://github.com/AnswerDotAI/fastllm/issues/31))
- Add timeout to `mk_client` ([#26](https://github.com/AnswerDotAI/fastllm/issues/26))
- Add retry logic with exponential backoff to acomplete ([#25](https://github.com/AnswerDotAI/fastllm/pull/25)), thanks to [@ncoop57](https://github.com/ncoop57)
- Yield tool calls JIT ([#19](https://github.com/AnswerDotAI/fastllm/issues/19))

### Bugs Squashed

- force enable `web_search` for codex models ([#22](https://github.com/AnswerDotAI/fastllm/pull/22)), thanks to [@jackhogan](https://github.com/jackhogan)


## 0.0.9

### New Features

- Add `approx_pricing` helper and fix Fireworks Kimi k2p6 model registration with pricing ([#21](https://github.com/AnswerDotAI/fastllm/issues/21))
- Refactor model info into registry; add `get_model_pricing`; add new gemini models ([#20](https://github.com/AnswerDotAI/fastllm/issues/20))

### Bugs Squashed

- Fix `accounts/fireworks/models/kimi-k2p5` registration ([#23](https://github.com/AnswerDotAI/fastllm/issues/23))
- force enable `web_search` for codex models ([#22](https://github.com/AnswerDotAI/fastllm/pull/22)), thanks to [@jackhogan](https://github.com/jackhogan)


## 0.0.8

### New Features

- Add `finalize_usage` to fix anthropic reasoning token tracking; Add debug `brief` mode ([#18](https://github.com/AnswerDotAI/fastllm/issues/18))

- make old web search tool `web_search_20250305` the default ([#16](https://github.com/AnswerDotAI/fastllm/issues/16))

- fastllm chat debug mode ([#15](https://github.com/AnswerDotAI/fastllm/issues/15))
  - <img width="1244" height="823" alt="Image" src="https://github.com/user-attachments/assets/4a13d627-d069-4fdf-9807-03162e559141" />

- Track reasoning tokens in Anthropic usage; handle token details in mk_msgs; fix _trunc_param escaping; add codex auth module ([#13](https://github.com/AnswerDotAI/fastllm/issues/13))

### Bugs Squashed

- `𝍁...𝍁` `print()` rstrip, summary truncation, non-ascii fix ([#17](https://github.com/AnswerDotAI/fastllm/issues/17))

- `claude-opus-4-7` thinking bug ([#14](https://github.com/AnswerDotAI/fastllm/issues/14))


## 0.0.7

### New Features

- AsyncChat callback system ([#11](https://github.com/AnswerDotAI/fastllm/issues/11))

### Bugs Squashed

- markdown='1' ([#12](https://github.com/AnswerDotAI/fastllm/issues/12))

- `stop` status getting reset in streaming loop ([#10](https://github.com/AnswerDotAI/fastllm/issues/10))
  - Fix is to only check the stop condition if it's not met yet, e.g. 

```py
if not stop: stop = stop_and_trim(part_accum, d, stop_callables)
```


## 0.0.6


### Bugs Squashed

- `model` param in `mk_client` should be optional ([#9](https://github.com/AnswerDotAI/fastllm/issues/9))


## 0.0.5


### Bugs Squashed

- `get_model_info` `strict` param ([#8](https://github.com/AnswerDotAI/fastllm/issues/8))
  - If a model can't be resolved in `get_model_info` you can pass `strict=False` to get placeholder price values to avoid errors with `AsyncChat`. For example, we use `strict=True` in solveit and `strict=False` in shell sage where users can pass their custom models with custom base urls.

- Unresolved model fixes ([#7](https://github.com/AnswerDotAI/fastllm/issues/7))
  - When a model and it's info can't be resolved:

- Default to `openai_chat` api if not provided
- Default max tokens to 32k
- Set pricing to codex values


## 0.0.4

### Bugs Squashed

- Code fence tool fixes ([#6](https://github.com/AnswerDotAI/fastllm/issues/6))
  - `_split_msg_on_fences` fix which now correctly handles mixed msg content, e.g. thinking + text, tool use + text etc..
  - `_fence_re` new line start check

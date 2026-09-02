# Release notes

<!-- do not remove -->

## 0.0.49

### New Features

- Add `oauth_token` support across vendors for per-user subscription auth, and apply flat subscription pricing to Claude Code models ([#95](https://github.com/AnswerDotAI/fastllm/issues/95))


## 0.0.48

### New Features

- Rewrite Codex backend with async token refresh, device-code login and fastllm-auth CLI; thread per-call kwargs through Responses turns ([#94](https://github.com/AnswerDotAI/fastllm/issues/94))


## 0.0.47

### New Features

- Server tools, effort levels, and batched calls in the Responses facade ([#93](https://github.com/AnswerDotAI/fastllm/pull/93)), thanks to [@jph00](https://github.com/jph00)


## 0.0.46

### New Features

- Add Fable 5.1 and DeepSeek V4 Flash Vision, with official DeepSeek pricing ([#92](https://github.com/AnswerDotAI/fastllm/pull/92)), thanks to [@jph00](https://github.com/jph00)


## 0.0.45

### New Features

- Import APIError from fasttransport, and drop the fastspec dependency ([#91](https://github.com/AnswerDotAI/fastllm/pull/91)), thanks to [@jph00](https://github.com/jph00)


## 0.0.44

### New Features

- Initial responses API support ([#90](https://github.com/AnswerDotAI/fastllm/issues/90))
- Add Responses API continuation via `previous_response_id` and Completion.`response_id`; use `call_id` for tool call IDs ([#89](https://github.com/AnswerDotAI/fastllm/issues/89))


## 0.0.43

### New Features

- Add glm-5.3-flash ([#88](https://github.com/AnswerDotAI/fastllm/issues/88))


## 0.0.42

### New Features

- Add `wrap_typed`/`unwrap_typed` to preserve str subclasses over the wire ([#87](https://github.com/AnswerDotAI/fastllm/issues/87))


## 0.0.41

### New Features

- Replace httpx with httpx2 ([#85](https://github.com/AnswerDotAI/fastllm/issues/85))


## 0.0.40

### New Features

- Rename `_query` kwarg to `query_` for SSE streaming in gemini ([#84](https://github.com/AnswerDotAI/fastllm/issues/84))


## 0.0.39

### New Features

- Move `mk_msg`/`mk_msgs` and Completion to aidialog, relocate prompt caching to anthropic module, drop prefill support ([#83](https://github.com/AnswerDotAI/fastllm/issues/83))


## 0.0.38

### New Features

- Add `norm_tr_parts` to anthropic ([#82](https://github.com/AnswerDotAI/fastllm/issues/82))
- Replace generic Part/ToolCall with typed part classes (Text, Thinking, ToolUse, ToolResult, Input...) across all providers ([#81](https://github.com/AnswerDotAI/fastllm/issues/81))
- Refactor streaming to yield typed Part/Status objects with `formatted` rendering, replacing dict chunks and StreamFormatter with StreamAccum/Refresh ([#80](https://github.com/AnswerDotAI/fastllm/issues/80))


## 0.0.37

### New Features

- Migrate from toolslm.funccall to fastcore.funccall and drop toolslm dependency ([#79](https://github.com/AnswerDotAI/fastllm/issues/79))


## 0.0.36

### New Features

- Add kimi-k3 model info and pricing for moonshot ([#78](https://github.com/AnswerDotAI/fastllm/issues/78))


## 0.0.35

### New Features

- Add tiered pricing to the OpenAI Responses and Gemini cost functions, with registry fixes for codex context limits and flat-rate pricing ([#77](https://github.com/AnswerDotAI/fastllm/issues/77))

### Bugs Squashed

- Move ToolResponse to aidialog ([#73](https://github.com/AnswerDotAI/fastllm/pull/73)), thanks to [@curtis-allan](https://github.com/curtis-allan)


## 0.0.34

### New Features

- Make APIRegistry a dict subclass, removing the .apis attribute ([#76](https://github.com/AnswerDotAI/fastllm/issues/76))
- Load api plugins on demand via `fastllm.apis` entry points; index ``api_registry`` directly ([#75](https://github.com/AnswerDotAI/fastllm/issues/75))
- Move media/content conversion helpers from fastllm.chat to aidialog.`msg_parts` ([#72](https://github.com/AnswerDotAI/fastllm/issues/72))

### Bugs Squashed

- codex client caches a stale access token ([#74](https://github.com/AnswerDotAI/fastllm/issues/74))


## 0.0.33

### New Features

- Update models ([#71](https://github.com/AnswerDotAI/fastllm/issues/71))
- Approx usage for interrupted streaming requests ([#55](https://github.com/AnswerDotAI/fastllm/pull/55)), thanks to [@KeremTurgutlu](https://github.com/KeremTurgutlu)


## 0.0.32

### New Features

- Move the message model to aidialog and depend on it ([#70](https://github.com/AnswerDotAI/fastllm/pull/70)), thanks to [@jph00](https://github.com/jph00)
- Replace HTML `<details>` tool/usage envelopes with fenced JSON wire format; refactor split_tools into parse_tools/strip_tools/conv_tools and add fastcore dependency ([#69](https://github.com/AnswerDotAI/fastllm/issues/69))


## 0.0.31

### New Features

- Add vendor-prefixed model strings (e.g. codex/gpt-5.5) with `split_vendor` resolution in `mk_client`, acomplete, and AsyncChat; track `last_req_use` for per-request usage stats ([#68](https://github.com/AnswerDotAI/fastllm/issues/68))


## 0.0.30

### Bugs Squashed

- Move early-return guard in `_trunc_str` ([#66](https://github.com/AnswerDotAI/fastllm/issues/66))


## 0.0.29

### New Features

- Add gpt-5.6 (sol/terra/luna) model registrations ([#65](https://github.com/AnswerDotAI/fastllm/issues/65))


## 0.0.28

### New Features

- Add hist2fmt inverse of fmt2hist, support mx=None for no truncation ([#64](https://github.com/AnswerDotAI/fastllm/issues/64))
- Add Meta AI vendor and muse-spark-1.1 model ([#62](https://github.com/AnswerDotAI/fastllm/pull/62)), thanks to [@ncoop57](https://github.com/ncoop57)
- Switch to anthropic api for mimo ([#60](https://github.com/AnswerDotAI/fastllm/issues/60))

## 0.0.27

- limit gpt context to 250k
- register cc sonn5 pricing

## 0.0.26

Register sonn5 with (30% less context) and refresh model prices json

## 0.0.25

### New Features

- GLM 5.2 fast ([#59](https://github.com/AnswerDotAI/fastllm/issues/59))

- Deepseek peak hour pricing ([#58](https://github.com/AnswerDotAI/fastllm/issues/58))

## 0.0.24

### New Features

- Add collapsible thinking tags and stripping in StreamFormatter and fmt2hist ([#57](https://github.com/AnswerDotAI/fastllm/issues/57))

- Add model name to usage details html ([#56](https://github.com/AnswerDotAI/fastllm/pull/56)), thanks to [@jackhogan](https://github.com/jackhogan)

- Register claude code models with codex pricing


## 0.0.24

### New Features

- Add collapsible thinking tags and stripping in StreamFormatter and fmt2hist ([#57](https://github.com/AnswerDotAI/fastllm/issues/57))

- Add model name to usage details html ([#56](https://github.com/AnswerDotAI/fastllm/pull/56)), thanks to [@jackhogan](https://github.com/jackhogan)

- Register claude code models with codex pricing


## 0.0.23

- Add claude code backend, see https://github.com/AnswerDotAI/fastllm-claude-code


## 0.0.22

### New Features

- Add qwen3p7-plus and glm-5p2 Fireworks model registrations ([#54](https://github.com/AnswerDotAI/fastllm/issues/54))


## 0.0.20

### New Features

- Add `PrettyString` as another `FullResponse` marker ([#53](https://github.com/AnswerDotAI/fastllm/issues/53))


## 0.0.19

### New Features

- Set default non-thinking `temp=None`, add `kimi-k2.7-code` ([#52](https://github.com/AnswerDotAI/fastllm/issues/52))

### Bugs Squashed

- Filter top level gemini tool schema fields such as `$refs`, `$defs`... ([#51](https://github.com/AnswerDotAI/fastllm/issues/51))


## 0.0.18

### New Features

- Add `mimo-v2.5-pro-ultraspeed` ([#50](https://github.com/AnswerDotAI/fastllm/issues/50))


## 0.0.17

### New Features

- Add claude-fable-5 model and stream stop-reason warnings ([#49](https://github.com/AnswerDotAI/fastllm/issues/49))
- fastspec `dict2obj -> obj2dict` handling ([#43](https://github.com/AnswerDotAI/fastllm/pull/43)), thanks to [@KeremTurgutlu](https://github.com/KeremTurgutlu)

### Bugs Squashed

- FullResponse/Safe tool results are truncated ([#47](https://github.com/AnswerDotAI/fastllm/issues/47))


## 0.0.16

### New Features

- fastspec `dict2obj -> obj2dict` handling ([#43](https://github.com/AnswerDotAI/fastllm/pull/43)), thanks to [@KeremTurgutlu](https://github.com/KeremTurgutlu)


## 0.0.15

### New Features

- Resize images returned from tool call (Anthropic) ([#41](https://github.com/AnswerDotAI/fastllm/issues/41))

### Bugs Squashed

- Anthropic `messages: text content blocks must be non-empty` ([#44](https://github.com/AnswerDotAI/fastllm/issues/44))
- Invalid 'input[95].`call_id`': string too long. Expected a string with maximum length 64, but got a string with length 116 instead. ([#45](https://github.com/AnswerDotAI/fastllm/issues/45))
- Fireworks generates `tool_call`.ids that result in Anthropic API calls failing ([#42](https://github.com/AnswerDotAI/fastllm/issues/42))
- Claude server `tool_result` yield bug ([#40](https://github.com/AnswerDotAI/fastllm/issues/40))
- fix truncation ([#39](https://github.com/AnswerDotAI/fastllm/pull/39)), thanks to [@RensDimmendaal](https://github.com/RensDimmendaal)


## 0.0.14

### New Features

- Anthropic `messages: text content blocks must be non-empty` ([#44](https://github.com/AnswerDotAI/fastllm/issues/44))
- Resize images returned from tool call (Anthropic) ([#41](https://github.com/AnswerDotAI/fastllm/issues/41))

### Bugs Squashed

- Invalid 'input[95].`call_id`': string too long. Expected a string with maximum length 64, but got a string with length 116 instead. ([#45](https://github.com/AnswerDotAI/fastllm/issues/45))
- Fireworks generates `tool_call`.ids that result in Anthropic API calls failing ([#42](https://github.com/AnswerDotAI/fastllm/issues/42))
- Claude server `tool_result` yield bug ([#40](https://github.com/AnswerDotAI/fastllm/issues/40))
- fix truncation ([#39](https://github.com/AnswerDotAI/fastllm/pull/39)), thanks to [@RensDimmendaal](https://github.com/RensDimmendaal)


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

# Release notes

<!-- do not remove -->

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

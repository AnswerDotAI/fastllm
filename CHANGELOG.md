# Release notes

<!-- do not remove -->

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

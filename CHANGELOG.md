# Release notes

<!-- do not remove -->

## 0.0.4

### Bugs Squashed

- Code fence tool fixes ([#6](https://github.com/AnswerDotAI/fastllm/issues/6))
  - `_split_msg_on_fences` fix which now correctly handles mixed msg content, e.g. thinking + text, tool use + text etc..
  - `_fence_re` new line start check

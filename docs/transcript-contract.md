# Transcript compactor: the model-visible contract

`TranscriptCompactor` (and `ContextRuntimeManager`) owns how removed content is marked and what a caller may ask about it.

* `is_placeholder(value)`: True for this manager's stand-ins (v2 `<<ARCHIVED-PLACEHOLDER ...>>` argument stub, legacy `[archived A...]` stubs). A harness must ask instead of hard-coding formats (agent-harness: `ContextAware`).
* `visibility(messages, tool_call_id) -> ResultView(state, archive_id, chars)`: `visible`, `clipped` (head and tail shown, middle archived by `admit`), `archived` (replaced by a stub), `unavailable`. For a clipped result that was archived later the id returned is the one holding the complete text, not the intermediate stub.
* `recovery_keep` (default 6): the output of `read_archive` / `search_context` is not archived again for the next six tool results, so evidence the model just asked for is still there while it works with it.
* Format and pure helpers live in `context_runtime/compression/contract.py` (no dependencies).

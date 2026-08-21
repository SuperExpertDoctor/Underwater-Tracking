# Task 3 Report: Real Embeddings and Structured Memory LLM

## Scope

- Added only the Task 3 memory primitives: real OpenAI-compatible embeddings,
  structured LLM memory reasoning, and bounded long-term retrieval.
- Added the three required memory test modules. No worker, service, API, or UI
  code was added.
- No local/hash/keyword embedding or memory-decision fallback exists in the
  implementation or tests.

## RED Evidence

Initial focused run after adding tests, before adding the memory package:

```text
$ PYTHONPATH=src pytest -q tests/memory/test_embeddings.py tests/memory/test_reasoner.py
E   ModuleNotFoundError: No module named 'underwater_tracking.memory'
2 errors in 0.22s
```

Strict typing exposed the first implementation integration issue:

```text
$ PYTHONPATH=src mypy src/underwater_tracking/memory
src/underwater_tracking/memory/reasoner.py:35: error: Type argument "object" of "StructuredLLM" must be a subtype of "BaseModel"  [type-var]
src/underwater_tracking/memory/reasoner.py:77: error: Argument 2 to "invoke_structured" of "StructuredLLM" has incompatible type ... [arg-type]
src/underwater_tracking/memory/retriever.py:150: error: Returning Any from function declared to return "float"  [no-any-return]
Found 8 errors in 2 files (checked 4 source files)
```

Grounding initially rejected valid non-Latin source text:

```text
$ PYTHONPATH=src pytest -q tests/memory/test_reasoner.py::test_extraction_accepts_grounded_non_latin_source_text
E   MemoryReasonerValidationError: summary introduces facts outside the supplied source text
1 failed in 0.16s
```

The source-payload bound regression first failed because the bounded payload
helper did not yet exist:

```text
$ PYTHONPATH=src pytest -q tests/memory/test_reasoner.py::test_source_payload_limits_current_source_text_to_memory_config
E   ImportError: cannot import name 'build_bounded_source_payload'
1 error in 0.16s
```

## GREEN Evidence

Focused Task 3 and persistence contract suite:

```text
$ PYTHONPATH=src pytest -q tests/memory/test_embeddings.py tests/memory/test_reasoner.py tests/persistence/test_memory_repository.py
23 passed in 3.21s
```

Credential-gated real provider suite:

```text
$ PYTHONPATH=src pytest -q -m real_llm tests/memory/test_real_llm_memory.py
1 skipped in 0.16s
```

The skip is correct: `UNDERWATER_TRACKING_API_KEY` is unavailable. The test
does not treat a degraded embedding result as a successful provider call.
When credentials are configured, it invokes the actual `/embeddings` endpoint,
`memory_filter`, `memory_extract`, and `short_term_compress`, validates the
responses, persists a real vector/version, retrieves it, and checks the audit
operations and final access count.

Static verification:

```text
$ ruff check src tests
All checks passed!

$ PYTHONPATH=src mypy src/underwater_tracking/memory
Success: no issues found in 4 source files

$ git diff --check
exit 0
```

Repository-wide `PYTHONPATH=src mypy src` remains red on pre-existing
unrelated modules in `domain`, `tracking`, `planning`, `runtime`, and existing
agent code (for example legacy `StrEnum` redefinitions and unparameterized
`ndarray` annotations). The new memory package has no mypy errors.

## Self-review

- `HTTPEmbeddingProvider` reads its configured environment variable at each
  call, posts only to `<base_url>/embeddings`, retries only typed transient
  provider errors, and records only hashes, model/version, latency, token
  count, and error category in `DecisionLedger`.
- Provider responses require exactly one finite non-empty vector, a matching
  model, and a bounded dimension. The configured vector version is carried to
  `MemoryVersion.embedding_version` and covered by repository persistence
  coverage.
- `MemoryReasoner` calls `StructuredLLM` for every filter/extract/compress
  operation; there is no keyword decision or static-summary path. Its prompts
  have explicit versions and contain bounded current source text, bounded
  short-term context, and only the repository's user-scoped candidate window.
- Filter update IDs must be supplied candidates for the same user; extraction
  references must be supplied source IDs; retained short-term messages must
  exactly match input; summaries are lexically grounded to supplied source
  material, preserve Unicode text, and respect configured length/token bounds.
- `MemoryRetriever` embeds the query through the real provider, asks the
  repository for a bounded active user-scoped candidate set with filters,
  rejects incompatible vector versions/dimensions, reranks with semantic
  similarity plus importance/recency/access frequency, hard-caps Top-K and
  context tokens, and updates access metrics only for final hits. Typed
  embedding failures return an empty degraded `MemoryContext`.

## Concerns

- The real-provider test is skipped in this environment because credentials
  are unavailable; no network success claim was made.
- Full-project mypy is currently blocked by unrelated existing errors. Scoped
  strict mypy for every new memory module is green.

## Review Fix: Aggregate Reasoner Context Budget

### RED Evidence

The focused budget regression suite initially failed against `2f3337a` because
the reasoner had no aggregate payload-budget accounting helper:

```text
$ PYTHONPATH=src pytest -q tests/memory/test_reasoner.py
E   ImportError: cannot import name 'estimate_memory_payload_tokens' from
    'underwater_tracking.memory.reasoner'
1 error in 0.22s
```

The review identified the underlying behavior: `filter` allocated a full
budget independently to source and short-term context before appending every
candidate, while compression sent the complete recent-message window and
validated only its summary.

### GREEN Evidence

The repaired reasoner uses one invocation-wide allocator for dynamic memory
context. It retains complete source texts, source IDs, short-term summaries,
messages, and candidate items only when each fits. Filter validation accepts
only candidate IDs that were actually sent; extraction validates against the
bounded source material; compression validates the returned summary and all
retained messages together.

```text
$ PYTHONPATH=src pytest -q tests/memory/test_embeddings.py tests/memory/test_reasoner.py tests/memory/test_real_llm_memory.py
..................s                                                      [100%]
18 passed, 1 skipped in 1.44s

$ ruff check src tests
All checks passed!

$ PYTHONPATH=src mypy src/underwater_tracking/memory
Success: no issues found in 4 source files

$ git diff --check
exit 0
```

### Review Notes

- Added focused regression coverage for aggregate filter context, complete
  source texts and reference IDs, complete candidate items, and short-term
  input/output budgets, including two 4,000-character messages under a
  one-token budget.
- The tests call `MemoryReasoner` through its structured LLM port and only
  record the payload boundary; no HTTP transport, provider fallback, worker,
  service, API, or UI code was added.

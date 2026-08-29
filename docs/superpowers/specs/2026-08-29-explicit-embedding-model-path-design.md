# Explicit Local Embedding Model Path

Date: 2026-08-29

## Context

The memory configuration currently names a SentenceTransformer model and a
cache directory, but the runtime still asks SentenceTransformers to resolve
the model by name. This makes the actual model source ambiguous. A partial
Hugging Face cache can also cause startup to fall through to a download attempt
or leave the runtime using the degraded memory retriever.

The configured local model is currently stored under `.cache` as the
SentenceTransformer snapshot for
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

## Goals

- Add an explicit `embedding_model_path` hyperparameter.
- Point the checked-in memory configuration at the complete local model
  snapshot, including its tokenizer, modules, configuration, and weight files.
- Make the real runtime load that directory locally before asynchronous work
  starts.
- Ensure memory writes and queries use vectors produced by that loaded model.
- Fail clearly when the configured model directory is missing or incomplete;
  do not silently download a different model or substitute synthetic vectors.

## Non-goals

- Do not change the ontology behavior or reintroduce ontology requests.
- Do not change the HTTP embedding provider contract.
- Do not copy or commit the 470 MB model weights into the repository.
- Do not redesign memory ranking, decay, or maintenance behavior.

## Design

### Configuration contract

Add `embedding_model_path` to `MemoryConfig`. It is required for an enabled
`sentence_transformers` provider and is not required by the HTTP provider.
Relative paths are repository-relative, not dependent on the process current
directory. The checked-in `configs/memory.yaml` will contain the current
snapshot directory explicitly:

`.cache/sentence-transformers/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/e8f8c211226b894fcb81acc59f3b34ba3efd5f42`

The path denotes a model directory rather than only `model.safetensors`,
because SentenceTransformers needs the directory metadata and tokenizer files
as well as the weights. `embedding_model` remains the logical model identifier
used in audit records and vector metadata.

`embedding_cache_dir` remains available for compatibility with existing HTTP
and legacy configuration, but it is not the source of truth for the explicit
local provider. `embedding_download_on_missing` must not enable a network
fallback when `embedding_model_path` is configured.

### Path resolution and validation

Resolve the configured path from the repository/configuration root and support
Git worktree execution by checking the repository common root as needed. The
resolver must select a directory only when it is a complete local
SentenceTransformer snapshot. Before constructing the provider, validate that
the directory exists and contains the required model metadata, module
definition, tokenizer/module files, and at least one supported weight file.

The resolved path is passed directly to `SentenceTransformer` with
`local_files_only=True`. The model-name lookup path and download retry path are
removed from this provider. The error must include the configured path and
identify whether the path is missing or incomplete.

### Runtime loading and data flow

`SentenceTransformerEmbeddingProvider.verify_ready()` remains the startup
readiness probe. It must load the model from `embedding_model_path` and encode
a probe string before the live worker is started. The provider caches that
loaded model for subsequent calls, and every memory insert/query continues to
call `model.encode` through the provider.

The strict live runtime (`llm_execution_required=True`) propagates a model
configuration/readiness error instead of constructing a degraded retriever.
Non-strict callers may retain the existing explicit degraded state for test
and diagnostic scenarios, but no successful live run may report a real memory
provider while using a substitute vector implementation.

The main and memory-worker provider instances both use the same resolved model
path. Their ledger records retain the logical model name and vector version.

### Error handling

- Missing or incomplete explicit path: `LLMConfigError` with the path.
- Missing `sentence-transformers` dependency: existing typed configuration
  error, with the explicit path included when available.
- Local model load or encode failure: existing typed provider error; no
  download fallback.
- HTTP provider: unchanged, including its API-key validation and retry rules.

## Tests and acceptance

1. Configuration tests verify that the YAML exposes the exact explicit path and
   that local-provider configuration without a path is rejected.
2. Provider unit tests verify that the explicit directory is passed to
   SentenceTransformer, `local_files_only=True` is used, and no download/name
   fallback is attempted.
3. A local provider readiness test exercises the available cached snapshot and
   asserts a non-empty, finite vector whose dimension matches the model output.
4. Memory retrieval tests verify that both stored text and the query use the
   configured provider, and that the agent loop exposes `MemoryRetriever` in
   the configured local path case.
5. Strict runtime tests verify that an invalid explicit path fails startup and
   cannot silently become `DegradedMemoryRetriever`.
6. Run the focused memory/config/runtime tests, then the existing related test
   set and lint checks. A short real `main.py` run must show the local provider
   readiness path succeeding before execution begins.

## Review checklist

- The YAML path is explicit, repository-relative, and points to a complete
  snapshot available in the current `.cache`.
- The provider receives a directory path, not merely a model name or cache
  directory.
- Strict live execution cannot proceed with a missing model or degraded
  substitute.
- No ontology code or request path is added.

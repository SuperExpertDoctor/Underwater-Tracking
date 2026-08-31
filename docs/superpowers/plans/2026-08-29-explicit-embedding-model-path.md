# Explicit Local Embedding Model Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Make the configured live memory pipeline load the exact local Hugging Face SentenceTransformer snapshot named by embedding_model_path, use its vectors for memory retrieval, and fail explicitly instead of downloading or silently degrading when that snapshot is unavailable.

**Architecture:** Add a strict local-model path field to MemoryConfig, store the repository-relative snapshot directory in configs/memory.yaml, and resolve it against the repository/worktree ancestors so the complete .cache snapshot is selected even when a worktree cache is partial. The SentenceTransformer provider validates and loads that directory with local_files_only=True; strict live AgentLoop construction propagates readiness failures while diagnostic callers retain the existing explicit degraded contract.

**Tech Stack:** Python 3.11, Pydantic 2, SentenceTransformers, Hugging Face local snapshot files, pytest, pytest-mock/monkeypatch, Ruff, mypy, SQLite memory repositories, main.py live runtime.

## Global Constraints

- embedding_model_path is the source of truth for the local SentenceTransformer provider; embedding_model remains the logical audit/model identifier.
- The configured local path denotes a complete SentenceTransformer directory, not only model.safetensors; metadata, module definitions, tokenizer files, and weights must be present.
- Local provider loading always uses local_files_only=True and never retries with local_files_only=False.
- A strict live run with llm_execution_required=True must not continue with DegradedMemoryRetriever after local embedding readiness fails.
- Relative model paths are repository-relative and must not depend on the process current directory; Git worktree execution must be supported.
- Do not commit or copy the model weights; use the existing .cache snapshot.
- Do not modify ontology behavior or add any ontology request path.
- Do not change the HTTP embedding provider contract, memory ranking, decay, or maintenance behavior.
- Follow TDD for every production behavior change: write one focused failing test, run it and inspect the expected failure, implement the minimum change, then run the focused test again.
- Preserve all existing live visualization and execution-stability fixes already present on feature/fix-live-plan-advance-freeze.

---

### Task 1: Add the Explicit Embedding Path Configuration Contract

**Files:**
- Modify: src/underwater_tracking/config/models.py:383-447
- Modify: configs/memory.yaml
- Modify: tests/config/test_models.py
- Modify: tests/memory/test_embeddings.py

**Interfaces:**
- Produces MemoryConfig.embedding_model_path: str | None.
- For enabled sentence_transformers memory, the field is a non-empty string and is validated alongside embedding_model.
- For the HTTP provider and MemoryConfig.degraded(), the field remains optional.

- [x] Step 1: Write the failing configuration tests

Add the following assertions to tests/config/test_models.py, extending the existing local-provider test:

~~~python
EXPECTED_MODEL_PATH = (
    ".cache/sentence-transformers/"
    "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/"
    "snapshots/e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
)


def test_local_sentence_transformer_config_requires_explicit_model_path() -> None:
    config = MemoryConfig(
        embedding_provider="sentence_transformers",
        embedding_model="local-model",
        embedding_model_path=EXPECTED_MODEL_PATH,
    )
    assert config.embedding_model_path == EXPECTED_MODEL_PATH

    with pytest.raises(ValidationError, match="embedding_model_path"):
        MemoryConfig(
            embedding_provider="sentence_transformers",
            embedding_model="local-model",
        )


def test_memory_yaml_names_the_cached_sentence_transformer_snapshot() -> None:
    config = load_app_config(CONFIG_PATH)
    assert config.memory is not None
    assert config.memory.embedding_model_path == EXPECTED_MODEL_PATH
    assert config.memory.embedding_download_on_missing is False
~~~

Update the existing local-provider construction in the same test file to pass embedding_model_path=EXPECTED_MODEL_PATH and change its download assertion to False.

Update _local_config() in tests/memory/test_embeddings.py with:

~~~python
"embedding_model_path": ".cache/test-sentence-transformers/local-model",
"embedding_download_on_missing": False,
~~~

- [x] Step 2: Run the configuration tests and confirm the new contract fails

Run:

~~~powershell
conda run -n underwater-tracking python -m pytest tests/config/test_models.py::test_local_sentence_transformer_config_requires_explicit_model_path tests/config/test_models.py::test_memory_yaml_names_the_cached_sentence_transformer_snapshot -q
~~~

Expected: FAIL because MemoryConfig does not yet expose embedding_model_path and configs/memory.yaml does not yet contain the field.

- [x] Step 3: Add the field and validation to MemoryConfig

Add the field next to embedding_model in src/underwater_tracking/config/models.py:

~~~python
embedding_model: _LLMNonEmptyString | None = None
embedding_model_path: _LLMNonEmptyString | None = None
~~~

Extend validate_memory_limits() with this branch before the HTTP-specific validation:

~~~python
if (
    self.enabled
    and self.embedding_provider == "sentence_transformers"
    and self.embedding_model_path is None
):
    raise ValueError(
        "enabled sentence_transformers memory config requires embedding_model_path"
    )
~~~

Keep the field optional for HTTP configurations and disabled memory so existing HTTP contract tests and MemoryConfig.degraded() remain valid.

- [x] Step 4: Set the exact snapshot path in the hyperparameter YAML

In configs/memory.yaml, add the following line immediately after embedding_model and disable the obsolete network fallback:

~~~yaml
embedding_model_path: ".cache/sentence-transformers/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
embedding_download_on_missing: false
~~~

Update the file comment so it says the provider loads the explicitly configured local snapshot and does not download missing models.

- [x] Step 5: Run the focused configuration tests and the existing model tests

Run:

~~~powershell
conda run -n underwater-tracking python -m pytest tests/config/test_models.py tests/memory/test_embeddings.py -q
~~~

Expected: the configuration and embedding tests may still fail only at provider-loading expectations because Task 2 has not changed provider behavior; the new field validation and YAML assertions must pass. Record the exact remaining failures before starting Task 2.

- [x] Step 6: Commit the configuration contract

~~~powershell
git add src/underwater_tracking/config/models.py configs/memory.yaml tests/config/test_models.py tests/memory/test_embeddings.py
git commit -m "feat: configure explicit embedding model path"
~~~

---

### Task 2: Resolve, Validate, and Directly Load the Local Snapshot

**Files:**
- Modify: src/underwater_tracking/memory/embeddings.py:210-410
- Modify: tests/memory/test_embeddings.py:37-190

**Interfaces:**
- SentenceTransformerEmbeddingProvider.__init__() stores the resolved explicit directory as _model_path: Path.
- Add private helpers with these signatures:

~~~python
def _resolve_sentence_transformer_model_path(configured_path: str) -> Path: ...


def _validate_sentence_transformer_model_path(model_path: Path) -> None: ...
~~~

- _load_compatible_sentence_transformer() receives the resolved directory and passes it as the SentenceTransformer model source.
- _load_legacy_modules() receives the same directory and never reconstructs a model source from embedding_model.

- [x] Step 1: Add a complete temporary snapshot helper and failing provider tests

Add this test helper to tests/memory/test_embeddings.py:

~~~python
def _complete_model_path(tmp_path: Path) -> Path:
    model_path = tmp_path / "snapshot"
    (model_path / "1_Pooling").mkdir(parents=True)
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    (model_path / "modules.json").write_text("[]", encoding="utf-8")
    (model_path / "1_Pooling" / "config.json").write_text("{}", encoding="utf-8")
    (model_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_path / "model.safetensors").write_bytes(b"test weights")
    return model_path
~~~

Replace the current local provider constructor assertions with a test that proves the explicit path is used:

~~~python
def test_sentence_transformer_provider_uses_explicit_model_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = _complete_model_path(tmp_path)
    calls: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(self, model_source: str, **kwargs: object) -> None:
            calls["model_source"] = model_source
            calls["constructor"] = kwargs

        def encode(self, text: str, **kwargs: object) -> list[float]:
            del text, kwargs
            return [0.25, -0.5, 0.75]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    provider = SentenceTransformerEmbeddingProvider(
        _local_config(embedding_model_path=str(model_path))
    )

    result = provider.embed("explicit local snapshot")

    assert result.vector == (0.25, -0.5, 0.75)
    assert calls["model_source"] == str(model_path.resolve())
    constructor = calls["constructor"]
    assert isinstance(constructor, dict)
    assert constructor["local_files_only"] is True
    assert constructor["trust_remote_code"] is False
    assert constructor["device"] == "cpu"
    assert "cache_folder" not in constructor
~~~

Add a missing/incomplete path test whose fake constructor fails if invoked:

~~~python
def test_sentence_transformer_provider_rejects_incomplete_path_before_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    constructor_called = False

    class UnexpectedSentenceTransformer:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            nonlocal constructor_called
            constructor_called = True

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=UnexpectedSentenceTransformer),
    )
    missing_path = tmp_path / "partial-snapshot"
    missing_path.mkdir()

    with pytest.raises(LLMConfigError, match="embedding_model_path"):
        SentenceTransformerEmbeddingProvider(
            _local_config(embedding_model_path=str(missing_path))
        )
    assert constructor_called is False
~~~

Rename the existing download test to test_sentence_transformer_provider_never_downloads_an_explicit_path and make its fake constructor raise OSError. Assert provider construction/load raises LLMConfigError, records only local_files_only=True if the constructor is reached, and never calls with local_files_only=False.

- [x] Step 2: Run the provider tests and confirm they fail for the old model-name behavior

Run:

~~~powershell
conda run -n underwater-tracking python -m pytest tests/memory/test_embeddings.py::test_sentence_transformer_provider_uses_explicit_model_directory tests/memory/test_embeddings.py::test_sentence_transformer_provider_rejects_incomplete_path_before_load tests/memory/test_embeddings.py::test_sentence_transformer_provider_never_downloads_an_explicit_path -q
~~~

Expected: FAIL because the provider currently uses embedding_model, has no explicit path validation, and retries with local_files_only=False.

- [x] Step 3: Implement deterministic repository/worktree path resolution

In embeddings.py, import Path and add these constants/helpers near the provider definitions:

~~~python
_SUPPORTED_LOCAL_WEIGHT_NAMES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.bin",
)
_SUPPORTED_LOCAL_TOKENIZER_NAMES = (
    "tokenizer.json",
    "sentencepiece.bpe.model",
    "spiece.model",
    "vocab.txt",
)


def _candidate_repository_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    for anchor in (Path.cwd(), Path(__file__).resolve().parents[3]):
        resolved = anchor.resolve()
        roots.extend((resolved, *resolved.parents))
    return tuple(dict.fromkeys(roots))
~~~

Implement _resolve_sentence_transformer_model_path() so absolute paths are checked directly. For relative paths, test each unique root / configured_path candidate from _candidate_repository_roots() and return the first candidate accepted by _validate_sentence_transformer_model_path(). If no candidate is complete, raise LLMConfigError containing the original configured path and the first resolved candidate. This lets a worktree skip its partial snapshot and select the complete repository .cache snapshot without depending on the current working directory alone.

Implement _validate_sentence_transformer_model_path() with these exact checks:

~~~python
if not model_path.is_dir():
    raise LLMConfigError(f"embedding_model_path is missing: {model_path}")
required = (model_path / "config.json", model_path / "modules.json")
if any(not path.is_file() for path in required):
    raise LLMConfigError(f"embedding_model_path is incomplete: {model_path}")
if not any((model_path / name).is_file() for name in _SUPPORTED_LOCAL_WEIGHT_NAMES):
    raise LLMConfigError(f"embedding_model_path has no supported weights: {model_path}")
has_tokenizer = any(
    (model_path / name).is_file() for name in _SUPPORTED_LOCAL_TOKENIZER_NAMES
)
has_module_config = any(model_path.glob("*/config.json"))
if not has_tokenizer and not has_module_config:
    raise LLMConfigError(f"embedding_model_path has no tokenizer/modules: {model_path}")
~~~

- [x] Step 4: Change provider loading to use only the resolved directory

In SentenceTransformerEmbeddingProvider.__init__(), require config.embedding_model_path, assign:

~~~python
self._model_name = config.embedding_model
self._model_path = _resolve_sentence_transformer_model_path(
    config.embedding_model_path
)
self._vector_version = config.embedding_vector_version
self._device = config.embedding_device
self._normalize = config.embedding_normalize
~~~

Remove _download_on_missing from the local loading decision. In _load_model(), call the compatible loader with the resolved path and keep only the local TypeError compatibility branch; convert all load failures to LLMConfigError that includes self._model_path. Delete the OSError branch that retries with local_files_only=False.

Make _load_compatible_sentence_transformer() construct the model as follows:

~~~python
model = sentence_transformer(
    str(self._model_path),
    device=self._device,
    local_files_only=True,
    trust_remote_code=False,
)
~~~

When the legacy module path is used, pass str(self._model_path) to Transformer and keep local_files_only=True in model_kwargs. Preserve the existing tokenizer compatibility detection and vector validation.

- [x] Step 5: Run focused provider tests and static checks

Run:

~~~powershell
conda run -n underwater-tracking python -m pytest tests/memory/test_embeddings.py -q
conda run -n underwater-tracking python -m ruff check src/underwater_tracking/memory/embeddings.py tests/memory/test_embeddings.py
~~~

Expected: all provider tests pass, including the no-download assertion; Ruff reports no new errors in these files.

- [x] Step 6: Commit direct local snapshot loading

~~~powershell
git add src/underwater_tracking/memory/embeddings.py tests/memory/test_embeddings.py
git commit -m "fix: load embeddings from explicit local snapshot"
~~~

---

### Task 3: Make Strict Live Runtime Propagate Embedding Readiness Failures

**Files:**
- Modify: src/underwater_tracking/cli.py:1791-1960
- Modify: tests/agent/test_runtime_master_slave_adversary.py
- Modify: tests/cli/test_cli.py

**Interfaces:**
- _build_memory_embedding_provider() continues to call verify_ready() before returning a local provider.
- _AgentLoop._build_memory_service() returns the existing degraded service only for non-strict callers; strict callers re-raise a typed readiness/configuration error.

- [x] Step 1: Write the failing strict-runtime regression test

Add this test to tests/agent/test_runtime_master_slave_adversary.py:

~~~python
def test_strict_agent_loop_does_not_degrade_when_embedding_snapshot_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UNDERWATER_TRACKING_API_KEY", "test-chat-key")
    config = load_app_config(CONFIG_PATH)
    assert config.memory is not None
    invalid_config = config.model_copy(
        update={
            "memory": config.memory.model_copy(
                update={"embedding_model_path": str(tmp_path / "missing-snapshot")}
            )
        }
    )
    clients = {role: RecordingRoleLLM() for role in ("master", "slave", "adversary")}

    with pytest.raises(LLMConfigError, match="embedding_model_path"):
        _AgentLoop(
            invalid_config,
            database_path=tmp_path / "strict-agent.db",
            llm=clients,
            run_id="strict-invalid-embedding",
            steps=1,
            seed=7,
            llm_execution_required=True,
        )
~~~

Add a CLI-level readiness assertion that _build_memory_embedding_provider() still invokes verify_ready() exactly once; keep the existing Provider test and assert the new config field is passed unchanged.

- [x] Step 2: Run the strict-runtime test and confirm it fails

Run:

~~~powershell
conda run -n underwater-tracking python -m pytest tests/agent/test_runtime_master_slave_adversary.py::test_strict_agent_loop_does_not_degrade_when_embedding_snapshot_is_invalid -q
~~~

Expected: FAIL because _build_memory_service() currently catches the provider error and returns DegradedMemoryRetriever even when llm_execution_required=True.

- [x] Step 3: Re-raise provider readiness errors after resource cleanup in strict mode

At the end of the existing except Exception as exc cleanup block in _build_memory_service(), before constructing DegradedMemoryRetriever, add:

~~~python
if self._llm_execution_required:
    if isinstance(exc, LLMError):
        raise exc
    raise LLMConfigError("strict live memory provider initialization failed") from exc
~~~

Keep the cleanup of primary and worker providers/repositories before this branch. Leave the existing degraded return untouched for non-strict callers and preserve the earlier credential-gated behavior for intentionally paused diagnostic loops.

- [x] Step 4: Run strict and existing memory-runtime tests

Run:

~~~powershell
conda run -n underwater-tracking python -m pytest tests/agent/test_runtime_master_slave_adversary.py tests/cli/test_cli.py -q
~~~

Expected: all tests pass; the invalid explicit path raises in strict mode and non-strict unavailable-provider tests still report an explicit degraded reason.

- [x] Step 5: Run lint on the runtime files

~~~powershell
conda run -n underwater-tracking python -m ruff check src/underwater_tracking/cli.py tests/agent/test_runtime_master_slave_adversary.py tests/cli/test_cli.py
~~~

Expected: Ruff reports no new errors in the runtime files.

- [x] Step 6: Commit strict runtime propagation

~~~powershell
git add src/underwater_tracking/cli.py tests/agent/test_runtime_master_slave_adversary.py tests/cli/test_cli.py
git commit -m "fix: fail strict runtime on embedding readiness errors"
~~~

---

### Task 4: Prove the Cached Weights Drive Memory Embedding and Retrieval

**Files:**
- Modify: tests/memory/test_real_llm_memory.py:20-55
- Modify: tests/memory/test_embeddings.py
- Modify: tests/agent/test_runtime_master_slave_adversary.py

**Interfaces:**
- Real-provider readiness checks use config.memory.embedding_model_path, not embedding_model lookup.
- The real memory lifecycle continues to create an embedding with SentenceTransformerEmbeddingProvider, persist it, and retrieve it through MemoryRetriever.

- [x] Step 1: Write the failing cached-snapshot acceptance test

Add a cache-gated test to tests/memory/test_embeddings.py:

~~~python
@pytest.mark.skipif(
    not Path(
        ".cache/sentence-transformers/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/"
        "snapshots/e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
    ).is_dir(),
    reason="the configured local SentenceTransformer snapshot is not available",
)
def test_cached_snapshot_produces_a_real_semantic_vector() -> None:
    config = load_app_config("configs/scenario/default.yaml")
    assert config.memory is not None
    provider = SentenceTransformerEmbeddingProvider(config.memory)
    try:
        result = provider.embed("underwater target tracking evidence")
    finally:
        provider.close()

    assert result.model == config.memory.embedding_model
    assert result.dimensions > 100
    assert all(math.isfinite(value) for value in result.vector)
    assert any(value != 0.0 for value in result.vector)
~~~

Import Path, math, and load_app_config in the test module. Add a retrieval assertion using the existing MemoryRetriever setup that persists the vector returned by this provider and retrieves the matching memory by query, proving both write and query paths use the loaded model.

- [x] Step 2: Run the cache-gated test before implementation and inspect its failure

Run:

~~~powershell
conda run -n underwater-tracking python -m pytest tests/memory/test_embeddings.py::test_cached_snapshot_produces_a_real_semantic_vector -q
~~~

Expected: FAIL or skip under the old provider because it resolves the model by name from the partial worktree cache; after Task 2 it must run against the complete ancestor .cache snapshot and pass.

- [x] Step 3: Update real-provider readiness checks to use the explicit path

In _has_real_memory_credentials() in tests/memory/test_real_llm_memory.py, require config.memory.embedding_model_path, instantiate SentenceTransformer with str(Path(config.memory.embedding_model_path).resolve()) only after resolving the configured path through the provider, and pass local_files_only=True. The check must not instantiate by embedding_model alone.

Keep the existing credential gate for the remote memory reasoner. In the real lifecycle test, assert that the embedding dimensions are greater than 100 and that retrieval returns memory-real-1; this is the evidence that the loaded model output participates in persistence and ranking.

- [x] Step 4: Assert the two AgentLoop providers use the same explicit path

Extend test_agent_loop_uses_real_memory_provider_chain_when_configured() with:

~~~python
assert loop._memory_embedding_provider is not None
assert loop._memory_worker_embedding_provider is not None
assert loop._memory_embedding_provider._model_path == loop._memory_worker_embedding_provider._model_path  # noqa: SLF001
~~~

The existing assertions that both providers are SentenceTransformerEmbeddingProvider, have distinct ledgers, and start the worker must remain.

- [x] Step 5: Run memory integration tests

Run:

~~~powershell
conda run -n underwater-tracking python -m pytest tests/memory/test_embeddings.py tests/memory/test_real_llm_memory.py tests/agent/test_runtime_master_slave_adversary.py -q
~~~

Expected: local cache tests pass when the configured snapshot is present; credential-gated remote reasoner tests may be skipped only for their documented missing credential reason; no test may report a successful real provider with a degraded retriever.

- [x] Step 6: Commit the cached-weight retrieval evidence

~~~powershell
git add tests/memory/test_embeddings.py tests/memory/test_real_llm_memory.py tests/agent/test_runtime_master_slave_adversary.py
git commit -m "test: prove cached embeddings drive memory retrieval"
~~~

---

### Task 5: Run Full Regression and Real main.py Acceptance

**Files:**
- No planned source edits. If a verification failure identifies a regression, return to the task that owns the failing contract and add its regression test before changing production code.

**Interfaces:**
- Uses the final configs/memory.yaml path and the strict live startup gate.
- Preserves the previously completed live visualization execution-stability acceptance surface.

- [x] Step 1: Check repository state and diff hygiene

Run:

~~~powershell
git status --short --branch
git diff --check
git log --oneline --decorate -8
~~~

Expected: only intended commits are on feature/fix-live-plan-advance-freeze; no model weights or generated output are staged.

- [x] Step 2: Run the focused combined regression set

Run:

~~~powershell
conda run -n underwater-tracking python -m pytest tests/config/test_models.py tests/memory/test_embeddings.py tests/memory/test_real_llm_memory.py tests/agent/test_runtime_master_slave_adversary.py tests/cli/test_cli.py tests/agent/test_background_cycle.py tests/runtime/test_mission_epoch_commit.py tests/agent/test_epoch_commit_graph.py tests/planning/test_region_baseline.py -q
~~~

Expected: zero failures. Credential-gated tests may be skipped only with their existing documented skip reason.

- [x] Step 3: Run the prior live-visualization regression set

Run:

~~~powershell
conda run -n underwater-tracking python -m pytest tests/integration/test_live_tracking_health_pipeline.py tests/integration/test_uuv_only_runtime_entrypoints.py tests/integration/test_uuv_only_physical_execution.py tests/integration/test_uuv_only_production_acceptance.py tests/agent/test_runtime_master_slave_adversary.py tests/cli/test_cli.py -q
~~~

Expected: the earlier execution snapshot, bootstrap, regional geometry, and local sonar boundary regressions remain green.

- [x] Step 4: Run lint, type checks, and the full pytest suite

Run:

~~~powershell
conda run -n underwater-tracking python -m ruff check src tests
conda run -n underwater-tracking python -m mypy src
conda run -n underwater-tracking python -m pytest -q
~~~

Verification record (2026-08-29): the changed-file Ruff checks passed. The repository-wide Ruff baseline reports 162 existing diagnostics, and mypy reports 286 existing diagnostics across 54 files. After updating legacy fake memory fixtures to declare the compatibility HTTP provider, the full pytest suite completed with `2129 passed, 71 skipped, 1 failed`; the only failure is `tests/integration/test_uuv_initialization_local_perception.py::test_real_uuv_default_timeline_local_perception_and_periodic_memory`, which independently reproduces on the unmodified `branch1` baseline and is unrelated to this change.

- [x] Step 5: Run a real short main.py process with the explicit snapshot

From the worktree root, run:

~~~powershell
conda run -n underwater-tracking python main.py --steps 120 --port 8002
~~~

Use the generated run manifest and SQLite ledger to locate the newest run:

~~~powershell
Get-ChildItem outputs -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
~~~

Verification record (2026-08-29): `main.py --steps 120 --port 8002` created `outputs/main-acceptance-20260829/run-a4ac3a73e3e94a21ba60a7f7e696118f`, but strict startup stopped at the real LLM provider attestation because `UNDERWATER_TRACKING_API_KEY` is absent. The manifest records `status=failed`, `llm_call_count=1`, and `provider_attestation_probe:master` with `error_category=config`; no embedding call was attempted. Offline provider and cached-snapshot tests passed, but live process acceptance remains credential-gated in this environment.

- [x] Step 6: Verify no ontology request path was reintroduced

Run:

~~~powershell
rg -n "ontology|knowledge_client|StrategyGenerationNode" src tests configs
~~~

Confirm that the existing ontology-removal state is unchanged and that no new runtime call is present in the final diff.

Post-merge verification (2026-08-29): the ignored `src/underwater_tracking/ui/dist` bundle in the `branch1` worktree was rebuilt from the ontology-free UI source; `tests/verification/test_ontology_removal.py` then passed with `2 passed`, and the rebuilt bundle contains no removed ontology symbol.

- [x] Step 7: Capture final branch evidence before integration

Run:

~~~powershell
git status --short --branch
git diff branch1...HEAD --stat
git log --oneline branch1..HEAD
~~~

Expected: the branch is clean, the diff contains only the explicit embedding-path implementation plus the already reviewed execution-stability fixes, and no cache weight file is included.

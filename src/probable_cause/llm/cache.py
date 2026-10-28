"""Content-addressed disk cache for completions, plus a hard budget on live calls.

The key is a SHA-256 of everything that determines the reply: backend, model, system
prompt, messages and schema. Each entry is one small JSON file, so the cache can be
committed alongside the benchmark and evaluation re-runs need no LLM at all.

Every live call (a cache miss) is appended to a ledger file. The ledger is shared by all
processes, so the budget holds across separate CLI invocations; past the cap, calls are
refused before anything is sent.
"""

import hashlib
import json
import sys
import threading
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from probable_cause.llm.base import Backend, BackendError, Completion, CompletionRequest


def request_key(backend_name: str, model: str, request: CompletionRequest) -> str:
    payload = {
        "backend": backend_name,
        "model": model,
        "system": request.system,
        "messages": [asdict(m) for m in request.messages],
        "schema": request.schema,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class CallBudget:
    """A persistent count of live LLM calls with a hard cap."""

    def __init__(self, ledger: Path, cap: int) -> None:
        self.ledger = ledger
        self.cap = cap
        self._lock = threading.Lock()

    def used(self) -> int:
        if not self.ledger.exists():
            return 0
        with self.ledger.open(encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())

    def spend(self, tag: str, model: str) -> int:
        """Record one call, or raise if the cap is reached. Returns the running count."""
        with self._lock:
            used = self.used()
            if used >= self.cap:
                raise BackendError(f"{tag}: LLM call budget exhausted ({used}/{self.cap})")
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            entry = {"n": used + 1, "tag": tag, "model": model,
                     "at": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip
            with self.ledger.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(entry) + "\n")
        print(f"[llm] live call {used + 1}/{self.cap}: {tag} ({model})", file=sys.stderr)
        return used + 1


class CachingBackend:
    """Wrap a backend; serve repeats from ``directory``. ``offline`` forbids live calls.

    ``model_aliases`` are other names the same model has been cached under (e.g. the CLI
    alias "sonnet" for "claude-sonnet-5-5"), so renaming the model does not re-ask.
    """

    def __init__(
        self,
        inner: Backend,
        directory: Path,
        *,
        offline: bool = False,
        budget: CallBudget | None = None,
        model_aliases: tuple[str, ...] = (),
    ) -> None:
        self._inner = inner
        self._dir = directory
        self._offline = offline
        self._budget = budget
        self._aliases = model_aliases
        self.name = inner.name
        self.model = inner.model
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self._dir / key[:2] / f"{key}.json"

    def complete(self, request: CompletionRequest) -> Completion:
        key = request_key(self._inner.name, self._inner.model, request)
        for model in (self._inner.model, *self._aliases):
            path = self._path(request_key(self._inner.name, model, request))
            if path.exists():
                self.hits += 1
                data = json.loads(path.read_text(encoding="utf-8"))
                return Completion(**data["completion"])
        if self._offline:
            raise BackendError(f"{request.tag}: not in the LLM cache and offline mode is on")
        if self._budget is not None:
            self._budget.spend(request.tag, self._inner.model)
        completion = self._inner.complete(request)
        self.misses += 1
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"tag": request.tag, "completion": asdict(completion)}
        path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return completion

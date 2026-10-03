"""Load AML's published pipelines byte-for-byte and run their own answer/evaluate.

Each published ``answer``/``evaluate`` opens its output file inside
``async with httpx.AsyncClient(...) as client, output.open(...) as handle``; a plain
file has no async context methods, so the published code raises before any request.
``bind`` gives the module a ``Path`` whose ``open`` returns a file that also supports
``async with``. No pipeline line is changed; prompts, parsing and scoring run as published.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import pathlib
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

AML_COMMIT = "1b8142bfe0f20f1c5218d6b554aa0012de34e504"
PIPELINES = {
    "longmemeval-s": "data/longmemeval-s/pipeline.py",
    "locomo-refined": "data/locomo-refined/pipeline.py",
    "beam": "data/beam/pipeline.py",
}
SHIM = "Path.open returns a file that also supports async with; pipeline source unchanged."


def load(name: str, root: str | Path) -> Any:
    root = Path(root)
    relative = PIPELINES[name]
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != AML_COMMIT:
        raise ValueError(f"AML source revision differs: {commit}")
    published = subprocess.check_output(["git", "-C", str(root), "show", f"{AML_COMMIT}:{relative}"])
    if (root / relative).read_bytes() != published:
        raise ValueError(f"AML pipeline {relative} has local modifications")
    spec = importlib.util.spec_from_file_location(f"aml_{name.replace('-', '_')}", root / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.AML_PIPELINE_SHA256 = hashlib.sha256(published).hexdigest()
    return module


class _AsyncFile:
    def __init__(self, handle):
        self._handle = handle

    def __getattr__(self, name):
        return getattr(self._handle, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._handle.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self._handle.close()


class _AsyncPath(type(pathlib.Path())):
    def open(self, *args, **kwargs):
        return _AsyncFile(super().open(*args, **kwargs))


def bind(module: Any, *, answer: tuple[str, str], judge: tuple[str, str], api_key: str = "local",
         transport: httpx.AsyncBaseTransport | None = None, request_timeout_s: float | None = None) -> None:
    """Point the module's api_config globals at the given endpoints; install the shims.

    ``request_timeout_s`` replaces the pipelines' hard-coded 120 s HTTP timeout, which a
    local model cannot meet on 25K+ token prompts; it changes how long we wait, never
    what is sent or how it is scored. It is recorded in every run report.
    """
    module.ANSWER_API_BASE, module.ANSWER_MODEL = answer[0].rstrip("/"), answer[1]
    module.JUDGE_API_BASE, module.JUDGE_MODEL = judge[0].rstrip("/"), judge[1]
    module.ANSWER_API_KEY = module.JUDGE_API_KEY = api_key
    module.Path = _AsyncPath
    module.request_timeout_s = request_timeout_s
    if transport is not None or request_timeout_s is not None:
        real = httpx.AsyncClient

        def client(**kwargs):
            if transport is not None:
                kwargs["transport"] = transport
            if request_timeout_s is not None:
                kwargs["timeout"] = httpx.Timeout(request_timeout_s)
            return real(**kwargs)

        module.httpx = SimpleNamespace(AsyncClient=client, TimeoutException=httpx.TimeoutException,
                                       TransportError=httpx.TransportError)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def write_rows(rows: list[dict[str, Any]], directory: str | Path) -> Path:
    """Write the pipeline input; a second call must carry identical rows (resume safety)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "input.jsonl"
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if source.exists() and source.read_text() != text:
        raise ValueError(f"{source} differs from this run's inputs")
    source.write_text(text)
    return source


def run(module: Any, name: str, rows: list[dict[str, Any]], directory: str | Path) -> dict[str, Any]:
    """Run the published answer then evaluate on ``rows``; resumable via their own checkpoints."""
    directory = Path(directory)
    source = write_rows(rows, directory)
    answers, scored = directory / "answers.jsonl", directory / "evaluation.jsonl"
    for argv in (["answer", "--input", str(source), "--output", str(answers)],
                 ["evaluate", "--input", str(source), "--answers", str(answers), "--output", str(scored)]):
        args = module.parser().parse_args(argv)
        asyncio.run(args.run(args))
    items = _jsonl(scored)
    if name == "beam":
        values = [item["llm_judge_score"] for item in items]
    else:
        values = [1.0 if item["is_correct"] else 0.0 for item in items]
    timeout = getattr(module, "request_timeout_s", None)
    shim = SHIM + (f" HTTP timeout raised from the published 120 s to {timeout:g} s." if timeout else "")
    return {"pipeline": PIPELINES[name], "aml_commit": AML_COMMIT, "pipeline_sha256": module.AML_PIPELINE_SHA256,
            "shim": shim, "request_timeout_s": timeout, "n": len(items), "score": 100.0 * sum(values) / len(values) if values else None,
            "items": items}

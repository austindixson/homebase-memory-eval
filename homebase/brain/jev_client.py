"""TypeSafe Jev System One client (hosted). Same wire shape as former local CLM."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Protocol

import httpx

DEFAULT_JEV_BASE = "https://api.typesafe.ai"
DEFAULT_JEV_MODEL = "jev-latest"


class SystemOneClient(Protocol):
    def system_one(self, state: str, questions: dict[str, Any]) -> dict[str, Any]: ...


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def resolve_jev_api_key() -> str:
    """TYPESAFE_API_KEY / JEV_API_KEY from env, then repo/.env, then ~/.homebase/secrets.env."""
    for env_name in ("TYPESAFE_API_KEY", "JEV_API_KEY"):
        v = (os.environ.get(env_name) or "").strip()
        if v:
            return v
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / ".env",  # repo root when installed as homebase/brain/
        Path.home() / ".homebase" / "secrets.env",
        Path.home() / "Desktop" / "CLM" / ".env",
        Path.home() / "Desktop" / "CLM-portable-20260924" / ".env",
    ]
    for p in candidates:
        _load_dotenv(p)
        for env_name in ("TYPESAFE_API_KEY", "JEV_API_KEY"):
            v = (os.environ.get(env_name) or "").strip()
            if v:
                return v
    return ""


class JevClient:
    """POST https://api.typesafe.ai/v1/systemone — Jev keep/drop / router / mem."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 60.0,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("TYPESAFE_BASE_URL")
            or os.environ.get("JEV_BASE_URL")
            or DEFAULT_JEV_BASE
        ).rstrip("/")
        self.timeout = timeout
        self.api_key = (api_key if api_key is not None else resolve_jev_api_key()).strip()
        self.model = (
            model
            or os.environ.get("TYPESAFE_MODEL")
            or os.environ.get("JEV_MODEL")
            or DEFAULT_JEV_MODEL
        )

    def configured(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        h = {"content-type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def system_one(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("TYPESAFE_API_KEY (or JEV_API_KEY) is not set")
        payload = {"model": self.model, "state": state, "questions": questions}
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    r = client.post(
                        f"{self.base_url}/v1/systemone",
                        headers=self._headers(),
                        json=payload,
                    )
                    if r.status_code in (429, 502, 503) and attempt < 2:
                        time.sleep(0.4 * (attempt + 1))
                        continue
                    r.raise_for_status()
                    body = r.json()
                return body.get("answers") or {}
            except Exception as exc:  # noqa: BLE001 — retry transient Jev/network errors
                last_exc = exc
                if attempt < 2:
                    time.sleep(0.4 * (attempt + 1))
                    continue
                raise
        if last_exc:
            raise last_exc
        return {}


class MockJevClient:
    """Deterministic answers for unit tests — keyed by question id."""

    def __init__(
        self,
        answers: dict[str, Any] | None = None,
        *,
        default_noul: float = 0.5,
        default_choice: str | None = None,
        default_score: float = 2.5,
    ) -> None:
        self.answers = answers or {}
        self.default_noul = default_noul
        self.default_choice = default_choice
        # Score-as-sort uses this for ranking (0–10 preferred; mocks may use 0–3).
        self.default_score = default_score
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def system_one(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((state, questions))
        out: dict[str, Any] = {}
        for qid, q in questions.items():
            if qid in self.answers:
                out[qid] = self.answers[qid]
                continue
            qtype = q.get("type") if isinstance(q, dict) else None
            if qtype == "noul":
                out[qid] = {"type": "noul", "noul": self.default_noul}
            elif qtype == "choice":
                criteria = list((q.get("criteria") or {}).keys())
                choice = self.default_choice or (criteria[0] if criteria else "unknown")
                if choice not in criteria and criteria:
                    choice = criteria[0]
                probs = {c: (1.0 if c == choice else 0.0) for c in criteria}
                out[qid] = {"type": "choice", "choice": choice, "probabilities": probs}
            elif qtype == "score":
                out[qid] = {"type": "score", "score": float(self.default_score)}
            else:
                out[qid] = {"type": "score", "score": float(self.default_score)}
        return out


# Back-compat aliases for older test imports.
MockClmClient = MockJevClient
ClmClient = JevClient

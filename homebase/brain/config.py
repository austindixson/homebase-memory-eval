"""Load brain registry.toml with env overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


DEFAULT_REGISTRY = Path(__file__).with_name("registry.toml")


@dataclass
class Registry:
    # Hosted TypeSafe Jev (System One) — no local CLM.
    jev_base: str = "https://api.typesafe.ai"
    jev_model: str = "jev-latest"
    dense38: str = "http://127.0.0.1:11234/v1"
    moe35: str = "http://ghost64s-macbook-pro:11350/v1"
    flash: str = "http://127.0.0.1:8080/v1"
    dense38_model: str = "Qwen3.8-27B-4bit"
    moe35_model: str = "Qwen3.6-35B-A3B-MLX-Serve-4bit"
    flash_model: str = "/Volumes/DRIVE/models/mlx/GLM-4.7-Flash-4bit"
    host: str = "0.0.0.0"
    port: int = 9081
    inject_memory: bool = True
    mem_top_k: int = 8
    fast_path_max_chars: int = 240
    compact_min_chars: int = 8000
    route_cache_ttl_s: float = 30.0
    keep_threshold: float = 0.5
    error_keep_threshold: float = 0.25
    min_candidate_chars: int = 2000
    pin_last_user_turns: int = 2
    # Hard ceiling on chars forwarded to dense after Jev keep/drop.
    # ~70k tokens @ ~4 chars/tok — dense --ctx-size 150000 with headroom.
    max_forward_chars: int = 280_000
    # Explicit /v1/compact always aims here (~40k tok), not just the ceiling.
    compact_target_chars: int = 160_000
    max_jev_candidates: int = 32
    # When over budget, cut to this fraction of max_forward_chars (big steps).
    trim_target_ratio: float = 0.45
    # Where to inject frozen memory: "first_user" (prefix-stable) | "tail" (legacy).
    memory_position: str = "first_user"
    # Per-turn Jev compact on the chat path — off so history stays append-only.
    chat_compact: bool = False
    # After a dense response, if prompt cache hit rate falls below this floor,
    # unload+reload the mlx-serve model to drop stale prefix/KV (no clear API).
    kv_clear_enabled: bool = False
    kv_clear_hit_rate_floor: float = 0.70
    kv_clear_min_prompt_tokens: int = 8192
    kv_clear_cooldown_s: float = 300.0
    router_default: str = "dense38"
    flash_criteria: str = "Direct lookups, short factual answers, simple extraction."
    dense38_criteria: str = "Coding and everyday chat on local dense (~70 tok/s)."
    moe35_criteria: str = "Long-context or hard reasoning on travel MoE (ghost64)."
    memory_only_criteria: str = "Recall a fact already stored in Home Base memory."
    deny_patterns: list[str] = field(default_factory=list)
    safe_tools: list[str] = field(default_factory=list)
    db_path: str = ""
    relation_threshold: float = 0.6
    evidence_sufficient: float = 0.9
    continue_useful_floor: float = 0.15
    max_hops: int = 1
    budget: int = 20
    write_candidate_k: int = 10
    # Mem v2 retrieve / filter knobs (GPTR-style Jev usefulness).
    mem_candidate_k: int = 48
    # Telemetry only — Score-as-sort does not use this as a keep cut.
    mem_jev_min_score: float = 0.0
    mem_jev_concurrency: int = 4
    mem_keyword_relative_threshold: float = 0.5
    mem_prove_floor: float = 3.5
    # Optional embedding expander (BRAIN_EMBED_URL).
    embed_url: str = ""
    embed_model: str = "qwen3-embed"
    # Frontier thrash handoff (OpenGrok spawn_subagent → Luna/Sol).
    frontier_enabled: bool = True
    frontier_handoff_models: list[str] = field(
        default_factory=lambda: [
            "gpt-6-astra",
            "gpt-6-sol",
            "gpt-5.6-sol",
            "gpt-6-luna",
            "gpt-5.6-luna",
        ]
    )
    frontier_probe_host: str = "chatgpt.com:443"
    frontier_client_signal_max_age_s: float = 120.0
    # Local VLM for OpenGrok paste/drag images (Flash is text-only).
    vision_enabled: bool = True
    vision_base_url: str = "http://ghost32.tailf163d8.ts.net:11434/v1"
    vision_model: str = "qwen2.5vl:3b"
    vision_timeout_s: float = 120.0
    # Orchestrator (plan queue + Jev lanes).
    orchestrate_enabled: bool = True
    orchestrate_t1_parallel: int = 3
    orchestrate_escalate_after_failures: int = 2
    orchestrate_catalog_max_age_s: float = 300.0
    orchestrate_db_path: str = ""
    orchestrate_planners: list[str] = field(
        default_factory=lambda: [
            "claude:claude-opus-5-5",
            "codex:gpt-6-astra",
            "codex:gpt-6-sol",
            "cursor:opus-5.5",
            "cursor:composer-2.5",
        ]
    )
    orchestrate_tiers: dict[str, list[str]] = field(
        default_factory=lambda: {
            "T0": ["homebase-brain"],
            "T1": [
                "nim-glm-5.3-flash",
                "nim-glm-5.3",
                "nim-kimi-k3",
                "nim-deepseek-v4-pro",
            ],
            "T2": ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-6-luna"],
            "T3": ["gpt-6-astra", "gpt-6-sol", "gpt-5.6-sol"],
        }
    )

    def upstream_base(self, route: str) -> str | None:
        if route == "dense38":
            return self.dense38.rstrip("/")
        if route == "moe35":
            return self.moe35.rstrip("/")
        if route == "flash":
            return self.flash.rstrip("/")
        if route == "memory_only":
            return None
        if self.router_default == "moe35":
            return self.moe35.rstrip("/")
        if self.router_default == "flash":
            return self.flash.rstrip("/")
        return self.dense38.rstrip("/")

    def upstream_model(self, route: str) -> str:
        if route == "flash":
            return self.flash_model
        if route == "dense38":
            return self.dense38_model
        if route == "moe35":
            return self.moe35_model
        return self.dense38_model


def _get(table: dict[str, Any], key: str, default: Any) -> Any:
    return table.get(key, default) if isinstance(table, dict) else default


def load_registry(path: Path | None = None) -> Registry:
    p = Path(os.environ.get("BRAIN_REGISTRY", path or DEFAULT_REGISTRY))
    data: dict[str, Any] = {}
    if p.is_file():
        with p.open("rb") as f:
            data = tomllib.load(f)

    up = data.get("upstreams") or {}
    px = data.get("proxy") or {}
    cp = data.get("compact") or {}
    rt = data.get("router") or {}
    am = data.get("automode") or {}
    mem = data.get("mem") or {}
    fr = data.get("frontier") or {}
    vis = data.get("vision") or {}
    orch = data.get("orchestrate") or {}
    orch_tiers = orch.get("tiers") if isinstance(orch.get("tiers"), dict) else {}

    default_db = str(Path.home() / ".homebase" / "mem" / "mem.sqlite")
    default_orch_db = str(Path.home() / ".homebase" / "orchestrate.sqlite")

    # Prefer max_jev_candidates; accept legacy max_clm_candidates from older toml.
    max_candidates = int(
        _get(cp, "max_jev_candidates", _get(cp, "max_clm_candidates", 24))
    )

    reg = Registry(
        jev_base=os.environ.get(
            "TYPESAFE_BASE_URL",
            os.environ.get("JEV_BASE_URL", _get(up, "jev_base", "https://api.typesafe.ai")),
        ).rstrip("/"),
        jev_model=os.environ.get(
            "TYPESAFE_MODEL",
            os.environ.get("JEV_MODEL", _get(up, "jev_model", "jev-latest")),
        ),
        dense38=os.environ.get(
            "BRAIN_DENSE38_URL", _get(up, "dense38", "http://127.0.0.1:11234/v1")
        ),
        moe35=os.environ.get(
            "BRAIN_MOE35_URL", _get(up, "moe35", "http://ghost64s-macbook-pro:11350/v1")
        ),
        flash=os.environ.get("BRAIN_FLASH_URL", _get(up, "flash", "http://127.0.0.1:8080/v1")),
        dense38_model=_get(up, "dense38_model", "Qwen3.8-27B-4bit"),
        moe35_model=_get(up, "moe35_model", "Qwen3.6-35B-A3B-MLX-Serve-4bit"),
        flash_model=_get(up, "flash_model", "/Volumes/DRIVE/models/mlx/GLM-4.7-Flash-4bit"),
        host=_get(px, "host", "0.0.0.0"),
        port=int(os.environ.get("BRAIN_PORT", _get(px, "port", 9081))),
        inject_memory=bool(_get(px, "inject_memory", True)),
        mem_top_k=int(_get(px, "mem_top_k", 8)),
        fast_path_max_chars=int(_get(px, "fast_path_max_chars", 240)),
        compact_min_chars=int(
            _get(px, "compact_min_chars", _get(cp, "compact_min_chars", 8000))
        ),
        route_cache_ttl_s=float(_get(px, "route_cache_ttl_s", 30)),
        keep_threshold=float(_get(cp, "keep_threshold", 0.65)),
        error_keep_threshold=float(_get(cp, "error_keep_threshold", 0.30)),
        min_candidate_chars=int(_get(cp, "min_candidate_chars", 800)),
        pin_last_user_turns=int(_get(cp, "pin_last_user_turns", 2)),
        max_forward_chars=int(
            os.environ.get(
                "BRAIN_MAX_FORWARD_CHARS",
                _get(cp, "max_forward_chars", 280_000),
            )
        ),
        compact_target_chars=int(
            os.environ.get(
                "BRAIN_COMPACT_TARGET_CHARS",
                _get(cp, "compact_target_chars", 160_000),
            )
        ),
        max_jev_candidates=max_candidates,
        trim_target_ratio=float(_get(cp, "trim_target_ratio", 0.45)),
        memory_position=str(_get(px, "memory_position", "first_user") or "first_user"),
        chat_compact=bool(_get(px, "chat_compact", False)),
        kv_clear_enabled=bool(_get(px, "kv_clear_enabled", False)),
        kv_clear_hit_rate_floor=float(_get(px, "kv_clear_hit_rate_floor", 0.70)),
        kv_clear_min_prompt_tokens=int(_get(px, "kv_clear_min_prompt_tokens", 8192)),
        kv_clear_cooldown_s=float(_get(px, "kv_clear_cooldown_s", 300.0)),
        router_default=_get(rt, "default", "dense38"),
        flash_criteria=_get(rt, "flash_criteria", ""),
        dense38_criteria=_get(rt, "dense38_criteria", ""),
        moe35_criteria=_get(rt, "moe35_criteria", ""),
        memory_only_criteria=_get(rt, "memory_only_criteria", ""),
        deny_patterns=list(_get(am, "deny_patterns", [])),
        safe_tools=list(_get(am, "safe_tools", [])),
        db_path=os.environ.get("BRAIN_MEM_DB", _get(mem, "db_path", default_db) or default_db),
        relation_threshold=float(_get(mem, "relation_threshold", 0.6)),
        evidence_sufficient=float(_get(mem, "evidence_sufficient", 0.9)),
        continue_useful_floor=float(_get(mem, "continue_useful_floor", 0.15)),
        max_hops=int(os.environ.get("BRAIN_MEM_MAX_HOPS", _get(mem, "max_hops", 1))),
        budget=int(_get(mem, "budget", 20)),
        write_candidate_k=int(_get(mem, "write_candidate_k", 10)),
        mem_candidate_k=int(
            os.environ.get("BRAIN_MEM_CANDIDATE_K", _get(mem, "mem_candidate_k", 48))
        ),
        mem_jev_min_score=float(
            os.environ.get("BRAIN_MEM_JEV_MIN_SCORE", _get(mem, "mem_jev_min_score", 0.0))
        ),
        mem_jev_concurrency=int(
            os.environ.get(
                "BRAIN_MEM_JEV_CONCURRENCY", _get(mem, "mem_jev_concurrency", 4)
            )
        ),
        mem_keyword_relative_threshold=float(
            _get(mem, "mem_keyword_relative_threshold", 0.5)
        ),
        mem_prove_floor=float(
            os.environ.get("BRAIN_MEM_PROVE_FLOOR", _get(mem, "mem_prove_floor", 4.0))
        ),
        embed_url=str(
            os.environ.get("BRAIN_EMBED_URL", _get(mem, "embed_url", "")) or ""
        ),
        embed_model=str(
            os.environ.get("BRAIN_EMBED_MODEL", _get(mem, "embed_model", "qwen3-embed"))
            or "qwen3-embed"
        ),
        frontier_enabled=bool(_get(fr, "enabled", True)),
        frontier_handoff_models=list(
            _get(
                fr,
                "handoff_models",
                [
                    "gpt-6-astra",
                    "gpt-6-sol",
                    "gpt-5.6-sol",
                    "gpt-6-luna",
                    "gpt-5.6-luna",
                ],
            )
        ),
        frontier_probe_host=str(_get(fr, "probe_host", "chatgpt.com:443")),
        frontier_client_signal_max_age_s=float(
            _get(fr, "client_signal_max_age_s", 120)
        ),
        vision_enabled=bool(_get(vis, "enabled", True)),
        vision_base_url=str(
            os.environ.get(
                "BRAIN_VISION_BASE_URL",
                _get(
                    vis,
                    "base_url",
                    "http://ghost32.tailf163d8.ts.net:11434/v1",
                ),
            )
        ),
        vision_model=str(
            os.environ.get(
                "BRAIN_VISION_MODEL",
                _get(vis, "model", "qwen2.5vl:3b"),
            )
        ),
        vision_timeout_s=float(
            os.environ.get(
                "BRAIN_VISION_TIMEOUT_S",
                _get(vis, "timeout_s", 120.0),
            )
        ),
        orchestrate_enabled=bool(_get(orch, "enabled", True)),
        orchestrate_t1_parallel=int(_get(orch, "t1_parallel", 3)),
        orchestrate_escalate_after_failures=int(
            _get(orch, "escalate_after_failures", 2)
        ),
        orchestrate_catalog_max_age_s=float(
            _get(orch, "catalog_max_age_s", 300)
        ),
        orchestrate_db_path=os.environ.get(
            "BRAIN_ORCHESTRATE_DB",
            _get(orch, "db_path", default_orch_db) or default_orch_db,
        ),
        orchestrate_planners=list(
            _get(
                orch,
                "planners",
                [
                    "claude:claude-opus-5-5",
                    "codex:gpt-6-astra",
                    "codex:gpt-6-sol",
                    "cursor:opus-5.5",
                    "cursor:composer-2.5",
                ],
            )
        ),
        orchestrate_tiers={
            "T0": list(
                orch_tiers.get("T0")
                or ["homebase-brain"]
            ),
            "T1": list(
                orch_tiers.get("T1")
                or [
                    "nim-glm-5.3-flash",
                    "nim-glm-5.3",
                    "nim-kimi-k3",
                    "nim-deepseek-v4-pro",
                ]
            ),
            "T2": list(
                orch_tiers.get("T2")
                or ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-6-luna"]
            ),
            "T3": list(
                orch_tiers.get("T3")
                or ["gpt-6-astra", "gpt-6-sol", "gpt-5.6-sol"]
            ),
        },
    )
    return reg

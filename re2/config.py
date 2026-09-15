"""Load config.toml into a typed Config object."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    corpus_root: Path
    data_dir: Path
    legacy_dir: Path
    thumbs_dir: Path
    archive_metadata: Path
    legacy_output_json: Path

    base_url: str
    vision_model: str
    chat_model: str
    vision_concurrency: int     # 0 = auto: sized from the server's slots and context
    vision_max_tokens: int
    vision_max_tokens_tall: int
    vision_max_side: int
    vision_max_tiles: int

    embed_url: str
    embed_model: str
    embed_dims: int
    embed_max_chars: int

    asr_model: str
    asr_device: str
    asr_compute_type: str

    server_host: str
    server_port: int

    chat_base_url: str
    chat_context_tokens: int

    @property
    def db_path(self) -> Path:
        return self.data_dir / "re2.sqlite"


def _resolve(p: str) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for key, value in over.items():
        out[key] = _merge(base[key], value) if isinstance(value, dict) and isinstance(base.get(key), dict) else value
    return out


LOCAL_OVERRIDES = REPO_ROOT / "config.local.toml"


def load(path: Path | str | None = None) -> Config:
    """config.toml, with config.local.toml (git-ignored) layered on top when loading the default."""
    explicit = path is not None
    path = Path(path) if path else REPO_ROOT / "config.toml"
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    if not explicit and LOCAL_OVERRIDES.is_file():
        with open(LOCAL_OVERRIDES, "rb") as f:
            raw = _merge(raw, tomllib.load(f))
    p, llm, emb, asr, srv = raw["paths"], raw["llm"], raw["embed"], raw["asr"], raw["server"]
    chat = raw.get("chat", {})
    return Config(
        corpus_root=_resolve(p["corpus_root"]),
        data_dir=_resolve(p["data_dir"]),
        legacy_dir=_resolve(p["legacy_dir"]),
        thumbs_dir=_resolve(p["thumbs_dir"]),
        archive_metadata=_resolve(p["archive_metadata"]),
        legacy_output_json=_resolve(p["legacy_output_json"]),
        base_url=llm["base_url"].rstrip("/"),
        vision_model=llm["vision_model"],
        vision_concurrency=0 if str(llm["vision_concurrency"]).lower() == "auto" else int(llm["vision_concurrency"]),
        vision_max_tokens=int(llm["vision_max_tokens"]),
        vision_max_tokens_tall=int(llm["vision_max_tokens_tall"]),
        vision_max_side=int(llm["vision_max_side"]),
        vision_max_tiles=int(llm["vision_max_tiles"]),
        embed_url=emb["url"],
        embed_model=emb["model"],
        embed_dims=int(emb["dims"]),
        embed_max_chars=int(emb["max_chars"]),
        asr_model=asr["model"],
        asr_device=asr["device"],
        asr_compute_type=asr["compute_type"],
        server_host=srv["host"],
        server_port=int(srv["port"]),
        chat_base_url=chat.get("base_url", llm["base_url"]).rstrip("/"),
        chat_model=chat.get("model", llm["chat_model"]),
        chat_context_tokens=int(chat.get("context_tokens", 32000)),
    )

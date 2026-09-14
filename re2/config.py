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
    vision_concurrency: int
    vision_max_tokens: int
    vision_max_side: int

    embed_url: str
    embed_model: str
    embed_dims: int
    embed_max_chars: int

    asr_model: str
    asr_device: str
    asr_compute_type: str

    server_host: str
    server_port: int

    @property
    def db_path(self) -> Path:
        return self.data_dir / "re2.sqlite"


def _resolve(p: str) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load(path: Path | str | None = None) -> Config:
    path = Path(path) if path else REPO_ROOT / "config.toml"
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    p, llm, emb, asr, srv = raw["paths"], raw["llm"], raw["embed"], raw["asr"], raw["server"]
    return Config(
        corpus_root=_resolve(p["corpus_root"]),
        data_dir=_resolve(p["data_dir"]),
        legacy_dir=_resolve(p["legacy_dir"]),
        thumbs_dir=_resolve(p["thumbs_dir"]),
        archive_metadata=_resolve(p["archive_metadata"]),
        legacy_output_json=_resolve(p["legacy_output_json"]),
        base_url=llm["base_url"].rstrip("/"),
        vision_model=llm["vision_model"],
        chat_model=llm["chat_model"],
        vision_concurrency=int(llm["vision_concurrency"]),
        vision_max_tokens=int(llm["vision_max_tokens"]),
        vision_max_side=int(llm["vision_max_side"]),
        embed_url=emb["url"],
        embed_model=emb["model"],
        embed_dims=int(emb["dims"]),
        embed_max_chars=int(emb["max_chars"]),
        asr_model=asr["model"],
        asr_device=asr["device"],
        asr_compute_type=asr["compute_type"],
        server_host=srv["host"],
        server_port=int(srv["port"]),
    )

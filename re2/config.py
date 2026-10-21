"""Load config.toml into a typed Config object."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
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

    wiki_base_url: str          # the text model that writes digests and wiki pages; defaults to [chat]
    wiki_model: str
    wiki_context_tokens: int    # what one request may use
    wiki_concurrency: int

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
    wiki = raw.get("wiki", {})
    chat_base_url = chat.get("base_url", llm["base_url"]).rstrip("/")
    chat_model = chat.get("model", llm["chat_model"])
    chat_context_tokens = int(chat.get("context_tokens", 32000))
    corpus_root = _resolve(os.environ.get("RE2_CORPUS_ROOT") or p["corpus_root"])
    return Config(
        corpus_root=corpus_root,
        data_dir=_resolve(p["data_dir"]),
        legacy_dir=_resolve(p["legacy_dir"]),
        thumbs_dir=_resolve(p["thumbs_dir"]),
        archive_metadata=corpus_root / Path(p["archive_metadata"]).expanduser(),
        legacy_output_json=corpus_root / Path(p["legacy_output_json"]).expanduser(),
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
        chat_base_url=chat_base_url,
        chat_model=chat_model,
        chat_context_tokens=chat_context_tokens,
        wiki_base_url=wiki.get("base_url", chat_base_url).rstrip("/"),
        wiki_model=wiki.get("model", chat_model),
        wiki_context_tokens=int(wiki.get("context_tokens", chat_context_tokens)),
        wiki_concurrency=max(1, int(wiki.get("concurrency", 1))),
    )


# ── Settings the dashboard can change ─────────────────────────────────────
# (section in config.local.toml, key there, Config field). Written to config.local.toml, never config.toml.
SERVICE_FIELDS = {
    "chat": [("chat", "base_url", "chat_base_url"), ("chat", "model", "chat_model"),
             ("chat", "context_tokens", "chat_context_tokens")],
    "wiki": [("wiki", "base_url", "wiki_base_url"), ("wiki", "model", "wiki_model"),
             ("wiki", "context_tokens", "wiki_context_tokens"), ("wiki", "concurrency", "wiki_concurrency")],
    "vision": [("llm", "base_url", "base_url"), ("llm", "vision_model", "vision_model"),
               ("llm", "vision_concurrency", "vision_concurrency")],
    "embed": [("embed", "url", "embed_url"), ("embed", "model", "embed_model")],
    "asr": [("asr", "model", "asr_model"), ("asr", "device", "asr_device"), ("asr", "compute_type", "asr_compute_type")],
}
INT_KEYS = {"context_tokens", "vision_concurrency", "concurrency"}


def services(cfg: Config) -> dict:
    return {name: {key: getattr(cfg, attr) for _, key, attr in fields} for name, fields in SERVICE_FIELDS.items()}


def save_services(cfg: Config, updates: dict) -> None:
    """Write changed service settings to config.local.toml (keeping its comments), reload, and apply them to the
    running cfg: the dashboard's Ask and search use them at once; batch jobs read them when next started."""
    import tomlkit

    path = LOCAL_OVERRIDES
    doc = tomlkit.parse(path.read_text()) if path.is_file() else tomlkit.document()
    current = services(cfg)
    for name, values in updates.items():
        if name not in SERVICE_FIELDS or not isinstance(values, dict):
            raise ValueError(f"unknown service {name!r}")
        for section, key, _ in SERVICE_FIELDS[name]:
            if key not in values or values[key] == current[name][key]:
                continue
            value = values[key]
            if key in INT_KEYS:
                value = int(value)
                if value < 0:
                    raise ValueError(f"{name}.{key} must not be negative")
            else:
                value = str(value).strip()
                if not value:
                    raise ValueError(f"{name}.{key} must not be empty")
                if key in ("base_url", "url") and not value.startswith(("http://", "https://")):
                    raise ValueError(f"{name}.{key} must be an http(s) URL")
            if section not in doc:
                doc.add(tomlkit.nl())
                doc.add(section, tomlkit.table())
            doc[section][key] = value
    path.write_text(tomlkit.dumps(doc))
    fresh = load()
    for fields in SERVICE_FIELDS.values():
        for _, _, attr in fields:
            setattr(cfg, attr, getattr(fresh, attr))

"""Named model servers ("presets") and the picker the model-using commands show on startup.

A preset is a server URL and a model name, with the requests-at-once and context size to use there:

    [presets.mlboy-gemma-mtp]
    base_url = "https://mlboy.tail99c274.ts.net:5000/v1"
    model = "gemma4-31b-it-mtp-mlboy"
    concurrency = "auto"        # or a number; "auto" fits the vision server's slots
    context_tokens = 32000      # chat and wiki only; what one request may use

They live in config.local.toml (git-ignored), or config.toml. Any preset can serve vision, chat or wiki;
choosing one copies its values onto that service, for one run or, when saved, as the default.
"""
from __future__ import annotations

import dataclasses
import difflib
import os
import re
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import httpx

from re2 import config, vlm

# Which service each model-using command talks to.
COMMAND_SERVICE = {"extract-images": "vision", "pdf-ocr": "vision", "digests": "wiki", "wiki-build": "wiki",
                   "tag-relations": "chat", "ask-coverage": "chat"}
SERVICE_TITLE = {"vision": "Vision", "chat": "Chat", "wiki": "Wiki writer"}
# preset key -> (key in config.services()[service], Config attribute)
SERVICE_KEYS = {
    "vision": {"base_url": ("base_url", "base_url"), "model": ("vision_model", "vision_model"),
               "concurrency": ("vision_concurrency", "vision_concurrency")},
    "chat": {"base_url": ("base_url", "chat_base_url"), "model": ("model", "chat_model"),
             "context_tokens": ("context_tokens", "chat_context_tokens")},
    "wiki": {"base_url": ("base_url", "wiki_base_url"), "model": ("model", "wiki_model"),
             "context_tokens": ("context_tokens", "wiki_context_tokens"),
             "concurrency": ("concurrency", "wiki_concurrency")},
}
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,59}$")
TIMEOUT = 8.0


# ── Storage ───────────────────────────────────────────────────────────────

def clean(raw: dict) -> dict:
    """A preset with a trimmed URL, concurrency as an int (0 = auto) and context_tokens as an int, if set."""
    base_url, model = str(raw.get("base_url", "")).strip().rstrip("/"), str(raw.get("model", "")).strip()
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("base_url must be an http(s) URL")
    if not model:
        raise ValueError("model must not be empty")
    out = {"base_url": base_url, "model": model}
    conc = raw.get("concurrency")
    if conc not in (None, ""):
        out["concurrency"] = 0 if str(conc).lower() == "auto" else int(conc)
        if out["concurrency"] < 0:
            raise ValueError("concurrency must not be negative")
    if raw.get("context_tokens") not in (None, "", 0):
        out["context_tokens"] = int(raw["context_tokens"])
        if out["context_tokens"] < 1000:
            raise ValueError("context_tokens is too small")
    return out


def load_all() -> dict[str, dict]:
    """Presets from config.toml, then config.local.toml (a name in both takes the local one)."""
    found: dict[str, dict] = {}
    for path in (config.REPO_ROOT / "config.toml", config.LOCAL_OVERRIDES):
        if path.is_file():
            with open(path, "rb") as f:
                found.update(tomllib.load(f).get("presets") or {})
    out = {}
    for name, raw in found.items():
        try:
            out[name] = clean(raw)
        except (ValueError, TypeError, AttributeError):
            continue   # a half-written preset is left out rather than stopping every command
    return out


def save(name: str, preset: dict) -> dict:
    """Add or replace a preset in config.local.toml, keeping its comments."""
    import tomlkit

    if not NAME_RE.match(name or ""):
        raise ValueError("a preset name is letters, digits, '.', '_' or '-', up to 60 characters")
    preset = clean(preset)
    path = config.LOCAL_OVERRIDES
    doc = tomlkit.parse(path.read_text()) if path.is_file() else tomlkit.document()
    if "presets" not in doc:
        doc.add(tomlkit.nl())
        doc.add("presets", tomlkit.table(is_super_table=True))
    table = tomlkit.table()
    for key, value in preset.items():
        table[key] = "auto" if key == "concurrency" and value == 0 else value
    doc["presets"][name] = table
    path.write_text(tomlkit.dumps(doc))
    return preset


def delete(name: str) -> bool:
    import tomlkit

    path = config.LOCAL_OVERRIDES
    if not path.is_file():
        return False
    doc = tomlkit.parse(path.read_text())
    if name not in doc.get("presets", {}):
        return False
    del doc["presets"][name]
    path.write_text(tomlkit.dumps(doc))
    return True


# ── Presets and services ──────────────────────────────────────────────────

def of_service(cfg, service: str) -> dict:
    """The service's current settings, as a preset."""
    return {key: getattr(cfg, attr) for key, (_, attr) in SERVICE_KEYS[service].items()}


def matching(cfg, service: str, presets: dict[str, dict]) -> str | None:
    """The name of the preset the service uses now (same server and model), if any."""
    cur = of_service(cfg, service)
    return next((n for n, p in presets.items()
                 if p["base_url"] == cur["base_url"].rstrip("/") and p["model"] == cur["model"]), None)


def service_values(service: str, preset: dict) -> dict:
    """The preset as `config.save_services` values for one service; keys the preset leaves out are kept."""
    return {SERVICE_KEYS[service][k][0]: v for k, v in preset.items() if k in SERVICE_KEYS[service]}


def apply(cfg, service: str, preset: dict):
    """A copy of cfg that uses the preset for the service."""
    changes = {SERVICE_KEYS[service][k][1]: v for k, v in preset.items() if k in SERVICE_KEYS[service]}
    if service == "wiki" and "wiki_concurrency" in changes:
        changes["wiki_concurrency"] = changes["wiki_concurrency"] or cfg.wiki_concurrency   # no auto for text
    return dataclasses.replace(cfg, **changes)


# ── What a server offers ──────────────────────────────────────────────────

def _root(base_url: str) -> str:
    return base_url[:-3] if base_url.endswith("/v1") else base_url


def _arg(args: list, *flags: str) -> str | None:
    return next((args[i + 1] for i, a in enumerate(args[:-1]) if a in flags), None)


def list_models(client: httpx.Client, base_url: str) -> list[dict]:
    """The models an OpenAI-compatible server offers, with load state, context, slots and whether it reads
    images when llama.cpp's router says. Raises httpx.HTTPError or ValueError when the server does not answer."""
    resp = client.get(f"{base_url.rstrip('/')}/models", timeout=TIMEOUT)
    resp.raise_for_status()
    models = []
    for m in resp.json().get("data") or []:
        status = m.get("status") if isinstance(m.get("status"), dict) else {}
        args = status.get("args") or []
        ctx, parallel = _arg(args, "--ctx-size", "-c"), _arg(args, "--parallel", "-np")
        ctx = int(ctx) if str(ctx or "").isdigit() else None
        parallel = int(parallel) if str(parallel or "").isdigit() and int(parallel) > 0 else None
        unified = "--kv-unified" in args or "-kvu" in args or not parallel   # llama.cpp's automatic parallel shares one cache
        models.append({"id": m.get("id"), "state": status.get("value"), "ctx": ctx, "parallel": parallel,
                       "request_ctx": ctx if ctx is None or unified else ctx // max(parallel, 1),
                       "vision": ("--mmproj" in args or "-mm" in args or "--mmproj-url" in args) if args else None})
    return models


@dataclass
class Probe:
    reachable: bool = False
    error: str = ""                 # why the preset cannot be used as it is
    models: list[dict] = field(default_factory=list)
    entry: dict | None = None       # the preset's model, as list_models gives it
    suggest: list[str] = field(default_factory=list)   # close model names when the model is missing
    others_loaded: list[str] = field(default_factory=list)
    limits: vlm.ServerLimits | None = None
    at_once: int | None = None      # requests the run will send at once (vision: as sized for the server)

    @property
    def ok(self) -> bool:
        return self.reachable and self.entry is not None and not self.error


def probe(client: httpx.Client, cfg, service: str, preset: dict) -> Probe:
    """Whether the server answers, has the model, can read images (vision) and how many requests fit."""
    try:
        models = list_models(client, preset["base_url"])
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        return Probe(error=f"cannot reach the server: {str(exc)[:160] or type(exc).__name__}")
    ids = [m["id"] for m in models]
    entry = next((m for m in models if m["id"] == preset["model"]), None)
    if entry is None:
        return Probe(reachable=True, models=models, error="the server has no model by that name",
                     suggest=difflib.get_close_matches(preset["model"], ids, n=3, cutoff=0.5))
    p = Probe(reachable=True, models=models, entry=entry,
              others_loaded=[m["id"] for m in models if m["state"] == "loaded" and m["id"] != entry["id"]])
    if entry["state"] in ("loaded", None):
        try:   # a running model says for itself whether it reads images
            mods = client.get(f"{_root(preset['base_url'])}/props", params={"model": entry["id"]},
                              timeout=TIMEOUT).json().get("modalities") or {}
            if "vision" in mods:
                entry["vision"] = bool(mods["vision"])
        except (httpx.HTTPError, ValueError, AttributeError):
            pass
    if service == "vision":
        if entry["vision"] is False:
            p.error = "this model cannot read images (no --mmproj)"
        run_cfg = apply(cfg, service, preset)
        p.limits = vlm.server_limits(client, run_cfg)
        p.at_once = vlm.plan_concurrency(run_cfg, p.limits, run_cfg.vision_concurrency)[0]
    else:
        p.at_once = apply(cfg, service, preset).wiki_concurrency if service == "wiki" else None
    return p


# ── Colour ────────────────────────────────────────────────────────────────
# Off unless the picker runs in a terminal (and NO_COLOR is unset), so logs and tests stay plain.
_color = False
STYLES = {"ok": "38;5;114", "bad": "38;5;203", "warn": "38;5;221", "key": "1;38;5;75", "head": "1;38;5;75",
          "name": "1", "dim": "2", "mark": "1;38;5;211"}


def paint(style: str, text) -> str:
    return f"\033[{STYLES[style]}m{text}\033[0m" if _color else str(text)


def describe(p: Probe) -> str:
    """One line: what the check found."""
    if not p.reachable:
        return paint("bad", f"✗ {p.error}")
    if p.entry is None:
        return paint("bad", f"✗ {p.error}") + \
            (f"{paint('dim', '; closest:')} {paint('warn', ', '.join(p.suggest))}" if p.suggest else "")
    e = p.entry
    sep = paint("dim", " · ")
    bits = [paint("ok", "loaded") if e["state"] == "loaded" else paint("dim", e["state"] or "available")]
    if e["vision"]:
        bits.append(paint("ok", "reads images"))
    if p.limits:
        bits.append(paint("dim", f"{p.limits.slots} slots {'sharing' if p.limits.shared else 'of'} "
                                 f"{p.limits.slot_ctx:,} tokens"))
    elif e["request_ctx"]:
        bits.append(paint("dim", f"{e['request_ctx']:,} tokens per request"
                                 + (f", {e['parallel']} slots" if e["parallel"] else "")))
    if p.at_once:
        bits.append(paint("key", f"{p.at_once} at once"))
    line = (paint("bad", f"✗ {p.error}") + sep if p.error else paint("ok", "✓ ")) + sep.join(bits)
    if e["state"] not in ("loaded", None) and p.others_loaded:
        line += "\n      " + paint("warn", f"⚠ loading it may unload {', '.join(p.others_loaded)}")
    return line


# ── The picker ────────────────────────────────────────────────────────────

class Quit(Exception):
    pass


def _ask(ask, prompt: str, default: str = "") -> str:
    try:
        answer = ask(f"{paint('name', prompt)}{paint('dim', f' [{default}]') if default else ''}: ").strip()
    except EOFError:
        raise Quit from None
    if answer.lower() in ("q", "quit"):
        raise Quit
    return answer or default


def _yes(ask, prompt: str, default: bool = False) -> bool:
    answer = _ask(ask, f"{prompt} ({'Y/n' if default else 'y/N'})").lower()
    return not answer.startswith("n") if default else answer.startswith("y")


def _find_server(client, url: str) -> tuple[str, list[dict]]:
    """The base URL that lists models, trying https:// and /v1 when they were left out."""
    url = url.strip().rstrip("/")
    bases = [url] if url.startswith(("http://", "https://")) else [f"https://{url}", f"http://{url}"]
    tries = [c for b in bases for c in ([b] if b.endswith("/v1") else [f"{b}/v1", b])]
    last = None
    for base in tries:
        try:
            return base, list_models(client, base)
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            last = exc
    raise ValueError(f"no model list at {url}: {str(last)[:160]}")


def new_preset(client, cfg, service: str, presets: dict, ask=input, say=print) -> tuple[str, dict]:
    """Ask for a server, list its models to choose from, check the choice and save it as a preset."""
    known = sorted({p["base_url"] for p in presets.values()} | {of_service(cfg, service)["base_url"]})
    say("\n" + paint("head", "New preset.") + " Known servers: " + ", ".join(paint("dim", k) for k in known))
    while True:
        url = _ask(ask, "Server URL (host:port is enough)")
        if not url:
            continue
        try:
            base_url, models = _find_server(client, url)
            break
        except ValueError as exc:
            say("  " + paint("bad", f"✗ {exc}"))
    if not models:
        raise ValueError(f"{base_url} lists no models")
    say(f"\n{paint('head', 'Models on')} {paint('dim', base_url)}")
    width = max(len(m["id"]) for m in models)
    for i, m in enumerate(models, 1):
        tag = "" if service != "vision" else (paint("ok", "  reads images") if m["vision"]
                                               else paint("warn", "  text only") if m["vision"] is False else "")
        state = paint("ok", "loaded  ") if m["state"] == "loaded" else paint("dim", f"{m['state'] or '':8}")
        ctx = paint("dim", f"  {m['request_ctx']:,} ctx") if m["request_ctx"] else ""
        say(f"  {paint('key', f'{i:>2}')}  {m['id']:<{width}}  {state}{tag}{ctx}")
    while True:
        pick = _ask(ask, "Model (number or name)")
        entry = models[int(pick) - 1] if pick.isdigit() and 0 < int(pick) <= len(models) else \
            next((m for m in models if m["id"] == pick), None)
        if entry is None:
            say("  " + paint("bad", "not one of the above"))
        elif service == "vision" and entry["vision"] is False and not _yes(ask, "  That model cannot read images. Use it anyway?"):
            continue
        else:
            break
    preset = {"base_url": base_url, "model": entry["id"]}
    if service == "vision":
        preset["concurrency"] = _ask(ask, "Requests at once (auto fits the server's slots)", "auto")
    elif service == "wiki":
        preset["concurrency"] = _ask(ask, "Requests at once", str(entry["parallel"] or 1))
    if service in ("chat", "wiki"):
        preset["context_tokens"] = _ask(ask, "Context tokens one request may use",
                                        str(min(32000, entry["request_ctx"] or 32000)))
    preset = clean(preset)
    host = re.sub(r"^https?://|[:/].*$", "", base_url).split(".")[0]
    name = entry["id"] if host in entry["id"] else f"{host}-{entry['id']}"
    while True:
        name = _ask(ask, "Name", name if NAME_RE.match(name) else "")
        if NAME_RE.match(name):
            break
        say("  " + paint("bad", "letters, digits, '.', '_' or '-', up to 60 characters"))
    save(name, preset)
    say(paint("ok", "✓ Saved preset ") + paint("name", name) + paint("dim", f" in {config.LOCAL_OVERRIDES}"))
    return name, preset


def pick(cfg, service: str, command: str, client_factory=httpx.Client, ask=input, say=print):
    """Show the presets that can serve this service with a live check of each, and return cfg using the
    one chosen (Enter keeps the current settings). `n` adds a preset; `q` quits."""
    with client_factory() as client:
        presets = load_all()
        current = matching(cfg, service, presets)
        while True:
            rows = [(name, p) for name, p in presets.items()]
            if current is None:
                rows.insert(0, ("(current settings)", of_service(cfg, service)))
            default = next(i for i, (n, _) in enumerate(rows, 1) if n in (current, "(current settings)"))
            say(paint("dim", f"Checking {len(rows)} server{'s' if len(rows) != 1 else ''} for {command} "
                             f"({SERVICE_TITLE[service].lower()})…"))
            with ThreadPoolExecutor(min(8, len(rows))) as pool:
                checks = list(pool.map(lambda r: probe(client, cfg, service, r[1]), rows))
            say(f"\n{paint('head', f'{SERVICE_TITLE[service]} server for {command}')}")
            wn = max(len(n) for n, _ in rows)
            wm = max(len("" if n == p["model"] else p["model"]) for n, p in rows)
            for i, ((name, p), check) in enumerate(zip(rows, checks), 1):
                mark = paint("mark", "●") if i == default else " "
                model = "" if name == p["model"] else p["model"]
                say(f" {mark}{paint('key', f'{i:>2}')}  {paint('name', f'{name:<{wn}}')}  {model:<{wm}}  "
                    f"{paint('dim', p['base_url'])}\n      {describe(check)}")
            say(f"   {paint('key', 'n')}  add a preset    {paint('key', 'q')}  quit")
            choice = _ask(ask, "Use", str(default))
            if choice.lower() == "n":
                try:
                    name, preset = new_preset(client, cfg, service, presets, ask, say)
                except ValueError as exc:
                    say("  " + paint("bad", f"✗ {exc}"))
                    continue
                presets[name] = preset
                rows.append((name, preset))
                choice, check = name, probe(client, cfg, service, preset)
            elif choice.isdigit() and 0 < int(choice) <= len(rows):
                check = checks[int(choice) - 1]
                choice = rows[int(choice) - 1][0]
            else:
                say("  " + paint("bad", "choose a number, n or q"))
                continue
            if choice == "(current settings)":
                if not check.ok and not _yes(ask, f"  {describe(check)}\n  Run anyway?"):
                    continue
                return cfg
            preset = presets[choice]
            if check.entry is None and check.suggest and \
                    _yes(ask, f"  {choice}: the server has no {preset['model']}. Use {check.suggest[0]}?", True):
                preset = save(choice, {**preset, "model": check.suggest[0]})
                presets[choice] = preset
                check = probe(client, cfg, service, preset)
                say(f"  {describe(check)}")
            if not check.ok and not _yes(ask, f"  {describe(check)}\n  Run anyway?"):
                continue
            if choice != current and _yes(ask, f"  Make {choice} the default for {service} (written to "
                                               f"{config.LOCAL_OVERRIDES.name})?"):
                config.save_services(cfg, {service: service_values(service, preset)})
            return apply(cfg, service, preset)


def choose(cfg, args) -> object:
    """cfg for a model-using command: --preset NAME, else the picker in a terminal, else as configured."""
    service = COMMAND_SERVICE.get(args.command)
    if service is None:
        return cfg
    name = getattr(args, "preset", None)
    if name:
        presets = load_all()
        if name not in presets:
            raise SystemExit(f"no preset {name!r}; presets: {', '.join(presets) or '(none)'}")
        return apply(cfg, service, presets[name])
    if getattr(args, "server", None) or getattr(args, "model", None) or getattr(args, "no_pick", False) \
            or not (sys.stdin.isatty() and sys.stdout.isatty()):
        return cfg
    global _color
    _color = not os.environ.get("NO_COLOR")
    try:
        return pick(cfg, service, args.command)
    except (Quit, KeyboardInterrupt):
        raise SystemExit(130) from None

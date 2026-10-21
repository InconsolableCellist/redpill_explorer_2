"""Be polite on a shared llama.cpp router: never take it from a model someone else is using.

The router holds one model at a time (models-max 1), and a request for another model unloads it. Two rules:
- may_start: a stage starts on its own only when its model is already loaded, or nothing is.
- Guard: a running stage stops (after the requests in flight) once another model has taken the router,
  instead of loading its own back. While its own model is loading or loaded, it carries on.
A server that is not a router (no per-model status) never blocks anything.
"""
from __future__ import annotations

import logging
import time

import httpx

from re2 import presets

log = logging.getLogger(__name__)

CHECK_EVERY_S = 30
BUSY = ("loaded", "loading")


def router_state(base_url: str, client_factory=httpx.Client) -> list[dict] | None:
    """The router's models with their state, or None when it does not answer or has no per-model state."""
    try:
        with client_factory() as client:
            models = presets.list_models(client, base_url)
    except (httpx.HTTPError, ValueError) as exc:
        log.debug("router %s: %s", base_url, exc)
        return None
    return models if any(m["state"] for m in models) else None


def others_using(models: list[dict] | None, model: str) -> list[str]:
    """Other models the router has loaded or is loading, when ours is neither."""
    if not models:
        return []
    ours = next((m["state"] for m in models if m["id"] == model), None)
    if ours in BUSY:
        return []
    return [m["id"] for m in models if m["id"] != model and m["state"] in BUSY]


def may_start(base_url: str, model: str, client_factory=httpx.Client) -> str | None:
    """None when model may be started on base_url now; otherwise why not."""
    models = router_state(base_url, client_factory)
    others = others_using(models, model)
    if others:
        return f"the router has {', '.join(others)} loaded; starting {model} would unload it"
    return None


class Guard:
    """runner.guard for one stage: a reason to stop once another model has the router, checked at most
    every CHECK_EVERY_S."""

    def __init__(self, base_url: str, model: str, client_factory=httpx.Client):
        self.base_url, self.model, self.client_factory = base_url, model, client_factory
        self.checked_at = 0.0
        self.reason: str | None = None

    def __call__(self) -> str | None:
        if time.monotonic() - self.checked_at >= CHECK_EVERY_S:
            self.checked_at = time.monotonic()
            others = others_using(router_state(self.base_url, self.client_factory), self.model)
            self.reason = f"{', '.join(others)} took the router from {self.model}" if others else None
        return self.reason

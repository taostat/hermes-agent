"""SayGM provider profile.

SayGM (https://saygm.com) is a multi-model inference gateway that fronts the
frontier labs (OpenAI, Anthropic, Gemini) plus confidential/open-weight TEE
SKUs. This profile consumes SayGM's **OpenAI-compatible chat surface**
(``POST /v1/chat/completions`` on ``https://api.saygm.com/v1`` with
``Authorization: Bearer SAYGM_API_KEY``).

Wire caveats verified live against the gateway (2026-08-07):

- The cc surface serves the OpenAI chat-model set only (gpt-5.4/5.5/5.6
  families, o3, o4-mini). Claude ids return ``unknown model`` — they are
  Anthropic-wire-only on SayGM (``/v1/messages``) and need a separate
  ``anthropic_messages`` profile; Gemini ids are Gemini-wire-only and were
  additionally marked ``available: false``. ``gpt-5.5-pro`` is Responses-only
  (``/v1/completions`` → 404). None of those belong on this cc profile.
- ``gpt-5.6-sol`` rejects function tools on cc unless ``reasoning_effort`` is
  explicitly ``"none"``. Hermes always sends tools on agent turns, so the
  profile pins ``reasoning_effort="none"`` for that one model (see
  :meth:`SayGMProfile.build_api_kwargs_extras`). Every other cc model either
  accepts tools without the field (gpt-5.4 family, o3, o4-mini) or rejects
  ``"none"`` outright (gpt-5.6-terra/luna, gpt-5.5), so the knob stays scoped.

The live model catalog at ``GET /v1/models`` is unauthenticated, so the
picker can probe it before the user has configured a key; the curated
:attr:`fallback_models` below is the cc-servable subset of the frontier set
that is currently available and works through this transport, and is what
offline / no-key flows show first.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

# Curated cc-servable frontier set (2026-08-07), verified to work through the
# OpenAI-compatible chat surface with function tools. Excludes Claude ids
# (Anthropic-wire-only), Gemini ids (Gemini-wire-only, plus available:false),
# and gpt-5.5-pro (Responses-only). Ordered with the default main model first.
SAYGM_CC_MODELS = (
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.5",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "o3",
    "o4-mini",
)


class SayGMProfile(ProviderProfile):
    """SayGM — OpenAI-compatible chat surface, cc-specific reasoning quirks."""

    # Per-model completion-token caps. SayGM's default (mirroring upstream) is
    # 128k for the gpt-5.x family, but it caps o3/o4-mini at 100,003
    # completion tokens and 400s anything larger ("max_tokens is too large").
    # Hermes then misclassifies that 400 as a context-overflow and, on a tiny
    # conversation, bails with "Context length exceeded (N tokens)". Pin the
    # cap below SayGM's limit so the request goes through.
    _MODEL_MAX_TOKENS: dict[str, int] = {
        "o3": 100000,
        "o4-mini": 100000,
    }

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Return only currently available Chat Completions models."""
        import json
        import urllib.request

        from hermes_cli.urllib_security import open_credentialed_url

        url = (base_url or self.base_url).rstrip("/") + "/models"
        req = urllib.request.Request(url)
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", _profile_user_agent())

        try:
            with open_credentialed_url(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode())
            items = payload if isinstance(payload, list) else payload.get("data", [])
            return [
                item["id"]
                for item in items
                if isinstance(item, dict)
                and isinstance(item.get("id"), str)
                and item.get("available") is True
                and isinstance(item.get("api_shapes"), list)
                and "chat.completions" in item["api_shapes"]
            ]
        except Exception:
            return None

    def get_max_tokens(self, model: str | None) -> int | None:
        cap = self._MODEL_MAX_TOKENS.get((model or "").strip().lower())
        if cap is not None:
            return cap
        return self.default_max_tokens

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        # SayGM's /v1/chat/completions rejects function tools for gpt-5.6-sol
        # unless reasoning_effort is "none" (HTTP 400 "Function tools with
        # reasoning_effort are not supported for gpt-5.6-sol in
        # /v1/chat/completions. To use function tools, use /v1/responses or
        # set reasoning_effort to 'none'."). Hermes always sends tools on
        # agent turns, so pin "none" for this one model — the tradeoff of
        # disabling its reasoning is required to use it agentically on cc.
        # The other cc models must NOT get the field: gpt-5.6-terra/luna,
        # gpt-5.5, o3 and o4-mini reject reasoning_effort="none" with a 400,
        # and gpt-5.4 family works without it.
        if (model or "").strip().lower() == "gpt-5.6-sol":
            top_level["reasoning_effort"] = "none"

        return extra_body, top_level


saygm = SayGMProfile(
    name="saygm",
    aliases=("saygm", "SayGM"),
    display_name="SayGM",
    description="SayGM — multi-model gateway (GPT, Claude, Gemini, open-weight)",
    signup_url="https://saygm.com",
    env_vars=("SAYGM_API_KEY",),
    base_url="https://api.saygm.com/v1",
    auth_type="api_key",
    # Frontier models cap output at 128k tokens; make that budget usable
    # instead of letting the endpoint truncate to its server default.
    default_max_tokens=128000,
    default_aux_model="gpt-5.4-mini",
    fallback_models=SAYGM_CC_MODELS,
)

register_provider(saygm)

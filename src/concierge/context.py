"""Agent context — pulled from LangSmith Context Hub at runtime.

The concierge's system prompt (``AGENTS.md``) lives in the LangSmith Context
Hub, NOT in this repo. By default the runtime pulls the commit promoted to the
``production`` environment tag (promote in the Context Hub UI, or via
``scripts/setup_context_hub.py``), so promoting a fix to Production is what
ships it — no code redeploy. Override the tag/commit with the
``CONTEXT_HUB_VERSION`` env var (e.g. ``staging``, a commit hash, or an empty
string to pull the latest commit).

Engine should look here to find where the agent's context comes from — but the
hub prompt itself is edited in the Context Hub UI. Local customer-access and
branch-location rules are applied to both the hub prompt and the offline fallback.
``concierge.prompts.SYSTEM_PROMPT`` is only the seed pushed to the hub and the
offline fallback used when the hub is unreachable or the tag doesn't exist yet;
it is not the runtime source of truth.
"""

from __future__ import annotations

import os

from langsmith import Client

CONTEXT_HUB_REPO = "banking-concierge-agent"

# Environment tag (or commit hash) the runtime resolves. Defaults to the
# promoted ``production`` commit; set CONTEXT_HUB_VERSION="" to pull the latest.
CONTEXT_HUB_VERSION = os.getenv("CONTEXT_HUB_VERSION", "production")


def get_prompt() -> str:
    """Return the hub prompt with local data-access rules, or the offline fallback."""
    from concierge.prompts import CUSTOMER_LOOKUP_RULES, SYSTEM_PROMPT

    try:
        version = CONTEXT_HUB_VERSION or None
        agent = Client().pull_agent(CONTEXT_HUB_REPO, version=version)
        content = getattr(agent.files["AGENTS.md"], "content", "")
        if content:
            if CUSTOMER_LOOKUP_RULES not in content:
                content += "\n\n" + CUSTOMER_LOOKUP_RULES
            return content
    except Exception:
        pass

    return SYSTEM_PROMPT

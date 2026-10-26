# Pre-workshop checklist

Goal: deploy the concierge once, so attendees can chat with it and see their own
traces in LangSmith with no setup on the day. Do all of this beforehand.

Traces are exported through the OpenTelemetry SDK to LangSmith (see
`.env.example`). All attendees share one deployment and one tracing project.

## 1. Deploy

Requires a LangSmith Plus plan or above and an API key for the target workspace.

```bash
npm --prefix frontend install && npm --prefix frontend run build   # UI is served from frontend/dist
uv run langgraph deploy
```

Or create the deployment in the LangSmith UI (Deployments > New) from your fork.

## 2. Deployment environment variables

Set these as secrets on the deployment. `.env` is not uploaded.

| Variable | Value |
|---|---|
| `LANGSMITH_API_KEY` | Shared service key. The agent uses it for the LLM gateway, Context Hub and embeddings. Do not hand this out. |
| `LANGSMITH_WORKSPACE_ID` | Workspace the deployment and Context Hub live in |
| `BASE_URL` | `https://gateway.smith.langchain.com/openai/v1` (or set `OPENAI_API_KEY` and leave this unset) |
| `CONCIERGE_MODEL` | Optional, defaults to `gpt-4o-mini` |
| `LANGSMITH_PROJECT` | Tracing project attendees will use, e.g. `banking-concierge` |
| `LANGSMITH_TRACING` | `true` |
| `LANGSMITH_OTEL_ENABLED` | `true` |
| `LANGSMITH_OTEL_ONLY` | `true` (drop this if the smoke test below shows missing or duplicate traces) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `https://api.smith.langchain.com/otel/v1/traces` |
| `OTEL_EXPORTER_OTLP_HEADERS` | `x-api-key=<key>,Langsmith-Project=<project>,X-Tenant-Id=<workspace id>` |

The OTel endpoint must be the full `/otel/v1/traces` URL, and the headers must
include `X-Tenant-Id` for org-scoped keys, or spans are dropped without an error.
`Langsmith-Project` must match `LANGSMITH_PROJECT`.

## 3. Seed Context Hub (new workspace only)

```bash
uv run python scripts/setup_context_hub.py
```

The agent pulls its system prompt from Context Hub. If the hub is empty or
unreachable it falls back to `src/concierge/prompts.py`.

## 4. Smoke test

1. Open `https://<deployment>.us.langgraph.app/app/` (Studio), send
   "What is the monthly fee on Everyday Checking?" and expect a grounded answer.
2. Open the tracing project in LangSmith and confirm the trace appears with
   the full tree (agent, ChatOpenAI, search_banking_docs).
3. Repeat with `{"user": "smoke-test"}` under the assistant's configurable
   settings, then filter the project on metadata `user = smoke-test`.
4. Confirm there are no duplicate traces for a single chat.
5. Optional: `uv run python scripts/load_generation.py --mode remote --n 30 --url <deployment>`
   to pre-seed the project with traces so it isn't empty when people arrive.

## 5. Give attendees access

- Invite every attendee to the workspace by email (or via org SSO) before the
  day, with at least a role that can view traces and use Studio.
- Send the Studio link: `https://<deployment>.us.langgraph.app/app/`. It signs
  in through LangSmith, so there are no keys or installs.
- The chat UI at `/concierge/` needs each person to paste their own LangSmith
  API key. Skip it unless it is part of the demo.

## 6. Attendee instructions (paste into the invite)

1. Open the Studio link and sign in to LangSmith.
2. In the assistant's configurable settings, set `user` to your name. This tags
   every run in your trace.
3. Chat with the agent.
4. In LangSmith, open project `<project>` and filter on metadata `user = <your name>`.

## Notes

- Attendees share one project and can see each other's traces.
- Traces land in the deployment's workspace, not in each attendee's own.
- `user` is read from `configurable` in `src/concierge/graph.py`
  (`_with_user_metadata`). Studio and the chat UI only set `configurable`, which
  LangGraph does not turn into metadata on its own.
- Check that `/concierge-api/feedback` is behind the same auth as the rest of the
  server before opening the deployment to a wider group.
- Dataset, Context Hub and project names are `banking-concierge-*`. Don't run
  other people's demos in the same workspace.

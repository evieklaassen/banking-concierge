"""Graph state schema for the concierge agent."""

from __future__ import annotations

from langgraph.graph import MessagesState


class ConciergeState(MessagesState):
    """Messages plus retrieval_calls, counting search_banking_docs invocations regardless of query count."""

    retrieval_calls: int

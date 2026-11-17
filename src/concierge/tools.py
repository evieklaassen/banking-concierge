"""Tools available to the Meridian National customer service concierge agent.

A few tools have deliberate rough edges so LangSmith Engine has something
to cluster after the load generator runs:

- search_banking_docs has a vague description so the model occasionally
  re-queries multiple times rephrasing
- customer-data tools reject IDs not supplied in the current representative request
- recent_transactions raises if the model passes a runaway limit
- find_branch raises on non-zip inputs
"""

from __future__ import annotations

import re
from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_core.tools import ToolException, tool
from langgraph.prebuilt import InjectedState

from concierge.mock_data import (
    BRANCHES,
    CUSTOMERS,
    TRANSACTIONS,
    find_branch_by_zip,
)
from concierge.retrieval import retrieve
from concierge.state import ConciergeState


@tool
def search_banking_docs(query: str, k: int = 4) -> str:
    """Search Meridian National banking documentation.

    Args:
        query: A natural-language search query.
        k: Number of relevant chunks to return. Defaults to 4.
    """
    chunks = retrieve(query, k=k)
    if not chunks:
        return "No relevant documentation found."
    blocks = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        blocks.append(f"[source: {source}]\n{chunk.page_content}")
    return "\n\n---\n\n".join(blocks)


def _current_request(state: ConciergeState) -> str:
    """Return only the latest representative request, excluding tool and model text."""
    request = next(
        (
            message
            for message in reversed(state["messages"])
            if isinstance(message, HumanMessage)
        ),
        None,
    )
    return (
        request.content
        if request is not None and isinstance(request.content, str)
        else ""
    )


def _validate_customer_id(customer_id: str, state: ConciergeState) -> None:
    """Require an unambiguous customer ID from the current representative request."""
    content = _current_request(state)
    supplied_ids = set(re.findall(r"(?<![\w-])CUST-[0-9]{4}(?![\w-])", content))
    if not re.fullmatch(r"CUST-[0-9]{4}", customer_id) or supplied_ids != {customer_id}:
        raise ToolException(
            "Customer ID is unverified. Ask the representative for a valid CUST-#### "
            "ID for the customer in the current request. Do not guess or try other IDs."
        )


@tool
def account_lookup(
    customer_id: str, state: Annotated[ConciergeState, InjectedState]
) -> dict:
    """Return accounts only for an explicit CUST-#### ID supplied by the representative for this customer in the current request; otherwise ask for it, never derive, guess, or enumerate IDs."""
    _validate_customer_id(customer_id, state)
    customer = CUSTOMERS.get(customer_id)
    if customer is None:
        raise ToolException(
            "Customer not found. Ask the representative for a valid CUST-#### ID "
            "for this customer. Do not guess or try other IDs."
        )
    return dict(customer)


@tool
def recent_transactions(
    customer_id: str, state: Annotated[ConciergeState, InjectedState], limit: int = 5
) -> list[dict]:
    """Return recent transactions for a representative-supplied CUST-#### ID for this customer in the current request; ask for the ID rather than guessing."""
    _validate_customer_id(customer_id, state)
    if limit <= 0:
        raise ValueError("limit must be positive")
    if limit > 50:
        raise ValueError(
            f"limit {limit} exceeds the maximum of 50. Pick a smaller number."
        )
    if customer_id not in CUSTOMERS:
        raise ToolException(
            "Customer not found. Ask the representative for a valid CUST-#### ID "
            "for this customer. Do not guess or try other IDs."
        )
    txs = TRANSACTIONS.get(customer_id, [])
    return [dict(t) for t in txs[:limit]]


@tool
def find_branch(zip_code: str, state: Annotated[ConciergeState, InjectedState]) -> dict:
    """Find a branch using a full five-digit ZIP; ask for incomplete ZIPs, disclose assumptions, and do not claim nearest without distance evidence."""
    supplied_zips = re.findall(r"(?<![\w-])[0-9]{5}(?![\w-])", _current_request(state))
    if not re.fullmatch(r"[0-9]{5}", zip_code) or zip_code not in supplied_zips:
        raise ToolException(
            "Ask the representative for the full 5-digit U.S. ZIP code."
        )
    branch = find_branch_by_zip(zip_code)
    if branch is None:
        return {
            "match": False,
            "message": "No branch found for that ZIP. The directory example is not a nearest-branch result; distance is unknown.",
            "nearest_known": BRANCHES[0],
        }
    return {
        "match": True,
        "message": "Directory match; distance and nearest branch are unknown.",
        **branch,
    }


@tool
def transfer_funds(from_account: str, to_account: str, amount: float) -> dict:
    """Initiate a transfer between two Meridian National accounts owned by the same customer.

    Args:
        from_account: The source account ID.
        to_account: The destination account ID.
        amount: The dollar amount to transfer.
    """
    if amount <= 0:
        raise ValueError("amount must be positive")
    confirmation = (
        f"MNB-XFER-{abs(hash((from_account, to_account, amount))) % 10_000_000:07d}"
    )
    return {
        "status": "submitted",
        "from_account": from_account,
        "to_account": to_account,
        "amount": round(amount, 2),
        "confirmation": confirmation,
        "estimated_post": "immediately",
    }


account_lookup.handle_tool_error = True
recent_transactions.handle_tool_error = True
find_branch.handle_tool_error = True

TOOLS = [
    search_banking_docs,
    account_lookup,
    recent_transactions,
    find_branch,
    transfer_funds,
]

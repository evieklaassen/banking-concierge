import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from concierge.mock_data import TRANSACTIONS
from concierge.state import ConciergeState
from concierge.tools import recent_transactions


class RecentTransactionsTests(unittest.TestCase):
    def test_default_and_ordinary_limits(self):
        for limit in (None, 1, 3):
            with self.subTest(limit=limit):
                args: dict[str, str | int] = {"customer_id": "CUST-0001"}
                if limit is not None:
                    args["limit"] = limit
                requested = 5 if limit is None else limit
                result = recent_transactions.invoke(args)
                self.assertEqual(
                    result,
                    {
                        "customer_id": "CUST-0001",
                        "requested": requested,
                        "returned": requested,
                        "max_limit": 50,
                        "transactions": TRANSACTIONS["CUST-0001"][:requested],
                    },
                )

    def test_maximum_and_oversized_requests(self):
        transactions = TRANSACTIONS["CUST-0001"] * 10
        with patch.dict(TRANSACTIONS, {"CUST-0001": transactions}):
            for limit in (50, 100):
                with self.subTest(limit=limit):
                    result = recent_transactions.invoke(
                        {"customer_id": "CUST-0001", "limit": limit}
                    )
                    self.assertEqual(result["requested"], limit)
                    self.assertEqual(result["returned"], 50)
                    self.assertEqual(result["max_limit"], 50)
                    self.assertEqual(result["transactions"], transactions[:50])

    def test_available_count_reaches_agent_through_tool_node(self):
        builder = StateGraph(ConciergeState)
        builder.add_node("tools", ToolNode([recent_transactions]))
        builder.add_edge(START, "tools")
        builder.add_edge("tools", END)
        graph = builder.compile()
        for limit in (50, 100):
            with self.subTest(limit=limit):
                output = graph.invoke(
                    ConciergeState(
                        retrieval_calls=0,
                        messages=[
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": "recent_transactions",
                                        "args": {
                                            "customer_id": "CUST-0001",
                                            "limit": limit,
                                        },
                                        "id": "transactions",
                                        "type": "tool_call",
                                    }
                                ],
                            )
                        ],
                    )
                )
                result = json.loads(output["messages"][-1].content)
                self.assertEqual(result["customer_id"], "CUST-0001")
                self.assertEqual(result["requested"], limit)
                self.assertEqual(result["returned"], 7)
                self.assertEqual(result["max_limit"], 50)
                self.assertEqual(result["transactions"], TRANSACTIONS["CUST-0001"])

    def test_nonpositive_limits_raise(self):
        for limit in (0, -1):
            with (
                self.subTest(limit=limit),
                self.assertRaisesRegex(ValueError, "limit must be positive"),
            ):
                recent_transactions.invoke({"customer_id": "CUST-0001", "limit": limit})


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from concierge.mock_data import TRANSACTIONS
from concierge.tools import recent_transactions


class RecentTransactionsTests(unittest.TestCase):
    def test_oversized_limit_is_capped(self):
        rows = TRANSACTIONS["CUST-0001"] * 10
        with patch.dict(TRANSACTIONS, {"CUST-0001": rows}):
            result = recent_transactions.invoke(
                {"customer_id": "CUST-0001", "limit": 100}
            )
        self.assertEqual(
            result,
            {
                "requested": 100,
                "returned": 50,
                "max_limit": 50,
                "transactions": rows[:50],
            },
        )

    def test_fewer_available_transactions(self):
        rows = TRANSACTIONS["CUST-0001"]
        for limit in (10, 50, 100):
            with self.subTest(limit=limit):
                result = recent_transactions.invoke(
                    {"customer_id": "CUST-0001", "limit": limit}
                )
                self.assertEqual(result["requested"], limit)
                self.assertEqual(result["transactions"], rows)
                self.assertEqual(result["returned"], len(rows))
                self.assertEqual(result["max_limit"], 50)

    def test_valid_limits_and_default_preserve_order(self):
        rows = TRANSACTIONS["CUST-0001"]
        for args, expected in (({}, 5), ({"limit": 1}, 1), ({"limit": 5}, 5)):
            with self.subTest(args=args):
                result = recent_transactions.invoke(
                    {"customer_id": "CUST-0001", **args}
                )
                self.assertEqual(result["requested"], expected)
                self.assertEqual(result["returned"], expected)
                self.assertEqual(result["transactions"], rows[:expected])
        description = recent_transactions.description
        self.assertIn("at most the requested number", description)
        self.assertIn("1-50 (default 5)", description)
        self.assertIn("Fewer are returned if the customer has fewer.", description)

    def test_no_transactions(self):
        with patch.dict(TRANSACTIONS, {"CUST-0001": []}):
            result = recent_transactions.invoke({"customer_id": "CUST-0001"})
        self.assertEqual(result["transactions"], [])
        self.assertEqual(result["returned"], 0)

    def test_invalid_inputs_still_raise(self):
        for limit in (0, -1):
            with (
                self.subTest(limit=limit),
                self.assertRaisesRegex(ValueError, "limit must be positive"),
            ):
                recent_transactions.invoke({"customer_id": "CUST-0001", "limit": limit})
        with self.assertRaisesRegex(ValueError, "No customer found"):
            recent_transactions.invoke({"customer_id": "unknown", "limit": 100})

    def test_tool_node_exposes_actual_count_to_agent(self):
        call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "recent_transactions",
                    "args": {"customer_id": "CUST-0001", "limit": 100},
                    "id": "transactions",
                    "type": "tool_call",
                }
            ],
        )
        builder = StateGraph(MessagesState)
        builder.add_node(
            "tools", ToolNode([recent_transactions], handle_tool_errors=True)
        )
        builder.add_edge(START, "tools")
        builder.add_edge("tools", END)
        result = builder.compile().invoke({"messages": [call]})
        message = result["messages"][-1]
        self.assertEqual(message.status, "success")
        payload = json.loads(message.content)
        self.assertEqual(payload["requested"], 100)
        self.assertEqual(payload["returned"], 7)
        self.assertEqual(payload["returned"], len(payload["transactions"]))
        self.assertEqual(payload["transactions"], TRANSACTIONS["CUST-0001"])


if __name__ == "__main__":
    unittest.main()

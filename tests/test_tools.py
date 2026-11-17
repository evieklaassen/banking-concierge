import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from concierge.mock_data import TRANSACTIONS
from concierge.tools import recent_transactions


class RecentTransactionsTests(unittest.TestCase):
    def test_requested_rows(self):
        result = recent_transactions.invoke({"customer_id": "CUST-0001", "limit": 3})
        self.assertEqual(
            result,
            {
                "customer_id": "CUST-0001",
                "requested": 3,
                "returned": 3,
                "max_limit": 50,
                "transactions": TRANSACTIONS["CUST-0001"][:3],
            },
        )

    def test_default_limit(self):
        result = recent_transactions.invoke({"customer_id": "CUST-0001"})
        self.assertEqual(result["requested"], 5)
        self.assertEqual(result["returned"], 5)
        self.assertEqual(result["transactions"], TRANSACTIONS["CUST-0001"][:5])

    def test_oversized_limit_is_capped(self):
        rows = TRANSACTIONS["CUST-0001"] * 10
        with patch.dict(TRANSACTIONS, {"CUST-0001": rows}):
            result = recent_transactions.invoke(
                {"customer_id": "CUST-0001", "limit": 100}
            )
        self.assertEqual(result["requested"], 100)
        self.assertEqual(result["returned"], 50)
        self.assertEqual(result["max_limit"], 50)
        self.assertEqual(result["transactions"], rows[:50])

    def test_fewer_rows_than_requested(self):
        result = recent_transactions.invoke({"customer_id": "CUST-0001", "limit": 50})
        self.assertEqual(result["returned"], len(TRANSACTIONS["CUST-0001"]))
        self.assertLess(result["returned"], result["requested"])
        self.assertEqual(result["transactions"], TRANSACTIONS["CUST-0001"])

    def test_no_transactions(self):
        with patch.dict(TRANSACTIONS, {"CUST-0001": []}):
            result = recent_transactions.invoke({"customer_id": "CUST-0001"})
        self.assertEqual(result["returned"], 0)
        self.assertEqual(result["transactions"], [])

    def test_non_positive_limit(self):
        for limit in (0, -1):
            with (
                self.subTest(limit=limit),
                self.assertRaisesRegex(ValueError, "^limit must be positive$"),
            ):
                recent_transactions.invoke({"customer_id": "CUST-0001", "limit": limit})

    def test_unknown_customer(self):
        with self.assertRaisesRegex(
            ValueError,
            "^No customer found with ID 'CUST-9999'. "
            "Customer IDs are in the format CUST-####.$",
        ):
            recent_transactions.invoke({"customer_id": "CUST-9999", "limit": 100})

    def test_tool_node_delivers_counts_without_error(self):
        builder = StateGraph(MessagesState)
        builder.add_node(
            "tools", ToolNode([recent_transactions], handle_tool_errors=True)
        )
        builder.add_edge(START, "tools")
        builder.add_edge("tools", END)
        result = builder.compile().invoke(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "recent_transactions",
                                "args": {"customer_id": "CUST-0001", "limit": 100},
                                "id": "transactions",
                            }
                        ],
                    )
                ]
            }
        )
        self.assertEqual(len(result["messages"]), 2)
        message = result["messages"][-1]
        self.assertEqual(message.status, "success")
        envelope = json.loads(message.content)
        self.assertEqual(envelope["requested"], 100)
        self.assertEqual(envelope["returned"], 7)
        self.assertEqual(len(envelope["transactions"]), envelope["returned"])


if __name__ == "__main__":
    unittest.main()

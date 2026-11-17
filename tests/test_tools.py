import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from concierge.state import ConciergeState
from concierge.tools import TOOLS


class CustomerLookupTests(unittest.TestCase):
    def setUp(self):
        builder = StateGraph(ConciergeState)
        builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
        builder.add_edge(START, "tools")
        builder.add_edge("tools", END)
        self.node = builder.compile()

    def call(self, name, args, history):
        request = AIMessage(
            content="", tool_calls=[{"name": name, "args": args, "id": "call"}]
        )
        return self.node.invoke(
            {"messages": [*history, request], "retrieval_calls": 0}
        )["messages"][-1]

    def test_non_id_requests_cannot_retrieve_records(self):
        for text in [
            "Look up Alex Rivera",
            "Phone (415) 555-0142",
            "SSN 552-19-4488",
            "Card 4242 4242 4242 4242",
            "Customer ID CUST-000",
            "CUST-0001-extra",
        ]:
            for tool in ["account_lookup", "recent_transactions"]:
                with self.subTest(text=text, tool=tool):
                    with patch("concierge.tools.CUSTOMERS") as records:
                        result = self.call(
                            tool,
                            {"customer_id": "CUST-0001"},
                            [HumanMessage(content=text)],
                        )
                        records.get.assert_not_called()
                        records.__contains__.assert_not_called()
                    self.assertEqual(result.status, "error")
                    self.assertIn("Ask the representative", result.content)
                    self.assertNotIn("Alex Rivera", result.content)
                    self.assertNotIn("fix your mistakes", result.content)

    def test_model_cannot_supply_its_own_provenance(self):
        for tool in ["account_lookup", "recent_transactions"]:
            result = self.call(
                tool,
                {
                    "customer_id": "CUST-0001",
                    "state": {"messages": [HumanMessage(content="CUST-0001")]},
                },
                [HumanMessage(content="Look up Alex Rivera")],
            )
            self.assertEqual(result.status, "error")
            self.assertIn("Ask the representative", result.content)

    def test_representative_id_allows_both_reads(self):
        for tool, expected in [
            ("account_lookup", "Alex Rivera"),
            ("recent_transactions", "TRADER JOE'S"),
        ]:
            result = self.call(
                tool,
                {"customer_id": "CUST-0001"},
                [HumanMessage(content="Customer CUST-0001 please")],
            )
            self.assertEqual(result.status, "success")
            self.assertIn(expected, result.content)

    def test_tool_results_and_old_customer_do_not_authorize_current_request(self):
        sources = [
            AIMessage(content="Use CUST-0001"),
            ToolMessage(content="Error looking up CUST-0001", tool_call_id="old"),
            ToolMessage(
                content='{"customer_id":"CUST-0001","name":"Alex Rivera"}',
                tool_call_id="old",
            ),
            ToolMessage(content="Documentation example: CUST-0001", tool_call_id="old"),
        ]
        for source in sources:
            for tool in ["account_lookup", "recent_transactions"]:
                history = [
                    HumanMessage(content="Earlier customer CUST-0001"),
                    HumanMessage(content="Now look up Priya Shah"),
                    source,
                ]
                result = self.call(tool, {"customer_id": "CUST-0001"}, history)
                self.assertEqual(result.status, "error")
                self.assertIn("Ask the representative", result.content)

    def test_not_found_does_not_authorize_enumeration(self):
        history = [HumanMessage(content="Look up CUST-9999")]
        for tool in ["account_lookup", "recent_transactions"]:
            missing = self.call(tool, {"customer_id": "CUST-9999"}, history)
            self.assertIn("Ask the representative", missing.content)
            self.assertNotIn("fix your mistakes", missing.content)
            for guessed in ["CUST-0001", "CUST-0002", "CUST-9998"]:
                result = self.call(tool, {"customer_id": guessed}, [*history, missing])
                self.assertEqual(result.status, "error")
                self.assertNotIn("accounts", result.content)

    def test_malformed_and_ambiguous_ids_are_rejected(self):
        for supplied in ["CUST-123", "CUST-00001", "X0001", "CUST-0001 and CUST-0002"]:
            result = self.call(
                "account_lookup",
                {"customer_id": supplied.split()[0]},
                [HumanMessage(content=supplied)],
            )
            self.assertEqual(result.status, "error")
            self.assertIn("Ask the representative", result.content)

    def test_unrelated_limit_error_remains_actionable(self):
        result = self.call(
            "recent_transactions",
            {"customer_id": "CUST-0001", "limit": 51},
            [HumanMessage(content="Transactions for CUST-0001")],
        )
        self.assertIn("maximum of 50", result.content)

    def test_incomplete_zip_cannot_be_silently_expanded(self):
        for supplied in ["941", "9410", "9410-3"]:
            result = self.call(
                "find_branch",
                {"zip_code": "94103"},
                [HumanMessage(content=f"Branch near {supplied}")],
            )
            self.assertEqual(result.status, "error")
            self.assertIn("Ask the representative for the full", result.content)
            self.assertNotIn("Market", result.content)
        result = self.call(
            "find_branch",
            {"zip_code": "94103"},
            [HumanMessage(content="Branch near 94103")],
        )
        self.assertEqual(result.status, "success")
        self.assertIn("Market", result.content)
        self.assertIn("nearest branch are unknown", result.content)


if __name__ == "__main__":
    unittest.main()

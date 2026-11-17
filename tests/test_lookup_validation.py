import importlib
import json
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage

with patch("langsmith.Client.pull_agent", side_effect=RuntimeError("offline")):
    graph_module = importlib.import_module("concierge.graph")

from concierge.mock_data import BRANCHES, CUSTOMERS
from concierge.tools import account_lookup, find_branch


class LookupValidationTests(unittest.TestCase):
    def test_invalid_customer_ids_do_not_access_records(self):
        for value in (
            "552-19-4488",
            "(415) 555-0142",
            "4242424242424242",
            "Alex Rivera",
            "0001",
            "CUST-001",
            "CUST-00001",
            "cust-0001",
            "CUST-０００１",
            "CUST-0001\n",
            "XCUST-0001",
            "",
        ):
            with (
                self.subTest(value=value),
                patch("concierge.tools.CUSTOMERS", {value: {"name": "Wrong record"}}),
            ):
                result = account_lookup.invoke({"customer_id": value})
                self.assertEqual(result["status"], "invalid_customer_id")
                self.assertFalse(result["retryable"])
                self.assertIn("Ask the representative", result["message"])
                self.assertNotIn("name", result)

    def test_valid_customer_lookup_is_unchanged(self):
        for customer_id, customer in CUSTOMERS.items():
            with self.subTest(customer_id=customer_id):
                self.assertEqual(
                    account_lookup.invoke({"customer_id": customer_id}), customer
                )

    def test_invalid_zips_do_not_return_a_branch(self):
        for value in ("9410", "941030", "9410a", "９４１０３", "94103\n", " 94103", ""):
            with (
                self.subTest(value=value),
                patch("concierge.tools.find_branch_by_zip", return_value=BRANCHES[0]),
            ):
                result = find_branch.invoke({"zip_code": value})
                self.assertEqual(result["status"], "invalid_zip")
                self.assertFalse(result["retryable"])
                self.assertIn("Ask the representative", result["message"])
                self.assertNotIn("address", result)

    def test_valid_branch_lookup_is_unchanged(self):
        for branch in BRANCHES:
            with self.subTest(zip_code=branch["zip_code"]):
                self.assertEqual(
                    find_branch.invoke({"zip_code": branch["zip_code"]}),
                    {"match": True, **branch},
                )
        self.assertFalse(find_branch.invoke({"zip_code": "00000"})["match"])

    def tool_result(self, name, args):
        model = Mock()
        model.invoke.side_effect = [
            AIMessage(
                content="", tool_calls=[{"name": name, "args": args, "id": "lookup"}]
            ),
            AIMessage(content="Please provide the correct lookup value."),
        ]
        with patch.object(graph_module, "_make_model", return_value=model):
            result = graph_module.graph.invoke(
                {"messages": [("user", "Look up this value")]}
            )
        return result["messages"][-2]

    def test_validation_results_reach_agent_without_generic_retry_instruction(self):
        for name, args, status in (
            ("account_lookup", {"customer_id": "Alex Rivera"}, "invalid_customer_id"),
            ("find_branch", {"zip_code": "9410"}, "invalid_zip"),
        ):
            with self.subTest(name=name):
                result = self.tool_result(name, args)
                content = json.loads(result.content)
                self.assertEqual(content["status"], status)
                self.assertFalse(content["retryable"])
                self.assertIn("Ask the representative", content["message"])

    def test_lookup_errors_request_correction_without_guessing(self):
        for args in ({"customer_id": "CUST-9999"}, {"customer_id": 1234}):
            with self.subTest(args=args):
                result = self.tool_result("account_lookup", args)
                self.assertEqual(result.status, "error")
                self.assertIn(
                    "ask the representative for the correct value", result.content
                )
                self.assertIn("Never guess", result.content)
                self.assertIn("does not establish identity", result.content)

    def test_service_error_does_not_suggest_changing_lookup_value(self):
        with patch(
            "concierge.tools.find_branch_by_zip", side_effect=RuntimeError("offline")
        ):
            result = self.tool_result("find_branch", {"zip_code": "94103"})
        self.assertEqual(result.status, "error")
        self.assertIn("without changing the lookup value", result.content)


if __name__ == "__main__":
    unittest.main()

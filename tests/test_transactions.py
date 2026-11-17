import importlib
import unittest
from datetime import UTC, datetime
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage

from concierge.mock_data import TRANSACTIONS
from concierge.prompts import SYSTEM_PROMPT
from concierge.tools import recent_transactions


class RecentTransactionsTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"date": day, "amount": -10, "type": "debit", "description": day}
            for day in (
                "2026-05-21",
                "2026-05-20",
                "2026-05-19",
                "2026-05-18",
                "2026-05-17",
            )
        ]
        patcher = patch.dict(TRANSACTIONS, {"CUST-0001": self.rows})
        patcher.start()
        self.addCleanup(patcher.stop)

    def lookup(self, **kwargs):
        return recent_transactions.invoke({"customer_id": "CUST-0001", **kwargs})

    def test_both_bounds_are_inclusive_and_filter_before_limit(self):
        bounds = {"start_date": "2026-05-18", "end_date": "2026-05-20"}
        self.assertEqual(self.lookup(**bounds), self.rows[1:4])
        self.assertEqual(self.lookup(limit=2, **bounds), self.rows[1:3])

    def test_start_only_is_inclusive(self):
        self.assertEqual(self.lookup(start_date="2026-05-19"), self.rows[:3])

    def test_end_only_is_inclusive(self):
        self.assertEqual(self.lookup(end_date="2026-05-19"), self.rows[2:])

    def test_omitted_bounds_preserve_order_and_limit(self):
        self.assertEqual(self.lookup(), self.rows)
        self.assertEqual(self.lookup(limit=2), self.rows[:2])
        result = self.lookup(limit=1)
        result[0]["amount"] = 0
        self.assertEqual(self.rows[0]["amount"], -10)

    def test_no_matching_transactions(self):
        self.assertEqual(
            self.lookup(start_date="2026-11-16", end_date="2026-11-17"), []
        )

    def test_invalid_dates_and_reversed_range(self):
        for bounds in (
            {"start_date": "2026-02-30"},
            {"end_date": "not-a-date"},
            {"start_date": "2026-05-20", "end_date": "2026-05-18"},
        ):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                self.lookup(**bounds)

    def test_existing_limit_and_customer_validation(self):
        for limit in (0, 51):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.lookup(limit=limit)
        with self.assertRaises(ValueError):
            recent_transactions.invoke({"customer_id": "missing"})


class AgentDateTests(unittest.TestCase):
    def test_current_utc_date_refreshes_on_each_model_call(self):
        with patch("concierge.context.get_prompt", return_value=SYSTEM_PROMPT):
            graph_module = importlib.import_module("concierge.graph")
        response = AIMessage(content="Ready")
        model = Mock()
        model.invoke.return_value = response
        question = HumanMessage(content="What did CUST-0002 spend this week?")
        with (
            patch.object(graph_module, "SYSTEM_PROMPT", SYSTEM_PROMPT),
            patch.object(graph_module, "_make_model", return_value=model),
            patch.object(graph_module, "datetime") as clock,
        ):
            clock.now.side_effect = [
                datetime(2026, 11, 17, 23, 59, tzinfo=UTC),
                datetime(2026, 11, 18, 0, 1, tzinfo=UTC),
            ]
            for day in ("2026-11-17", "2026-11-18"):
                result = graph_module.agent_node({"messages": [question]})
                messages = model.invoke.call_args.args[0]
                self.assertIn(f"Current UTC date: {day}", messages[0].content)
                self.assertIn(SYSTEM_PROMPT, messages[0].content)
                self.assertEqual(messages[1:], [question])
                self.assertEqual(result["messages"], [response])
            self.assertEqual(clock.now.call_args.args, (UTC,))


if __name__ == "__main__":
    unittest.main()

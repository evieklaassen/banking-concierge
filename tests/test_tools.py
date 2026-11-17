import json
import unittest
from copy import deepcopy
from unittest.mock import patch

from concierge.mock_data import CUSTOMERS
from concierge.tools import account_lookup


class AccountLookupTests(unittest.TestCase):
    def test_lookup_retains_allowed_details_and_masks_identifiers(self):
        for customer_id, customer in CUSTOMERS.items():
            with self.subTest(customer_id=customer_id):
                result = account_lookup.invoke({"customer_id": customer_id})
                self.assertEqual(
                    set(result),
                    {
                        "customer_id",
                        "name",
                        "phone",
                        "email",
                        "ssn_last4",
                        "accounts",
                        "credit_cards",
                    },
                )
                for field in ("customer_id", "name", "phone", "email", "accounts"):
                    self.assertEqual(result[field], customer[field])
                self.assertEqual(result["ssn_last4"], customer["ssn"][-4:])
                self.assertEqual(
                    len(result["credit_cards"]), len(customer["credit_cards"])
                )
                for actual, card in zip(
                    result["credit_cards"], customer["credit_cards"]
                ):
                    self.assertEqual(
                        actual,
                        {
                            "brand": card["brand"],
                            "last4": card["number"].replace(" ", "")[-4:],
                            "exp": card["exp"],
                        },
                    )
                serialized = json.dumps(result)
                self.assertNotIn(customer["ssn"], serialized)
                self.assertNotIn('"ssn"', serialized)
                self.assertNotIn('"number"', serialized)
                self.assertNotIn('"cvv"', serialized)
                for card in customer["credit_cards"]:
                    self.assertNotIn(card["number"], serialized)
                    self.assertNotIn(card["number"].replace(" ", ""), serialized)

    def test_card_last_four_normalizes_spaces_without_changing_source(self):
        for number in ("1234567890123456", "1234 5678 9012 3 4 5 6 "):
            with self.subTest(number=number):
                customer = deepcopy(CUSTOMERS["CUST-0001"])
                customer["credit_cards"][0]["number"] = number
                original = deepcopy(customer)
                with patch.dict(CUSTOMERS, {"CUST-0001": customer}):
                    result = account_lookup.invoke({"customer_id": "CUST-0001"})
                self.assertEqual(
                    result["credit_cards"],
                    [{"brand": "Visa", "last4": "3456", "exp": "08/29"}],
                )
                self.assertEqual(customer, original)

    def test_lookup_errors_do_not_return_records(self):
        with self.assertRaises(ValueError):
            account_lookup.invoke({"customer_id": "unknown"})
        with self.assertRaises(RuntimeError):
            account_lookup.invoke({"customer_id": "X-0001"})


if __name__ == "__main__":
    unittest.main()

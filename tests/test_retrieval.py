import importlib
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from concierge.tools import search_banking_docs


class SearchBankingDocsTests(unittest.TestCase):
    def test_overlapping_queries_preserve_first_seen_order(self):
        first = Document(
            page_content="Monthly fee is $10.", metadata={"source": "fees.md"}
        )
        second = Document(
            page_content="Waived with $500 in deposits.",
            metadata={"source": "checking.md"},
        )
        third = Document(
            page_content="Transfers post immediately.",
            metadata={"source": "transfers.md"},
        )
        duplicate = Document(page_content=first.page_content, metadata=first.metadata)
        with patch(
            "concierge.tools.retrieve",
            side_effect=[[first, second], [duplicate, third, second]],
        ):
            result = search_banking_docs.invoke(
                {"queries": ["checking fees", "checking transfers"]}
            )
        self.assertEqual(
            result,
            "[source: fees.md]\nMonthly fee is $10.\n\n---\n\n[source: checking.md]\nWaived with $500 in deposits.\n\n---\n\n[source: transfers.md]\nTransfers post immediately.",
        )

    def test_deduplication_requires_matching_source_and_content(self):
        chunks = [
            Document(page_content="Shared text", metadata={"source": "first.md"}),
            Document(page_content="Shared text", metadata={"source": "second.md"}),
            Document(page_content="Other text", metadata={"source": "first.md"}),
        ]
        with patch("concierge.tools.retrieve", return_value=chunks):
            result = search_banking_docs.invoke({"queries": ["fees"]})
        self.assertEqual(
            result,
            "[source: first.md]\nShared text\n\n---\n\n[source: second.md]\nShared text\n\n---\n\n[source: first.md]\nOther text",
        )

    def test_single_query_defaults_to_four_chunks_and_source_blocks(self):
        chunks = [
            Document(page_content=str(i), metadata={"source": "fees.md"})
            for i in range(5)
        ]
        with patch("concierge.tools.retrieve", side_effect=lambda query, k: chunks[:k]):
            result = search_banking_docs.invoke({"queries": ["fees"]})
        self.assertEqual(
            result,
            "[source: fees.md]\n0\n\n---\n\n[source: fees.md]\n1\n\n---\n\n[source: fees.md]\n2\n\n---\n\n[source: fees.md]\n3",
        )

    def test_disjoint_queries_preserve_query_and_retrieval_order(self):
        results = {
            "fees": [Document(page_content="fee 1"), Document(page_content="fee 2")],
            "transfers": [
                Document(page_content="transfer 1"),
                Document(page_content="transfer 2"),
            ],
        }
        with patch(
            "concierge.tools.retrieve", side_effect=lambda query, k: results[query][:k]
        ):
            result = search_banking_docs.invoke(
                {"queries": ["fees", "transfers"], "k": 2}
            )
        self.assertEqual(
            result,
            "[source: unknown]\nfee 1\n\n---\n\n[source: unknown]\nfee 2\n\n---\n\n[source: unknown]\ntransfer 1\n\n---\n\n[source: unknown]\ntransfer 2",
        )

    def test_empty_results(self):
        with patch("concierge.tools.retrieve", return_value=[]):
            self.assertEqual(
                search_banking_docs.invoke({"queries": ["missing"]}),
                "No relevant documentation found.",
            )
            self.assertEqual(
                search_banking_docs.invoke({"queries": []}),
                "No relevant documentation found.",
            )

    def test_graph_passes_unique_context_and_counts_tool_invocations(self):
        with patch(
            "concierge.context.get_prompt",
            return_value="Answer using retrieved documentation.",
        ):
            graph_module = importlib.import_module("concierge.graph")
        with patch("langchain_openai.ChatOpenAI"):
            evaluators = importlib.import_module("evals.evaluators")
        fee = Document(
            page_content="Monthly fee is $10.", metadata={"source": "checking.md"}
        )
        waiver = Document(
            page_content="Waived with $500 in deposits.",
            metadata={"source": "checking.md"},
        )
        model = Mock()

        def respond(messages):
            if not isinstance(messages[-1], ToolMessage):
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_banking_docs",
                            "args": {
                                "queries": ["checking fee", "checking fee waiver"]
                            },
                            "id": "docs",
                            "type": "tool_call",
                        }
                    ],
                )
            self.assertEqual(
                messages[-1].content,
                "[source: checking.md]\nMonthly fee is $10.\n\n---\n\n[source: checking.md]\nWaived with $500 in deposits.",
            )
            return AIMessage(
                content="Monthly fee is $10. Waived with $500 in deposits."
            )

        model.invoke.side_effect = respond
        with (
            patch.object(graph_module, "_make_model", return_value=model),
            patch("concierge.tools.retrieve", side_effect=[[fee], [fee, waiver]]),
        ):
            result = graph_module.graph.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content="What is the checking fee and how is it waived?"
                        )
                    ],
                    "retrieval_calls": 2,
                }
            )
        self.assertEqual(result["retrieval_calls"], 3)
        self.assertEqual(
            result["messages"][-1].content,
            "Monthly fee is $10. Waived with $500 in deposits.",
        )
        self.assertEqual(
            evaluators._context_for_hallucination(result["messages"]),
            result["messages"][-2].content,
        )
        reference = evaluators._synthesize_reference_trajectory(
            "What is the checking fee and how is it waived?",
            ["search_banking_docs"],
            result["messages"][-1].content,
        )
        self.assertEqual(len(reference[1]["tool_calls"]), 1)
        self.assertIn('"queries":', evaluators._render_trajectory(result["messages"]))

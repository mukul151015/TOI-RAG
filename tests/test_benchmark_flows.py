import unittest
from contextlib import ExitStack
from unittest.mock import patch

from app.schemas import QueryResponse, RoutedQuery
from app.services import chat_service, query_router


def _article(
    *,
    article_id: str,
    headline: str,
    edition: str,
    section: str,
    excerpt: str,
    issue_date: str = "2026-03-11",
) -> dict:
    return {
        "id": int(article_id),
        "external_article_id": article_id,
        "headline": headline,
        "edition": edition,
        "section": section,
        "issue_date": issue_date,
        "excerpt": excerpt,
        "matched_chunk": excerpt,
        "similarity": 0.82,
    }


class QueryRouterBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.publications = [
            {"id": "delhi", "publication_name": "TOIDelhiBS - Toi_Noida_Digital"},
            {"id": "mumbai", "publication_name": "TOIMumbaiBS - MumbaiCity_Digital"},
            {"id": "kolkata", "publication_name": "TOIKolkataBS - KolkataCity_Digital"},
        ]
        self.sections = ["FrontPage", "Sports", "World", "Edit", "Oped", "Business", "Nation", "City"]

    def _route(self, question: str):
        with (
            patch("app.services.query_router.fetch_publication_catalog", return_value=self.publications),
            patch("app.services.query_router.fetch_section_catalog", return_value=self.sections),
        ):
            return query_router.route_query(question, "2026-03-11")

    def test_sql_prompt_routes_to_sql_with_ambiguous_delhi_family(self):
        routed = self._route("Show me all articles published in the Delhi edition on March 11")
        self.assertEqual(routed.mode, "sql")
        self.assertEqual(routed.edition, "Delhi")

    def test_section_count_prompt_routes_to_sql(self):
        routed = self._route("Which sections had the most articles on March 11?")
        self.assertEqual(routed.mode, "sql")
        self.assertIsNone(routed.section)

    def test_list_sports_prompt_routes_to_sql_with_sports_section(self):
        routed = self._route("List all articles from the Sports section")
        self.assertEqual(routed.mode, "sql")
        self.assertEqual(routed.section, "Sports")

    def test_iran_war_prompt_routes_to_semantic(self):
        routed = self._route("Find articles related to the Iran war")
        self.assertEqual(routed.mode, "semantic")
        self.assertIsNone(routed.section)

    def test_world_cup_prompt_routes_to_hybrid_and_sports(self):
        routed = self._route("Which stories covered India's World Cup win?")
        self.assertEqual(routed.mode, "hybrid")
        self.assertEqual(routed.section, "Sports")

    def test_budget_prompt_routes_to_hybrid_business(self):
        routed = self._route("What was written about the budget impact on the middle class?")
        self.assertEqual(routed.mode, "hybrid")
        self.assertEqual(routed.section, "Business")

    def test_iran_mumbai_prompt_routes_to_hybrid(self):
        routed = self._route("Find articles about the Iran conflict published in the Mumbai edition")
        self.assertEqual(routed.mode, "hybrid")
        self.assertEqual(routed.edition, "TOIMumbaiBS - MumbaiCity_Digital")

    def test_sports_world_cup_prompt_routes_to_hybrid(self):
        routed = self._route("Which Sports section articles covered India winning the World Cup?")
        self.assertEqual(routed.mode, "hybrid")
        self.assertEqual(routed.section, "Sports")

    def test_editorial_prompt_routes_to_edit_section(self):
        routed = self._route("Show me opinion pieces from the Editorial section that discuss geopolitical tensions")
        self.assertEqual(routed.mode, "hybrid")
        self.assertEqual(routed.section, "Edit")


class ChatBenchmarkTests(unittest.TestCase):
    def _route_for(self, *, mode: str, edition: str | None = None, section: str | None = None, semantic_query: str | None = None):
        return RoutedQuery(
            mode=mode,
            issue_date="2026-03-11",
            edition=edition,
            section=section,
            semantic_query=semantic_query,
        )

    def _catalog_publications(self):
        return [
            {"id": "delhi", "publication_name": "TOIDelhiBS - Toi_Noida_Digital"},
            {"id": "mumbai", "publication_name": "TOIMumbaiBS - MumbaiCity_Digital"},
            {"id": "kolkata", "publication_name": "TOIKolkataBS - KolkataCity_Digital"},
        ]

    def _answer(self, question: str, query_response: QueryResponse, *, routed: RoutedQuery, history=None, session_context=None, extra_patches=None):
        extra_patches = extra_patches or []
        with ExitStack() as stack:
            stack.enter_context(patch("app.services.chat_service.route_query", return_value=routed))
            stack.enter_context(patch("app.services.chat_service.run_query", return_value=query_response))
            stack.enter_context(patch("app.services.chat_service.fetch_publication_catalog", return_value=self._catalog_publications()))
            for item in extra_patches:
                stack.enter_context(item)
            return chat_service.answer_question(
                question,
                "2026-03-11",
                10,
                history=history,
                session_context=session_context,
            )

    def test_count_question_uses_exact_sql_count_and_no_citations(self):
        query_response = QueryResponse(
            mode="sql",
            filters={"issue_date": "2026-03-11", "edition": "TOIDelhiBS - Toi_Noida_Digital"},
            results=[],
        )
        response = self._answer(
            "How many articles were there in the Delhi edition?",
            query_response,
            routed=self._route_for(mode="sql", edition="TOIDelhiBS - Toi_Noida_Digital"),
            extra_patches=[patch("app.services.chat_service.fetch_sql_article_count", return_value=171)],
        )
        self.assertIn("171", response.answer)
        self.assertEqual(response.citations, [])

    def test_ambiguous_delhi_listing_returns_clarification_instead_of_noida(self):
        query_response = QueryResponse(
            mode="sql",
            filters={"issue_date": "2026-03-11", "edition": "Delhi"},
            results=[],
        )
        response = self._answer(
            "Show me all articles published in the Delhi edition on March 11",
            query_response,
            routed=self._route_for(mode="sql", edition="Delhi"),
            extra_patches=[
                patch(
                    "app.services.chat_service.fetch_matching_publications",
                    return_value=[
                        {"publication_name": "TOIDelhiBS - Agra_Digital", "article_count": 134},
                        {"publication_name": "TOIDelhiBS - Bareilly_Digital", "article_count": 134},
                        {"publication_name": "TOIDelhiBS - Dehradun_Digital", "article_count": 132},
                        {"publication_name": "TOIDelhiBS - Toi_Noida_Digital", "article_count": 171},
                    ],
                )
            ],
        )
        self.assertIn("I don't see a single exact edition named Delhi", response.answer)
        self.assertIn("TOIDelhiBS - Toi_Noida_Digital", response.answer)

    def test_section_count_answer_names_top_section_first(self):
        query_response = QueryResponse(
            mode="sql",
            filters={"issue_date": "2026-03-11"},
            results=[
                {"section": "Nation", "article_count": 1392},
                {"section": "City", "article_count": 1314},
            ],
        )
        response = self._answer(
            "Which sections had the most articles on March 11?",
            query_response,
            routed=self._route_for(mode="sql"),
        )
        self.assertIn("Nation had the most articles", response.answer)
        self.assertIn("City: 1314", response.answer)

    def test_explicit_article_list_returns_requested_count(self):
        rows = [
            _article(
                article_id=str(index),
                headline=f"Sports Story {index}",
                edition="TOIMumbaiBS - MumbaiCity_Digital",
                section="Sports",
                excerpt="India lifted the World Cup title.",
            )
            for index in range(1, 7)
        ]
        query_response = QueryResponse(mode="sql", filters={"section": "Sports"}, results=rows)
        response = self._answer(
            "Show me 5 articles from the Sports section",
            query_response,
            routed=self._route_for(mode="sql", section="Sports"),
        )
        self.assertIn("Here are 5 worth looking at.", response.answer)
        self.assertEqual(len(response.citations), 5)

    def test_semantic_iran_prompt_returns_summary_without_citations(self):
        query_response = QueryResponse(
            mode="semantic",
            filters={"issue_date": "2026-03-11"},
            results=[
                _article(
                    article_id="1",
                    headline="Iran launches new attacks targeting Israel, Gulf nations",
                    edition="TOIMumbaiBS - MumbaiCity_Digital",
                    section="World",
                    excerpt="Iran escalated the conflict with fresh attacks across the region.",
                ),
                _article(
                    article_id="2",
                    headline="War brings new water crises to parched Iran",
                    edition="TOIDelhiBS - Toi_Noida_Digital",
                    section="World",
                    excerpt="The war deepened Iran's water shortage and humanitarian strain.",
                ),
            ],
        )
        response = self._answer(
            "Find articles related to the Iran war",
            query_response,
            routed=self._route_for(mode="semantic", semantic_query="iran war"),
            extra_patches=[
                patch(
                    "app.services.chat_service.chat_completion",
                    return_value="The coverage focuses on military escalation and humanitarian fallout in Iran.",
                )
            ],
        )
        self.assertIn("military escalation", response.answer)
        self.assertEqual(response.citations, [])

    def test_world_cup_story_prompt_summarizes_not_world_section_noise(self):
        query_response = QueryResponse(
            mode="hybrid",
            filters={"issue_date": "2026-03-11", "section": "Sports"},
            results=[
                _article(
                    article_id="10",
                    headline="BCCI announces 131 cr reward for World Champions",
                    edition="TOIKolkataBS - KolkataCity_Digital",
                    section="Sports",
                    excerpt="BCCI announced a cash reward after India's World Cup triumph.",
                ),
                _article(
                    article_id="11",
                    headline="BCCI announces 131 cr cash reward for entire WT20 winning squad",
                    edition="TOIMumbaiBS - MumbaiCity_Digital",
                    section="Sports",
                    excerpt="India's World Cup-winning squad was rewarded by the board.",
                ),
            ],
        )
        response = self._answer(
            "Which stories covered India's World Cup win?",
            query_response,
            routed=self._route_for(mode="hybrid", section="Sports", semantic_query="india world cup win"),
            extra_patches=[
                patch(
                    "app.services.chat_service.chat_completion",
                    return_value="The main story is the BCCI reward announcement following India's World Cup win.",
                )
            ],
        )
        self.assertIn("BCCI reward announcement", response.answer)
        self.assertEqual(response.citations, [])

    def test_budget_prompt_returns_summary(self):
        query_response = QueryResponse(
            mode="semantic",
            filters={"issue_date": "2026-03-11"},
            results=[
                _article(
                    article_id="20",
                    headline="Middle class braces for budget pinch",
                    edition="TOIMumbaiBS - MumbaiCity_Digital",
                    section="Business",
                    excerpt="Households weighed the tax and price implications of the budget.",
                ),
            ],
        )
        response = self._answer(
            "What was written about the budget impact on the middle class?",
            query_response,
            routed=self._route_for(mode="semantic", semantic_query="budget middle class"),
            extra_patches=[
                patch(
                    "app.services.chat_service.chat_completion",
                    return_value="The coverage focused on how the budget could pressure household spending and savings.",
                )
            ],
        )
        self.assertIn("household spending", response.answer)

    def test_hybrid_mumbai_iran_prompt_returns_summary_without_edition_clutter(self):
        query_response = QueryResponse(
            mode="hybrid",
            filters={"issue_date": "2026-03-11", "edition": "TOIMumbaiBS - MumbaiCity_Digital"},
            results=[
                _article(
                    article_id="30",
                    headline="Iran launches new attacks targeting Israel, Gulf nations",
                    edition="TOIMumbaiBS - MumbaiCity_Digital",
                    section="World",
                    excerpt="The piece tracks the latest regional escalation around Iran.",
                ),
            ],
        )
        response = self._answer(
            "Find articles about the Iran conflict published in the Mumbai edition",
            query_response,
            routed=self._route_for(
                mode="hybrid",
                edition="TOIMumbaiBS - MumbaiCity_Digital",
                semantic_query="iran conflict",
            ),
            extra_patches=[
                patch(
                    "app.services.chat_service.chat_completion",
                    return_value="Mumbai-edition coverage focused on Iran's latest escalation and regional fallout.",
                )
            ],
        )
        self.assertIn("regional fallout", response.answer)
        self.assertEqual(response.citations, [])

    def test_editorial_geopolitical_prompt_returns_summary(self):
        query_response = QueryResponse(
            mode="hybrid",
            filters={"issue_date": "2026-03-11", "section": "Edit"},
            results=[
                _article(
                    article_id="40",
                    headline="Five Ways The Iran War Can End",
                    edition="TOIMumbaiBS - MumbaiCity_Digital",
                    section="Edit",
                    excerpt="The piece examines the strategic endgames around the Iran war.",
                ),
                _article(
                    article_id="41",
                    headline="World’s Newest Beijing Problem",
                    edition="TOIDelhiBS - Toi_Noida_Digital",
                    section="Edit",
                    excerpt="The editorial argues that China's political ambitions create a fresh geopolitical fault line.",
                ),
            ],
        )
        response = self._answer(
            "Show me opinion pieces from the Editorial section that discuss geopolitical tensions",
            query_response,
            routed=self._route_for(mode="hybrid", section="Edit", semantic_query="geopolitical tensions"),
            extra_patches=[
                patch(
                    "app.services.chat_service.chat_completion",
                    return_value="The editorials focus on the Iran conflict and China's strategic challenge as the two dominant geopolitical themes.",
                )
            ],
        )
        self.assertIn("Iran conflict", response.answer)
        self.assertEqual(response.citations, [])

    def test_follow_up_article_text_uses_previous_topic(self):
        query_response = QueryResponse(
            mode="semantic",
            filters={"issue_date": "2026-03-11"},
            results=[
                _article(
                    article_id="50",
                    headline="Iran launches new attacks targeting Israel, Gulf nations",
                    edition="TOIMumbaiBS - MumbaiCity_Digital",
                    section="World",
                    excerpt="Iran widened the confrontation with fresh regional strikes.",
                )
            ],
        )
        response = self._answer(
            "show any one of article",
            query_response,
            routed=self._route_for(mode="semantic", semantic_query="iran war"),
            history=[
                {"role": "user", "content": "Find articles related to the Iran war"},
                {"role": "assistant", "content": "The coverage centers on the Iran conflict."},
            ],
            session_context={"last_topic": "Iran launches new attacks targeting Israel, Gulf nations"},
        )
        self.assertIn("Here is one relevant article excerpt", response.answer)
        self.assertIn("Iran launches new attacks", response.answer)
        self.assertEqual(len(response.citations), 1)

    def test_fresh_follow_up_question_does_not_inherit_old_section(self):
        with ExitStack() as stack:
            mocked_run_query = stack.enter_context(patch("app.services.chat_service.run_query"))
            stack.enter_context(
                patch(
                    "app.services.chat_service.route_query",
                    return_value=self._route_for(mode="semantic", semantic_query="rahul gandhi"),
                )
            )
            stack.enter_context(
                patch("app.services.chat_service.fetch_publication_catalog", return_value=self._catalog_publications())
            )
            mocked_run_query.return_value = QueryResponse(
                mode="semantic",
                filters={"issue_date": "2026-03-11"},
                results=[],
            )
            chat_service.answer_question(
                "can you tell me news about rahul gandhi",
                "2026-03-11",
                10,
                history=[
                    {"role": "user", "content": "List all articles from the Sports section"},
                    {"role": "assistant", "content": "I found several sports articles."},
                ],
                session_context={"section": "Sports", "last_topic": "India World Cup win"},
            )
        args, kwargs = mocked_run_query.call_args
        self.assertEqual(args[0], "can you tell me news about rahul gandhi")
        self.assertIsNone(kwargs["section"])

    def test_follow_up_editorial_article_uses_previous_story_title(self):
        with ExitStack() as stack:
            mocked_run_query = stack.enter_context(patch("app.services.chat_service.run_query"))
            stack.enter_context(
                patch(
                    "app.services.chat_service.route_query",
                    return_value=self._route_for(mode="hybrid", section="Edit", semantic_query="world newest beijing problem"),
                )
            )
            stack.enter_context(
                patch("app.services.chat_service.fetch_publication_catalog", return_value=self._catalog_publications())
            )
            mocked_run_query.return_value = QueryResponse(
                mode="hybrid",
                filters={"issue_date": "2026-03-11", "section": "Edit"},
                results=[],
            )
            chat_service.answer_question(
                "can you give me article regarding world's newest brijing proble,m what you have shared above as",
                "2026-03-11",
                10,
                history=[
                    {"role": "user", "content": "Show me opinion pieces from the Editorial section that discuss geopolitical tensions"},
                    {"role": "assistant", "content": "Another significant thread is World’s Newest Beijing Problem."},
                ],
                session_context={"story_titles": ["World’s Newest Beijing Problem"]},
            )
        args, kwargs = mocked_run_query.call_args
        self.assertIn("World’s Newest Beijing Problem", args[0])


if __name__ == "__main__":
    unittest.main()

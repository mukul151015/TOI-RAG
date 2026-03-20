import unittest
from unittest.mock import patch

from app.services.query_analyzer import analyze_query, expand_person_alias_terms
from app.services.rag_v3.planner import parse_user_intent


class EntityGeneralizationTests(unittest.TestCase):
    def test_generic_person_alias_expansion_handles_unlisted_name_variants(self):
        aliases = expand_person_alias_terms("Manish Sisodiya")
        self.assertIn("Manish Sisodiya", aliases)
        self.assertIn("Sisodiya", aliases)
        self.assertIn("Sisodia", aliases)
        self.assertIn("Manish Sisodia", aliases)

    def test_query_analyzer_extracts_lowercase_place_without_known_city_list_dependency(self):
        with patch("app.services.query_analyzer.llm_analyze_query", return_value={}):
            analysis = analyze_query("is there any news about pune flooding", "2026-03-11")
        self.assertIn("pune", analysis.entities["places"])

    def test_v3_planner_emergency_entity_terms_handle_generic_lowercase_person_query(self):
        with (
            patch(
                "app.services.rag_v3.planner.rewrite_question",
                return_value={
                    "standalone_question": "is there any news about vijay rupani",
                    "references_session_context": False,
                    "reasoning": "same question",
                },
            ),
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.chat_completion", return_value="{}"),
        ):
            _intent, plan = parse_user_intent("is there any news about vijay rupani", "2026-03-11")
        self.assertIn("vijay rupani", [term.lower() for term in plan.entity_terms])

    def test_v3_planner_extracts_generic_lowercase_topic_phrase(self):
        with (
            patch(
                "app.services.rag_v3.planner.rewrite_question",
                return_value={
                    "standalone_question": "how many articles about iran usa war and what were the key points",
                    "references_session_context": False,
                    "reasoning": "same question",
                },
            ),
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.chat_completion", return_value="{}"),
        ):
            _intent, plan = parse_user_intent("how many articles about iran usa war and what were the key points", "2026-03-11")
        lowered_terms = [term.lower() for term in plan.entity_terms]
        self.assertIn("iran usa war", lowered_terms)

    def test_v3_planner_emergency_fallback_keeps_generic_place_news_query_as_summary(self):
        with (
            patch(
                "app.services.rag_v3.planner.rewrite_question",
                return_value={
                    "standalone_question": "is there any news about pune flooding",
                    "references_session_context": False,
                    "reasoning": "same question",
                },
            ),
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.chat_completion", return_value="{}"),
        ):
            intent, plan = parse_user_intent("is there any news about pune flooding", "2026-03-11")
        self.assertEqual(intent.intent, "lookup")
        self.assertFalse(intent.needs_count)
        self.assertEqual(plan.task_type, "summary")

    def test_v3_planner_prefers_hybrid_for_named_topic_summary_queries(self):
        with (
            patch(
                "app.services.rag_v3.planner.rewrite_question",
                return_value={
                    "standalone_question": "what was written about the iran usa war and israel",
                    "references_session_context": False,
                    "reasoning": "same question",
                },
            ),
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=[]),
            patch(
                "app.services.rag_v3.planner.chat_completion",
                return_value='{"task_type":"summary","answer_shape":"summary","intent":"lookup","mode":"semantic","needs_summary":true,"query_focus":"iran usa war israel","entity_terms":["iran usa war","israel"],"retrieval_tools":["semantic_chunks","story_clusters"],"fallback_order":["headline_keyword"],"reasoning":"topic summary"}',
            ),
        ):
            intent, plan = parse_user_intent("what was written about the iran usa war and israel", "2026-03-11")
        self.assertEqual(intent.mode, "hybrid")
        self.assertIn("structured_articles", plan.retrieval_tools)

    def test_v3_planner_does_not_extract_world_section_from_world_cup_phrase(self):
        with (
            patch(
                "app.services.rag_v3.planner.rewrite_question",
                return_value={
                    "standalone_question": "news about india's victory in t-20 world cup",
                    "references_session_context": False,
                    "reasoning": "same question",
                },
            ),
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=["World", "Sports"]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=[]),
            patch(
                "app.services.rag_v3.planner.chat_completion",
                return_value='{"task_type":"summary","answer_shape":"summary","intent":"lookup","mode":"hybrid","needs_summary":true,"query_focus":"India T-20 World Cup victory news","entity_terms":["India","T-20 World Cup","victory"],"retrieval_tools":["structured_articles","semantic_chunks","story_clusters"],"fallback_order":["headline_keyword"],"reasoning":"topic summary"}',
            ),
        ):
            intent, plan = parse_user_intent("news about india's victory in t-20 world cup", "2026-03-11")
        self.assertIsNone(plan.section)
        self.assertEqual(intent.mode, "hybrid")

    def test_v3_planner_can_use_llm_suggested_section_without_section_keyword(self):
        with (
            patch(
                "app.services.rag_v3.planner.rewrite_question",
                return_value={
                    "standalone_question": "what was written in world",
                    "references_session_context": False,
                    "reasoning": "same question",
                },
            ),
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=["World", "Sports"]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=[]),
            patch(
                "app.services.rag_v3.planner.chat_completion",
                return_value='{"task_type":"summary","answer_shape":"summary","intent":"lookup","mode":"sql","section":"World","needs_summary":true,"query_focus":"World section coverage","entity_terms":["World"],"retrieval_tools":["structured_articles"],"fallback_order":["semantic_chunks"],"reasoning":"section understood semantically"}',
            ),
        ):
            intent, plan = parse_user_intent("what was written in world", "2026-03-11")
        self.assertEqual(plan.section, "World")
        self.assertEqual(intent.filters["section"], "World")

    def test_v3_planner_falls_back_to_raw_edition_phrase_when_catalog_match_is_ambiguous(self):
        with (
            patch(
                "app.services.rag_v3.planner.rewrite_question",
                return_value={
                    "standalone_question": "what was written in the delhi edition",
                    "references_session_context": False,
                    "reasoning": "same question",
                },
            ),
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=[]),
            patch(
                "app.services.rag_v3.planner.chat_completion",
                return_value='{"task_type":"summary","answer_shape":"summary","intent":"lookup","mode":"semantic","needs_summary":true,"query_focus":"summary of news in Delhi edition on 2026-03-11","entity_terms":[],"retrieval_tools":["semantic_chunks","story_clusters"],"fallback_order":["headline_keyword"],"reasoning":"edition summary"}',
            ),
        ):
            intent, plan = parse_user_intent("what was written in the delhi edition", "2026-03-11")
        self.assertEqual(plan.edition, "Delhi")
        self.assertEqual(intent.filters["edition"], "Delhi")

    def test_v3_planner_ignores_junk_followup_tokens_in_entity_terms(self):
        with (
            patch(
                "app.services.rag_v3.planner.rewrite_question",
                return_value={
                    "standalone_question": "Can you tell me about Novak Djokovic and Carlos Alcaraz's performances at the Indian Wells tennis tournament?",
                    "references_session_context": True,
                    "reasoning": "resolved follow-up",
                },
            ),
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=["Sports", "World"]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=[]),
            patch(
                "app.services.rag_v3.planner.chat_completion",
                return_value='{"task_type":"summary","answer_shape":"summary","intent":"lookup","mode":"hybrid","needs_summary":true,"query_focus":"Novak Djokovic and Carlos Alcaraz performances Indian Wells tennis 2026-03-11","entity_terms":["Novak Djokovic","Carlos Alcaraz","Indian Wells tennis tournament"],"retrieval_tools":["structured_articles","semantic_chunks","headline_keyword","story_clusters"],"fallback_order":["structured_count"],"reasoning":"follow-up summary"}',
            ),
        ):
            _intent, plan = parse_user_intent("can you tell me about them", "2026-03-11")
        lowered_terms = [term.lower() for term in plan.entity_terms]
        self.assertNotIn("can", lowered_terms)
        self.assertNotIn("tournament", lowered_terms)

    def test_v3_planner_sets_single_issue_time_scope_from_explicit_date_phrase(self):
        with (
            patch(
                "app.services.rag_v3.planner.rewrite_question",
                return_value={
                    "standalone_question": "what was written about modi on March 11 2026",
                    "references_session_context": False,
                    "reasoning": "same question",
                },
            ),
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.chat_completion", return_value="{}"),
        ):
            intent, plan = parse_user_intent("what was written about modi on March 11 2026", None)
        self.assertEqual(plan.time_scope, "single_issue")
        self.assertEqual(plan.issue_date, "2026-03-11")
        self.assertEqual(intent.filters["issue_date"], "2026-03-11")

    def test_v3_planner_sets_all_time_scope_for_archive_queries(self):
        with (
            patch(
                "app.services.rag_v3.planner.rewrite_question",
                return_value={
                    "standalone_question": "what was written about modi in the archive",
                    "references_session_context": False,
                    "reasoning": "same question",
                },
            ),
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.chat_completion", return_value="{}"),
        ):
            intent, plan = parse_user_intent("what was written about modi in the archive", "2026-03-11")
        self.assertEqual(plan.time_scope, "all_time")
        self.assertIsNone(plan.issue_date)
        self.assertEqual(intent.filters["time_scope"], "all_time")

    def test_v3_planner_reuses_session_issue_date_when_question_has_no_new_date(self):
        with (
            patch(
                "app.services.rag_v3.planner.rewrite_question",
                return_value={
                    "standalone_question": "what about the context of articles about Narendra Modi",
                    "references_session_context": True,
                    "reasoning": "used prior session",
                },
            ),
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.chat_completion", return_value="{}"),
        ):
            intent, plan = parse_user_intent(
                "what about the context",
                None,
                session_context={"last_issue_date": "2026-03-11", "last_time_scope": "single_issue", "last_topic": "Narendra Modi"},
            )
        self.assertEqual(plan.time_scope, "single_issue")
        self.assertEqual(plan.issue_date, "2026-03-11")
        self.assertEqual(intent.filters["issue_date"], "2026-03-11")

    def test_v3_planner_sets_year_range_for_explicit_year_queries(self):
        with (
            patch(
                "app.services.rag_v3.planner.rewrite_question",
                return_value={
                    "standalone_question": "what was written about modi in 2024",
                    "references_session_context": False,
                    "reasoning": "same question",
                },
            ),
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.chat_completion", return_value="{}"),
        ):
            intent, plan = parse_user_intent("what was written about modi in 2024", None)
        self.assertEqual(plan.time_scope, "date_range")
        self.assertEqual(plan.start_date, "2024-01-01")
        self.assertEqual(plan.end_date, "2024-12-31")
        self.assertIsNone(plan.issue_date)
        self.assertEqual(intent.filters["time_scope"], "date_range")

    def test_v3_planner_reuses_session_date_range_when_followup_has_no_new_time(self):
        with (
            patch(
                "app.services.rag_v3.planner.rewrite_question",
                return_value={
                    "standalone_question": "what happened after that regarding Modi",
                    "references_session_context": True,
                    "reasoning": "used prior session",
                },
            ),
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.chat_completion", return_value="{}"),
        ):
            intent, plan = parse_user_intent(
                "what happened after that",
                None,
                session_context={
                    "last_time_scope": "date_range",
                    "last_start_date": "2024-01-01",
                    "last_end_date": "2024-12-31",
                    "last_topic": "Modi",
                },
            )
        self.assertEqual(plan.time_scope, "date_range")
        self.assertEqual(plan.start_date, "2024-01-01")
        self.assertEqual(plan.end_date, "2024-12-31")
        self.assertEqual(intent.filters["time_scope"], "date_range")


if __name__ == "__main__":
    unittest.main()

import unittest
from datetime import date
from unittest.mock import patch

from app.core.config import get_settings
from app.schemas import AnswerDraft, DistilledEvidence, EvidenceBundle, EvidenceItem, RetrievalPlan, UserIntent
from app.services.repository import _is_plausible_author_name
from app.services.rag_v3 import pipeline
from app.services.rag_v3.answer_generator import generate_answer
from app.services.rag_v3.distiller import distill_evidence
from app.services.rag_v3.evals import load_eval_cases, summarize_eval_results
from app.services.rag_v3.executor import execute_plan
from app.services.rag_v3.retrievers import _contains_term_match, retrieve_with_tool
from app.services.rag_v3.verifier import _contains_approximate_term, verify_answer


def _intent(**overrides):
    base = UserIntent(
        original_question="how many articles about modi and what were the key points",
        standalone_question="how many articles about modi and what were the key points",
        intent="topic_count",
        mode="hybrid",
        needs_count=True,
        needs_summary=True,
        needs_listing=False,
        needs_article_text=False,
        entities={"entity_terms": ["Narendra Modi", "Modi"]},
        filters={"issue_date": "2026-03-11", "edition": None, "section": None, "author": None},
        reasoning="mixed intent",
    )
    return base.model_copy(update=overrides)


def _plan(**overrides):
    base = RetrievalPlan(
        mode="hybrid",
        intent="topic_count",
        task_type="count",
        answer_shape="count_plus_summary",
        issue_date="2026-03-11",
        semantic_query="Narendra Modi",
        entity_terms=["Narendra Modi", "Modi"],
        retrieval_tools=["structured_count", "story_clusters", "semantic_chunks"],
        fallback_order=["headline_keyword", "structured_articles"],
        confidence=0.7,
        reasoning="mixed intent",
    )
    return base.model_copy(update=overrides)


class RagV3PipelineTests(unittest.TestCase):
    def test_v3_is_enabled_in_live_config(self):
        settings = get_settings()
        self.assertTrue(settings.rag_v3_enabled)

    def test_author_name_cleaner_rejects_malformed_bylines(self):
        self.assertFalse(_is_plausible_author_name("– Donald Trump, Nov 30, 2011"))
        self.assertTrue(_is_plausible_author_name("Ajay Sura"))

    def test_approximate_entity_match_handles_small_name_typos(self):
        self.assertTrue(_contains_term_match("Rijiju slams Cong for motion against Birla", "rijju"))
        self.assertTrue(_contains_term_match("Kiren Rijiju says Owaisi's claim irrelevant", "kiran rijju"))
        self.assertFalse(_contains_term_match("Completely unrelated story about inflation", "rijju"))

    def test_verifier_approximate_match_handles_small_name_typos(self):
        self.assertTrue(_contains_approximate_term("rijiju slams cong for motion against birla", "rijju"))
        self.assertTrue(_contains_approximate_term("kiren rijiju says owaisi claim irrelevant", "rijiju"))
        self.assertFalse(_contains_approximate_term("inflation and gdp target discussion", "rijju"))

    def test_planner_mixed_intent_uses_structured_count_first(self):
        with (
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=[]),
            patch(
                "app.services.rag_v3.planner.chat_completion",
                return_value='{"task_type":"count","answer_shape":"count_plus_summary","intent":"topic_count","mode":"hybrid","needs_count":true,"needs_summary":true,"query_focus":"Narendra Modi","entity_terms":["Narendra Modi"],"retrieval_tools":["structured_count","story_clusters"],"fallback_order":["semantic_chunks"],"reasoning":"mixed intent"}',
            ),
        ):
            intent, plan = pipeline.parse_user_intent("how many articles about modi and what were the key points", "2026-03-11")
        self.assertEqual(intent.intent, "topic_count")
        self.assertTrue(intent.needs_count)
        self.assertTrue(intent.needs_summary)
        self.assertEqual(plan.retrieval_tools[0], "structured_count")
        self.assertIn("story_clusters", plan.retrieval_tools)

    def test_planner_normalizes_free_text_llm_intent_to_allowed_value(self):
        with (
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=[]),
            patch(
                "app.services.rag_v3.planner.chat_completion",
                return_value='{"task_type":"count","answer_shape":"count_plus_summary","intent":"Determine the number of articles about Modi and summarize their key points.","mode":"hybrid","needs_count":true,"needs_summary":true,"query_focus":"Narendra Modi","entity_terms":["Narendra Modi"],"retrieval_tools":["structured_count","story_clusters"],"fallback_order":["semantic_chunks"],"reasoning":"mixed intent"}',
            ),
        ):
            intent, _plan_result = pipeline.parse_user_intent("how many articles about modi and what were the key points", "2026-03-11")
        self.assertEqual(intent.intent, "topic_count")

    def test_planner_typo_count_query_still_normalizes_to_topic_count(self):
        with (
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=[]),
            patch(
                "app.services.rag_v3.planner.chat_completion",
                return_value='{"task_type":"summary","answer_shape":"count_plus_summary","intent":"lookup","mode":"hybrid","needs_count":true,"needs_summary":true,"query_focus":"Modi","entity_terms":["Modi"],"retrieval_tools":["structured_count","story_clusters"],"fallback_order":["semantic_chunks"],"reasoning":"typo query"}',
            ),
        ):
            intent, plan = pipeline.parse_user_intent("how amny articles abput modi and what were the key points", "2026-03-11")
        self.assertEqual(intent.intent, "topic_count")
        self.assertEqual(plan.task_type, "count")

    def test_person_topic_query_does_not_false_match_author_filter(self):
        with (
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=["Rajeev Mani", "Times News Network"]),
            patch(
                "app.services.rag_v3.planner.chat_completion",
                return_value='{"task_type":"summary","answer_shape":"summary","intent":"lookup","mode":"semantic","needs_summary":true,"query_focus":"Manish Sisodiya","entity_terms":["Manish Sisodiya"],"retrieval_tools":["semantic_chunks","structured_articles"],"fallback_order":["headline_keyword"],"reasoning":"person topic query"}',
            ),
        ):
            intent, plan = pipeline.parse_user_intent("is there any news about manish sisodiya", "2026-03-11")
        self.assertIsNone(plan.author)
        self.assertIn("Sisodiya", plan.entity_terms)
        self.assertIn("Sisodia", plan.entity_terms)
        self.assertEqual(plan.mode, "hybrid")
        self.assertIn("structured_articles", plan.retrieval_tools)

    def test_existential_news_query_is_not_forced_into_count_only(self):
        with (
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=[]),
            patch(
                "app.services.rag_v3.planner.chat_completion",
                return_value='{"task_type":"count","answer_shape":"count_only","intent":"article_count","mode":"hybrid","needs_count":true,"needs_summary":false,"query_focus":"news about Manish Sisodiya on 2026-03-11","entity_terms":["Manish Sisodiya"],"retrieval_tools":["structured_count","semantic_chunks"],"fallback_order":[],"reasoning":"bad count classification"}',
            ),
        ):
            intent, plan = pipeline.parse_user_intent("is there any news about manish sisodiya", "2026-03-11")
        self.assertEqual(intent.intent, "lookup")
        self.assertFalse(intent.needs_count)
        self.assertTrue(intent.needs_summary)
        self.assertEqual(plan.task_type, "summary")
        self.assertEqual(plan.answer_shape, "summary")

    def test_existential_news_query_is_not_forced_into_list(self):
        with (
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=[]),
            patch(
                "app.services.rag_v3.planner.chat_completion",
                return_value='{"task_type":"list","answer_shape":"list","intent":"lookup","mode":"semantic","needs_count":false,"needs_summary":false,"needs_listing":true,"query_focus":"news about Manish Sisodiya on 2026-03-11","entity_terms":["Manish Sisodiya"],"retrieval_tools":["semantic_chunks","story_clusters","structured_articles"],"fallback_order":["headline_keyword"],"reasoning":"bad list classification"}',
            ),
        ):
            intent, plan = pipeline.parse_user_intent("is there any news about manish sisodiya", "2026-03-11")
        self.assertEqual(intent.intent, "lookup")
        self.assertFalse(intent.needs_listing)
        self.assertTrue(intent.needs_summary)
        self.assertEqual(plan.task_type, "summary")
        self.assertEqual(plan.answer_shape, "summary")

    def test_author_query_extracts_author_filter_from_write_about_wording(self):
        with (
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=["Ajay Sura", "Rajeev Mani"]),
            patch(
                "app.services.rag_v3.planner.chat_completion",
                return_value='{"task_type":"summary","answer_shape":"summary","intent":"author_lookup","mode":"sql","needs_summary":true,"query_focus":"Ajay Sura","entity_terms":["Ajay Sura"],"retrieval_tools":["structured_articles"],"fallback_order":["semantic_chunks"],"reasoning":"author query"}',
            ),
        ):
            intent, plan = pipeline.parse_user_intent("what did Ajay Sura write about", "2026-03-11")
        self.assertEqual(plan.author, "Ajay Sura")
        self.assertEqual(intent.intent, "author_lookup")
        self.assertEqual(intent.mode, "sql")
        self.assertIn("structured_articles", plan.retrieval_tools)
        self.assertNotIn("semantic_chunks", plan.retrieval_tools)

    def test_planner_can_use_llm_suggested_author_without_explicit_author_cue_words(self):
        with (
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=["Ajay Sura", "Rajeev Mani"]),
            patch(
                "app.services.rag_v3.planner.chat_completion",
                return_value='{"task_type":"summary","answer_shape":"summary","intent":"author_lookup","mode":"sql","author":"Ajay Sura","needs_summary":true,"query_focus":"Ajay Sura recent coverage","entity_terms":["Ajay Sura"],"retrieval_tools":["structured_articles"],"fallback_order":["semantic_chunks"],"reasoning":"author understood semantically"}',
            ),
        ):
            intent, plan = pipeline.parse_user_intent("ajay sura latest coverage", "2026-03-11")
        self.assertEqual(plan.author, "Ajay Sura")
        self.assertEqual(intent.intent, "author_lookup")
        self.assertEqual(intent.mode, "sql")

    def test_summary_answer_shape_for_entity_query_forces_summary_and_hybrid_tools(self):
        with (
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=[]),
            patch(
                "app.services.rag_v3.planner.chat_completion",
                return_value='{"task_type":"summary","answer_shape":"summary","intent":"lookup","mode":"semantic","needs_summary":false,"query_focus":"news about Manish Sisodiya on 2026-03-11","entity_terms":["Manish Sisodiya"],"retrieval_tools":["semantic_chunks"],"fallback_order":["structured_articles"],"reasoning":"bad summary flags"}',
            ),
        ):
            intent, plan = pipeline.parse_user_intent("is there any news about manish sisodiya", "2026-03-11")
        self.assertTrue(intent.needs_summary)
        self.assertEqual(intent.mode, "hybrid")
        self.assertIn("structured_articles", plan.retrieval_tools)

    def test_out_of_domain_summary_is_normalized_to_standard_abstention_text(self):
        draft = generate_answer(
            _intent(intent="lookup", needs_count=False, needs_summary=True, entities={"entity_terms": ["Professor Lupin"]}),
            EvidenceBundle(
                question="what did professor lupin teach in this archive",
                mode="hybrid",
                plan=_plan(intent="lookup", task_type="summary", answer_shape="summary"),
                items=[EvidenceItem(article_id="1", headline="Unrelated story", excerpt="Some unrelated excerpt")],
                retrieval_confidence=0.3,
            ),
            DistilledEvidence(summary="The provided evidence does not contain any information about what Professor Lupin taught in the archive."),
        )
        self.assertEqual(
            draft.answer,
            "I couldn't find enough grounded evidence in the current dataset to answer that confidently.",
        )

    def test_ranking_query_uses_structured_count_and_returns_top_section_answer(self):
        with (
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=["Nation", "City", "FrontPage"]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=[]),
            patch(
                "app.services.rag_v3.planner.chat_completion",
                return_value='{"task_type":"ranking","answer_shape":"ranking","intent":"lookup","mode":"sql","needs_summary":true,"query_focus":"most articles section","entity_terms":["sections"],"retrieval_tools":["structured_count"],"fallback_order":["structured_articles"],"reasoning":"ranking output"}',
            ),
            patch(
                "app.services.rag_v3.retrievers.fetch_section_counts",
                return_value=[
                    {"section": "Nation", "article_count": 1392},
                    {"section": "City", "article_count": 1314},
                    {"section": "FrontPage", "article_count": 1183},
                ],
            ),
        ):
            response = pipeline.answer_question("which section had the most articles", "2026-03-11", 6)
        self.assertIn("Nation", response.answer)
        self.assertIn("1392", response.answer)
        self.assertEqual(response.mode, "sql")

    def test_ranking_query_normalizes_needs_count_false_even_if_llm_sets_it_true(self):
        with (
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=[]),
            patch(
                "app.services.rag_v3.planner.chat_completion",
                return_value='{"task_type":"ranking","answer_shape":"ranking","intent":"lookup","mode":"sql","needs_count":true,"needs_summary":true,"query_focus":"coverage of Modi by year","entity_terms":["Modi"],"retrieval_tools":["structured_count"],"fallback_order":[],"reasoning":"ranking output"}',
            ),
        ):
            intent, plan = pipeline.parse_user_intent("which year had the most coverage of Modi", None)
        self.assertFalse(intent.needs_count)
        self.assertEqual(plan.task_type, "ranking")
        self.assertEqual(plan.answer_shape, "ranking")

    def test_ranking_query_can_return_least_section_answer(self):
        draft = generate_answer(
            _intent(intent="lookup", needs_count=False, needs_summary=True, entities={"entity_terms": ["section"]}),
            EvidenceBundle(
                question="which section had the least articles",
                mode="sql",
                plan=_plan(intent="lookup", task_type="ranking", answer_shape="ranking", retrieval_tools=["structured_count"]),
                items=[],
                raw_filters={
                    "section_counts": [
                        {"section": "Nation", "article_count": 1392},
                        {"section": "City", "article_count": 1314},
                        {"section": "Oped", "article_count": 1},
                    ]
                },
            ),
            DistilledEvidence(),
        )
        self.assertIn("least articles was Oped with 1 articles", draft.answer)

    def test_ranking_query_can_return_top_edition_answer(self):
        draft = generate_answer(
            _intent(intent="lookup", needs_count=False, needs_summary=True, entities={"entity_terms": ["edition"]}),
            EvidenceBundle(
                question="which edition had the most articles",
                mode="sql",
                plan=_plan(intent="lookup", task_type="ranking", answer_shape="ranking", retrieval_tools=["structured_count"]),
                items=[],
                raw_filters={
                    "ranking_kind": "edition",
                    "publication_counts": [
                        {"publication_name": "TOIChandigarhBS - Chandigarh_Digital", "article_count": 220},
                        {"publication_name": "TOIBangaloreBS - BangaloreCity_Digital", "article_count": 210},
                    ],
                },
            ),
            DistilledEvidence(),
        )
        self.assertIn("edition with the most articles was TOIChandigarhBS - Chandigarh_Digital with 220 articles", draft.answer)

    def test_ranking_query_can_return_top_author_answer(self):
        draft = generate_answer(
            _intent(intent="lookup", needs_count=False, needs_summary=True, entities={"entity_terms": ["author"]}),
            EvidenceBundle(
                question="which author wrote the most articles",
                mode="sql",
                plan=_plan(intent="lookup", task_type="ranking", answer_shape="ranking", retrieval_tools=["structured_count"]),
                items=[],
                raw_filters={
                    "ranking_kind": "author",
                    "author_counts": [
                        {"author": "Ajay Sura", "article_count": 5},
                        {"author": "A Selvaraj", "article_count": 4},
                    ],
                },
            ),
            DistilledEvidence(),
        )
        self.assertIn("author with the most articles was Ajay Sura with 5 articles", draft.answer)

    def test_ranking_query_can_return_top_year_answer(self):
        draft = generate_answer(
            _intent(intent="lookup", needs_count=False, needs_summary=True, entities={"entity_terms": ["Modi"]}),
            EvidenceBundle(
                question="which year had the most coverage of Modi",
                mode="sql",
                plan=_plan(intent="lookup", task_type="ranking", answer_shape="ranking", retrieval_tools=["structured_count"]),
                items=[],
                raw_filters={
                    "ranking_kind": "year",
                    "year_counts": [
                        {"year": 2026, "article_count": 105},
                        {"year": 2025, "article_count": 44},
                    ],
                },
            ),
            DistilledEvidence(),
        )
        self.assertIn("year with the most articles was 2026 with 105 articles", draft.answer)

    def test_structured_count_uses_date_range_repository_calls(self):
        plan = _plan(
            time_scope="date_range",
            issue_date=None,
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
        with (
            patch("app.services.rag_v3.retrievers.fetch_entity_mention_articles_in_range", return_value=[]),
            patch("app.services.rag_v3.retrievers.fetch_entity_mention_count_in_range", return_value=12),
            patch(
                "app.services.rag_v3.retrievers.fetch_entity_mention_contexts_in_range",
                return_value=[{"headline": "Modi in 2024", "article_count": 12, "section": "Nation", "excerpt": "Context"}],
            ),
        ):
            bundle = retrieve_with_tool("what was written about modi in 2024", plan, "structured_count", 5)
        self.assertEqual(bundle.raw_filters["exact_article_count"], 12)
        self.assertEqual(bundle.raw_filters["time_scope"], "date_range")
        self.assertEqual(bundle.raw_filters["start_date"], "2024-01-01")
        self.assertEqual(bundle.raw_filters["end_date"], "2024-12-31")

    def test_semantic_keyword_tools_include_date_range_filters(self):
        plan = _plan(
            time_scope="date_range",
            issue_date=None,
            start_date="2024-01-01",
            end_date="2024-12-31",
            intent="lookup",
            task_type="summary",
            answer_shape="summary",
            retrieval_tools=["semantic_chunks", "headline_keyword"],
        )
        with (
            patch("app.services.rag_v3.retrievers.embed_texts", return_value=[[0.1, 0.2]]),
            patch("app.services.rag_v3.retrievers.semantic_search", return_value=[]),
        ):
            bundle = retrieve_with_tool("what was written about modi in 2024", plan, "semantic_chunks", 5)
        self.assertEqual(bundle.raw_filters["time_scope"], "date_range")
        self.assertEqual(bundle.raw_filters["start_date"], "2024-01-01")
        self.assertEqual(bundle.raw_filters["end_date"], "2024-12-31")

    def test_executor_keeps_successful_bundles_when_one_tool_fails(self):
        intent = _intent(intent="lookup", needs_count=False, needs_summary=True)
        plan = _plan(
            intent="lookup",
            task_type="summary",
            answer_shape="summary",
            retrieval_tools=["structured_articles", "semantic_chunks"],
            fallback_order=[],
        )
        structured_bundle = EvidenceBundle(
            question="what is the latest one",
            mode="sql",
            plan=plan,
            items=[EvidenceItem(article_id="1", headline="Modi latest", excerpt="Latest Modi story", source_type="structured_articles")],
            retrieval_confidence=0.95,
            applied_tools=["structured_articles"],
        )
        with (
            patch("app.services.rag_v3.executor.retrieve_with_tool", side_effect=[structured_bundle, RuntimeError("Connection error")]),
            patch(
                "app.services.rag_v3.executor.distill_evidence",
                return_value=DistilledEvidence(summary="Latest Modi story", supporting_article_ids=["1"]),
            ),
        ):
            bundle, state = execute_plan("what is the latest one", intent, plan, 6, "trace_test")
        self.assertEqual(len(bundle.items), 1)
        self.assertIn("structured_articles", bundle.applied_tools)
        self.assertTrue(any(step.status == "failed" and step.tool == "semantic_chunks" for step in state.steps))

    def test_fallback_distillation_formats_latest_summary_cleanly(self):
        intent = _intent(
            original_question="what is the latest one",
            standalone_question="What is the latest news about Modi?",
            intent="lookup",
            needs_count=False,
            needs_summary=True,
        )
        bundle = EvidenceBundle(
            question="what is the latest one",
            mode="hybrid",
            plan=_plan(intent="lookup", task_type="summary", answer_shape="summary"),
            items=[
                EvidenceItem(
                    article_id="1",
                    headline="Modi says DMK is rattled by NDA’s popularity in Tamil Nadu",
                    excerpt="A day ahead of his visit to poll-bound Tamil Nadu, PM Narendra Modi raised the campaign pitch.",
                ),
                EvidenceItem(
                    article_id="2",
                    headline="Modi: India actively promoting animation",
                    excerpt="Prime Minister Narendra Modi said India is actively promoting the AVGC sector.",
                ),
            ],
        )
        with patch("app.services.rag_v3.distiller.chat_completion", side_effect=RuntimeError("model down")):
            distilled = distill_evidence(intent, bundle)
        self.assertIn("Latest coverage includes:", distilled.summary)
        self.assertNotIn("A day ahead of his visit to poll-bound Tamil Nadu, PM Narendra Modi raised the campaign pitch. Prime Minister", distilled.summary)

    def test_archive_overview_distillation_prefers_overview_for_all_time_queries(self):
        intent = _intent(
            original_question="what was written about Modi in the archive",
            standalone_question="What was written about Modi in the archive?",
            intent="lookup",
            needs_count=False,
            needs_summary=True,
        )
        bundle = EvidenceBundle(
            question="what was written about Modi in the archive",
            mode="hybrid",
            plan=_plan(intent="lookup", task_type="summary", answer_shape="summary", time_scope="all_time", issue_date=None),
            items=[
                EvidenceItem(
                    article_id="1",
                    headline="Modi says DMK is rattled by NDA’s popularity in Tamil Nadu",
                    issue_date="2026-03-11",
                    excerpt="A day ahead of his visit to poll-bound Tamil Nadu, PM Narendra Modi raised the campaign pitch.",
                ),
                EvidenceItem(
                    article_id="2",
                    headline="DMK rattled by NDA’s popularity: Modi",
                    issue_date="2026-03-11",
                    excerpt="Raising the campaign pitch a day ahead of his visit to Tamil Nadu, Prime Minister Narendra Modi said the DMK was rattled.",
                ),
                EvidenceItem(
                    article_id="3",
                    headline="Modi: India actively promoting animation",
                    issue_date="2026-03-11",
                    excerpt="Prime Minister Narendra Modi said India is actively promoting the AVGC sector.",
                ),
                EvidenceItem(
                    article_id="4",
                    headline="Modi addresses development agenda",
                    issue_date="2025-01-09",
                    excerpt="Modi focused on development and infrastructure priorities.",
                ),
            ],
        )
        distilled = distill_evidence(intent, bundle)
        self.assertEqual(distilled.coverage, "archive_overview")
        self.assertIn("Archive coverage", distilled.summary)
        self.assertIn("2025", distilled.summary)
        self.assertIn("2026", distilled.summary)
        self.assertTrue(any("related reports" in point for point in distilled.key_points))

    def test_timeline_distillation_prefers_latest_update_shape_for_archive_followup(self):
        intent = _intent(
            original_question="what is the latest one",
            standalone_question="What is the latest news about Modi?",
            intent="lookup",
            needs_count=False,
            needs_summary=True,
        )
        bundle = EvidenceBundle(
            question="what is the latest one",
            mode="hybrid",
            plan=_plan(intent="lookup", task_type="summary", answer_shape="summary", time_scope="all_time", issue_date=None),
            items=[
                EvidenceItem(
                    article_id="1",
                    headline="Modi says DMK is rattled by NDA’s popularity in Tamil Nadu",
                    issue_date="2026-03-11",
                    excerpt="A day ahead of his visit to poll-bound Tamil Nadu, PM Narendra Modi raised the campaign pitch.",
                ),
                EvidenceItem(
                    article_id="2",
                    headline="DMK rattled by NDA’s popularity: Modi",
                    issue_date="2026-03-11",
                    excerpt="Raising the campaign pitch a day ahead of his visit to Tamil Nadu, Prime Minister Narendra Modi said the DMK was rattled.",
                ),
                EvidenceItem(
                    article_id="3",
                    headline="Modi addresses development agenda",
                    issue_date="2025-01-09",
                    excerpt="Modi focused on development and infrastructure priorities.",
                ),
            ],
        )
        distilled = distill_evidence(intent, bundle)
        self.assertEqual(distilled.coverage, "timeline_overview")
        self.assertIn("Latest update", distilled.summary)
        self.assertIn("Earlier related developments include", distilled.summary)
        self.assertTrue(any(point.startswith("2026-03-11:") for point in distilled.key_points))
        self.assertNotIn("Latest update on 2026-03-11: 2026-03-11:", distilled.summary)

    def test_compare_question_uses_llm_compare_task(self):
        with (
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=["Business", "World"]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=[]),
            patch(
                "app.services.rag_v3.planner.chat_completion",
                return_value='{"task_type":"compare","answer_shape":"compare","intent":"lookup","mode":"hybrid","needs_summary":true,"query_focus":"business world coverage","entity_terms":["Business","World"],"retrieval_tools":["structured_articles","semantic_chunks"],"fallback_order":["story_clusters"],"reasoning":"compare output"}',
            ),
        ):
            intent, plan = pipeline.parse_user_intent("compare business and world coverage", "2026-03-11")
        self.assertEqual(plan.task_type, "compare")
        self.assertEqual(plan.answer_shape, "compare")
        self.assertTrue(intent.needs_summary)
        self.assertIn("structured_articles", plan.retrieval_tools)

    def test_author_ranking_query_uses_structured_count(self):
        with (
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=["Ajay Sura", "A Selvaraj"]),
            patch(
                "app.services.rag_v3.planner.chat_completion",
                return_value='{"task_type":"ranking","answer_shape":"ranking","intent":"author_lookup","mode":"sql","needs_summary":true,"query_focus":"authors most articles","entity_terms":["authors"],"retrieval_tools":["structured_count"],"fallback_order":["structured_articles"],"reasoning":"author ranking output"}',
            ),
        ):
            intent, plan = pipeline.parse_user_intent("which author wrote the most articles", "2026-03-11")
        self.assertEqual(plan.task_type, "ranking")
        self.assertEqual(plan.retrieval_tools, ["structured_count"])
        self.assertEqual(intent.mode, "sql")

    def test_section_summary_question_uses_summary_fallback_when_planner_missing(self):
        with (
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=["World", "Nation"]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.chat_completion", return_value="{}"),
        ):
            intent, plan = pipeline.parse_user_intent("what was written in the world section", "2026-03-11")
        self.assertFalse(intent.needs_article_text)
        self.assertEqual(plan.task_type, "summary")

    def test_compare_answer_shape_normalizes_to_compare_task(self):
        with (
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=[]),
            patch(
                "app.services.rag_v3.planner.chat_completion",
                return_value='{"task_type":"summary","answer_shape":"compare","intent":"lookup","mode":"hybrid","needs_summary":true,"query_focus":"compare iran israel","entity_terms":["Iran","Israel"],"retrieval_tools":["structured_articles","semantic_chunks"],"fallback_order":["story_clusters"],"reasoning":"compare query"}',
            ),
        ):
            intent, plan = pipeline.parse_user_intent("compare iran and israel coverage", "2026-03-11")
        self.assertEqual(plan.task_type, "compare")
        self.assertTrue(intent.needs_summary)

    def test_compare_answer_generation_uses_llm_comparison_when_available(self):
        with patch("app.services.rag_v3.answer_generator.chat_completion", return_value="Coverage of Iran focused on attacks and endurance, while coverage of Israel focused on military action and regime-change goals."):
            draft = generate_answer(
                _intent(intent="lookup", needs_count=False, needs_summary=True, entities={"entity_terms": ["Iran", "Israel"]}),
                EvidenceBundle(
                    question="compare iran and israel coverage",
                    mode="hybrid",
                    plan=_plan(intent="lookup", task_type="compare", answer_shape="compare", retrieval_tools=["structured_articles", "semantic_chunks"]),
                    items=[],
                ),
                DistilledEvidence(
                    summary="Conflict stories discussed Iran enduring attacks and Israel pursuing military action.",
                    key_points=["Iran endured attacks.", "Israel pursued military action."],
                ),
            )
        self.assertIn("Coverage of Iran focused", draft.answer)

    def test_verifier_rejects_generic_count_for_topic_query(self):
        bundle = EvidenceBundle(
            question="how many articles about modi and what were the key points",
            mode="sql",
            plan=_plan(mode="sql"),
            items=[
                EvidenceItem(
                    article_id="1",
                    headline="Modi speaks on economy",
                    issue_date="2026-03-11",
                    metadata={"external_article_id": "1"},
                )
            ],
            raw_filters={"exact_article_count": 7},
            retrieval_confidence=1.0,
        )
        draft = AnswerDraft(answer="There were 23 articles on 2026-03-11.", mode="sql")
        report = verify_answer(_intent(), bundle, draft)
        self.assertFalse(report.answer_accepted)
        self.assertTrue(any("generic count form" in item for item in report.unsupported_claims))

    def test_verifier_allows_deterministic_count_only_answer_with_zero_preview_items(self):
        bundle = EvidenceBundle(
            question="how many articles were there in the delhi edition",
            mode="sql",
            plan=_plan(mode="sql", intent="article_count", answer_shape="count_only", retrieval_tools=["structured_count"]),
            items=[],
            raw_filters={"exact_article_count": 0},
            retrieval_confidence=1.0,
        )
        draft = AnswerDraft(answer="I found 0 relevant articles.", mode="sql")
        report = verify_answer(
            _intent(intent="article_count", needs_summary=False, entities={}, filters={"issue_date": "2026-03-11", "edition": "Delhi", "section": None, "author": None}),
            bundle,
            draft,
        )
        self.assertTrue(report.answer_accepted)

    def test_verifier_does_not_reject_author_summary_when_author_filter_drove_retrieval(self):
        bundle = EvidenceBundle(
            question="what did Ajay Sura write about",
            mode="hybrid",
            plan=_plan(
                mode="hybrid",
                intent="lookup",
                task_type="summary",
                answer_shape="summary",
                author="Ajay Sura",
                entity_terms=["Ajay Sura", "Sura"],
                retrieval_tools=["structured_articles", "story_clusters"],
            ),
            items=[
                EvidenceItem(
                    article_id="1",
                    headline="Driver’s gun photos on social media can’t implicate licensed owners: HC",
                    edition="TOIChandigarhBS - Chandigarh-Upcountry_Digital",
                    section="City",
                    issue_date="2026-03-11",
                    excerpt="Chandigarh: The Punjab and Haryana high court quashed an FIR against Bajrang Dass Garg and his son.",
                    metadata={"author": "Ajay Sura"},
                )
            ],
            raw_filters={"author": "Ajay Sura"},
            retrieval_confidence=0.7,
        )
        draft = AnswerDraft(
            answer="Ajay Sura wrote about a high court ruling that quashed an FIR linked to licensed weapons and social media photos.",
            mode="hybrid",
        )
        report = verify_answer(
            _intent(
                intent="author_lookup",
                needs_count=False,
                needs_summary=True,
                entities={"entity_terms": ["Ajay Sura", "Sura"]},
                filters={"issue_date": "2026-03-11", "edition": None, "section": None, "author": "Ajay Sura"},
            ),
            bundle,
            draft,
        )
        self.assertTrue(report.answer_accepted)

    def test_pipeline_failure_trace_always_contains_stage(self):
        with (
            patch("app.services.rag_v3.pipeline.parse_user_intent", return_value=(_intent(), _plan())),
            patch("app.services.rag_v3.pipeline.execute_plan", side_effect=pipeline.V3ExecutionError("retrieve", __import__("app.schemas").schemas.ExecutionState(plan=_plan()), RuntimeError("db timeout"))),
        ):
            response = pipeline.answer_question("how many articles about modi", "2026-03-11", 6)
        self.assertEqual(response.debug_trace["failure"]["stage"], "retrieve")
        self.assertIn("db timeout", response.debug_trace["failure"]["message"])

    def test_pipeline_success_returns_trace_envelope_and_verification(self):
        bundle = EvidenceBundle(
            question="how many articles about modi and what were the key points",
            mode="sql",
            plan=_plan(mode="sql"),
            items=[
                EvidenceItem(
                    article_id="1",
                    headline="Modi speaks on economy",
                    edition="Delhi",
                    section="Nation",
                    issue_date=date(2026, 3, 11).isoformat(),
                    excerpt="The story focused on growth and inflation.",
                    metadata={"external_article_id": "1"},
                )
            ],
            raw_filters={"exact_article_count": 1},
            retrieval_confidence=1.0,
            applied_tools=["structured_count"],
        )
        state = __import__("app.schemas").schemas.ExecutionState(plan=_plan(mode="sql"), trace_id="trace_123")
        state.steps = [__import__("app.schemas").schemas.ExecutionStep(step_type="retrieve", tool="structured_count", status="completed")]
        with (
            patch("app.services.rag_v3.pipeline.parse_user_intent", return_value=(_intent(), _plan(mode="sql"))),
            patch("app.services.rag_v3.pipeline.execute_plan", return_value=(bundle, state)),
            patch("app.services.rag_v3.pipeline.distill_evidence", return_value=DistilledEvidence(summary="growth and inflation", key_points=["growth and inflation"], supporting_article_ids=["1"])),
        ):
            response = pipeline.answer_question("how many articles about modi and what were the key points", "2026-03-11", 6)
        self.assertIn("I found 1 relevant article", response.answer)
        self.assertTrue(str(response.debug_trace["trace_id"]).startswith("trace_"))
        self.assertEqual(response.debug_trace["evidence_summary"]["item_count"], 1)
        self.assertTrue(response.verification.answer_accepted)

    def test_query_route_uses_v3_only(self):
        from app.api import routes

        payload = type("Payload", (), {"query": "modi", "issue_date": "2026-03-11", "limit": 5})()
        with (
            patch("app.api.routes.require_authenticated_user"),
            patch("app.api.routes.execute_query_v3", return_value={"path": "v3"}) as execute_query_v3,
        ):
            result = routes.query_route(payload, object())
        self.assertEqual(result["path"], "v3")
        execute_query_v3.assert_called_once()

    def test_eval_loader_and_summary_support_v3_dataset(self):
        cases = load_eval_cases("benchmarks/rag_v3_eval_cases.json")
        self.assertGreaterEqual(len(cases), 8)
        summary = summarize_eval_results([])
        self.assertEqual(summary["pass_rate"], 0.0)
        followup = next(case for case in cases if case.case_id == "followup_001")
        self.assertTrue(followup.history)
        self.assertTrue(followup.session_context)

    def test_count_summary_formatter_strips_excerpt_noise_from_structured_contexts(self):
        draft = generate_answer(
            _intent(),
            EvidenceBundle(
                question="how many articles about modi and what were the key points",
                mode="hybrid",
                plan=_plan(),
                items=[],
                raw_filters={"exact_article_count": 105},
            ),
            DistilledEvidence(
                summary="",
                key_points=[
                    "Modi says DMK is rattled by NDA’s popularity in Tamil Nadu (4 articles) in Nation A day ahead of his visit to poll-bound Tamil Nadu, PM Narendra Modi raised the campaign pitch",
                    "Modi: India actively promoting animation (4 articles) in Nation Prime Minister Narendra Modi has emphasised that India is actively promoting the AVGC sector",
                ],
                coverage="structured_contexts",
            ),
        )
        self.assertIn("Modi says DMK is rattled by NDA’s popularity in Tamil Nadu (4 articles)", draft.answer)
        self.assertNotIn("A day ahead of his visit", draft.answer)

    def test_followup_rewriter_uses_session_context(self):
        with patch(
            "app.services.rag_v3.query_rewriter.chat_completion",
            return_value='{"standalone_question":"What about the context of articles about Narendra Modi?","references_session_context":true,"reasoning":"Used prior topic."}',
        ):
            result = __import__("app.services.rag_v3.query_rewriter", fromlist=["rewrite_question"]).rewrite_question(
                "what about the context",
                history=[{"role": "user", "content": "how many articles about modi"}],
                session_context={"last_topic": "Narendra Modi"},
            )
        self.assertIn("Narendra Modi", result["standalone_question"])

    def test_followup_rewriter_fallback_preserves_single_issue_scope(self):
        result = __import__("app.services.rag_v3.query_rewriter", fromlist=["rewrite_question"]).rewrite_question(
            "what about the context",
            history=[],
            session_context={
                "last_topic": "Narendra Modi",
                "last_issue_date": "2026-03-11",
                "last_time_scope": "single_issue",
            },
        )
        self.assertIn("Narendra Modi", result["standalone_question"])
        self.assertIn("2026-03-11", result["standalone_question"])

    def test_followup_rewriter_fallback_handles_latest_for_archive_scope(self):
        result = __import__("app.services.rag_v3.query_rewriter", fromlist=["rewrite_question"]).rewrite_question(
            "what is the latest one",
            history=[],
            session_context={
                "last_topic": "Kiren Rijiju",
                "last_time_scope": "all_time",
                "last_entity_terms": ["Kiren Rijiju", "Rijiju"],
            },
        )
        self.assertIn("latest news about Kiren Rijiju", result["standalone_question"])

    def test_reranker_reorders_semantic_evidence_when_enabled(self):
        items = [
            EvidenceItem(article_id="1", headline="Weak", excerpt="weak", source_type="semantic_chunks"),
            EvidenceItem(article_id="2", headline="Strong", excerpt="strong", source_type="semantic_chunks"),
        ]
        with (
            patch("app.services.rag_v3.reranker._settings.reranking_enabled", True),
            patch("app.services.reranker.rerank", return_value=[1, 0]),
        ):
            reranked, used = __import__("app.services.rag_v3.reranker", fromlist=["rerank_items"]).rerank_items("modi", items)
        self.assertTrue(used)
        self.assertEqual(reranked[0].headline, "Strong")

    def test_structured_count_bundle_is_not_reranked(self):
        bundle = EvidenceBundle(
            question="how many articles about modi",
            mode="sql",
            plan=_plan(mode="sql", retrieval_tools=["structured_count"]),
            items=[
                EvidenceItem(article_id="1", headline="One", source_type="structured_count"),
                EvidenceItem(article_id="2", headline="Two", source_type="structured_count"),
            ],
            retrieval_confidence=1.0,
            applied_tools=["structured_count"],
        )
        with patch("app.services.rag_v3.retrievers.rerank_items", return_value=(bundle.items[::-1], True)) as rerank_items:
            merged = __import__("app.services.rag_v3.retrievers", fromlist=["merge_bundles"]).merge_bundles(
                "how many articles about modi",
                bundle.plan,
                [bundle],
            )
        rerank_items.assert_not_called()
        self.assertEqual([item.headline for item in merged.items], ["One", "Two"])

    def test_hyde_is_skipped_for_exact_entity_summary_queries(self):
        plan = _plan(
            mode="hybrid",
            intent="lookup",
            task_type="summary",
            answer_shape="summary",
            entity_terms=["Beijing"],
            semantic_query="Beijing 2026-03-11",
        )
        with patch("app.services.rag_v3.retrievers._generate_hyde_query", return_value="hyde draft") as hyde:
            queries = __import__("app.services.rag_v3.retrievers", fromlist=["_semantic_queries"])._semantic_queries(
                "what was written about beijing",
                plan,
            )
        hyde.assert_not_called()
        self.assertNotIn("hyde draft", queries)

    def test_merge_bundles_reports_hybrid_mode_when_structured_and_semantic_tools_both_used(self):
        structured = EvidenceBundle(
            question="is there any news about manish sisodiya",
            mode="sql",
            plan=_plan(mode="hybrid", intent="lookup", task_type="summary", answer_shape="summary"),
            items=[EvidenceItem(article_id="1", headline="Structured hit", source_type="structured_articles")],
            retrieval_confidence=0.9,
            applied_tools=["structured_articles"],
        )
        semantic = EvidenceBundle(
            question="is there any news about manish sisodiya",
            mode="semantic",
            plan=_plan(mode="hybrid", intent="lookup", task_type="summary", answer_shape="summary"),
            items=[EvidenceItem(article_id="2", headline="Semantic hit", source_type="semantic_chunks")],
            retrieval_confidence=0.6,
            applied_tools=["semantic_chunks"],
        )
        merged = __import__("app.services.rag_v3.retrievers", fromlist=["merge_bundles"]).merge_bundles(
            "is there any news about manish sisodiya",
            structured.plan,
            [structured, semantic],
        )
        self.assertEqual(merged.mode, "hybrid")

    def test_native_semantic_retrieval_does_not_call_old_run_query(self):
        plan = _plan(
            mode="semantic",
            intent="lookup",
            task_type="summary",
            answer_shape="summary",
            retrieval_tools=["semantic_chunks"],
            entity_terms=["Narendra Modi"],
            semantic_query="Narendra Modi economy",
        )
        with (
            patch("app.services.rag_v3.retrievers.embed_texts", return_value=[[0.1, 0.2, 0.3]]),
            patch("app.services.rag_v3.retrievers.semantic_search", return_value=[{"article_id": 11, "chunk_text": "Narendra Modi discussed the economy", "similarity": 0.91}]),
            patch(
                "app.services.rag_v3.retrievers.fetch_articles_for_ids",
                return_value=[
                    {
                        "id": 11,
                        "external_article_id": "11",
                        "headline": "Modi discusses economy",
                        "edition": "Delhi",
                        "section": "Nation",
                        "issue_date": date(2026, 3, 11),
                        "excerpt": "Narendra Modi discussed the economy in Delhi.",
                    }
                ],
            ),
            patch("app.services.rag_v3.retrievers.run_query", side_effect=AssertionError("old run_query should not be used"), create=True),
        ):
            bundle = __import__("app.services.rag_v3.retrievers", fromlist=["retrieve_with_tool"]).retrieve_with_tool(
                "articles about modi",
                plan,
                "semantic_chunks",
                5,
            )
        self.assertEqual(len(bundle.items), 1)
        self.assertEqual(bundle.items[0].headline, "Modi discusses economy")

    def test_structured_articles_person_query_not_polluted_by_author_filter(self):
        plan = _plan(
            mode="hybrid",
            intent="lookup",
            task_type="summary",
            answer_shape="summary",
            author=None,
            entity_terms=["Manish Sisodiya", "Sisodiya"],
            retrieval_tools=["structured_articles"],
        )
        with patch(
            "app.services.rag_v3.retrievers.fetch_entity_mention_articles",
            return_value=[
                {
                    "id": 21,
                    "external_article_id": "21",
                    "headline": "Excise policy case: Delhi HC seeks Kejriwal, Sisodia's response on ED plea to expunge remarks",
                    "edition": "Delhi",
                    "section": "Nation",
                    "issue_date": date(2026, 3, 11),
                    "excerpt": "Delhi HC seeks Kejriwal and Sisodia response.",
                }
            ],
        ):
            bundle = __import__("app.services.rag_v3.retrievers", fromlist=["retrieve_with_tool"]).retrieve_with_tool(
                "is there any news about manish sisodiya",
                plan,
                "structured_articles",
                6,
            )
        self.assertEqual(bundle.items[0].headline, "Excise policy case: Delhi HC seeks Kejriwal, Sisodia's response on ED plea to expunge remarks")

    def test_structured_count_uses_broad_mentions_for_topic_queries(self):
        plan = _plan(
            mode="hybrid",
            intent="topic_count",
            task_type="count",
            answer_shape="count_plus_summary",
            entity_terms=["Narendra Modi", "Modi"],
            retrieval_tools=["structured_count"],
        )
        with (
            patch(
                "app.services.rag_v3.retrievers.fetch_entity_mention_articles",
                return_value=[
                    {
                        "id": 31,
                        "external_article_id": "31",
                        "headline": "Ensure Indian consumers not impacted by conflict: Modi",
                        "edition": "Delhi",
                        "section": "Nation",
                        "issue_date": date(2026, 3, 11),
                        "excerpt": "Modi says consumers should not be impacted by conflict.",
                    }
                ],
            ) as fetch_articles,
            patch("app.services.rag_v3.retrievers.fetch_entity_mention_count", return_value=105) as fetch_count,
            patch("app.services.rag_v3.retrievers.fetch_entity_mention_contexts", return_value=["Conflict and consumer protection featured in Modi coverage."]) as fetch_contexts,
        ):
            bundle = __import__("app.services.rag_v3.retrievers", fromlist=["retrieve_with_tool"]).retrieve_with_tool(
                "how many articles about modi and what were the key points",
                plan,
                "structured_count",
                6,
            )
        self.assertFalse(fetch_articles.call_args.kwargs["headline_priority_only"])
        self.assertFalse(fetch_count.call_args.kwargs["headline_priority_only"])
        self.assertFalse(fetch_contexts.call_args.kwargs["headline_priority_only"])
        self.assertEqual(bundle.raw_filters["exact_article_count"], 105)
        self.assertEqual(bundle.raw_filters["count_scope"], "broad_mentions")

    def test_count_plus_summary_answer_does_not_emit_article_text_citations(self):
        bundle = EvidenceBundle(
            question="how many articles about modi and what were the key points",
            mode="hybrid",
            plan=_plan(),
            items=[
                EvidenceItem(
                    article_id="31",
                    headline="Ensure Indian consumers not impacted by conflict: Modi",
                    edition="Delhi",
                    section="Nation",
                    issue_date="2026-03-11",
                    excerpt="This excerpt should not reach the summary UI.",
                    source_type="structured_count",
                )
            ],
            raw_filters={"exact_article_count": 105, "count_scope": "broad_mentions"},
            retrieval_confidence=1.0,
        )
        draft = generate_answer(
            _intent(),
            bundle,
            DistilledEvidence(
                summary="Coverage focused on conflict, consumer impact, and political messaging.",
                key_points=[
                    "Modi says DMK is rattled by NDA’s popularity in Tamil Nadu (4 articles) in Nation: Trichy coverage",
                    "Modi: India actively promoting animation (4 articles) in Nation: New Delhi coverage",
                ],
                coverage="structured_contexts",
            ),
        )
        self.assertIn("I found 105 relevant articles.", draft.answer)
        self.assertIn("Modi says DMK is rattled by NDA’s popularity in Tamil Nadu (4 articles)", draft.answer)
        self.assertNotIn("Trichy coverage", draft.answer)
        self.assertEqual(draft.citations, [])

    def test_non_explicit_what_was_written_question_is_not_forced_into_article_text(self):
        with (
            patch("app.services.rag_v3.planner.fetch_publication_catalog", return_value=[]),
            patch("app.services.rag_v3.planner.fetch_section_catalog", return_value=["World"]),
            patch("app.services.rag_v3.planner.fetch_author_catalog", return_value=[]),
            patch(
                "app.services.rag_v3.planner.chat_completion",
                return_value='{"task_type":"article_text","answer_shape":"article_text","intent":"fact_lookup","mode":"hybrid","section":"World","needs_article_text":true,"query_focus":"World section coverage","entity_terms":["World"],"retrieval_tools":["structured_articles"],"fallback_order":["semantic_chunks"],"reasoning":"bad article text classification"}',
            ),
        ):
            intent, plan = pipeline.parse_user_intent("what was written in the world section", "2026-03-11")
        self.assertFalse(intent.needs_article_text)
        self.assertTrue(intent.needs_summary)
        self.assertEqual(plan.task_type, "summary")
        self.assertEqual(plan.answer_shape, "summary")

    def test_summary_answer_does_not_emit_article_text_citations(self):
        bundle = EvidenceBundle(
            question="is there any news about kejriwal",
            mode="hybrid",
            plan=_plan(intent="lookup", task_type="summary", answer_shape="summary"),
            items=[
                EvidenceItem(
                    article_id="44",
                    headline="Excise case: HC seeks stand of Kejriwal & 22 others on ED plea",
                    edition="Delhi",
                    section="Nation",
                    issue_date="2026-03-11",
                    excerpt="This excerpt should stay out of summary-mode citations.",
                    source_type="semantic_chunks",
                )
            ],
            retrieval_confidence=0.8,
        )
        draft = generate_answer(
            _intent(
                original_question="is there any news about kejriwal",
                standalone_question="is there any news about kejriwal",
                intent="lookup",
                mode="hybrid",
                needs_count=False,
                needs_summary=True,
                entities={"entity_terms": ["Arvind Kejriwal", "Kejriwal"]},
            ),
            bundle,
            DistilledEvidence(summary="Yes. Coverage centered on the excise case and court proceedings involving Kejriwal.", key_points=[]),
        )
        self.assertEqual(draft.citations, [])

    def test_verifier_rejects_weak_semantic_match_for_lookup_query(self):
        intent = _intent(
            original_question="what did professor lupin teach in this archive",
            standalone_question="what did professor lupin teach in this archive",
            intent="lookup",
            needs_count=False,
            needs_summary=True,
            entities={"entity_terms": ["professor lupin"]},
        )
        bundle = EvidenceBundle(
            question=intent.standalone_question,
            mode="semantic",
            plan=_plan(mode="semantic", intent="lookup", task_type="summary", answer_shape="summary"),
            items=[
                EvidenceItem(
                    article_id="8",
                    headline="Market volatility grows after oil shock",
                    excerpt="Economists said inflation concerns were rising.",
                    source_type="semantic_chunks",
                )
            ],
            retrieval_confidence=0.64,
        )
        draft = AnswerDraft(answer="The archive suggests he taught Defence Against the Dark Arts.", mode="semantic")
        report = verify_answer(intent, bundle, draft)
        self.assertFalse(report.answer_accepted)
        self.assertTrue(any("core query terms" in note for note in report.unsupported_claims))

    def test_verifier_can_accept_semantically_grounded_answer_when_token_overlap_is_weak(self):
        intent = _intent(
            original_question="what was the news about Kiran Rijju",
            standalone_question="what was the news about Kiran Rijju",
            intent="lookup",
            needs_count=False,
            needs_summary=True,
            entities={"entity_terms": ["Kiran Rijju", "Rijju"]},
        )
        bundle = EvidenceBundle(
            question=intent.standalone_question,
            mode="hybrid",
            plan=_plan(mode="hybrid", intent="lookup", task_type="summary", answer_shape="summary"),
            items=[
                EvidenceItem(
                    article_id="9",
                    headline="Rijiju slams Cong for motion against Birla",
                    excerpt="Kiren Rijiju criticised Congress for a no-confidence move against Om Birla.",
                    source_type="semantic_chunks",
                )
            ],
            retrieval_confidence=0.62,
        )
        draft = AnswerDraft(
            answer="Kiren Rijiju criticised Congress for bringing a no-confidence motion against Om Birla.",
            mode="hybrid",
        )
        with patch(
            "app.services.rag_v3.verifier.chat_completion",
            return_value='{"grounded": true, "rationale": "The answer matches the retrieved evidence."}',
        ):
            report = verify_answer(intent, bundle, draft)
        self.assertTrue(report.answer_accepted)
        self.assertFalse(report.unsupported_claims)

    def test_verifier_semantic_rejection_blocks_bad_summary_even_when_evidence_exists(self):
        intent = _intent(
            original_question="is there any news about manish sisodiya",
            standalone_question="is there any news about manish sisodiya",
            intent="lookup",
            needs_count=False,
            needs_summary=True,
            entities={"entity_terms": ["Manish Sisodiya", "Sisodia"]},
        )
        bundle = EvidenceBundle(
            question=intent.standalone_question,
            mode="hybrid",
            plan=_plan(mode="hybrid", intent="lookup", task_type="summary", answer_shape="summary"),
            items=[
                EvidenceItem(
                    article_id="10",
                    headline="Excise case: HC seeks stand of Kejriwal & 22 others on ED plea",
                    excerpt="Delhi High Court sought the stand of former chief minister Arvind Kejriwal and ex-deputy CM Manish Sisodia.",
                    source_type="structured_articles",
                )
            ],
            retrieval_confidence=0.88,
        )
        draft = AnswerDraft(
            answer="The coverage was about economic growth and stock market momentum.",
            mode="hybrid",
        )
        with patch(
            "app.services.rag_v3.verifier.chat_completion",
            return_value='{"grounded": false, "rationale": "The answer is not supported by the evidence."}',
        ):
            report = verify_answer(intent, bundle, draft)
        self.assertFalse(report.answer_accepted)
        self.assertTrue(any("Semantic verification found" in note for note in report.unsupported_claims))

    def test_verifier_accepts_explicit_abstention_for_weak_lookup_query(self):
        intent = _intent(
            original_question="what did professor lupin teach in this archive",
            standalone_question="what did professor lupin teach in this archive",
            intent="lookup",
            needs_count=False,
            needs_summary=True,
            entities={"entity_terms": ["professor lupin"]},
        )
        bundle = EvidenceBundle(
            question=intent.standalone_question,
            mode="semantic",
            plan=_plan(mode="semantic", intent="lookup", task_type="summary", answer_shape="summary"),
            items=[
                EvidenceItem(
                    article_id="8",
                    headline="Market volatility grows after oil shock",
                    excerpt="Economists said inflation concerns were rising.",
                    source_type="semantic_chunks",
                )
            ],
            retrieval_confidence=0.64,
        )
        draft = AnswerDraft(
            answer="I couldn't find enough grounded evidence in the current dataset to answer that confidently.",
            mode="semantic",
            grounded=False,
        )
        report = verify_answer(intent, bundle, draft)
        self.assertTrue(report.answer_accepted)
        self.assertTrue(any("abstained" in note.lower() for note in report.supported_claims))

    def test_distiller_uses_deterministic_exact_contexts_for_count_plus_summary(self):
        bundle = EvidenceBundle(
            question="how many articles about modi and what were the key points",
            mode="sql",
            plan=_plan(mode="sql", retrieval_tools=["structured_count"], answer_shape="count_plus_summary"),
            items=[
                EvidenceItem(
                    article_id="31",
                    headline="Ensure Indian consumers not impacted by conflict: Modi",
                    source_type="structured_count",
                )
            ],
            raw_filters={
                "exact_article_count": 105,
                "exact_contexts": [
                    {
                        "headline": "Modi says DMK is rattled by NDA’s popularity in Tamil Nadu",
                        "article_count": 4,
                        "section": "Nation",
                        "excerpt": "Coverage focused on DMK and NDA politics in Tamil Nadu.",
                    },
                    {
                        "headline": "Modi: India actively promoting animation",
                        "article_count": 4,
                        "section": "Nation",
                        "excerpt": "Several stories discussed animation and creative economy promotion.",
                    },
                ],
            },
            retrieval_confidence=1.0,
        )
        distilled = distill_evidence(_intent(), bundle)
        self.assertEqual(distilled.coverage, "structured_contexts")
        self.assertIn("Modi says DMK is rattled", distilled.summary)
        self.assertNotIn("{'headline'", distilled.summary)
        self.assertIn("; ", distilled.summary)
        self.assertIn("(4 articles)", distilled.summary)
        self.assertEqual(
            distilled.notes,
            ["Used deterministic entity contexts without LLM distillation."],
        )

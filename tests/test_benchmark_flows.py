import unittest
from contextlib import ExitStack
from unittest.mock import patch

from app.schemas import QueryResponse, RoutedQuery
from app.services import chat_service, query_analyzer, query_router


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
            patch("app.services.query_analyzer.fetch_publication_catalog", return_value=self.publications),
            patch("app.services.query_analyzer.fetch_section_catalog", return_value=self.sections),
            patch("app.services.query_analyzer.fetch_author_catalog", return_value=["Abhinav Garg"]),
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

    def test_author_prompt_routes_to_author_lookup(self):
        routed = self._route("tell me article which author abhinav garg wrote")
        self.assertEqual(routed.mode, "sql")
        self.assertEqual(routed.intent, "author_lookup")
        self.assertEqual(routed.author, "Abhinav Garg")

    def test_author_prompt_with_typo_still_resolves_author(self):
        routed = self._route("how many articles abhimnav garg has written and what they about")
        self.assertEqual(routed.intent, "author_count")
        self.assertEqual(routed.author, "Abhinav Garg")

    def test_fact_count_question_routes_to_fact_lookup(self):
        routed = self._route("How many households in Ludhiana are registered for LPG cylinders?")
        self.assertEqual(routed.mode, "semantic")
        self.assertEqual(routed.intent, "fact_lookup")

    def test_topic_extraction_strips_context_suffix_noise(self):
        with (
            patch("app.services.query_analyzer.fetch_publication_catalog", return_value=self.publications),
            patch("app.services.query_analyzer.fetch_section_catalog", return_value=self.sections),
            patch("app.services.query_analyzer.fetch_author_catalog", return_value=[]),
        ):
            analysis = query_analyzer.analyze_query(
                "how many articles about rahul gandhi and and in what context they are",
                "2026-03-11",
            )
        self.assertEqual(analysis.routed.semantic_query, "rahul gandhi")

    def test_topic_count_handles_singular_article_and_about_typo(self):
        with (
            patch("app.services.query_analyzer.fetch_publication_catalog", return_value=self.publications),
            patch("app.services.query_analyzer.fetch_section_catalog", return_value=self.sections),
            patch("app.services.query_analyzer.fetch_author_catalog", return_value=[]),
        ):
            analysis = query_analyzer.analyze_query(
                "how many article aboput narendra modi",
                "2026-03-11",
            )
        self.assertEqual(analysis.routed.intent, "topic_count")
        self.assertEqual(analysis.routed.semantic_query, "narendra modi")

    def test_analyzer_extracts_people_places_and_orgs(self):
        with (
            patch("app.services.query_analyzer.fetch_publication_catalog", return_value=self.publications),
            patch("app.services.query_analyzer.fetch_section_catalog", return_value=self.sections),
            patch("app.services.query_analyzer.fetch_author_catalog", return_value=["Abhinav Garg"]),
        ):
            analysis = query_analyzer.analyze_query(
                "How many times Modi name appeared in BCCI coverage in Ludhiana?",
                "2026-03-11",
            )
        self.assertIn("Narendra Modi", analysis.entities["people"])
        self.assertIn("ludhiana", analysis.entities["places"])
        self.assertIn("bcci", analysis.entities["organizations"])
        self.assertIn("Narendra Modi", analysis.entities["content_people"])
        self.assertIn("ludhiana", analysis.entities["content_locations"])

    def test_analyzer_treats_ludhiana_as_edition_filter_when_edition_is_asked(self):
        publications = self.publications + [
            {"id": "ludhiana", "publication_name": "TOIChandigarhBS - TimesOfLudhiana_Digital"},
        ]
        with (
            patch("app.services.query_analyzer.fetch_publication_catalog", return_value=publications),
            patch("app.services.query_analyzer.fetch_section_catalog", return_value=self.sections),
            patch("app.services.query_analyzer.fetch_author_catalog", return_value=[]),
        ):
            analysis = query_analyzer.analyze_query(
                "how many articles are there in ludhiana edition",
                "2026-03-11",
            )
        self.assertIn("TOIChandigarhBS - TimesOfLudhiana_Digital", analysis.entities["edition_filters"])
        self.assertNotIn("ludhiana", analysis.entities["content_locations"])

    def test_analyzer_treats_ludhiana_as_content_location_when_not_a_filter(self):
        publications = self.publications + [
            {"id": "ludhiana", "publication_name": "TOIChandigarhBS - TimesOfLudhiana_Digital"},
        ]
        with (
            patch("app.services.query_analyzer.fetch_publication_catalog", return_value=publications),
            patch("app.services.query_analyzer.fetch_section_catalog", return_value=self.sections),
            patch("app.services.query_analyzer.fetch_author_catalog", return_value=[]),
        ):
            analysis = query_analyzer.analyze_query(
                "How many households in Ludhiana are registered for LPG cylinders?",
                "2026-03-11",
            )
        self.assertEqual(analysis.entities["edition_filters"], [])
        self.assertIn("ludhiana", analysis.entities["content_locations"])


class ChatBenchmarkTests(unittest.TestCase):
    def _analysis_for(self, question: str, routed: RoutedQuery):
        return query_analyzer.QueryAnalysis(
            raw_query=question,
            normalized_query=question.lower(),
            lowered_query=question.lower(),
            routed=routed,
            entities={},
            ambiguous_edition=False,
        )

    def _route_for(
        self,
        *,
        mode: str,
        intent: str = "lookup",
        edition: str | None = None,
        section: str | None = None,
        author: str | None = None,
        semantic_query: str | None = None,
    ):
        return RoutedQuery(
            mode=mode,
            intent=intent,
            issue_date="2026-03-11",
            edition=edition,
            section=section,
            author=author,
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
            stack.enter_context(
                patch(
                    "app.services.chat_service.analyze_query",
                    side_effect=lambda question, issue_date=None: self._analysis_for(question, routed),
                )
            )
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
            routed=self._route_for(mode="sql", intent="article_count", edition="TOIDelhiBS - Toi_Noida_Digital"),
            extra_patches=[patch("app.services.chat_service.fetch_sql_article_count", return_value=171)],
        )
        self.assertIn("171", response.answer)
        self.assertEqual(response.citations, [])
        self.assertEqual(response.debug_trace["answer_path"], "count_answer")

    def test_author_count_question_returns_author_count(self):
        query_response = QueryResponse(
            mode="sql",
            filters={"issue_date": "2026-03-11", "author": "Abhinav Garg"},
            results=[{"author_article_count": 5}],
        )
        response = self._answer(
            "How many articles did Abhinav Garg write?",
            query_response,
            routed=self._route_for(mode="sql", intent="author_count", author="Abhinav Garg"),
        )
        self.assertIn("5 articles by Abhinav Garg", response.answer)
        self.assertEqual(response.citations, [])

    def test_author_count_with_context_returns_topics(self):
        query_response = QueryResponse(
            mode="sql",
            filters={"issue_date": "2026-03-11", "author": "Abhinav Garg"},
            results=[
                {
                    **_article(
                        article_id="71",
                        headline="Excise case: HC seeks stand of Kejriwal & 22 others on ED plea",
                        edition="TOIDelhiBS - Toi_Noida_Digital",
                        section="Nation",
                        excerpt="The report follows the HC hearing in the excise case.",
                    ),
                    "author_article_count": 5,
                },
                {
                    **_article(
                        article_id="72",
                        headline="Don’t demolish Uttam Nagar houses till hearing today: HC",
                        edition="TOIDelhiBS - Toi_Noida_Digital",
                        section="City",
                        excerpt="The story tracks a demolition dispute before the HC.",
                    ),
                    "author_article_count": 5,
                },
            ],
        )
        response = self._answer(
            "how many article author abhinav garg wrote and what they about",
            query_response,
            routed=self._route_for(mode="sql", intent="author_count", author="Abhinav Garg"),
        )
        self.assertIn("5 articles by Abhinav Garg", response.answer)
        self.assertIn("They are mainly about", response.answer)

    def test_contextual_followup_uses_session_context_without_new_search(self):
        routed = self._route_for(mode="sql", intent="author_count", author="Abhinav Garg")
        with patch(
            "app.services.chat_service.analyze_query",
            side_effect=lambda question, issue_date=None: self._analysis_for(question, routed),
        ):
            response = chat_service.answer_question(
                "and what they were about",
                "2026-03-11",
                10,
                session_context={
                    "author": "Abhinav Garg",
                    "last_mode": "sql",
                    "result_count": 5,
                    "story_candidates": [
                        {"headline": "Excise case: HC seeks stand of Kejriwal & 22 others on ED plea"},
                        {"headline": "Don’t demolish Uttam Nagar houses till hearing today: HC"},
                    ],
                },
            )
        self.assertIn("I found 5 articles by Abhinav Garg", response.answer)
        self.assertIn("Excise case", response.answer)

    def test_author_lookup_returns_summary(self):
        query_response = QueryResponse(
            mode="sql",
            filters={"issue_date": "2026-03-11", "author": "Abhinav Garg"},
            results=[
                {
                    **_article(
                        article_id="61",
                        headline="Excise case: HC seeks stand of Kejriwal & 22 others on ED plea",
                        edition="TOIDelhiBS - Toi_Noida_Digital",
                        section="Nation",
                        excerpt="Abhinav Garg reported on the HC proceedings in the excise case.",
                    ),
                    "author_article_count": 5,
                },
                {
                    **_article(
                        article_id="62",
                        headline="Don’t demolish Uttam Nagar houses till hearing today: HC",
                        edition="TOIDelhiBS - Toi_Noida_Digital",
                        section="City",
                        excerpt="The story followed a court stay around demolitions in Uttam Nagar.",
                    ),
                    "author_article_count": 5,
                },
            ],
        )
        response = self._answer(
            "tell me article which author abhinav garg wrote",
            query_response,
            routed=self._route_for(mode="sql", intent="author_lookup", author="Abhinav Garg"),
        )
        self.assertIn("5 articles by Abhinav Garg", response.answer)
        self.assertIn("Excise case", response.answer)

    def test_fact_lookup_question_does_not_fall_back_to_day_count(self):
        query_response = QueryResponse(
            mode="semantic",
            filters={"issue_date": "2026-03-11"},
            results=[
                _article(
                    article_id="63",
                    headline="In crunch, govt says domestic PNG, CNG, LPG prodn priority",
                    edition="TOIChandigarhBS - Chandigarh_Digital",
                    section="Business",
                    excerpt="The article says 8.6 lakh households in Ludhiana are registered for LPG cylinders.",
                ),
            ],
        )
        response = self._answer(
            "How many households in Ludhiana are registered for LPG cylinders?",
            query_response,
            routed=self._route_for(mode="semantic", intent="fact_lookup", semantic_query="lpg households ludhiana"),
            extra_patches=[
                patch(
                    "app.services.chat_service.chat_completion",
                    return_value="The article says 8.6 lakh households in Ludhiana are registered for LPG cylinders.",
                )
            ],
        )
        self.assertIn("8.6 lakh households", response.answer)
        self.assertNotIn("5607", response.answer)

    def test_topic_count_uses_exact_article_count_when_available(self):
        query_response = QueryResponse(
            mode="sql",
            filters={
                "issue_date": "2026-03-11",
                "exact_article_count": 83,
                "retrieval_strategy": "exact_entity_mentions",
                "entity_label": "Rahul Gandhi",
                "exact_contexts": [
                    {"headline": "Rahul attacking institutions as he can’t win polls: Min", "article_count": 20},
                    {"headline": "Om Birla’s behaviour partisan, LoP cut off 20 times, says Gogoi", "article_count": 13},
                ],
            },
            results=[
                _article(
                    article_id="81",
                    headline="Rahul attacking institutions as he can’t win polls: Min",
                    edition="TOIDelhiBS - Toi_Noida_Digital",
                    section="Nation",
                    excerpt="A minister accused Rahul Gandhi of attacking institutions.",
                ),
                _article(
                    article_id="82",
                    headline="Om Birla’s behaviour partisan, LoP cut off 20 times, says Gogoi",
                    edition="TOIDelhiBS - Toi_Noida_Digital",
                    section="Nation",
                    excerpt="Congress leaders defended Rahul Gandhi during the no-trust debate.",
                ),
            ],
        )
        response = self._answer(
            "how many articles about rahul gandhi and in what context they are",
            query_response,
            routed=self._route_for(mode="sql", intent="topic_count"),
        )
        self.assertIn("83 relevant articles", response.answer)
        self.assertIn("Rahul Gandhi", response.answer)

    def test_topic_count_uses_canonical_modi_label_and_exact_contexts(self):
        query_response = QueryResponse(
            mode="sql",
            filters={
                "issue_date": "2026-03-11",
                "exact_article_count": 277,
                "retrieval_strategy": "exact_entity_mentions",
                "entity_label": "Narendra Modi",
                "exact_contexts": [
                    {"headline": "In crunch, govt says domestic PNG, CNG, LPG prodn priority", "article_count": 24},
                    {"headline": "Modi says DMK is rattled by NDA’s popularity in Tamil Nadu", "article_count": 4},
                ],
            },
            results=[],
        )
        response = self._answer(
            "how many articles regarding modi and in what context",
            query_response,
            routed=self._route_for(mode="sql", intent="topic_count"),
        )
        self.assertIn("277 relevant articles mentioning Narendra Modi", response.answer)
        self.assertNotIn("Modi And", response.answer)
        self.assertIn("domestic PNG, CNG, LPG", response.answer)

    def test_topic_count_handles_in_which_context_wording(self):
        query_response = QueryResponse(
            mode="sql",
            filters={
                "issue_date": "2026-03-11",
                "exact_article_count": 38,
                "retrieval_strategy": "exact_entity_mentions",
                "entity_label": "Narendra Modi",
                "exact_contexts": [
                    {"headline": "Modi addresses NDA workers ahead of state push", "article_count": 7},
                    {"headline": "Fuel, LPG and PNG policy remains a recurring Modi mention", "article_count": 5},
                ],
            },
            results=[],
        )
        response = self._answer(
            "how many articles about modi and in which context",
            query_response,
            routed=self._route_for(mode="sql", intent="topic_count"),
        )
        self.assertIn("38 relevant articles mentioning Narendra Modi", response.answer)
        self.assertNotIn("Narendra Modi And", response.answer)
        self.assertIn("mainly in the context", response.answer)

    def test_topic_count_question_cleanup_strips_in_which_context_suffix(self):
        self.assertEqual(
            chat_service._extract_topic_from_question("how many articles about modi and in which context they are"),
            "Narendra Modi",
        )

    def test_person_topic_query_marks_subject_strict(self):
        routed = self._route_for(mode="sql", intent="topic_count", semantic_query="narendra modi")
        analysis = self._analysis_for(
            "how many articles about modi and in which context they are",
            routed,
        )
        analysis.entities = {
            "content_people": ["Narendra Modi"],
            "people": ["Narendra Modi"],
            "content_organizations": [],
            "organizations": [],
        }
        with patch("app.services.query_service.fetch_entity_mention_articles", return_value=[]), patch(
            "app.services.query_service.fetch_entity_mention_count", return_value=12
        ), patch(
            "app.services.query_service.fetch_entity_mention_contexts",
            return_value=[{"headline": "Modi says DMK is rattled by NDA’s popularity in Tamil Nadu", "article_count": 4}],
        ), patch("app.services.query_service.analyze_query", return_value=analysis), patch(
            "app.services.query_service.embed_texts", return_value=[[0.0] * 5]
        ):
            response = chat_service.run_query(
                "how many articles about modi and in which context they are",
                "2026-03-11",
                10,
            )
        self.assertTrue(response.filters["subject_strict"])
        self.assertEqual(response.filters["entity_label"], "Narendra Modi")

    def test_topic_count_display_label_normalizes_about_typo(self):
        query_response = QueryResponse(
            mode="sql",
            filters={
                "issue_date": "2026-03-11",
                "exact_article_count": 62,
                "retrieval_strategy": "exact_entity_mentions",
            },
            results=[],
        )
        response = self._answer(
            "how many article aboput narendra modi",
            query_response,
            routed=self._route_for(mode="sql", intent="topic_count"),
        )
        self.assertIn("about Narendra Modi", response.answer)
        self.assertNotIn("Aboput Narendra Modi", response.answer)

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

    def test_llm_prompt_uses_layered_answer_structure(self):
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
            ],
        )
        with patch("app.services.chat_service.chat_completion", return_value="Layered answer") as mock_chat:
            response = self._answer(
                "Find articles related to the Iran war with references",
                query_response,
                routed=self._route_for(mode="semantic", semantic_query="iran war"),
            )
        self.assertEqual(response.answer, "Layered answer")
        system_prompt, user_prompt = mock_chat.call_args.args
        self.assertIn("Role:", system_prompt)
        self.assertIn("Grounding:", system_prompt)
        self.assertIn("Layer 1 - User question:", user_prompt)
        self.assertIn("Layer 4 - Evidence:", user_prompt)
        self.assertIn("[Evidence 1]", user_prompt)
        self.assertIn("Layer 5 - Edge-case policy:", user_prompt)

    def test_story_summary_prompt_uses_layered_structure(self):
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
                    headline="BCCI announces 131 cr reward for World Champions",
                    edition="TOIMumbaiBS - MumbaiCity_Digital",
                    section="Sports",
                    excerpt="India's World Cup-winning squad was rewarded by the board.",
                ),
            ],
        )
        with patch("app.services.chat_service.chat_completion", return_value="Story summary") as mock_chat:
            response = self._answer(
                "Which stories covered India's World Cup win?",
                query_response,
                routed=self._route_for(mode="hybrid", section="Sports", semantic_query="india world cup win"),
            )
        self.assertEqual(response.answer, "Story summary")
        _, user_prompt = mock_chat.call_args.args
        self.assertIn("Layer 3 - Story evidence:", user_prompt)
        self.assertIn("[Story 1]", user_prompt)
        self.assertIn("Layer 4 - Summary policy:", user_prompt)

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
            routed = self._route_for(mode="semantic", semantic_query="rahul gandhi")
            stack.enter_context(
                patch(
                    "app.services.chat_service.analyze_query",
                    side_effect=lambda question, issue_date=None: self._analysis_for(question, routed),
                )
            )
            stack.enter_context(
                patch(
                    "app.services.chat_service.route_query",
                    return_value=routed,
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
            routed = self._route_for(mode="hybrid", section="Edit", semantic_query="world newest beijing problem")
            stack.enter_context(
                patch(
                    "app.services.chat_service.analyze_query",
                    side_effect=lambda question, issue_date=None: self._analysis_for(question, routed),
                )
            )
            stack.enter_context(
                patch(
                    "app.services.chat_service.route_query",
                    return_value=routed,
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

    def test_author_context_supports_three_turn_chain(self):
        query_response = QueryResponse(
            mode="sql",
            filters={"issue_date": "2026-03-11", "author": "Abhinav Garg"},
            results=[
                {
                    **_article(
                        article_id="91",
                        headline="Excise case: HC seeks stand of Kejriwal & 22 others on ED plea",
                        edition="TOIDelhiBS - Toi_Noida_Digital",
                        section="Nation",
                        excerpt="The report follows the HC hearing in the excise case.",
                    ),
                    "author_article_count": 5,
                },
                {
                    **_article(
                        article_id="92",
                        headline="Don’t demolish Uttam Nagar houses till hearing today: HC",
                        edition="TOIDelhiBS - Toi_Noida_Digital",
                        section="City",
                        excerpt="The story tracks a demolition dispute before the HC.",
                    ),
                    "author_article_count": 5,
                },
            ],
        )
        first = self._answer(
            "how many article author abhinav garg wrote and what they about",
            query_response,
            routed=self._route_for(mode="sql", intent="author_count", author="Abhinav Garg"),
        )
        second = chat_service.answer_question(
            "and what they were about",
            "2026-03-11",
            10,
            session_context=first.session_context,
        )
        third = chat_service.answer_question(
            "show any one article",
            "2026-03-11",
            10,
            session_context=second.session_context,
        )
        self.assertIn("5 articles by Abhinav Garg", first.answer)
        self.assertIn("They were mainly about", second.answer)
        self.assertIn("Excise case", second.answer)
        self.assertIn("Here is one relevant article excerpt", third.answer)
        self.assertEqual(len(third.citations), 1)

    def test_topic_context_supports_three_turn_chain(self):
        query_response = QueryResponse(
            mode="sql",
            filters={
                "issue_date": "2026-03-11",
                "exact_article_count": 38,
                "retrieval_strategy": "exact_entity_mentions",
                "entity_label": "Narendra Modi",
                "exact_contexts": [
                    {"headline": "Modi addresses NDA workers ahead of state push", "article_count": 7},
                    {"headline": "Fuel, LPG and PNG policy remains a recurring Modi mention", "article_count": 5},
                ],
            },
            results=[
                _article(
                    article_id="93",
                    headline="Modi addresses NDA workers ahead of state push",
                    edition="TOIMumbaiBS - MumbaiCity_Digital",
                    section="Nation",
                    excerpt="The story focuses on campaign positioning and alliance messaging.",
                ),
                _article(
                    article_id="94",
                    headline="Fuel, LPG and PNG policy remains a recurring Modi mention",
                    edition="TOIDelhiBS - Toi_Noida_Digital",
                    section="Business",
                    excerpt="The coverage links Modi mentions to fuel supply and LPG policy debates.",
                ),
            ],
        )
        first = self._answer(
            "how many articles about modi and in which context",
            query_response,
            routed=self._route_for(mode="sql", intent="topic_count"),
        )
        second = chat_service.answer_question(
            "what were those about",
            "2026-03-11",
            10,
            session_context=first.session_context,
        )
        third = chat_service.answer_question(
            "show any article",
            "2026-03-11",
            10,
            session_context=second.session_context,
        )
        self.assertIn("38 relevant articles mentioning Narendra Modi", first.answer)
        self.assertIn("They were mainly about", second.answer)
        self.assertIn("Modi addresses NDA workers", second.answer)
        self.assertIn("Here is one relevant article excerpt", third.answer)
        self.assertEqual(len(third.citations), 1)

    def test_edition_context_supports_follow_up_clarification_chain(self):
        session_context = {
            "last_mode": "sql",
            "issue_date": "2026-03-11",
            "ambiguous_edition": "Delhi",
            "ambiguous_publications": [
                {"publication_name": "TOIDelhiBS - Agra_Digital", "article_count": 134},
                {"publication_name": "TOIDelhiBS - Bareilly_Digital", "article_count": 134},
                {"publication_name": "TOIDelhiBS - Toi_Noida_Digital", "article_count": 171},
            ],
            "edition": "TOIDelhiBS - Toi_Noida_Digital",
        }
        first = chat_service.answer_question(
            "which exact editions are available",
            "2026-03-11",
            10,
            session_context=session_context,
        )
        second = chat_service.answer_question(
            "what edition did you use",
            "2026-03-11",
            10,
            session_context={**session_context, "ambiguous_publications": []},
        )
        third = chat_service.answer_question(
            "which edition was used",
            "2026-03-11",
            10,
            session_context={**session_context, "ambiguous_publications": [], "last_mode": "hybrid"},
        )
        self.assertIn("available editions", first.answer)
        self.assertIn("TOIDelhiBS - Toi_Noida_Digital", first.answer)
        self.assertIn("I used the edition filter TOIDelhiBS - Toi_Noida_Digital", second.answer)
        self.assertIn("I used the edition filter TOIDelhiBS - Toi_Noida_Digital", third.answer)


if __name__ == "__main__":
    unittest.main()

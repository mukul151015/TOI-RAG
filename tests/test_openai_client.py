import unittest
from unittest.mock import patch

from app.services import openai_client


class _FakeResponses:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return type("Response", (), {"output_text": f"ok-{self.calls}"})()


class _FakeEmbeddings:
    def create(self, **kwargs):
        return type("EmbeddingsResponse", (), {"data": []})()


class _FakeClient:
    def __init__(self):
        self.responses = _FakeResponses()
        self.embeddings = _FakeEmbeddings()


class _FakeHttpClient:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class OpenAIClientTests(unittest.TestCase):
    def setUp(self):
        openai_client.close_openai_client()
        openai_client._chat_completion_cached.cache_clear()
        openai_client._embed_texts_cached.cache_clear()

    def tearDown(self):
        openai_client.close_openai_client()
        openai_client._chat_completion_cached.cache_clear()
        openai_client._embed_texts_cached.cache_clear()

    def test_chat_completion_uses_cache_for_identical_prompt(self):
        fake_client = _FakeClient()
        with patch("app.services.openai_client.get_client", return_value=fake_client):
            first = openai_client.chat_completion("system", "user", model="gpt-test", timeout=5.0)
            second = openai_client.chat_completion("system", "user", model="gpt-test", timeout=5.0)
        self.assertEqual(first, "ok-1")
        self.assertEqual(second, "ok-1")
        self.assertEqual(fake_client.responses.calls, 1)

    def test_close_openai_client_closes_http_client(self):
        fake_http = _FakeHttpClient()
        openai_client._http_client = fake_http
        openai_client._client = object()
        openai_client.close_openai_client()
        self.assertTrue(fake_http.closed)
        self.assertIsNone(openai_client._http_client)
        self.assertIsNone(openai_client._client)


if __name__ == "__main__":
    unittest.main()

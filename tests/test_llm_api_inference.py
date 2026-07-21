from unittest.mock import MagicMock, patch

import pytest

from src.self_rag.llm_api_inference import LLMAPIInference
from src.self_rag.self_healing_graph import build_self_healing_pipeline


class TestLLMAPIInferenceInit:
    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No API key"):
            LLMAPIInference(provider="openai")

    def test_reads_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        inference = LLMAPIInference(provider="openai")
        assert inference.api_key == "sk-test"

    def test_explicit_api_key_overrides_env(self):
        inference = LLMAPIInference(provider="openai", api_key="explicit-key")
        assert inference.api_key == "explicit-key"

    def test_default_models(self):
        assert LLMAPIInference(provider="openai", api_key="x").model == "gpt-4o-mini"
        assert LLMAPIInference(provider="anthropic", api_key="x").model == "claude-3-5-haiku-20241022"

    def test_unknown_provider_rejected(self):
        with pytest.raises(ValueError):
            LLMAPIInference(provider="not-a-real-provider", api_key="x")


class TestParsing:
    def test_parses_well_formed_json(self):
        raw = '{"answer": "30 days notice.", "isrel": "Relevant", "issup": "Fully supported"}'
        out = LLMAPIInference._parse(raw)
        assert out.answer == "30 days notice."
        assert out.isrel == "[Relevant]"
        assert out.issup == "[Fully supported]"

    def test_parses_markdown_fenced_json(self):
        raw = '```json\n{"answer": "x", "isrel": "Irrelevant", "issup": "No support"}\n```'
        out = LLMAPIInference._parse(raw)
        assert out.answer == "x"
        assert out.isrel == "[Irrelevant]"
        assert "No support" in out.issup

    def test_malformed_json_falls_back_to_raw_text(self):
        out = LLMAPIInference._parse("The model just said something without JSON.")
        assert out.answer == "The model just said something without JSON."
        # Falls back to a non-blocking-but-not-confident critique, not silently "fully supported"
        assert out.issup == "[Partially supported]"

    def test_missing_fields_default_sensibly(self):
        out = LLMAPIInference._parse('{"answer": "an answer"}')
        assert out.answer == "an answer"
        assert out.isrel == "[Relevant]"
        assert out.issup == "[Partially supported]"


class TestGenerate:
    def test_openai_call_shape_and_response_parsing(self):
        inference = LLMAPIInference(provider="openai", api_key="fake-key")
        fake_json = '{"answer": "12 months.", "isrel": "Relevant", "issup": "Fully supported"}'

        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"choices": [{"message": {"content": fake_json}}]}
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            result = inference.generate("What is the contract duration?", passage="TERM: 12 months.")

            assert result.answer == "12 months."
            assert result.issup == "[Fully supported]"

            call_kwargs = mock_post.call_args
            assert call_kwargs.args[0] == "https://api.openai.com/v1/chat/completions"
            body = call_kwargs.kwargs["json"]
            assert body["model"] == "gpt-4o-mini"
            assert "TERM: 12 months." in body["messages"][1]["content"]
            assert "What is the contract duration?" in body["messages"][1]["content"]

    def test_anthropic_call_shape_and_response_parsing(self):
        inference = LLMAPIInference(provider="anthropic", api_key="fake-key")
        fake_json = '{"answer": "12 months.", "isrel": "Relevant", "issup": "Fully supported"}'

        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"content": [{"text": fake_json}]}
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            result = inference.generate("What is the contract duration?", passage="TERM: 12 months.")

            assert result.answer == "12 months."
            call_kwargs = mock_post.call_args
            assert call_kwargs.args[0] == "https://api.anthropic.com/v1/messages"
            assert call_kwargs.kwargs["headers"]["x-api-key"] == "fake-key"

    def test_no_passage_still_produces_a_prompt(self):
        inference = LLMAPIInference(provider="openai", api_key="fake-key")
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": '{"answer": "I don\'t know.", "isrel": "Irrelevant", "issup": "No support"}'}}]
            }
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            result = inference.generate("What is the governing law?", passage=None)
            body = mock_post.call_args.kwargs["json"]
            assert "No passage was retrieved" in body["messages"][1]["content"]
            assert result.isrel == "[Irrelevant]"


class TestGenerateJson:
    def test_returns_parsed_dict_with_custom_shape(self):
        inference = LLMAPIInference(provider="openai", api_key="fake-key")
        fake_json = '{"parties": ["Acme", "Beta"], "dates": ["2024-01-01"], "risks": []}'

        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"choices": [{"message": {"content": fake_json}}]}
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            result = inference.generate_json("Extract parties, dates, risks as JSON.", context="some contract text")

            assert result == {"parties": ["Acme", "Beta"], "dates": ["2024-01-01"], "risks": []}
            body = mock_post.call_args.kwargs["json"]
            # Must NOT use the Self-RAG critique system prompt for a generic extraction call
            assert "isrel" not in body["messages"][0]["content"]
            assert "some contract text" in body["messages"][1]["content"]

    def test_returns_none_on_invalid_json(self):
        inference = LLMAPIInference(provider="openai", api_key="fake-key")
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"choices": [{"message": {"content": "not json"}}]}
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            result = inference.generate_json("Extract something.")
            assert result is None


class TestPluggableIntoSelfHealingGraph:
    def test_build_self_healing_pipeline_accepts_llm_api_inference(self):
        """LLMAPIInference must satisfy the exact interface
        build_self_healing_pipeline expects from SelfRAGGGUFInference --
        this is the whole point of matching the .generate() signature."""
        inference = LLMAPIInference(provider="openai", api_key="fake-key")

        class FakeRetriever:
            def retrieve(self, query, top_k=3):
                return [{"text": "TERMINATION: 30 days written notice.", "score": 0.9}]

        fake_json = '{"answer": "30 days notice.", "isrel": "Relevant", "issup": "Fully supported"}'
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"choices": [{"message": {"content": fake_json}}]}
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            pipeline = build_self_healing_pipeline(inference, FakeRetriever(), max_attempts=2)
            state = pipeline.run("What is the termination clause?")

            assert state["accepted"] is True
            assert state["result"].answer == "30 days notice."

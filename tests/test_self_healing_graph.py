from src.self_rag.self_healing_graph import (
    FALLBACK_MESSAGE,
    GenerationResult,
    SelfHealingRAG,
    default_reformulate,
)


def make_pipeline(generate_fn, retrieve_fn=None, max_attempts=2):
    if retrieve_fn is None:
        def retrieve_fn(query, top_k):
            return [{"text": f"passage for {query}", "score": 0.5}]
    return SelfHealingRAG(generate_fn=generate_fn, retrieve_fn=retrieve_fn, max_attempts=max_attempts)


class TestSelfHealingRAG:
    def test_accepts_a_good_first_answer_without_retrying(self):
        calls = {"n": 0}

        def generate_fn(question, passage):
            calls["n"] += 1
            return GenerationResult(answer="Good answer.", isrel="[Relevant]", issup="[Fully supported]")

        pipeline = make_pipeline(generate_fn)
        state = pipeline.run("What is the termination clause?")

        assert state["accepted"] is True
        assert state["used_fallback"] is False
        assert calls["n"] == 1
        assert len(state["trace"]) == 1

    def test_retries_after_rejection_then_accepts(self):
        calls = {"n": 0}

        def generate_fn(question, passage):
            calls["n"] += 1
            if calls["n"] == 1:
                return GenerationResult(answer="bad", isrel="[Irrelevant]", issup="[No support / Contradictory]")
            return GenerationResult(answer="good", isrel="[Relevant]", issup="[Fully supported]")

        pipeline = make_pipeline(generate_fn, max_attempts=2)
        state = pipeline.run("What is the payment schedule?")

        assert state["accepted"] is True
        assert state["result"].answer == "good"
        assert calls["n"] == 2
        assert len(state["trace"]) == 2
        assert state["trace"][0]["accepted"] is False
        assert state["trace"][1]["accepted"] is True

    def test_falls_back_gracefully_after_exhausting_attempts(self):
        def generate_fn(question, passage):
            return GenerationResult(answer="always wrong", isrel="[Irrelevant]", issup="[No support / Contradictory]")

        pipeline = make_pipeline(generate_fn, max_attempts=2)
        state = pipeline.run("What is the confidentiality period?")

        assert state["used_fallback"] is True
        assert state["result"].answer == FALLBACK_MESSAGE
        # 2 rejected attempts before falling back
        assert len(state["trace"]) == 2
        assert all(not t["accepted"] for t in state["trace"])

    def test_high_eigenscore_triggers_rejection(self):
        def generate_fn(question, passage):
            return GenerationResult(answer="inconsistent answer", isrel="[Relevant]",
                                     issup="[Fully supported]", eigenscore=2.0)

        pipeline = make_pipeline(generate_fn, max_attempts=1)
        state = pipeline.run("What is the liability cap?")

        assert state["used_fallback"] is True
        assert "EigenScore" in state["trace"][0]["reasons"][0]

    def test_reformulate_changes_the_retrieval_query(self):
        queries_seen = []

        def retrieve_fn(query, top_k):
            queries_seen.append(query)
            return [{"text": "irrelevant", "score": 0.1}]

        def generate_fn(question, passage):
            return GenerationResult(answer="bad", isrel="[Irrelevant]")

        pipeline = make_pipeline(generate_fn, retrieve_fn=retrieve_fn, max_attempts=2)
        pipeline.run("What are the indemnification provisions?")

        assert len(queries_seen) == 2
        assert queries_seen[0] != queries_seen[1]

    def test_custom_reformulate_fn_is_used(self):
        def custom_reformulate(question, result):
            return "CUSTOM: " + question

        queries_seen = []

        def retrieve_fn(query, top_k):
            queries_seen.append(query)
            return [{"text": "irrelevant", "score": 0.1}]

        def generate_fn(question, passage):
            return GenerationResult(answer="bad", isrel="[Irrelevant]")

        pipeline = SelfHealingRAG(
            generate_fn=generate_fn, retrieve_fn=retrieve_fn,
            reformulate_fn=custom_reformulate, max_attempts=2,
        )
        pipeline.run("What is the governing law?")

        assert queries_seen[1].startswith("CUSTOM: ")

    def test_default_reformulate_strips_question_lead_words(self):
        result = GenerationResult(answer="bad")
        reformulated = default_reformulate("What is the termination clause?", result)
        assert reformulated.lower().startswith("the termination clause")
        assert "contract clause section terms" in reformulated

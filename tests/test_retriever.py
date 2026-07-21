import numpy as np

from src.retrieval.retriever import LegalRetriever


class FakeEmbeddingModel:
    """Deterministic fake: embeds each text as a one-hot-ish vector keyed by its content,
    so retrieval behavior is predictable without loading a real model."""

    DIM = 4

    def get_embedding_dim(self):
        return self.DIM

    def encode(self, texts, show_progress_bar=False):
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for t in texts:
            v = np.zeros(self.DIM, dtype=np.float32)
            v[abs(hash(t)) % self.DIM] = 1.0
            vectors.append(v)
        return np.array(vectors, dtype=np.float32)


def make_retriever():
    return LegalRetriever(embedding_model=FakeEmbeddingModel(), top_k=5)


class TestIndexReplaceDefault:
    def test_second_index_call_replaces_the_first_by_default(self):
        retriever = make_retriever()
        retriever.index_documents([{"text": "contract A about widgets"}], chunk_documents=False)
        retriever.index_documents([{"text": "contract B about gadgets"}], chunk_documents=False)

        results = retriever.retrieve("anything", top_k=5)
        texts = [r["text"] for r in results]

        assert "contract B about gadgets" in texts
        assert "contract A about widgets" not in texts, (
            "Regression: retriever leaked a previously indexed document into a "
            "later request's results -- this was a real cross-request contamination "
            "bug found live on the deployed backend (unrelated hallucinated content "
            "answering an unrelated question)."
        )

    def test_replace_false_accumulates_documents(self):
        retriever = make_retriever()
        retriever.index_documents([{"text": "contract A about widgets"}], chunk_documents=False, replace=False)
        retriever.index_documents([{"text": "contract B about gadgets"}], chunk_documents=False, replace=False)

        results = retriever.retrieve("anything", top_k=5)
        texts = [r["text"] for r in results]

        assert "contract A about widgets" in texts
        assert "contract B about gadgets" in texts

    def test_num_documents_reflects_replace_not_accumulate(self):
        retriever = make_retriever()
        retriever.index_documents([{"text": "doc one"}, {"text": "doc two"}], chunk_documents=False)
        assert retriever.get_num_documents() == 2

        retriever.index_documents([{"text": "doc three"}], chunk_documents=False)
        assert retriever.get_num_documents() == 1


class TestPerRequestRetrieverIsolation:
    """Mirrors backend/api.py's _request_retriever() pattern: a single global
    LegalRetriever's index is mutable state shared across concurrent HTTP
    requests, so reusing it directly lets one in-flight request's
    index_documents() wipe out another's between that other request's
    index_documents() and retrieve() calls (the Flask dev server used in
    production handles requests concurrently, not one-at-a-time). Giving
    each request its own LegalRetriever -- sharing only the read-only,
    expensive-to-load embedding model -- avoids that."""

    def test_interleaved_indexing_on_isolated_retrievers_does_not_corrupt_either(self):
        shared_embedding_model = FakeEmbeddingModel()
        retriever_a = LegalRetriever(embedding_model=shared_embedding_model, top_k=5)
        retriever_b = LegalRetriever(embedding_model=shared_embedding_model, top_k=5)

        # Simulates two requests overlapping in time: B indexes its document
        # on its own retriever in between A's index and A's retrieve --
        # this exact interleaving would corrupt a single shared retriever.
        retriever_a.index_documents([{"text": "contract A about widgets"}], chunk_documents=False)
        retriever_b.index_documents([{"text": "contract B about gadgets"}], chunk_documents=False)

        texts_a = [r["text"] for r in retriever_a.retrieve("anything", top_k=5)]
        texts_b = [r["text"] for r in retriever_b.retrieve("anything", top_k=5)]

        assert texts_a == ["contract A about widgets"]
        assert texts_b == ["contract B about gadgets"]

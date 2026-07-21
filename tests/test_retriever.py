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

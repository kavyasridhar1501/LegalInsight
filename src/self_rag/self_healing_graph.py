"""
Self-Healing RAG Pipeline

Models retrieve -> generate -> critique as a stateful, cyclical LangGraph
workflow instead of a linear chain. When the critic rejects an answer (not
grounded in the retrieved passage, or high semantic divergence across
resamples), the query is reformulated and retrieval runs again. If the
answer is still rejected after `max_attempts`, the pipeline returns a
graceful "insufficient information" response instead of a fabricated one.

The graph is built against small callables (`GenerateFn`, `RetrieveFn`,
`ReformulateFn`) rather than concrete model classes, so it can be unit
tested with fakes and wired to the real SelfRAGGGUFInference + LegalRetriever
in production via `build_self_healing_pipeline`.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import END, StateGraph

FALLBACK_MESSAGE = (
    "I don't have enough information in the retrieved contract text to "
    "answer this confidently. Please rephrase the question or point me to "
    "the relevant section."
)


@dataclass
class GenerationResult:
    """Minimal shape the graph needs from a generation call."""
    answer: str
    isrel: Optional[str] = None
    issup: Optional[str] = None
    isuse: Optional[str] = None
    eigenscore: Optional[float] = None


GenerateFn = Callable[[str, Optional[str]], GenerationResult]
RetrieveFn = Callable[[str, int], List[Dict[str, Any]]]
ReformulateFn = Callable[[str, GenerationResult], str]


class HealingState(TypedDict, total=False):
    question: str
    retrieval_query: str
    passage: Optional[str]
    passage_score: Optional[float]
    result: GenerationResult
    attempt: int
    max_attempts: int
    accepted: bool
    used_fallback: bool
    rejection_reasons: List[str]
    trace: List[Dict[str, Any]]


def default_reformulate(question: str, result: GenerationResult) -> str:
    """
    Heuristic query reformulation used when no LLM-based reformulator is
    supplied: broaden the query by stripping question words and appending
    terms the critic flagged as unsupported, which tends to pull in
    different passages on re-retrieval.
    """
    stripped = question
    for lead in ("what is", "what are", "how does", "how do", "why does", "why is", "does", "is", "are"):
        if stripped.lower().startswith(lead):
            stripped = stripped[len(lead):].strip()
            break
    stripped = stripped.rstrip("?").strip()
    return f"{stripped} contract clause section terms"


def _default_critique(result: GenerationResult, eigenscore_threshold: float) -> Tuple[bool, List[str]]:
    reasons = []

    if result.isrel is not None and "Irrelevant" in result.isrel:
        reasons.append("retrieved passage judged irrelevant (ISREL)")

    if result.issup is not None and (
        "No support" in result.issup or "Contradictory" in result.issup
    ):
        reasons.append("answer not supported by retrieved passage (ISSUP)")

    if result.eigenscore is not None and result.eigenscore > eigenscore_threshold:
        reasons.append(f"high semantic divergence across resamples (EigenScore={result.eigenscore:.2f})")

    if not result.answer or not result.answer.strip():
        reasons.append("empty answer")

    return len(reasons) == 0, reasons


class SelfHealingRAG:
    """Compiles and runs the retrieve -> generate -> critique -> heal LangGraph workflow."""

    def __init__(
        self,
        generate_fn: GenerateFn,
        retrieve_fn: RetrieveFn,
        reformulate_fn: ReformulateFn = default_reformulate,
        max_attempts: int = 2,
        top_k: int = 3,
        eigenscore_threshold: float = -2.0,
    ):
        """
        Args:
            max_attempts: Maximum number of retrieve+generate attempts before
                falling back to "insufficient information", including the
                first attempt (e.g. max_attempts=2 allows one retry).
        """
        self.generate_fn = generate_fn
        self.retrieve_fn = retrieve_fn
        self.reformulate_fn = reformulate_fn
        self.max_attempts = max_attempts
        self.top_k = top_k
        self.eigenscore_threshold = eigenscore_threshold
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(HealingState)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("generate", self._generate_node)
        graph.add_node("critique", self._critique_node)
        graph.add_node("reformulate", self._reformulate_node)
        graph.add_node("fallback", self._fallback_node)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "critique")
        graph.add_conditional_edges(
            "critique",
            self._route_after_critique,
            {"accept": END, "retry": "reformulate", "fallback": "fallback"},
        )
        graph.add_edge("reformulate", "retrieve")
        graph.add_edge("fallback", END)

        return graph.compile()

    def _retrieve_node(self, state: HealingState) -> HealingState:
        query = state.get("retrieval_query", state["question"])
        results = self.retrieve_fn(query, self.top_k)
        passage = results[0]["text"] if results else None
        passage_score = results[0].get("score") if results else None
        return {"passage": passage, "passage_score": passage_score}

    def _generate_node(self, state: HealingState) -> HealingState:
        result = self.generate_fn(state["question"], state.get("passage"))
        return {"result": result}

    def _critique_node(self, state: HealingState) -> HealingState:
        accepted, reasons = _default_critique(state["result"], self.eigenscore_threshold)
        trace_entry = {
            "attempt": state.get("attempt", 0),
            "retrieval_query": state.get("retrieval_query", state["question"]),
            "answer": state["result"].answer,
            "accepted": accepted,
            "reasons": reasons,
        }
        trace = state.get("trace", []) + [trace_entry]
        return {"accepted": accepted, "rejection_reasons": reasons, "trace": trace}

    def _reformulate_node(self, state: HealingState) -> HealingState:
        new_query = self.reformulate_fn(state["question"], state["result"])
        return {
            "retrieval_query": new_query,
            "attempt": state.get("attempt", 0) + 1,
        }

    def _fallback_node(self, state: HealingState) -> HealingState:
        fallback_result = GenerationResult(answer=FALLBACK_MESSAGE)
        return {"result": fallback_result, "used_fallback": True}

    def _route_after_critique(self, state: HealingState) -> str:
        if state["accepted"]:
            return "accept"
        attempts_done = state.get("attempt", 0) + 1
        if attempts_done >= self.max_attempts:
            return "fallback"
        return "retry"

    def run(self, question: str) -> HealingState:
        initial: HealingState = {
            "question": question,
            "retrieval_query": question,
            "attempt": 0,
            "max_attempts": self.max_attempts,
            "used_fallback": False,
            "trace": [],
        }
        return self._graph.invoke(initial)


def build_self_healing_pipeline(
    inference_model: Any,
    retriever: Any,
    max_attempts: int = 2,
    top_k: int = 3,
    eigenscore_threshold: float = -2.0,
) -> SelfHealingRAG:
    """
    Wire a real SelfRAGGGUFInference model and LegalRetriever into the
    self-healing graph.
    """

    def generate_fn(question: str, passage: Optional[str]) -> GenerationResult:
        output = inference_model.generate(question, passage=passage)
        return GenerationResult(
            answer=output.answer,
            isrel=output.isrel,
            issup=output.issup,
            isuse=output.isuse,
        )

    def retrieve_fn(query: str, top_k: int) -> List[Dict[str, Any]]:
        return retriever.retrieve(query, top_k=top_k)

    def reformulate_fn(question: str, result: GenerationResult) -> str:
        reformulate_prompt = (
            f"Rewrite this legal question to be more specific and easier to "
            f"match against contract text. Return only the rewritten question.\n\n"
            f"Question: {question}"
        )
        try:
            reformulated = inference_model.generate(reformulate_prompt, passage=None)
            candidate = reformulated.answer.strip()
            if candidate:
                return candidate
        except Exception:
            pass
        return default_reformulate(question, result)

    return SelfHealingRAG(
        generate_fn=generate_fn,
        retrieve_fn=retrieve_fn,
        reformulate_fn=reformulate_fn,
        max_attempts=max_attempts,
        top_k=top_k,
        eigenscore_threshold=eigenscore_threshold,
    )

"""Unit tests for RRF, filter parsing/SQL, the citation verifier, metrics, and the call budget."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from probable_cause.eval import metrics
from probable_cause.llm.base import BackendError, Completion, CompletionRequest, Message
from probable_cause.llm.cache import CachingBackend, CallBudget
from probable_cause.rag.chunking import Strategy
from probable_cause.rag.filters import Filters, QueryPlan, filter_sql
from probable_cause.rag.retrieve import RetrievalConfig, RetrievedChunk, Retriever, rrf
from probable_cause.rag.verify import parse_citations, split_sentences, verify

# -- RRF ---------------------------------------------------------------------------------


def test_rrf_rewards_agreement() -> None:
    fused = rrf([[1, 2, 3], [3, 1, 4]], k=60)
    order = [item for item, _ in fused]
    assert order[:2] == [1, 3]  # both lists rank them highly
    assert set(order) == {1, 2, 3, 4}
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_single_list_preserves_order() -> None:
    assert [i for i, _ in rrf([[9, 7, 8]])] == [9, 7, 8]


# -- filters -----------------------------------------------------------------------------


def test_filters_normalize_model_output() -> None:
    f = Filters.model_validate(
        {
            "year_from": 2021, "year_to": 2019, "far_parts": ["Part 91", "121", "91K"],
            "injury_severities": ["fatal"], "states": ["ohio", "tx"],
            "aircraft_categories": ["helicopters"], "phases": ["initial climb"],
        }
    )  # fmt: skip
    assert (f.year_from, f.year_to) == (2019, 2021)
    assert f.far_parts == ["091", "121", "091K"]
    assert f.injury_severities == ["Fatal"]
    assert f.states == ["OH", "TX"]
    assert f.aircraft_categories == ["Helicopter"]
    assert f.phases == ["Initial Climb"]


@pytest.mark.parametrize(
    "bad",
    [{"states": ["Atlantis"]}, {"far_parts": ["Part 999"]}, {"phases": ["Cruise"]},
     {"weather": "foggy"}, {"unknown_field": 1}],
)  # fmt: skip
def test_filters_reject_unknown_values(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Filters.model_validate(bad)


def test_query_plan_parses_llm_json() -> None:
    plan = QueryPlan.model_validate_json(
        '{"text_query": "carburetor icing", "filters": {"year_from": 2019, "year_to": null,'
        ' "aircraft_categories": [], "far_parts": [], "phases": [], "injury_severities": [],'
        ' "states": [], "weather": null, "night": null, "aircraft_make": null}}'
    )
    assert plan.text_query == "carburetor icing"
    assert not plan.filters.is_empty()
    assert Filters().is_empty()


def test_filter_sql_builds_only_requested_clauses() -> None:
    query, params = filter_sql(Filters(year_from=2019, far_parts=["121"], night=True))
    assert "make_date(%(year_from)s" in query
    assert "a.far_part = ANY(%(parts)s)" in query
    assert "Night%%" in query
    assert "states" not in params
    assert filter_sql(Filters())[0].endswith("WHERE true")


# -- citation verifier -------------------------------------------------------------------

RETRIEVED = {
    "20210102102480:probable_cause": "The pilot's failure to maintain adequate airspeed, "
    "which resulted in an aerodynamic stall.",
    "20190815X12345:analysis": "Carburetor icing caused a partial loss of engine power.",
}


def test_parse_citations_handles_groups_and_malformed() -> None:
    keys, bad = parse_citations(
        "A stall [20210102102480:probable_cause; 20190815X12345:analysis] [p.3]"
    )
    assert keys == ["20210102102480:probable_cause", "20190815X12345:analysis"]
    assert bad == ["p.3"]


def test_split_sentences_keeps_citations_with_their_sentence() -> None:
    text = "The airplane stalled. [20210102102480:analysis] Then it hit trees [x:y]. Done."
    assert split_sentences(text) == [
        "The airplane stalled. [20210102102480:analysis]",
        "Then it hit trees [x:y].",
        "Done.",
    ]


def test_verify_classifies_each_sentence() -> None:
    answer = (
        "The pilot failed to maintain adequate airspeed, resulting in an aerodynamic stall "
        "[20210102102480:probable_cause]. "
        "The engine lost power from carburetor icing [20200101X99999:analysis]. "
        "The pilot had been drinking heavily before the flight [20190815X12345:analysis]. "
        "This was a tragic event."
    )
    v = verify(answer, RETRIEVED)
    assert [s.status for s in v.sentences] == [
        "supported", "invalid_citation", "unsupported", "uncited",
    ]  # fmt: skip
    assert v.citation_validity == pytest.approx(2 / 3)
    assert v.citation_precision == pytest.approx(1 / 3)
    assert v.supported_fraction == pytest.approx(1 / 4)


# -- metrics -----------------------------------------------------------------------------


def test_retrieval_metrics() -> None:
    ranked = ["a", "x", "b", "y", "z"]
    assert metrics.recall_at_k(ranked, ["a", "b"], 2) == 0.5
    assert metrics.recall_at_k(ranked, ["a", "b"], 5) == 1.0
    # capped: 30 gold events, 2 of the top 5 are gold -> 2/5
    gold = ["a", "b", *[f"g{i}" for i in range(28)]]
    assert metrics.recall_at_k(ranked, gold, 5) == pytest.approx(2 / 5)
    assert metrics.reciprocal_rank(ranked, ["b"]) == pytest.approx(1 / 3)
    assert metrics.reciprocal_rank(ranked, ["q"]) == 0.0
    assert metrics.ndcg_at_k(["a", "b"], ["a", "b"], 10) == pytest.approx(1.0)
    assert 0 < metrics.ndcg_at_k(["x", "a"], ["a"], 10) < 1


def test_cohens_kappa() -> None:
    assert metrics.cohens_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert metrics.cohens_kappa([True, True, False, False], [True, False, True, False]) == 0.0


# -- LLM cache and budget ----------------------------------------------------------------


class _Echo:
    name = "fake"
    model = "m-new"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request: CompletionRequest) -> Completion:
        self.calls += 1
        return Completion(text='{"ok": true}', model=self.model)


def _request(text: str) -> CompletionRequest:
    return CompletionRequest("t", "sys", (Message("user", text),), {"type": "object"})


def test_cache_serves_repeats_and_budget_caps_live_calls(tmp_path: Path) -> None:
    inner = _Echo()
    budget = CallBudget(tmp_path / "calls.jsonl", cap=2)
    backend = CachingBackend(inner, tmp_path, budget=budget)
    backend.complete(_request("a"))
    backend.complete(_request("a"))  # cached
    backend.complete(_request("b"))
    assert inner.calls == 2
    assert budget.used() == 2
    with pytest.raises(BackendError, match="budget exhausted"):
        backend.complete(_request("c"))
    assert inner.calls == 2


def test_cache_finds_entries_recorded_under_a_model_alias(tmp_path: Path) -> None:
    old = _Echo()
    old.model = "m-alias"
    CachingBackend(old, tmp_path).complete(_request("a"))
    renamed = _Echo()
    backend = CachingBackend(renamed, tmp_path, model_aliases=("m-alias",), offline=True)
    assert backend.complete(_request("a")).text == '{"ok": true}'
    assert renamed.calls == 0


def test_verify_paragraph_citation_covers_preceding_sentences() -> None:
    answer = (
        "An aerodynamic stall resulted. "
        "The pilot failed to maintain adequate airspeed [20210102102480:probable_cause].\n"
        "A separate paragraph without any citation."
    )
    v = verify(answer, RETRIEVED)
    assert [(s.status, s.inherited) for s in v.sentences] == [
        ("supported", True), ("supported", False), ("uncited", False),
    ]  # fmt: skip


# -- retriever wiring --------------------------------------------------------------------


class _StubRetriever(Retriever):
    """Records the text each retriever receives; no database or index."""

    def __init__(self) -> None:
        super().__init__(conn=None, index_dir=Path())  # type: ignore[arg-type]
        self.seen: list[tuple[str, str, set[str] | None]] = []

    def allowed_events(self, plan: QueryPlan | None) -> set[str] | None:
        return {"E1"} if plan and not plan.filters.is_empty() else None

    def bm25_search(
        self, strategy: Strategy, text: str, k: int, allowed: set[str] | None
    ) -> list[tuple[int, float]]:
        self.seen.append(("bm25", text, allowed))
        return [(1, 2.0), (2, 1.0)]

    def dense_search(
        self, strategy: Strategy, text: str, k: int, allowed: set[str] | None
    ) -> list[tuple[int, float]]:
        self.seen.append(("dense", text, allowed))
        return [(2, 0.9), (3, 0.8)]

    def _load_chunks(self, strategy: Strategy, ids: list[int]) -> list[RetrievedChunk]:
        return [RetrievedChunk(i, f"E{i}", "analysis", None, f"t{i}", f"t{i}", i, 0.0) for i in ids]


PLAN = QueryPlan(text_query="stall on base turn", filters=Filters(year_from=2020))


def test_rewrite_off_ignores_the_plan_entirely() -> None:
    retriever = _StubRetriever()
    result = retriever.retrieve("Why did it stall?", RetrievalConfig("section", "bm25"), PLAN)
    assert retriever.seen == [("bm25", "Why did it stall?", None)]
    assert result.plan is None
    assert result.events == ["E1", "E2"]


def test_rewrite_on_uses_text_query_and_filters_and_hybrid_fuses() -> None:
    retriever = _StubRetriever()
    config = RetrievalConfig("section", "hybrid", rewrite=True)
    result = retriever.retrieve("Why did it stall?", config, PLAN)
    assert {(name, text) for name, text, _ in retriever.seen} == {
        ("bm25", "stall on base turn"), ("dense", "stall on base turn"),
    }  # fmt: skip
    assert all(allowed == {"E1"} for _, _, allowed in retriever.seen)
    assert result.events[0] == "E2"  # ranked by both retrievers
    assert result.filtered_events == 1

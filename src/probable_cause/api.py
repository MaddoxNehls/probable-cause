"""HTTP API: ``POST /ask`` and ``GET /events/{ev_id}``.

Run with ``uvicorn probable_cause.api:app``. The LLM backend comes from PC_LLM_BACKEND
(default claude-cli); every completion is cached on disk.
"""

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import date
from typing import Annotated, Any, Literal

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from probable_cause import tracing
from probable_cause.config import load_settings
from probable_cause.llm import LLM, BackendError, llm_from_env
from probable_cause.rag.pipeline import ask, debug_payload
from probable_cause.rag.retrieve import RetrievalConfig, Retriever
from probable_cause.rag.verify import parse_citations


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    strategy: Literal["fixed", "section", "parent_child"] = "section"
    retrieval: Literal["bm25", "dense", "hybrid"] = "hybrid"
    rerank: bool = True
    rewrite: bool = True
    llm_verify: bool = False


class Citation(BaseModel):
    key: str
    ev_id: str
    section: str
    ntsb_no: str | None
    date: date
    snippet: str
    url: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    refusal_reason: str | None
    citations: list[Citation]
    verification: dict[str, Any] | None
    debug: dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    tracing.init()
    app.state.settings = settings
    app.state.llm = llm_from_env()
    yield
    tracing.flush()


app = FastAPI(title="Probable Cause", version="0.2.0", lifespan=lifespan)


def get_conn(request: Request) -> Iterator[psycopg.Connection[tuple[object, ...]]]:
    with psycopg.connect(request.app.state.settings.database_url) as conn:
        yield conn


Conn = Annotated[psycopg.Connection[tuple[object, ...]], Depends(get_conn)]


@app.post("/ask")
def post_ask(body: AskRequest, request: Request, conn: Conn) -> AskResponse:
    llm: LLM = request.app.state.llm
    retriever = Retriever(conn, request.app.state.settings.index_dir)
    config = RetrievalConfig(
        strategy=body.strategy, method=body.retrieval, rerank=body.rerank, rewrite=body.rewrite
    )
    result = ask(body.question, config, retriever, llm, llm_verify=body.llm_verify)
    citations = _citations(conn, result.answer, result.retrieval.chunks)
    verification = None
    if result.verification:
        v = result.verification
        verification = {
            "citation_validity": v.citation_validity,
            "citation_precision": v.citation_precision,
            "supported_fraction": v.supported_fraction,
            "sentences": [
                {
                    "text": s.text,
                    "status": s.status,
                    "overlap": round(s.overlap, 3),
                    "invalid_citations": s.invalid,
                }
                for s in v.sentences
            ],
            "draft": result.draft,
        }
    return AskResponse(
        answer=result.answer,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        citations=citations,
        verification=verification,
        debug=debug_payload(result),
    )


def _citations(conn: Conn, answer: str, chunks: list[Any]) -> list[Citation]:
    keys = list(dict.fromkeys(k for s in answer.split("\n") for k in parse_citations(s)[0]))
    if not keys:
        return []
    by_key: dict[str, Any] = {}
    for chunk in chunks:
        by_key.setdefault(chunk.citation_key, chunk)
    rows = conn.execute(
        "SELECT ev_id, ntsb_no, ev_date, coalesce(report_url, docket_url) FROM events "
        "WHERE ev_id = ANY(%s)",
        ([k.split(":")[0] for k in keys],),
    ).fetchall()
    events = {str(r[0]): r for r in rows}
    out = []
    for key in keys:
        ev_id, section = key.split(":", 1)
        if ev_id not in events or key not in by_key:
            continue
        row = events[ev_id]
        out.append(
            Citation(
                key=key,
                ev_id=ev_id,
                section=section,
                ntsb_no=None if row[1] is None else str(row[1]),
                date=row[2],
                snippet=by_key[key].match_text[:400],
                url=str(row[3]),
            )
        )
    return out


@app.get("/events/{ev_id}")
def get_event(ev_id: str, conn: Conn) -> dict[str, Any]:
    conn.row_factory = psycopg.rows.dict_row  # type: ignore[assignment]
    event = conn.execute("SELECT * FROM events WHERE ev_id = %s", (ev_id,)).fetchone()
    if event is None:
        raise HTTPException(404, f"no event {ev_id}")

    def rows(query: str) -> list[Any]:
        return conn.execute(query, (ev_id,)).fetchall()

    return {
        "event": event,
        "aircraft": rows("SELECT * FROM aircraft WHERE ev_id = %s ORDER BY aircraft_key"),
        "occurrences": rows(
            "SELECT aircraft_key, occurrence_no, phase, event, is_defining FROM occurrences "
            "WHERE ev_id = %s ORDER BY aircraft_key, occurrence_no"
        ),
        "findings": rows(
            "SELECT aircraft_key, finding_no, description, cause_factor, in_probable_cause "
            "FROM findings WHERE ev_id = %s ORDER BY aircraft_key, finding_no"
        ),
        "narratives": rows(
            "SELECT aircraft_key, factual, analysis, probable_cause FROM narratives "
            "WHERE ev_id = %s ORDER BY aircraft_key"
        ),
    }


@app.exception_handler(BackendError)
def backend_error(request: Request, exc: BackendError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": f"LLM backend unavailable: {exc}"})

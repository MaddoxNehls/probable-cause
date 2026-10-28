"""Answer generation with mandatory inline citations."""

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from probable_cause import tracing
from probable_cause.llm import LLM, Message
from probable_cause.rag.retrieve import RetrievedChunk

INSUFFICIENT = "Insufficient evidence."
CONTEXT_WORD_LIMIT = 350

SYSTEM = """\
You answer questions about aviation accidents using only the numbered excerpts from NTSB
investigation records provided by the user. You have no other knowledge.

Rules:
- End every sentence that states a fact with one or more citations of the excerpts it came
  from, written exactly as the excerpt's key in square brackets, e.g. [20210102102480:analysis].
  Cite several as [key1][key2]. Never invent a key.
- Use only facts in the excerpts. No outside knowledge, no speculation.
- When asked to count or list, count only the events present in the excerpts and say so.
- If the excerpts do not answer the question, set insufficient_evidence to true and answer
  exactly "Insufficient evidence."
- Be concise: at most six sentences."""


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    insufficient_evidence: bool
    answer: str


@dataclass(frozen=True)
class EventInfo:
    ev_id: str
    ntsb_no: str | None
    date: str
    location: str
    aircraft: str
    highest_injury: str | None


def format_context(chunks: list[RetrievedChunk], events: dict[str, EventInfo]) -> str:
    blocks = []
    for chunk in chunks:
        info = events.get(chunk.ev_id)
        meta = (
            f"{info.ntsb_no or '-'} | {info.date} | {info.location} | {info.aircraft} | "
            f"highest injury: {info.highest_injury or 'unknown'}"
            if info
            else ""
        )
        words = chunk.context_text.split()
        text = " ".join(words[:CONTEXT_WORD_LIMIT]) + (
            " ..." if len(words) > CONTEXT_WORD_LIMIT else ""
        )
        blocks.append(f"[{chunk.citation_key}] {meta}\n{text}")
    return "\n\n".join(blocks)


def generate(
    llm: LLM, question: str, chunks: list[RetrievedChunk], events: dict[str, EventInfo]
) -> Answer:
    prompt = f"Excerpts:\n\n{format_context(chunks, events)}\n\nQuestion: {question}"
    with tracing.span("generate", input=question):
        return llm.structured("answer", SYSTEM, [Message("user", prompt)], Answer)

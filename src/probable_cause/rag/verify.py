"""Citation verifier: is every citation real, and is every claim supported by what it cites?

Two checks per sentence:
1. *Validity*: each ``[ev_id:section]`` citation must name a chunk that was actually
   retrieved and shown to the model. Anything else is a fabricated citation.
2. *Support*: the sentence's content words (stemmed, stopwords removed) must mostly appear
   in the cited text. This lexical check is cheap and deterministic but misses paraphrase,
   so sentences it rejects can optionally be re-checked by an LLM.
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal

import Stemmer
from pydantic import BaseModel, ConfigDict

from probable_cause.llm import LLM, Message

CITATION = re.compile(r"\[([^\[\]]+)\]")
_KEY = re.compile(r"^\s*(\d{8}[0-9X]\d{5}|\d{14}):([a-z_]+)\s*$")
# A sentence ends at . ! or ?, plus any citations placed after the punctuation.
_SENTENCE_END = re.compile(r"[.!?](?:\s*\[[^\[\]]+\])*(?=\s+[A-Z0-9\"'(])")
_TOKEN = re.compile(r"[a-z0-9]+")
_STEMMER = Stemmer.Stemmer("english")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "were",
        "which",
        "while",
        "who",
        "will",
        "with",
        "would",
        "not",
        "no",
        "also",
        "after",
        "before",
        "during",
        "about",
        "over",
        "under",
        "than",
        "both",
        "each",
        "other",
        "such",
        "any",
        "all",
        "some",
        "most",
        "more",
        "one",
        "two",
        "three",
    ]
)
SUPPORT_THRESHOLD = 0.6

type Status = Literal["supported", "supported_llm", "unsupported", "uncited", "invalid_citation"]


@dataclass
class SentenceCheck:
    text: str
    citations: list[str]
    invalid: list[str]
    overlap: float
    status: Status
    inherited: bool = False  # support checked against a later sentence's citations


@dataclass
class Verification:
    sentences: list[SentenceCheck] = field(default_factory=list)

    @property
    def citations(self) -> list[str]:
        return [c for s in self.sentences for c in s.citations]

    @property
    def citation_validity(self) -> float:
        """Share of citations that name a retrieved chunk."""
        cites = [(c, s) for s in self.sentences for c in s.citations]
        if not cites:
            return 0.0
        return sum(c not in s.invalid for c, s in cites) / len(cites)

    @property
    def citation_precision(self) -> float:
        """Share of citations that are valid *and* attached to a supported sentence."""
        cites = [(c, s) for s in self.sentences for c in s.citations]
        if not cites:
            return 0.0
        good = sum(
            c not in s.invalid and s.status in ("supported", "supported_llm") for c, s in cites
        )
        return good / len(cites)

    @property
    def supported_fraction(self) -> float:
        if not self.sentences:
            return 0.0
        ok = sum(s.status in ("supported", "supported_llm") for s in self.sentences)
        return ok / len(self.sentences)


def split_sentences(text: str) -> list[str]:
    sentences, start = [], 0
    for match in _SENTENCE_END.finditer(text):
        sentences.append(text[start : match.end()].strip())
        start = match.end()
    sentences.append(text[start:].strip())
    return [s for s in sentences if s]


def parse_citations(sentence: str) -> tuple[list[str], list[str]]:
    """Return (well-formed citation keys, malformed bracket contents)."""
    keys: list[str] = []
    malformed: list[str] = []
    for group in CITATION.findall(sentence):
        for part in re.split(r"[;,]", group):
            match = _KEY.match(part)
            if match:
                keys.append(f"{match[1]}:{match[2]}")
            elif part.strip():
                malformed.append(part.strip())
    return keys, malformed


def content_terms(text: str) -> set[str]:
    words = [w for w in _TOKEN.findall(CITATION.sub(" ", text.lower())) if w not in STOPWORDS]
    return set(_STEMMER.stemWords([w for w in words if len(w) > 2 or w.isdigit()]))


def lexical_support(sentence: str, sources: Iterable[str]) -> float:
    claim = content_terms(sentence)
    if not claim:
        return 1.0
    evidence: set[str] = set()
    for source in sources:
        evidence |= content_terms(source)
    return len(claim & evidence) / len(claim)


def verify(answer: str, retrieved: Mapping[str, str]) -> Verification:
    """`retrieved` maps citation key -> the text shown to the model under that key.

    Models often cite once at the end of a paragraph. An uncited sentence therefore borrows
    the valid citations of the next cited sentence in the same paragraph; its support is
    still checked against that text, and it is flagged ``inherited``.
    """
    result = Verification()
    for paragraph in (p for p in answer.splitlines() if p.strip()):
        checks: list[SentenceCheck] = []
        for sentence in split_sentences(paragraph):
            keys, malformed = parse_citations(sentence)
            invalid = [k for k in keys if k not in retrieved] + malformed
            checks.append(SentenceCheck(sentence, keys + malformed, invalid, 0.0, "uncited"))
        carried: list[str] = []
        for check in reversed(checks):
            valid = [k for k in check.citations if k in retrieved]
            if check.citations:
                carried = valid
            elif carried:
                valid, check.inherited = carried, True
            if not valid:
                check.status = "invalid_citation" if check.citations else "uncited"
                continue
            check.overlap = lexical_support(check.text, (retrieved[k] for k in valid))
            check.status = "supported" if check.overlap >= SUPPORT_THRESHOLD else "unsupported"
        result.sentences += checks
    return result


class Judgement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int
    supported: bool


class Judgements(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[Judgement]


LLM_CHECK_SYSTEM = """\
You check whether claims are supported by evidence excerpts from NTSB accident reports.
A claim is supported only if the excerpt states it or directly implies it. Paraphrase is
fine; added facts, numbers or causes that the excerpt does not contain are not."""


def llm_recheck(llm: LLM, verification: Verification, retrieved: Mapping[str, str]) -> None:
    """Ask an LLM about sentences the lexical check rejected; upgrade the ones it accepts."""
    todo = [s for s in verification.sentences if s.status == "unsupported"]
    if not todo:
        return
    blocks = []
    for i, s in enumerate(todo):
        evidence = "\n".join(retrieved[c][:2000] for c in s.citations if c in retrieved)
        blocks.append(
            f"### Claim {i}\n{CITATION.sub('', s.text).strip()}\n### Evidence {i}\n{evidence}"
        )
    result = llm.structured(
        "verify", LLM_CHECK_SYSTEM, [Message("user", "\n\n".join(blocks))], Judgements
    )
    for judgement in result.items:
        if 0 <= judgement.id < len(todo) and judgement.supported:
            todo[judgement.id].status = "supported_llm"

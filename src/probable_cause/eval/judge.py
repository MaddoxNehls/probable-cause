"""LLM-as-judge for answer correctness, batched and cached.

Rules applied before any LLM call:
* unanswerable question -> correct iff the system refused;
* answerable question that the system refused -> incorrect.
Only the remaining answers go to the judge, fifteen per call.
"""

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from probable_cause.eval.dataset import Item
from probable_cause.llm import LLM, Message
from probable_cause.rag.pipeline import AskResult
from probable_cause.rag.verify import CITATION

JUDGE_SYSTEM = """\
You grade answers from a question-answering system over NTSB aviation accident reports.

For each item you get the question, a reference answer, and the system's answer. Mark the
answer correct when it conveys the substance of the reference answer:
- lookup questions: the same probable cause (wording may differ; missing minor
  contributing factors is fine; a different or contradicting cause is incorrect);
- count/list questions: the same number or the same set of accidents (identified by NTSB
  number or unambiguous description); an answer that only covers some of them is incorrect;
- analytical questions: the main causes/themes in the reference, without contradicting it.
Ignore citation markers and style. Be strict about wrong facts."""

BATCH = 15


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    correct: bool
    reason: str


class Verdicts(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdicts: list[Verdict]


def judge_answers(
    llm: LLM, items: Sequence[Item], results: Sequence[AskResult], *, tag: str
) -> list[dict[str, Any]]:
    verdicts: list[dict[str, Any] | None] = []
    pending: list[tuple[int, Item, str]] = []
    for i, (item, result) in enumerate(zip(items, results, strict=True)):
        if not item.answerable:
            verdicts.append(
                {"correct": result.refused, "reason": "rule: unanswerable", "by": "rule"}
            )
        elif result.refused:
            verdicts.append({"correct": False, "reason": "rule: refused", "by": "rule"})
        else:
            verdicts.append(None)
            pending.append((i, item, CITATION.sub("", result.answer).strip()))
    for start in range(0, len(pending), BATCH):
        batch = pending[start : start + BATCH]
        verdicts_by_id = judge_batch(llm, [(item, answer) for _, item, answer in batch],
                                     tag=f"{tag}/{start}")  # fmt: skip
        for i, item, _ in batch:
            verdict = verdicts_by_id.get(item.id)
            verdicts[i] = (
                {"correct": verdict.correct, "reason": verdict.reason, "by": "llm"}
                if verdict
                else {"correct": False, "reason": "judge returned no verdict", "by": "llm"}
            )
    return [v for v in verdicts if v is not None]


def judge_batch(llm: LLM, pairs: Sequence[tuple[Item, str]], *, tag: str) -> dict[str, Verdict]:
    blocks = [
        f"### Item {item.id} ({item.type})\nQuestion: {item.question}\n"
        f"Reference answer: {item.gold_answer}\nSystem answer: {answer}"
        for item, answer in pairs
    ]
    prompt = "Grade every item; echo each item id exactly.\n\n" + "\n\n".join(blocks)
    result = llm.structured(tag, JUDGE_SYSTEM, [Message("user", prompt)], Verdicts)
    return {v.id: v for v in result.verdicts}

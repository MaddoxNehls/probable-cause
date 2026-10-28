"""LLM query rewriting: a natural-language question becomes structured filters plus a text
query. Batched for evaluation (many questions per call) and single for the API; both go
through the same pydantic validation."""

from pydantic import BaseModel, ConfigDict

from probable_cause import tracing
from probable_cause.llm import LLM, Message
from probable_cause.rag.filters import QueryPlan

SYSTEM = """\
You translate questions about NTSB aviation accident investigations (2008 onward) into a
search plan for a database that has structured fields and narrative text.

Put a constraint in `filters` only when the question states it explicitly; never guess.
- year_from / year_to: event years ("in 2019" means both 2019; "since 2015" means year_from 2015).
- aircraft_categories: Airplane, Helicopter, Glider, Balloon, Gyrocraft, Weight-Shift,
  Powered Parachute, Ultralight, Powered-Lift.
- far_parts: operating rule as a 3-digit code: "091" general aviation/private/personal,
  "121" airline/scheduled air carrier, "135" charter/air taxi/commuter, "137" agricultural/
  crop dusting, "133" external load, "129" foreign air carrier, "107" small drone.
- phases: phase of flight of the defining event: Standing, Taxi, Takeoff, Initial Climb,
  Enroute, Maneuvering, Approach, Landing, Emergency Descent, Uncontrolled Descent.
- injury_severities: highest injury of the event: Fatal, Serious, Minor, None.
  "fatal accident" means ["Fatal"].
- states: two-letter US state codes.
- weather: VMC or IMC, only if the question names the weather conditions class.
- night: true for night, false for daylight, null otherwise.
- aircraft_make: manufacturer name as written in the registry (e.g. "Cessna", "Piper",
  "Robinson", "Boeing"), only if named.

`text_query` keeps everything else that should be matched against narrative text: causes,
circumstances, systems, aircraft models ("172", "R44"), and phenomena ("carburetor icing").
Remove the words already captured by filters. Use an empty string if nothing remains."""


class PlanItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    plan: QueryPlan


class PlanBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plans: list[PlanItem]


def rewrite(llm: LLM, question: str) -> QueryPlan:
    with tracing.span("rewrite", input=question):
        return llm.structured("rewrite", SYSTEM, [Message("user", question)], QueryPlan)


def rewrite_batch(llm: LLM, questions: dict[str, str], *, tag: str) -> dict[str, QueryPlan]:
    """Rewrite many questions in one call. Keys are caller-chosen ids."""
    listing = "\n".join(f"[{qid}] {text}" for qid, text in questions.items())
    prompt = (
        "Produce one plan per question below, echoing each question's id exactly.\n\n" + listing
    )
    batch = llm.structured(tag, SYSTEM, [Message("user", prompt)], PlanBatch)
    plans = {item.id: item.plan for item in batch.plans}
    missing = set(questions) - set(plans)
    if missing:
        raise ValueError(f"{tag}: rewrite batch is missing ids {sorted(missing)}")
    return plans

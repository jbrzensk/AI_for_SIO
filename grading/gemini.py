"""One Gemini call that judges an answer point by point.

The model judges each rubric point and suggests one next step; it never
decides the overall verdict (grading/__init__.py counts the points).
"""
import json
import os
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

MODEL_NAME = "gemini-3.5-flash-lite"

ERROR_KINDS = ("no_key", "invalid_key", "offline", "rate_limit", "unreadable", "service")


class GradingError(RuntimeError):
    """The answer couldn't be graded. `kind` (one of ERROR_KINDS) selects
    the fix shown on the card; `detail` is the underlying message."""

    def __init__(self, kind: str, detail: str = ""):
        if kind not in ERROR_KINDS:
            raise ValueError(f"unknown GradingError kind: {kind!r}")
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind
        self.detail = detail


@dataclass(frozen=True)
class PointResult:
    name: str
    met: bool
    note: str


@dataclass(frozen=True)
class Judgement:
    points: list[PointResult]  # in the order of the rubric's points
    next_step: str


def _get_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise GradingError("no_key")
    return genai.Client(api_key=api_key)


def _schema(names) -> dict:
    return {
        "type": "object",
        "properties": {
            "points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": list(names)},
                        "met": {"type": "boolean"},
                        "note": {"type": "string"},
                    },
                    "required": ["name", "met", "note"],
                },
            },
            "next_step": {"type": "string"},
        },
        "required": ["points", "next_step"],
    }


def _prompt(question: str, points: dict[str, str], answer: str) -> str:
    listed = "\n".join(f"- {name}: {description}" for name, description in points.items())
    return (
        "You are grading a research-group member's exercise answer, one point "
        "at a time. Be strict but fair: mark a point met only if the answer "
        "satisfies its substance, not just its wording.\n\n"
        f"Question:\n{question}\n\n"
        f"Points to judge:\n{listed}\n\n"
        f"Learner answer:\n{answer}\n\n"
        "For each point, give met (true or false) and a one-sentence note on "
        "what the answer does, or is missing, for that point. Then give "
        "next_step: one concrete action that would most improve the answer "
        "(if every point is met, the next improvement). Judge every point "
        "exactly once, using exactly these names: " + ", ".join(points) + "."
    )


def _api_error(exc: errors.APIError) -> GradingError:
    code = getattr(exc, "code", None)
    text = str(exc)
    if code == 429:
        return GradingError("rate_limit", text)
    if code in (401, 403) or (code == 400 and "api key" in text.lower()):
        return GradingError("invalid_key", text)
    return GradingError("service", text)


def _parse(text, points: dict[str, str]) -> Judgement:
    try:
        data = json.loads(text)
        by_name = {}
        for item in data["points"]:
            name = item["name"]
            if name in by_name:
                raise ValueError(f"point judged twice: {name}")
            if not isinstance(item["met"], bool):
                raise ValueError(f"met is not a boolean for {name}")
            by_name[name] = PointResult(name=name, met=item["met"], note=str(item["note"]))
        if set(by_name) != set(points):
            raise ValueError(f"expected points {sorted(points)}, got {sorted(by_name)}")
        return Judgement(points=[by_name[name] for name in points], next_step=str(data["next_step"]))
    except (TypeError, KeyError, ValueError) as exc:  # json.JSONDecodeError is a ValueError
        raise GradingError("unreadable", f"{exc}; reply was {text!r}") from exc


def judge(question: str, points: dict[str, str], answer: str) -> Judgement:
    """Ask Gemini to judge `answer` against each of `points` (name -> what
    a met point looks like). Raises GradingError for anything the learner
    can fix or retry."""
    client = _get_client()
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=_prompt(question, points, answer),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_schema(points),
                # Same answer, same judgement: sampling at the default
                # temperature flipped borderline answers between runs.
                temperature=0,
                # No tools are used; leaving this on makes the SDK print a
                # confusing warning into the learner's notebook.
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        text = response.text
    except errors.APIError as exc:
        raise _api_error(exc) from exc
    except httpx.HTTPError as exc:
        raise GradingError("offline", str(exc)) from exc
    except errors.UnknownApiResponseError as exc:
        # An HTTP 200 body that isn't JSON (e.g. a proxy or captive-portal
        # page); the SDK raises this as a plain ValueError subclass, not APIError.
        raise GradingError("service", str(exc)) from exc
    return _parse(text, points)

import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

MODEL_NAME = "gemini-3.5-flash-lite"

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "pass": {"type": "boolean"},
        "feedback": {"type": "string"},
    },
    "required": ["pass", "feedback"],
}


class GradingError(RuntimeError):
    """Raised when the exercise can't be graded (missing or invalid key,
    API error, bad response)."""


def _get_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise GradingError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env, add a "
            "free API key from https://aistudio.google.com/apikey, and "
            "restart the notebook kernel."
        )
    return genai.Client(api_key=api_key)


def grade(question: str, rubric: str, answer: str) -> dict:
    """Grade `answer` to `question` against `rubric` using the Gemini
    free-tier API as an LLM judge. Returns {"pass": bool, "feedback": str}.
    """
    client = _get_client()
    prompt = (
        "You are grading a research-group member's exercise answer. "
        "Be strict but fair: mark pass only if the answer satisfies the "
        "rubric's substance, not just its wording.\n\n"
        f"Question:\n{question}\n\nRubric:\n{rubric}\n\n"
        f"Learner answer:\n{answer}\n\n"
        "Respond with JSON matching the schema: "
        '{"pass": boolean, "feedback": string}. '
        "feedback should be 1-3 sentences, specific and actionable."
    )
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_RESPONSE_SCHEMA,
            ),
        )
    except errors.APIError as exc:
        raise GradingError(
            f"The grading service returned an error: {exc}\n"
            "If it says the API key is invalid, check GOOGLE_API_KEY in your "
            ".env file (get a free key at https://aistudio.google.com/apikey) "
            "and restart the notebook kernel."
        ) from exc
    try:
        result = json.loads(response.text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise GradingError(
            f"Grader returned an unparseable response: {response.text!r}"
        ) from exc
    return {"pass": bool(result["pass"]), "feedback": str(result["feedback"])}

import json

import httpx
import pytest
from google.genai import errors

import grading.gemini as gemini

POINTS = {
    "Goal": "says what to produce",
    "Context": "names the data",
    "Output": "gives a format",
    "Boundaries": "says what to verify",
}


def reply(points=None, next_step="Add what the model must verify."):
    if points is None:
        points = [
            {"name": "Goal", "met": True, "note": "Clear goal."},
            {"name": "Context", "met": True, "note": "Names the 1959 data."},
            {"name": "Output", "met": True, "note": "PDF and PNG plots."},
            {"name": "Boundaries", "met": False, "note": "Nothing to verify."},
        ]
    return json.dumps({"points": points, "next_step": next_step})


@pytest.fixture
def fake_gemini(monkeypatch):
    """Replace the Gemini client. Call it with text=<reply> or exc=<exception>.
    Returns a dict that captures what was sent."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-test")
    sent = {}

    def install(text=None, exc=None):
        class FakeResponse:
            pass

        FakeResponse.text = text

        class FakeModels:
            def generate_content(self, model, contents, config):
                sent.update(model=model, contents=contents, config=config)
                if exc is not None:
                    raise exc
                return FakeResponse()

        class FakeClient:
            models = FakeModels()

        monkeypatch.setattr(gemini, "_get_client", lambda: FakeClient())
        return sent

    return install


def kind_of(callable_):
    with pytest.raises(gemini.GradingError) as excinfo:
        callable_()
    return excinfo.value.kind


def test_missing_key_raises_no_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert kind_of(lambda: gemini.judge("q", POINTS, "a")) == "no_key"


def test_judgement_comes_back_in_rubric_order(fake_gemini):
    fake_gemini(text=reply())
    result = gemini.judge("q", POINTS, "a")
    assert [p.name for p in result.points] == list(POINTS)
    assert [p.met for p in result.points] == [True, True, True, False]
    assert result.points[1].note == "Names the 1959 data."
    assert result.next_step == "Add what the model must verify."


def test_points_in_a_different_order_are_reordered(fake_gemini):
    shuffled = [{"name": n, "met": True, "note": n} for n in ("Boundaries", "Goal", "Output", "Context")]
    fake_gemini(text=reply(shuffled))
    assert [p.name for p in gemini.judge("q", POINTS, "a").points] == list(POINTS)


@pytest.mark.parametrize(
    "points",
    [
        [{"name": "Goal", "met": True, "note": ""}],
        [{"name": n, "met": True, "note": ""} for n in ("Goal", "Goal", "Context", "Output", "Boundaries")],
        [{"name": n, "met": True, "note": ""} for n in ("Goal", "Context", "Output", "Scope")],
        [{"name": n, "met": "yes", "note": ""} for n in POINTS],
    ],
    ids=["missing", "duplicate", "unknown-name", "met-not-boolean"],
)
def test_mismatched_or_malformed_points_are_unreadable(fake_gemini, points):
    fake_gemini(text=reply(points))
    assert kind_of(lambda: gemini.judge("q", POINTS, "a")) == "unreadable"


@pytest.mark.parametrize(
    "text",
    ["not json", None, json.dumps({"points": []}), json.dumps(["a", "list"])],
    ids=["not-json", "no-text", "no-points", "wrong-type"],
)
def test_unparseable_replies_are_unreadable(fake_gemini, text):
    fake_gemini(text=text)
    assert kind_of(lambda: gemini.judge("q", POINTS, "a")) == "unreadable"


def test_invalid_key_is_reported_with_detail(fake_gemini):
    fake_gemini(exc=errors.ClientError(400, {"error": {
        "code": 400, "message": "API key not valid. Please pass a valid API key.", "status": "INVALID_ARGUMENT"}}))
    with pytest.raises(gemini.GradingError) as excinfo:
        gemini.judge("q", POINTS, "a")
    assert excinfo.value.kind == "invalid_key"
    assert "API key not valid" in excinfo.value.detail


def test_rate_limit_is_reported(fake_gemini):
    fake_gemini(exc=errors.ClientError(429, {"error": {
        "code": 429, "message": "Resource has been exhausted", "status": "RESOURCE_EXHAUSTED"}}))
    assert kind_of(lambda: gemini.judge("q", POINTS, "a")) == "rate_limit"


def test_other_api_errors_are_service_errors(fake_gemini):
    fake_gemini(exc=errors.ServerError(500, {"error": {
        "code": 500, "message": "Internal error", "status": "INTERNAL"}}))
    assert kind_of(lambda: gemini.judge("q", POINTS, "a")) == "service"


def test_network_failure_is_offline(fake_gemini):
    fake_gemini(exc=httpx.ConnectError("[Errno 111] Connection refused"))
    assert kind_of(lambda: gemini.judge("q", POINTS, "a")) == "offline"


def test_request_settings(fake_gemini):
    sent = fake_gemini(text=reply())
    gemini.judge("The question", POINTS, "The answer")
    config = sent["config"]
    assert sent["model"] == "gemini-3.5-flash-lite"
    assert config.temperature == 0
    assert config.automatic_function_calling.disable is True
    assert config.response_mime_type == "application/json"
    assert config.response_schema is not None


def test_schema_restricts_point_names_to_the_rubric():
    schema = gemini._schema(POINTS)
    assert schema["properties"]["points"]["items"]["properties"]["name"]["enum"] == list(POINTS)
    assert schema["required"] == ["points", "next_step"]


def test_prompt_lists_every_point_and_never_asks_for_a_verdict(fake_gemini):
    sent = fake_gemini(text=reply())
    gemini.judge("The question", POINTS, "The answer")
    for name, description in POINTS.items():
        assert f"- {name}: {description}" in sent["contents"]
    assert "The question" in sent["contents"] and "The answer" in sent["contents"]
    assert "overall" not in sent["contents"].lower()


def test_unknown_error_kind_is_rejected():
    with pytest.raises(ValueError):
        gemini.GradingError("oops")

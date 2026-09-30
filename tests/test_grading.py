import pytest

import grading


def test_grade_raises_clear_error_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(grading.GradingError, match="GOOGLE_API_KEY"):
        grading.grade("question", "rubric", "answer")


def test_grade_returns_pass_and_feedback_from_mocked_response(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-test")

    class FakeResponse:
        text = '{"pass": true, "feedback": "Covers goal, context, and output."}'

    class FakeModels:
        def generate_content(self, model, contents, config):
            return FakeResponse()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(grading, "_get_client", lambda: FakeClient())

    result = grading.grade("question", "rubric", "answer")

    assert result == {
        "pass": True,
        "feedback": "Covers goal, context, and output.",
    }


def test_grade_raises_grading_error_on_unparseable_response(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-test")

    class FakeResponse:
        text = "not json"

    class FakeModels:
        def generate_content(self, model, contents, config):
            return FakeResponse()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(grading, "_get_client", lambda: FakeClient())

    with pytest.raises(grading.GradingError, match="unparseable"):
        grading.grade("question", "rubric", "answer")


def test_grade_raises_clear_error_when_api_rejects_key(monkeypatch):
    from google.genai import errors

    monkeypatch.setenv("GOOGLE_API_KEY", "invalid-key-for-test")

    class FakeModels:
        def generate_content(self, model, contents, config):
            raise errors.ClientError(
                400,
                {
                    "error": {
                        "code": 400,
                        "message": "API key not valid. Please pass a valid API key.",
                        "status": "INVALID_ARGUMENT",
                    }
                },
            )

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(grading, "_get_client", lambda: FakeClient())

    with pytest.raises(grading.GradingError, match="GOOGLE_API_KEY") as excinfo:
        grading.grade("question", "rubric", "answer")
    assert "API key not valid" in str(excinfo.value)


def test_grade_raises_clear_error_when_network_unreachable(monkeypatch):
    import httpx

    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-test")

    class FakeModels:
        def generate_content(self, model, contents, config):
            raise httpx.ConnectError("[Errno 111] Connection refused")

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(grading, "_get_client", lambda: FakeClient())

    with pytest.raises(grading.GradingError, match="internet connection"):
        grading.grade("question", "rubric", "answer")


def test_grade_raises_grading_error_on_response_missing_feedback_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-test")

    class FakeResponse:
        text = '{"pass": true}'

    class FakeModels:
        def generate_content(self, model, contents, config):
            return FakeResponse()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(grading, "_get_client", lambda: FakeClient())

    with pytest.raises(grading.GradingError, match="unparseable"):
        grading.grade("question", "rubric", "answer")


def test_grade_disables_automatic_function_calling(monkeypatch):
    # With automatic function calling left on, the SDK prints a confusing
    # "Direct use of automatic function calling (AFC)..." warning into the
    # learner's notebook on the first grade() call.
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-test")
    sent = {}

    class FakeResponse:
        text = '{"pass": true, "feedback": "ok"}'

    class FakeModels:
        def generate_content(self, model, contents, config):
            sent["config"] = config
            return FakeResponse()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(grading, "_get_client", lambda: FakeClient())

    grading.grade("question", "rubric", "answer")

    afc = sent["config"].automatic_function_calling
    assert afc is not None and afc.disable is True

import pytest

import grading
from grading import progress
from grading.gemini import GradingError, Judgement, PointResult

POINTS = {"Goal": "g", "Context": "c", "Output": "o", "Boundaries": "b"}


@pytest.fixture
def shown(monkeypatch):
    cards = []
    monkeypatch.setattr(grading, "_show", cards.append)
    return cards


def make_module(tmp_path, monkeypatch, folder_name, heading):
    folder = tmp_path / folder_name
    folder.mkdir()
    (folder / "README.md").write_text(f"# {heading}\n")
    monkeypatch.chdir(folder)
    pfile = tmp_path / ".progress.json"
    monkeypatch.setattr(progress, "PROGRESS_FILE", pfile)
    return pfile


@pytest.fixture
def module02(tmp_path, monkeypatch):
    return make_module(tmp_path, monkeypatch, "02-prompt-engineering", "Module 02: Prompt Engineering")


def fake_judge(monkeypatch, *mets, exc=None):
    def judge(question, points, answer):
        if exc is not None:
            raise exc
        return Judgement([PointResult(n, m, f"note {n}") for n, m in zip(points, mets)], "next step")

    monkeypatch.setattr(grading, "judge", judge)


def test_pass_shows_the_pass_card_and_records_progress(monkeypatch, shown, module02):
    fake_judge(monkeypatch, True, True, True, False)
    assert grading.check("q", POINTS, 3, "my answer") is None
    assert len(shown) == 1
    assert "Passed: Module 02, Prompt Engineering" in shown[0]
    assert "MODULE 02<br>COMPLETE" in shown[0]
    assert progress.completed_modules(path=module02) == {"02"}


def test_not_yet_then_pass_counts_attempts(monkeypatch, shown, module02):
    fake_judge(monkeypatch, True, False, False, False)
    grading.check("q", POINTS, 3, "a")
    assert "Not yet: Module 02, Prompt Engineering" in shown[0] and "attempt 1" in shown[0]
    fake_judge(monkeypatch, True, True, True, True)
    grading.check("q", POINTS, 3, "a")
    assert "Passed: Module 02, Prompt Engineering" in shown[1] and "attempt 2" in shown[1]


def test_progress_strip_includes_the_module_just_completed(monkeypatch, shown, module02):
    fake_judge(monkeypatch, True, True, True, True)
    grading.check("q", POINTS, 3, "a")
    assert '<span class="aisio-badge aisio-done aisio-now">02 ✓</span>' in shown[0]


def test_grading_errors_show_an_error_card_and_do_not_count(monkeypatch, shown, module02):
    fake_judge(monkeypatch, exc=GradingError("offline", "connection refused"))
    grading.check("q", POINTS, 3, "a")
    assert 'class="aisio-card aisio-error"' in shown[0]
    assert "Check your internet connection" in shown[0]
    assert not module02.exists()


@pytest.mark.parametrize("points, needed", [
    ({}, 1), (POINTS, 0), (POINTS, 5), (POINTS, True), (POINTS, 2.0), (["Goal"], 1), ({"Goal": 1}, 1),
], ids=["empty", "zero", "too-many", "bool", "float", "list", "non-str-description"])
def test_wrong_arguments_raise_value_error(monkeypatch, shown, module02, points, needed):
    fake_judge(monkeypatch, True)
    with pytest.raises(ValueError):
        grading.check("q", points, needed, "a")
    assert shown == []


def test_outside_a_module_folder_nothing_is_recorded(monkeypatch, shown, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(progress, "PROGRESS_FILE", tmp_path / ".progress.json")
    fake_judge(monkeypatch, True, True, True, True)
    grading.check("q", POINTS, 3, "a")
    assert "Passed: Exercise" in shown[0] and "Your progress" not in shown[0]
    assert not (tmp_path / ".progress.json").exists()


def test_check_code_then_explanation_completes_module_04(monkeypatch, shown, tmp_path):
    pfile = make_module(tmp_path, monkeypatch, "04-coding-practices", "Module 04: Coding with AI")

    def failing():
        assert False, "Expected a lazy Dask array before .compute()"

    grading.check_code(failing)
    assert "Code test not passing yet: Module 04, Coding with AI" in shown[0]
    assert "AssertionError: Expected a lazy Dask array before .compute()" in shown[0]

    grading.check_code(lambda: None)
    assert "Code test passed: Module 04, Coding with AI" in shown[1] and "COMPLETE" not in shown[1]

    fake_judge(monkeypatch, True, True)
    grading.check("q", {"Eager": "e", "Why": "w"}, 2, "explanation")
    assert "MODULE 04<br>COMPLETE" in shown[2]
    assert progress.completed_modules(path=pfile) == {"04"}


def test_check_code_shows_any_exception_from_the_learners_code(monkeypatch, shown, tmp_path):
    make_module(tmp_path, monkeypatch, "04-coding-practices", "Module 04: Coding with AI")
    grading.check_code(lambda: {}["missing"])
    assert "KeyError: &#x27;missing&#x27;" in shown[0]


def test_check_code_requires_a_callable(shown):
    with pytest.raises(TypeError):
        grading.check_code("not a function")


def test_reset_progress(monkeypatch, shown, module02, capsys):
    fake_judge(monkeypatch, True, True, True, True)
    grading.check("q", POINTS, 3, "a")
    grading.reset_progress()
    assert not module02.exists()
    assert "reset" in capsys.readouterr().out


def test_show_uses_ipython_display(monkeypatch):
    shown = []
    import IPython.display
    monkeypatch.setattr(IPython.display, "display", shown.append)
    grading._show("<b>hi</b>")
    assert shown[0].data == "<b>hi</b>"

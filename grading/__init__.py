"""Grading for the exercise notebooks.

    from grading import check           # grade an answer, show the card, record progress
    from grading import check_code      # run a deterministic test as a card (module 04)
    from grading import reset_progress  # start your progress over
"""
from pathlib import Path

from grading import card, progress
from grading.gemini import GradingError, judge

__all__ = ["check", "check_code", "reset_progress", "GradingError"]


def _show(html: str) -> None:
    from IPython.display import HTML, display

    display(HTML(html))


def _validate(question, points, needed, answer) -> None:
    if not isinstance(points, dict) or not points:
        raise ValueError("points must be a non-empty dict of {point name: what a met point looks like}")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in points.items()):
        raise ValueError("points must map point names (str) to descriptions (str)")
    if isinstance(needed, bool) or not isinstance(needed, int) or not 1 <= needed <= len(points):
        raise ValueError(f"needed must be a whole number from 1 to {len(points)}")
    if not isinstance(question, str) or not isinstance(answer, str):
        raise ValueError("question and answer must be strings")


def _record(module: str | None, check_name: str, passed: bool):
    """(attempt, module_complete, progress view); all empty outside a module.

    Grading never fails because of the progress file: anything unexpected
    here still lets the card show, just without the progress strip.
    """
    if module is None:
        return None, False, None
    try:
        result = progress.record(module, check_name, passed)
        complete = progress.completed_modules() | ({module} if result.module_complete else set())
        view = card.ProgressView(modules=progress.list_modules(), complete=complete, current=module, note=result.note)
        return result.attempt, result.module_complete, view
    except Exception:
        return None, False, None


def check(question: str, points: dict[str, str], needed: int, answer: str) -> None:
    """Grade `answer`, show the result card, and record progress."""
    _validate(question, points, needed, answer)
    module, title = progress.module_info(Path.cwd())
    try:
        judgement = judge(question, points, answer)
    except GradingError as exc:
        _show(card.error_card(exc.kind, exc.detail))
        return
    passed = sum(p.met for p in judgement.points) >= needed
    attempt, module_complete, view = _record(module, "answer", passed)
    _show(card.verdict_card(module=module, title=title, judgement=judgement, needed=needed, passed=passed,
                            attempt=attempt, module_complete=module_complete, progress=view))


def check_code(test_fn) -> None:
    """Run `test_fn()`; returning normally passes, any exception is "not yet"."""
    if not callable(test_fn):
        raise TypeError("check_code needs a function to call, e.g. check_code(run_test)")
    module, title = progress.module_info(Path.cwd())
    try:
        test_fn()
        passed, error = True, None
    except Exception as exc:  # the learner's code failing is the point of the test
        passed, error = False, f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    attempt, module_complete, view = _record(module, "code", passed)
    _show(card.code_card(module=module, title=title, passed=passed, error=error, attempt=attempt,
                         module_complete=module_complete, progress=view))


def reset_progress() -> None:
    """Delete your saved progress so every module starts over."""
    progress.reset()
    print("Your exercise progress has been reset.")

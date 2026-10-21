import re

import pytest

from grading import card
from grading.gemini import Judgement, PointResult

NAMES = ["Goal", "Context", "Output", "Boundaries"]
VIEW = card.ProgressView(modules=["01", "02", "03", "04"], complete={"01", "02"}, current="02")


def judgement(*mets, next_step="Add what the model must verify."):
    return Judgement(
        points=[PointResult(name, met, f"note about {name}") for name, met in zip(NAMES, mets)],
        next_step=next_step,
    )


def pass_card(**overrides):
    args = dict(module="02", title="Prompt Engineering", judgement=judgement(True, True, True, False),
                needed=3, passed=True, attempt=2, module_complete=True, progress=VIEW)
    args.update(overrides)
    return card.verdict_card(**args)


def test_pass_card_has_header_points_next_improvement_and_both_celebrations():
    html = pass_card()
    assert "Passed: Module 02, Prompt Engineering" in html
    assert "3 of 4 points (needed 3) · attempt 2" in html
    assert html.count("✅") == 3 and "➖" in html and "❌" not in html
    assert "Next improvement:" in html
    assert 'class="aisio-confetti"' in html
    assert "MODULE 02<br>COMPLETE" in html


def test_stamp_only_when_the_module_is_complete():
    html = pass_card(module_complete=False)
    assert 'class="aisio-confetti"' in html
    assert "COMPLETE" not in html


def test_not_yet_card():
    html = card.verdict_card(
        module="02", title="Prompt Engineering", judgement=judgement(True, False, False, False), needed=3,
        passed=False, attempt=1, module_complete=False,
        progress=card.ProgressView(["01", "02", "03", "04"], {"01"}, "02"))
    assert "Not yet: Module 02, Prompt Engineering" in html
    assert "1 of 4 points (needed 3) · attempt 1" in html
    assert html.count("❌") == 3
    assert "Your next step:" in html
    assert ("Most people need two or three tries. Edit your answer above, run that cell, "
            "then run this one again.") in html
    assert 'class="aisio-confetti"' not in html


def test_passing_explanation_without_the_code_test_explains_why_the_module_isnt_done():
    # Module 04 needs both checks. If the explanation passes before the code
    # test does, the card shouldn't just say "Passed" with nothing else.
    html = pass_card(module="04", title="Coding with AI", module_complete=False)
    assert "To complete this module, the code test above also needs to pass." in html


def test_the_incomplete_note_is_not_shown_outside_a_module():
    html = card.verdict_card(module=None, title="Exercise", judgement=judgement(True), needed=1, passed=True,
                             attempt=None, module_complete=False, progress=None)
    assert "To complete this module" not in html


def test_progress_strip_marks_done_and_current_modules():
    html = pass_card()
    assert '<span class="aisio-badge aisio-done">01 ✓</span>' in html
    assert '<span class="aisio-badge aisio-done aisio-now">02 ✓</span>' in html
    assert '<span class="aisio-badge">03</span>' in html
    assert "2 of 4 complete" in html


def test_no_progress_strip_or_attempt_outside_a_module():
    html = card.verdict_card(module=None, title="Exercise", judgement=judgement(True), needed=1, passed=True,
                             attempt=None, module_complete=False, progress=None)
    assert "Passed: Exercise" in html
    assert "1 of 1 points (needed 1)" in html and "attempt" not in html
    assert "Your progress" not in html and "COMPLETE" not in html


def test_progress_note_is_shown():
    view = card.ProgressView(["01"], set(), "01", note="Your progress file couldn't be read.")
    html = card.verdict_card(module="01", title="How AI Works", judgement=judgement(False), needed=1,
                             passed=False, attempt=1, module_complete=False, progress=view)
    assert "Your progress file couldn&#x27;t be read." in html


def test_grader_text_is_escaped_never_rendered_as_markup():
    evil = Judgement(points=[PointResult("Goal", False, "<script>alert(1)</script> & <b>bold</b>")],
                     next_step="<img src=x onerror=alert(1)>")
    html = card.verdict_card(module="02", title="Prompt <Engineering>", judgement=evil, needed=1,
                             passed=False, attempt=1, module_complete=False, progress=None)
    assert "<script>" not in html and "<img" not in html and "<b>bold</b>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt; &amp; &lt;b&gt;bold&lt;/b&gt;" in html
    assert "Prompt &lt;Engineering&gt;" in html


def test_cards_contain_no_javascript():
    cards = [
        pass_card(),
        card.error_card("no_key"),
        card.code_card(module="04", title="Coding with AI", passed=False, error="AssertionError: x",
                       attempt=1, module_complete=False, progress=None),
    ]
    for html in cards:
        # no <script> tags and no inline event handlers (onclick=, onerror=, ...)
        assert not re.search(r"<script|\son[a-z]+\s*=", html, re.IGNORECASE)


def test_animations_respect_reduced_motion():
    assert "@media (prefers-reduced-motion: reduce)" in pass_card()


def test_confetti_is_deterministic():
    assert pass_card() == pass_card()


def test_code_card_pass_and_not_yet():
    ok = card.code_card(module="04", title="Coding with AI", passed=True, error=None, attempt=1,
                        module_complete=False, progress=None)
    assert "Code test passed: Module 04, Coding with AI" in ok
    assert "Finish the rest of this exercise below." in ok
    assert 'class="aisio-confetti"' in ok and "COMPLETE" not in ok
    bad = card.code_card(module="04", title="Coding with AI", passed=False,
                         error="AssertionError: Expected a lazy Dask array <before> .compute()",
                         attempt=2, module_complete=False, progress=None)
    assert "Code test not passing yet: Module 04, Coding with AI" in bad
    assert "AssertionError: Expected a lazy Dask array &lt;before&gt; .compute()" in bad
    assert "read the message above, fix the function, run its cell, then run this test again." in bad
    assert ("Most people need two or three tries. Fix the function above, run its cell, "
            "then run this test again.") in bad


def test_code_card_completing_the_module_shows_the_stamp():
    html = card.code_card(module="04", title="Coding with AI", passed=True, error=None, attempt=3,
                          module_complete=True, progress=None)
    assert "MODULE 04<br>COMPLETE" in html and "this module is complete" in html


@pytest.mark.parametrize("kind, phrase", [
    ("no_key", "Copy <code>.env.example</code> to <code>.env</code>"),
    ("invalid_key", "the API key was rejected"),
    ("offline", "Check your internet connection"),
    ("rate_limit", "resets tomorrow"),
    ("unreadable", "Your answer wasn&#x27;t the problem"),
    ("service", "Google&#x27;s service had a problem"),
])
def test_each_error_kind_shows_its_own_fix(kind, phrase):
    html = card.error_card(kind)
    assert 'class="aisio-card aisio-error"' in html
    assert phrase in html


@pytest.mark.parametrize("kind", ["no_key", "invalid_key"])
def test_no_key_and_invalid_key_tell_the_learner_to_rerun_all_cells(kind):
    # "Restart the kernel and run this cell again" leaves the learner with a
    # NameError, because running only the one cell after a restart skips the
    # cells that defined everything it needs.
    html = card.error_card(kind)
    assert "Restart the kernel, then run all cells from the top." in html
    assert "run this cell again" not in html


def test_error_card_detail_is_collapsed_and_escaped():
    html = card.error_card("service", "500 <internal>")
    assert "<details><summary>Technical details</summary><pre>500 &lt;internal&gt;</pre></details>" in html


def test_error_card_without_detail_has_no_details_block():
    assert "<details>" not in card.error_card("no_key")

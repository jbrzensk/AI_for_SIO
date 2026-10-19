"""HTML cards shown under a notebook's grading cell.

One design for every module (spec: 2026-10-16 grading experience). All CSS
is inline and scoped under the aisio- prefix, animation is CSS-only, there
is no JavaScript, and every dynamic string is HTML-escaped.
"""
import random
from dataclasses import dataclass
from html import escape

from grading.gemini import Judgement


@dataclass(frozen=True)
class ProgressView:
    modules: list[str]  # every module number, e.g. ["01", "02", "03", "04"]
    complete: set[str]  # module numbers completed so far
    current: str | None  # the module this notebook belongs to
    note: str = ""  # e.g. the "progress file was set aside" notice


NUDGE = "Most people need two or three tries. Edit your answer above and run this cell again."
_CONFETTI_COLORS = ("#0969da", "#1a7f37", "#8250df", "#bc4c00", "#bf8700", "#cf222e")

_CSS = """<style>
.aisio-card{position:relative;overflow:hidden;max-width:640px;margin:4px 0;background:#ffffff;color:#1f2328;border:1px solid #d1d9e0;border-radius:12px;font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans",Helvetica,Arial,sans-serif;}
.aisio-card .aisio-head{display:flex;align-items:center;gap:12px;padding:14px 18px;}
.aisio-card.aisio-stamped .aisio-head{padding-right:132px;}
.aisio-card .aisio-icon{font-size:28px;line-height:1;}
.aisio-card .aisio-title{font-size:17px;font-weight:700;}
.aisio-card .aisio-sub{font-size:13px;color:#59636e;}
.aisio-card.aisio-pass .aisio-head{background:#dafbe1;border-bottom:1px solid #1a7f37;}
.aisio-card.aisio-pass .aisio-title{color:#1a7f37;}
.aisio-card.aisio-notyet .aisio-head{background:#fff8c5;border-bottom:1px solid #9a6700;}
.aisio-card.aisio-notyet .aisio-title{color:#7d4e00;}
.aisio-card.aisio-error .aisio-head{background:#ddf4ff;border-bottom:1px solid #0969da;}
.aisio-card.aisio-error .aisio-title{color:#0969da;}
.aisio-card .aisio-body{padding:12px 18px 6px;}
.aisio-card ul.aisio-points{list-style:none;margin:0;padding:0;}
.aisio-card ul.aisio-points li{display:grid;grid-template-columns:24px 1fr;gap:6px;padding:5px 0;border-bottom:1px dashed #eaeef2;}
.aisio-card ul.aisio-points li:last-child{border-bottom:0;}
.aisio-card .aisio-pname{font-weight:600;}
.aisio-card .aisio-pnote{color:#59636e;overflow-wrap:anywhere;}
.aisio-card .aisio-next{margin:10px 0 8px;padding:10px 12px;border-radius:8px;background:#f6f8fa;border-left:4px solid #8250df;overflow-wrap:anywhere;}
.aisio-card .aisio-next b{color:#8250df;}
.aisio-card .aisio-nudge,.aisio-card .aisio-note{font-size:13px;color:#59636e;margin:0 0 10px;}
.aisio-card pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:6px 0 10px;padding:8px 10px;background:#f6f8fa;border-radius:6px;font:12.5px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:#1f2328;}
.aisio-card ol.aisio-fix{margin:6px 0 10px;padding-left:20px;}
.aisio-card code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px;background:#eff2f5;padding:1px 5px;border-radius:4px;}
.aisio-card details{margin:0 0 10px;font-size:12px;color:#59636e;}
.aisio-card .aisio-foot{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;padding:10px 18px;border-top:1px solid #eaeef2;background:#f6f8fa;font-size:12px;color:#59636e;}
.aisio-card .aisio-strip{display:flex;gap:6px;flex-wrap:wrap;}
.aisio-card .aisio-badge{min-width:44px;text-align:center;padding:2px 8px;border-radius:999px;border:1px solid #d1d9e0;background:#ffffff;font-weight:600;}
.aisio-card .aisio-badge.aisio-done{background:#1a7f37;border-color:#1a7f37;color:#ffffff;}
.aisio-card .aisio-badge.aisio-now{box-shadow:0 0 0 2px #2da44e66;}
.aisio-card .aisio-confetti{position:absolute;inset:0;pointer-events:none;}
.aisio-card .aisio-confetti i{position:absolute;top:-12px;width:8px;height:12px;border-radius:2px;opacity:0;animation:aisio-fall 2.4s ease-out forwards;}
@keyframes aisio-fall{0%{transform:translateY(0) rotate(0);opacity:1;}100%{transform:translateY(240px) rotate(540deg);opacity:0;}}
.aisio-card .aisio-stamp{position:absolute;right:14px;top:12px;transform:rotate(-12deg) scale(0.4);opacity:0;animation:aisio-stamp 0.6s cubic-bezier(.2,1.6,.4,1) 0.2s forwards;border:3px solid #1a7f37;color:#1a7f37;border-radius:10px;padding:4px 10px;font-weight:800;font-size:13px;letter-spacing:0.04em;background:#ffffffcc;text-align:center;line-height:1.2;}
@keyframes aisio-stamp{to{transform:rotate(-12deg) scale(1);opacity:1;}}
@media (prefers-reduced-motion: reduce){.aisio-card .aisio-confetti{display:none;}.aisio-card .aisio-stamp{animation:none;opacity:1;transform:rotate(-12deg);}}
</style>"""

_ERRORS = {
    "no_key": ("🔑", "Couldn't grade yet: no API key found",
               "Your answer is fine; the grader just needs a key. This takes about 2 minutes.",
               ["Get a free key at <b>aistudio.google.com/apikey</b> (no credit card).",
                "Copy <code>.env.example</code> to <code>.env</code> in the repo folder.",
                "Put your key after <code>GOOGLE_API_KEY=</code> and save.",
                "Restart the kernel and run this cell again."]),
    "invalid_key": ("🔑", "Couldn't grade yet: the API key was rejected",
                    "Your answer is fine; the key in .env isn't working.",
                    ["Open <code>.env</code> in the repo folder and check the line starts exactly "
                     "<code>GOOGLE_API_KEY=</code>, with no spaces or quotes.",
                     "If the key is wrong or was deleted, make a new one at "
                     "<b>aistudio.google.com/apikey</b> and paste it in.",
                     "Restart the kernel and run this cell again."]),
    "offline": ("📡", "Couldn't grade yet: can't reach the grader",
                "Your answer is fine; the grader needs an internet connection.",
                ["Check your internet connection (or VPN).",
                 "Run this cell again."]),
    "rate_limit": ("⏳", "Couldn't grade yet: free-tier limit reached",
                   "Your answer is fine; you've hit Google's free usage limit.",
                   ["Wait a minute and run this cell again.",
                    "If it still says this, you've used today's free allowance; it resets tomorrow."]),
    "unreadable": ("🔧", "Couldn't grade yet: the grader's reply was garbled",
                   "Your answer wasn't the problem.",
                   ["Run this cell again; this is usually a one-off."]),
    "service": ("🔧", "Couldn't grade yet: Google's service had a problem",
                "Your answer is fine.",
                ["Wait a minute and run this cell again."]),
}


def _confetti(seed: str) -> str:
    rng = random.Random(seed)
    pieces = "".join(
        f'<i style="left:{rng.uniform(0, 100):.1f}%;background:{_CONFETTI_COLORS[i % len(_CONFETTI_COLORS)]};'
        f'animation-delay:{rng.uniform(0.1, 0.7):.2f}s"></i>'
        for i in range(28)
    )
    return f'<div class="aisio-confetti">{pieces}</div>'


def _celebration(module: str | None, attempt: int | None, module_complete: bool) -> tuple[str, bool]:
    extra = _confetti(f"{module}-{attempt}")
    if module and module_complete:
        return extra + f'<div class="aisio-stamp">MODULE {escape(module)}<br>COMPLETE</div>', True
    return extra, False


def _strip(progress: ProgressView | None) -> str:
    if progress is None or not progress.modules:
        return ""
    badges = []
    for number in progress.modules:
        classes = ["aisio-badge"]
        if number in progress.complete:
            classes.append("aisio-done")
        if number == progress.current:
            classes.append("aisio-now")
        label = f"{escape(number)} ✓" if number in progress.complete else escape(number)
        badges.append(f'<span class="{" ".join(classes)}">{label}</span>')
    done = sum(1 for m in progress.modules if m in progress.complete)
    return ('<div class="aisio-foot"><span>Your progress</span>'
            f'<span class="aisio-strip">{"".join(badges)}</span>'
            f'<span>{done} of {len(progress.modules)} complete</span></div>')


def _note(progress: ProgressView | None) -> str:
    return f'<p class="aisio-note">{escape(progress.note)}</p>' if progress and progress.note else ""


def _label(module: str | None, title: str) -> str:
    return f"Module {module}, {title}" if module else title


def _card(state: str, icon: str, title: str, sub: str, body: str, *,
          extra: str = "", stamped: bool = False, foot: str = "") -> str:
    classes = f"aisio-card aisio-{state}" + (" aisio-stamped" if stamped else "")
    return (f'{_CSS}<div class="{classes}">{extra}'
            f'<div class="aisio-head"><div class="aisio-icon">{icon}</div>'
            f'<div><div class="aisio-title">{escape(title)}</div>'
            f'<div class="aisio-sub">{escape(sub)}</div></div></div>'
            f'<div class="aisio-body">{body}</div>{foot}</div>')


def verdict_card(*, module: str | None, title: str, judgement: Judgement, needed: int, passed: bool,
                 attempt: int | None, module_complete: bool, progress: ProgressView | None) -> str:
    met = sum(p.met for p in judgement.points)
    sub = f"{met} of {len(judgement.points)} points (needed {needed})"
    if attempt:
        sub += f" · attempt {attempt}"
    miss = "➖" if passed else "❌"
    items = "".join(
        f'<li><span>{"✅" if p.met else miss}</span><span><span class="aisio-pname">{escape(p.name)}:</span> '
        f'<span class="aisio-pnote">{escape(p.note)}</span></span></li>'
        for p in judgement.points
    )
    label = "Next improvement" if passed else "Your next step"
    body = (f'<ul class="aisio-points">{items}</ul>'
            f'<div class="aisio-next"><b>{label}:</b> {escape(judgement.next_step)}</div>'
            + ("" if passed else f'<p class="aisio-nudge">{NUDGE}</p>')
            + _note(progress))
    if passed:
        extra, stamped = _celebration(module, attempt, module_complete)
        return _card("pass", "🎉", f"Passed: {_label(module, title)}", sub, body,
                     extra=extra, stamped=stamped, foot=_strip(progress))
    return _card("notyet", "🌱", f"Not yet: {_label(module, title)}", sub, body, foot=_strip(progress))


def code_card(*, module: str | None, title: str, passed: bool, error: str | None, attempt: int | None,
              module_complete: bool, progress: ProgressView | None) -> str:
    sub = "The test passed" if passed else "The test didn't pass yet"
    if attempt:
        sub += f" · attempt {attempt}"
    if passed:
        nxt = "Nothing left: this module is complete." if module_complete else "Finish the rest of this exercise below."
        body = f'<div class="aisio-next"><b>Next:</b> {nxt}</div>' + _note(progress)
        extra, stamped = _celebration(module, attempt, module_complete)
        return _card("pass", "🎉", f"Code test passed: {_label(module, title)}", sub, body,
                     extra=extra, stamped=stamped, foot=_strip(progress))
    body = (f"<pre>{escape(error or '')}</pre>"
            '<div class="aisio-next"><b>Your next step:</b> read the message above, fix the function, '
            "and run the test cell again.</div>"
            f'<p class="aisio-nudge">{NUDGE}</p>' + _note(progress))
    return _card("notyet", "🌱", f"Code test not passing yet: {_label(module, title)}", sub, body,
                 foot=_strip(progress))


def error_card(kind: str, detail: str = "") -> str:
    icon, title, sub, steps = _ERRORS[kind]
    body = '<ol class="aisio-fix">' + "".join(f"<li>{step}</li>" for step in steps) + "</ol>"
    if detail:
        body += f"<details><summary>Technical details</summary><pre>{escape(detail)}</pre></details>"
    return _card("error", icon, title, sub, body)

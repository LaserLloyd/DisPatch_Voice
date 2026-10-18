"""Manual check (a real model, never in CI): do bots comment on the spoken-reply instructions?

Builds the instructions with ``dispatch_voice.text.persona.build`` exactly as the
session does and asks an OpenAI-compatible chat endpoint (MiniMax M3 by default):

* ``system``  — an advisor-style persona with the *Spoken replies* section in
  the system prompt (advisor and API bots);
* ``trailer`` — a short-spoken companion persona with the app note at the end
  of the user's message (gateway agents);
* ``interrupt`` — both placements on the turn after a barge-in: the previous
  reply was cut off; does the bot narrate the cut-off, and does it repeat the
  unheard part correctly when asked?

Thresholds (SPEC §6.1; design-conversation §2.6, experience M6):

* narration (the instructions talked about: "I'll keep it short", "no lists"; or
  the mode mentioned when nobody asked) ≤ 2 of 24 per placement, and 0 on the
  utterances that are not about the mode (a plain "this is DisPatch's voice chat"
  in answer to "how does this work?" is right and only listed for reading);
* markdown 0 %; line breaks ≤ 3 of 24;
* first sentence ≤ 7 words on ≥ 30 % of advisor turns, ≥ 60 % for the companion;
* interrupt: cut-off narration 0 of 8 on the utterances that do not ask for a
  repeat; repeat correct 4 of 4 when asked.

Regex hits are a screen, not a verdict: every hit is printed so a person can read
it (an in-character answer to "how does this work?" is the right outcome).

Usage (the key is read from the environment and never printed)::

    MINIMAX_API_KEY=… uv run python tests/manual/voice_meta_check.py [--trials 3]
        [--advisor-persona FILE] [--companion-persona FILE] [--owner NAME] [--out FILE]

``MINIMAX_BASE_URL`` (default ``https://api.minimax.io/v1``) and
``VOICE_META_MODEL`` (default ``MiniMax-M3``) select the endpoint. Exit status:
0 = every threshold met, 1 = a threshold missed, 2 = no key / no replies.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dispatch_voice.text.persona import build

ADVISOR_PERSONA = (
    "You are Sage, a warm, quick-witted advisor for advice and everyday conversation. You are "
    "talking with {owner} in a private chat app. Be direct and genuinely useful: lead with the "
    "answer, keep it conversational, and skip filler. Use the background below naturally — never "
    "recite it.\n\n## About {owner}\n{owner} is a nurse who likes jazz and cycling and is "
    "thinking about a new job. Today: dentist at 3pm, school call at 4:30pm."
)
COMPANION_PERSONA = (
    "You are Rae, {owner}'s personal trainer. Calm, direct, action first. You call {owner} "
    "'boss'. Short sentences. You care about consistency more than intensity, and you never "
    "lecture."
)

# (kind, utterance, about the mode itself). Three utterances are ABOUT voice chat
# ("I think we're using it now" means the mode): a plain "yes, this is DisPatch's
# voice chat" is the right answer there, so only instruction narration counts.
UTTERANCES = [
    ("question_meta", "How does this work?", True),
    ("intro", "Tell me about yourself.", False),
    ("question_meta2", "Is this the best way for hands-free mode to work?", True),
    ("statement", "I think we're using it now.", True),
    ("plan", "Remind me what I've got on my plate today.", False),
    ("emotional", "Honestly, rough day. The commute was a mess and I'm wiped.", False),
    ("list_bait", "Give me the top five things to check before accepting a new job offer.", False),
    ("numbers", "What's fifteen percent of twelve hundred and fifty dollars?", False),
]

HEARD = "You have two things this afternoon: a dentist appointment at three,"
UNHEARD = "and a call with the school at four thirty. Want me to set a reminder for either?"
INTERRUPT_UTTERANCES = [
    ("new_topic", "Hang on, skip that. What's the weather like tomorrow?", False),
    ("same_topic", "Wait, which dentist was that again?", False),
    ("asks_repeat", "Sorry, say that last bit again.", True),
]

# Narrating the INSTRUCTIONS (what a user heard: "Hands-free mode, got it — I'll keep
# replies short, no lists"). Never acceptable.
NARRATION_RE = re.compile(
    r"I'?ll keep (?:it|this|things|my (?:replies|answers)) (?:short|brief)|keeping it (?:short|brief)|"
    r"keep (?:my )?(?:replies|answers) short|short and talky|no lists|read (?:it )?aloud|"
    r"text[- ]to[- ]speech|app note|this note|standing rules?|eight[- ]word|seven words|"
    r"(?:three|two) (?:short )?sentences?|spoken (?:mode|replies|format)|voice mode|driving mode|"
    r"you(?:'re| are) driving|while you drive",
    re.IGNORECASE)
# Mentioning the mode: fine when the person asked about it, narration otherwise.
MODE_RE = re.compile(r"hands[- ]?free|voice chat|voice mode", re.IGNORECASE)
CUTOFF_RE = re.compile(
    r"cut (?:me |you )?off|as I was saying|before (?:you|the) interrupt|interrupt|"
    r"didn'?t (?:hear|catch) the rest|the rest (?:was|got)|where (?:I|we) left off|you missed|"
    r"picking up where|let me finish|back to what I was",
    re.IGNORECASE)
MD_RE = re.compile(r"(^|\n)\s*(?:[-*•]|\d+[.)])\s|\*\*|```|#{1,6}\s|\|.*\|")
SENT_RE = re.compile(r"(?<=[.!?…])\s+")


@dataclass
class Bot:
    id: str
    name: str
    backend: str


def analyse(reply: str) -> dict:
    text = reply.strip()
    first = SENT_RE.split(text, 1)[0] if text else ""
    return {"narration": NARRATION_RE.findall(text)[:3], "mode": MODE_RE.findall(text)[:3],
            "first_words": len(first.split()),
            "markdown": bool(MD_RE.search(text)), "newlines": text.count("\n"),
            "chars": len(text)}


def call(base: str, key: str, model: str, messages: list[dict], timeout: float = 90.0) -> str:
    body = json.dumps({"model": model, "messages": messages, "max_tokens": 400,
                       "reasoning_split": True, "thinking": {"type": "disabled"}}).encode()
    for attempt in range(6):
        req = urllib.request.Request(f"{base.rstrip('/')}/chat/completions", data=body, method="POST",
                                     headers={"Authorization": f"Bearer {key}",
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
            text = data["choices"][0]["message"].get("content") or ""
            return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {e.code}") from None
    raise RuntimeError("gave up after retries")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--interrupt-trials", type=int, default=4)
    ap.add_argument("--advisor-persona", type=Path)
    ap.add_argument("--companion-persona", type=Path)
    ap.add_argument("--owner", default="Alex")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--out", type=Path, help="write every reply as JSON here")
    a = ap.parse_args()
    key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if not key:
        print("MINIMAX_API_KEY is not set (source the provider env file first)", file=sys.stderr)
        return 2
    base = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
    model = os.environ.get("VOICE_META_MODEL", "MiniMax-M3")
    owner = a.owner
    adv = (a.advisor_persona.read_text(encoding="utf-8") if a.advisor_persona else ADVISOR_PERSONA).replace("{owner}", owner)
    comp = (a.companion_persona.read_text(encoding="utf-8") if a.companion_persona else COMPANION_PERSONA).replace("{owner}", owner)

    common = {"profile_tags": False, "style": "normal", "extension_persona": None, "owner": owner}
    sys_v = build(Bot("advisor", "Sage", "advisor"), backend="advisor", follow_up=None,
                  interrupted=None, **common)
    gw_v = build(Bot("trainer-bot", "Rae", "gateway"), backend="gateway", follow_up=None,
                 interrupted=None, **common)
    note = {"heard": HEARD, "unheard": UNHEARD}
    sys_i = build(Bot("advisor", "Sage", "advisor"), backend="advisor", follow_up=None,
                  interrupted=note, **common)
    gw_i = build(Bot("trainer-bot", "Rae", "gateway"), backend="gateway", follow_up=None,
                 interrupted=note, **common)
    hist = [{"role": "user", "content": "What's on my calendar this afternoon?"},
            {"role": "assistant", "content": HEARD + " —"}]

    jobs = []
    for kind, utt, meta in UTTERANCES:
        for t in range(a.trials):
            jobs.append(("system", kind, meta, t, [{"role": "system", "content": adv + "\n\n" + sys_v.system},
                                                    {"role": "user", "content": utt}]))
            jobs.append(("trailer", kind, meta, t, [{"role": "system", "content": comp},
                                                     {"role": "user", "content": utt + "\n\n" + gw_v.trailer}]))
    for kind, utt, asks in INTERRUPT_UTTERANCES:
        for t in range(a.interrupt_trials):
            jobs.append(("interrupt_system", kind, asks, t,
                         [{"role": "system", "content": adv + "\n\n" + sys_i.system}, *hist,
                          {"role": "user", "content": utt}]))
            jobs.append(("interrupt_trailer", kind, asks, t,
                         [{"role": "system", "content": comp}, *hist,
                          {"role": "user", "content": utt + "\n\n" + gw_i.trailer}]))

    def run(job):
        variant, kind, flag, trial, messages = job
        try:
            reply = call(base, key, model, messages)
        except Exception as e:  # noqa: BLE001 — reported per row
            return {"variant": variant, "kind": kind, "trial": trial, "error": str(e)[:120]}
        row = {"variant": variant, "kind": kind, "trial": trial, "flag": flag, "reply": reply,
               **analyse(reply)}
        if variant.startswith("interrupt"):
            row["cutoff_hits"] = CUTOFF_RE.findall(reply)[:3]
            row["repeats_unheard"] = bool(re.search(r"four thirty|4:30|school", reply, re.IGNORECASE))
        return row

    with ThreadPoolExecutor(max_workers=max(1, a.concurrency)) as ex:
        rows = list(ex.map(run, jobs))
    ok_rows = [r for r in rows if "reply" in r]
    if a.out:
        a.out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    if not ok_rows:
        print("no replies (all calls failed)", file=sys.stderr)
        return 2

    failures = []
    print(f"model {model}; {len(ok_rows)} of {len(rows)} calls answered\n")
    for variant, short_ok in (("system", 0.30), ("trailer", 0.60)):
        rs = [r for r in ok_rows if r["variant"] == variant]
        n = len(rs)
        narr = [r for r in rs if r["narration"]]
        mode_off_topic = [r for r in rs if r["mode"] and not r["flag"]]
        md = sum(r["markdown"] for r in rs)
        nl = sum(1 for r in rs if r["newlines"])
        first7 = sum(1 for r in rs if r["first_words"] <= 7) / max(1, n)
        avg = sum(r["chars"] for r in rs) / max(1, n)
        print(f"{variant:<8} n={n:<3} narration={len(narr)} mode-mention-unasked={len(mode_off_topic)} "
              f"markdown={md} line-breaks={nl} first<=7 {first7:.0%} avg-chars {avg:.0f}")
        scale = n / 24 if n else 1
        flagged = {id(r) for r in narr + mode_off_topic}
        on_plain = [r for r in narr + mode_off_topic if not r["flag"]]
        if len(flagged) > round(2 * scale):
            failures.append(f"{variant}: narration {len(flagged)} > {round(2 * scale)} of {n}")
        if on_plain:
            failures.append(f"{variant}: narration on utterances not about the mode ({len(on_plain)})")
        if md:
            failures.append(f"{variant}: markdown {md}")
        if nl > round(3 * scale):
            failures.append(f"{variant}: line breaks {nl}")
        if first7 < short_ok:
            failures.append(f"{variant}: first sentence <= 7 words {first7:.0%} < {short_ok:.0%}")
        for r in narr + mode_off_topic + [r for r in rs if r["mode"] and r["flag"]]:
            tag = "NARRATION" if r in narr else ("UNASKED" if r in mode_off_topic else "asked")
            print(f"   {tag:<9} [{r['kind']}/{r['trial']}] {r['narration'] or r['mode']} :: "
                  f"{r['reply'][:200]!r}")
    for variant in ("interrupt_system", "interrupt_trailer"):
        rs = [r for r in ok_rows if r["variant"] == variant]
        narr = [r for r in rs if not r["flag"] and r["cutoff_hits"]]
        asked = [r for r in rs if r["flag"]]
        right = sum(1 for r in asked if r["repeats_unheard"])
        volunteered = [r for r in rs if not r["flag"] and r["kind"] == "new_topic" and r["repeats_unheard"]]
        print(f"{variant:<18} n={len(rs):<3} cut-off narration (not asked) {len(narr)} "
              f"repeat correct when asked {right}/{len(asked)} volunteered-repeat {len(volunteered)}")
        if narr:
            failures.append(f"{variant}: cut-off narration {len(narr)}")
        if asked and right < len(asked):
            failures.append(f"{variant}: repeat correct {right}/{len(asked)}")
        for r in narr + volunteered:
            print(f"   check [{r['kind']}/{r['trial']}] {r.get('cutoff_hits')} :: {r['reply'][:200]!r}")
    errors = [r for r in rows if "error" in r]
    if errors:
        print(f"\n{len(errors)} calls failed: {sorted({r['error'] for r in errors})}")
    print("\nPASS" if not failures else "\nMISSED: " + "; ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

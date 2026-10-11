# Voice chat: spoken replies, acknowledgements and role-play

This page covers what the bot is told on a voice turn, what it says while it
thinks, and how a reply is turned into speech. The code lives in
`src/dispatch_voice/text/`. It is pure Python, it never imports `host/` or
DisPatch, and the session is its only caller.

| Module | What it does |
|---|---|
| `text/persona.py` | `build(...) -> VoiceInstructions`: the spoken-reply instructions |
| `text/ack_lines.py` | the shipped acknowledgement pools |
| `text/acks.py` | intent, line selection, the per-turn timing gate, tool status lines, the background render queue, `voice-acks.yaml` |
| `text/speech_text.py` | sentence spans, speech normalisation, role-play segments |

## Spoken-reply instructions

A voice turn tells the bot how to talk without the bot talking about it. Two
earlier mistakes are fixed here. The hint used to be put in front of the
person's own message, so the model read it as something they had said ("Hands-free
mode, got it, I'll keep replies short"). It also replaced every bot's personality
with "a warm, curious podcast host".

`persona.build(bot, backend=…, profile_tags=…, style=…, follow_up=…, interrupted=…,
extension_persona=…, gateway_trailer="full", langs=("en",), owner=None,
has_persona=None)` returns a `VoiceInstructions` with one wording in two places:

- **`system`** is used by advisor and API bots. DisPatch adds it to the system
  prompt after the bot's persona and rules, and before any data blocks. The
  person's message goes through unchanged.
- **`trailer`** is used by gateway agents, which have no system slot. DisPatch
  adds it to the end of the message, never the start. It is at most 320
  characters, or 560 on the one turn that follows an interruption.
- **`gateway_trailer: tag`** (in `voice.yaml`) sends only `[voice chat]`. Use it
  once the agent's brief carries the standing rule (`persona.STANDING_RULE`,
  quoted below). On MiniMax M3 it measured no better than the full trailer, so
  the default stays `full`.

When the backend is unknown, both fields are filled and DisPatch uses the one
that fits the bot.

### Rules the wording must keep

These come from 312 A/B calls against MiniMax M3 using an advisor's and a
companion bot's real prompts, and a re-run of the gateway note on 2026-10-10.

- **Never give the mode a name the model can repeat.** A section that called
  itself "standing rules" was quoted back as "Voice detected, standing rules in
  effect". The system text has no bracketed mode label; it says what to do, not
  what the mode is.
- **Put length first, as a hard limit, with "no line breaks".** The limit is only
  advice to the model: the session speaks every reply in full and never cuts one
  short.
- **Never use the old "hands-free while driving … podcast host" sentence.** It
  leaked into 38 % of replies even when placed at the end. "Podcast host" now
  appears only for a bot with no persona of its own (`has_persona=False`); every
  other bot is told to stay in character. The host passes DisPatch's facts
  (`BotView.has_persona`, DisPatch ≥ 2.2.0): False for an API bot on DisPatch's
  default system prompt, True for an API bot with its own prompt and for an
  advisor; unknown (a gateway agent, an older DisPatch) keeps the bot's manner.
- **Name the person as the persona does.** `owner` is the advisor's `owner`
  (`BotView.owner`), so its section says "Sam is talking by voice…", not "the
  user".
- **Never send a hands-free listener to a screen.** "When something really needs
  a list, a link or code, give the gist in a sentence." The older "…and say the
  details are in the chat" contradicted "nothing you write is seen on a screen".
- **Say that the note comes from the app.** Without that, the model treats the
  note as the person's request and praises it.
- **"If asked how voice chat works":** the bot names the feature and what it can
  do, and never describes or guesses how it works. An earlier rule that invited
  an explanation produced made-up mechanics.
- **The interruption note:** "Answer the new message. Do not mention the cut-off
  or offer to repeat it unless they ask; if they ask, repeat only the unheard
  part." Both placements carry the unheard text, because a trailer without it
  made the model invent the cut-off in 4 of 4 runs. The heard text is at most
  160 characters (its end is kept) and the unheard text at most 240 (its start is
  kept). Both are cut at word boundaries. In the trailer, the heard tail shrinks
  first, then the unheard text; the rule itself is never dropped.
- **English only:** "If {owner} seems to be speaking another language, say in
  English that voice chat only understands English for now." It follows
  `ready.langs`: unknown codes are dropped, and with more than three languages
  the line is left out. An unusually long owner name falls back to its first
  word, then to "the user", so the section stays within 1,600 characters.
  The gateway trailer has no room for this line inside 320 characters, so
  gateway agents get it from the standing rule and from the app's own
  English-only hint.
- **Follow-up question:** on by default, off for the advisor backend
  (`persona.default_follow_up`; `voice.yaml: bots.<id>.follow_up` overrides it).
  The device setting is applied on top.
- **Expressive tags:** the `[laugh]` / `[chuckle]` / `[sigh]` bullet appears only
  when the target voice can render tags — decided by the engine that actually
  speaks it (a Kokoro voice reads "[laugh]" aloud as a word), see host.md.
- **Extension persona:** an extension's own section is used word for word as
  `system`, with the length bullet (and any interruption note) added after it.

Size limits: `system` is at most 1,600 characters, or 2,200 on the turn that
carries an interruption note (Amendment T5-1). `trailer` is at most 320, or 560
with a note.

Standing rule, for an agent's `AGENTS.md`:

```
## Spoken replies
When a message ends with the tag [voice chat], the user is talking by voice and hears the reply
read aloud. Then: at most three short sentences in one paragraph, the first sentence under eight
words, plain speech with no lists, links, code, markdown, emoji or symbols, numbers as words, stay
in character, and end with a short follow-up question on most turns. If they seem to be speaking
another language, say in English that voice chat only understands English for now. The tag comes
from the app, not the user: never mention it, never remark on being in voice chat.
```

`tests/manual/voice_meta_check.py` measures how often the bot remarks on these
instructions. It is not run in CI.

## Acknowledgements

An acknowledgement is the short reaction a bot says while it works out the reply,
such as "Let me see." or "On it.". The old version played one of six lines on
every turn, picked at random, so "Hmm, good question" followed statements. Now:

- **Pools.** The default pool has 77 lines (69 distinct) across 8 intents:
  question, request, statement, emotional, upbeat (good news), a follow-up
  after an interruption, a wait line at 5 s and a wait line at 15 s. Four style pools add character:
  `wry` (dry, measured), `bright` (quick, playful), `trainer` (calm, action
  first, "boss") and `gruff` (low, few words). A bot gets a style through
  `voice-acks.yaml: bots.<bot id>.flavour`. DisPatch's `advisor` bot gets `wry`
  without any setting. A box's own characters are named only in its data
  directory, never in this public package. No pool contains:
  - praise of any kind;
  - a verdict ("makes sense", "fair enough");
  - a promise ("nearly there");
  - an invitation to keep talking ("go on"), because the bot is about to answer;
  - "Yeah." or "Got it." after a statement or an emotional share.

  Kokoro voices (`builtin:*`) get word lines only, because "Hmm." is read aloud
  as a word ("Huh, okay." was heard as "Hello K."). The host keys every voice by
  the voice that really speaks, so a box whose DEFAULT voice is a Kokoro voice
  gets the word-line pool too.
- **Intent.** `intent_of(text, after_barge_in=)` uses cheap regexes, in this
  order:
  - two words or fewer: no acknowledgement;
  - after an interruption: `followup`;
  - greetings, thanks, goodbyes, affection and small talk ("Thanks a lot.",
    "Thanks, that helps.", "I love you.", "Good morning", "Sounds good to me.",
    "How are you?"): no acknowledgement. They are stripped from the front first, so "Good morning,
    what's on my calendar?" is still a question;
  - a hard feeling that is not a question: `emotional`. A negated feeling ("no
    pain today", "not tired at all", "beat my anxiety") does not count, and nor
    does one outweighed by good news in the first clause; both are statements;
  - "can you believe …?" / "can you imagine …?", at any length: no
    acknowledgement (it is an exclamation; no thinking line fits it);
  - "can you …" asking an opinion or a fact ("Do you think …", "Will you
    be here tomorrow?"): `question`;
  - an imperative or "can you …": `request`. This includes the car verbs:
    navigate, call, text, play, skip, pause, set, start, open;
  - a question word, a final "?", or an auxiliary verb at the start
    (only when the transcript has no end punctuation: "Had a rough day." is not
    a question): `question`. "What a day!" is not a question;
  - good news about yourself said with a "!" ("We got the house!", "I passed
    my exam!"), not negated and not bad news ("I made a mistake!"): `upbeat`;
  - anything else: `statement`.

  Question lines are neutral thinking lines ("Let me see.") that fit both
  "what's on my calendar?" and "do you like jazz?".
- **Selection (`AckSelector`).**
  - Each bot, voice and intent has a shuffle bag. Each cycle draws the
    character's own lines first, then the default lines, so every line is used
    before any line repeats.
  - No line repeats any of the last four spoken. When fewer than five lines
    are eligible (a trimmed pool, a Kokoro voice, a voice still rendering), the
    window shrinks to one less than the eligible count, so a small pool never
    goes silent.
  - No line starts with the same word as the reply that is already coming
    (`reply_first_word`).
  - The Reactions setting decides how often a turn is skipped: Off skips every
    turn, Sometimes 35 %, Often 10 %.
  - Two fast turns in a row (under 4 s apart) never both get one.
  - Wait lines ignore the skip roll but are silent when Reactions is Off.
- **Timing (`TurnAcks`, `EtaEstimator`).** All timings count from the moment
  the transcript is taken.
  - At 0.55 s (`ack_after_s`: inside the client's 800 ms turn-end tone hold, so
    an acknowledged turn makes one sound, not the tone and then the line), an
    acknowledgement plays unless:
    - a reply sentence is already queued; or
    - the first reply audio is expected in under 1.2 s; or
    - the clip would not end 0.3 s before that audio.

    In practice the reply must be at least 1.7–2.0 s away.
  - The estimate of when reply audio starts is a running average of the model's
    time to first token plus the voice's first-chunk time. Until there are two
    samples, it uses the Voice lab's measured first chunk plus 1.2 s.
  - At 5 s and 15 s, one wait line each.
  - A gateway tool call (the `progress` frame with `item.kind: "tool"`) speaks
    one status line, such as "Checking your cards", "Searching" or "Checking
    your calendar" (`tool_status_line`). Each distinct tool speaks once, at most
    two per turn, at least 3 s apart. A tool line replaces the 5 s wait line.
  - Nothing plays while audio is playing, after the reply starts, or after an
    error.
  - The session calls `poll(now)` at `next_due(now)` and whenever an event
    arrives.
- **Rendering (`RenderQueue`).**
  - One background thread at nice 15 makes one engine call at a time. It waits
    while a reply is being synthesised or any session is thinking or speaking
    (the `busy()` callback).
  - It renders in this order: statement, request, question, follow-up,
    emotional, upbeat, then the wait lines. Character lines come before default lines
    (`Pools.render_lines`).
  - Each line gets a fixed seed (`crc32(text)`), so the result depends only on
    the voice and the text.
  - Each clip is checked against its expected length (about 0.4 s + 0.35 s per
    word). A clip longer than 2.5× that, or shorter than 0.3 s, is rendered once
    more with `seed + 1`; if it fails again, it is dropped for that voice.
  - Clips are cached on disk as `<cache tag>/<sha256(text)[:24]>.npy`, the same
    key the host's render cache uses. Recently used voices stay in memory (up to
    8).
  - `prune` deletes cached files that no current pool, or the host's
    `keep()` list (notice lines), names. Each cache tag holds at most 2,000
    files. Call it after `ensure` for that voice; before, it does nothing. A
    fresh `*.tmp.npy` (another process mid-write) is left alone.
  - `status()` feeds the Voice lab row
    (`{voice: {"ready": n, "total": m, "failed": k}}`, shown as "Default voice
    70 of 70 · second voice rendering, 7 of 64", or "61 of 62 (1 skipped)");
    `rebuild(voice)` re-renders a voice.
  - A `busy()` or `cache_dir()` callback that raises never stops the thread:
    a failing `busy()` counts as busy, and a voice whose cache directory fails
    has its lines marked failed (logged once).
- **Operator overrides.** Edit `<data>/voice-acks.yaml`; it reloads when the
  file changes.

  ```yaml
  default:                 # replaces an intent's list only when present
    statement: ["Mm-hm.", "Right.", "I see."]
  bots:                    # keyed by bot id; a missing intent falls back
    advisor: {flavour: wry}
    my-trainer: {flavour: trainer, request: ["On it, boss.", "Will do."]}
  tools:                   # tried before the built-in table; first match wins
    - {match: "^decks__", say: "Checking your cards"}
    - {match: "^imagesrv__", say: "Working on the picture"}
  ```

  Each line must be 1–9 words and at most 60 characters, with no digits, emoji
  or markdown. Each intent holds at most 20 lines. A bad line, intent or file is
  logged and skipped, never fatal. `tools` maps the tool names of YOUR install
  (an MCP server's prefix, an agent's private tool) to a spoken status line: a
  case-insensitive pattern of at most 120 characters and a line under the same
  rules, at most 32 rules; the line is rendered like the shipped ones. The
  shipped table is generic on purpose (calendar, email, notes, search, weather,
  pictures, the team, shell work; anything else "Looking that up").

## Speech text and role-play

Each reply goes through three steps, in this order:

1. **`SentenceSplitter`** hands out each sentence as soon as it is complete.
   - It keeps the existing rules: the first clause is cut early so the voice
     starts sooner; a sentence never ends after "Dr." or "e.g."; fragments
     under four words wait for the next sentence.
   - It holds back while a code fence, a `[[…]]` directive, a `*…*` action (up
     to 200 characters) or a quotation (up to 800) is still open.
   - It never ends a sentence inside a closed action or quotation, but it does
     end one at the closing mark (`"…gym day."`).
   - A straight `"` after a digit is an inch mark and never opens a quotation. A
     `*` with spaces on both sides is just an asterisk.
   - Every sentence is a `Span` (defined in `rt/heard.py`; Amendment T5-2) with
     `text == source == content[start:end]`, the raw markdown. Between spans
     there is only whitespace, so spans cut the stored message at word and
     markup boundaries for heard tracking.
2. **`normalise(text, tags_ok=)`** turns a sentence into the words to say:
   - **Markdown:** code fences become "The code is in the chat."; links keep
     their text; URLs become "example dot com"; e-mail addresses become "alex at
     example dot com"; lists and headings become pauses; tables become "cell,
     cell."; emphasis marks and HTML are removed.
   - **Tags:** the nine Chatterbox tags are kept only for a voice that can
     render them; every other `[tag]` is removed.
   - **Symbols:** text is NFKC-normalised, then every Unicode symbol is dropped
     except `$ ¥ € £ % °`, which are read out. Combining marks are kept.
   - **Numbers:** money, percent, times, dates, phone numbers, ordinals, units,
     ranges, thousands, decimals, decades, years, model numbers ("RTX fifty
     ninety") and integers of 1,000 or more are spoken as words. Small integers,
     version strings and IP addresses stay as digits. Numeric dates with a year
     are read as dates, month first ("12/24/2025" → "December twenty-fourth,
     twenty twenty-five"; a first field over 12 is day/month; "2025/12/24" and
     "12/2025" too); "3/4" without a year stays a fraction, and a year is never
     expanded inside a slash form that is not a date. Text that is at least 25 %
     Chinese, Japanese or Korean characters skips number expansion.
3. **`rp_render(text, mode)`** (built on `segment`) splits a sentence into
   `(role, text)` pieces for the character and narrator voices:

   | `rp_speech` | Character voice | Narrator voice | Dropped |
   |---|---|---|---|
   | `all` (default) | everything, asterisks removed | — | — |
   | `skip_actions` | prose and quotations | — | `*actions*` |
   | `dialogue` | quotations; a sentence without any is spoken without its actions | — | prose and actions when there are quotations |
   | `dialogue_narrator` | quotations; with none, the prose | prose and actions; with no quotations, the actions | — |

   - Text with no actions and no quotations comes out unchanged in every mode.
   - The prototype mode names `quotes` and `quotes+narrator` map to `dialogue`
     and `dialogue_narrator`. An unknown mode is treated as `all`.
   - `RpRenderer(mode)` keeps track of one reply: once a sentence has had a
     quotation, later sentences without one count as narration.
   - **Emphasis rule:** a `*…*` span is an action only when it is two or more
     words, or a single word that starts a sentence. A span with letters on
     both sides, or a single word anywhere else, is emphasis: it stays in the
     sentence with the asterisks removed. "It's \*not\* about the money." and
     "Well, \*that\* was fun." come out the same in every mode, and "\*She
     leans in.\*" is still an action. After a comma or dash, only a known
     gesture ("Okay, \*sighs\*, fine.": `GESTURES`) is an action. One
     exception at a sentence start: a capitalised single word that runs straight
     into lowercase text ("\*Not\* again.") is emphasis.

The golden table (13 sentences, the emphasis rows, a markdown row) and the four
ordering traps are in `tests/text/test_speech_text.py`. The traps are:

- a bare domain needs a letters-only top-level domain, or "2.5 GB" would be read
  as a link;
- a single-letter unit needs a space, or "1990s" would become "seconds";
- money must not swallow the space after it;
- text that is mostly Chinese, Japanese or Korean keeps its digits.

## Tests

`tests/text/`: `test_speech_text.py`, `test_splitter.py`, `test_persona.py`,
`test_acks.py` and `test_rename.py`. Run them with `uv run pytest tests/text`.
They need no models and no network.

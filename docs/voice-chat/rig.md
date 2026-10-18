# The GPU rig (optional)

Voice chat runs entirely on the computer that hosts DisPatch. If you also have a machine
with an NVIDIA GPU running **SpeechForge** (a speech server that speaks the `sf.1` session
protocol), the add-on can hand it the heavy work: speech recognition, the end-of-turn score
and the reply voice. Replies then start sooner. This computer stays in charge of the
conversation (turn-taking, interruptions, acknowledgements, what was heard) and takes over
the speech work whenever the rig is slow, busy or away.

It is **off by default**, and it stays off until three things are true: the switch is on,
an address is set, and the token is present. Without them the add-on behaves exactly as it
does without a rig.

## Set it up

Everything below can be done in Settings → Voice (the **Speech to text** section; choose *Speech
rig (GPU)* as the provider, or open *Set up the speech rig*), without a terminal. The files and
variables are for servers that are configured as code.

1. **The token.** Paste it in Settings → Voice → *Rig token*. DisPatch keeps it in its own
   credential store (the OS keyring when there is one, else a private file in its data folder),
   so it survives an image update and works with a read-only container root. A DisPatch without
   a credential store gets a private file in the add-on's state folder instead
   (`<data>/ext/voice/speechforge.env`: mode 0600, or an owner-only ACL on Windows). It is
   write-only: no page, route or log line ever shows it. The 1.6.0 location
   (`~/.config/local-chat/speechforge.env`) is still read. On a server you configure as code:

   ```sh
   mkdir -p <data>/ext/voice
   (umask 077 && printf 'SPEECHFORGE_TOKEN=%s\n' '<the token from the rig>' > <data>/ext/voice/speechforge.env)
   ```

   `SPEECHFORGE_TOKEN` in DisPatch's environment works too and wins over the file (the page then
   says the environment provides it). A file that other users can read is ignored (the status
   says so; on Windows "readable by others" means the ACL grants Everyone, Users or
   Authenticated Users access). The token is read when each voice chat opens, so a rotated token
   needs no restart. It is sent only in the connection's `Authorization` header; it is never
   written to `voice.yaml`, never logged (the websockets library's own DEBUG output is redacted)
   and never shown in the app.

2. **The address**, in Settings → Voice → *Rig address*, in `<data>/voice.yaml`, or in
   `DISPATCH_VOICE_RIG_URL` (the environment wins and the page shows *Set by environment*):

   ```yaml
   rig:
     enabled: true
     url: ws://<rig>:1236/v1/realtime    # wss:// whenever it crosses a network you do not control
   ```

   **Which addresses are allowed.** The address receives the token and your live microphone, so
   it is checked when you save it and again when it is used: loopback, private-network, tailnet
   (100.64.0.0/10) and plain LAN names (`rig`, `rig.local`, `rig.lan`, a tailnet's MagicDNS names, ...) are accepted;
   link-local (169.254.x.x, fe80::) and cloud-metadata addresses are always refused; a **public**
   address or domain is refused unless the administrator lists it in `rig.allow` (or
   `DISPATCH_VOICE_RIG_ALLOW`, comma separated: a host, `*.example.org` or a CIDR). A plain
   `ws://` to anything but this machine is accepted with a warning: use `wss://` or keep the rig
   on a VPN or tailnet. A name that resolves to a forbidden address is not dialled.

   Optional tuning keys (defaults shown; the app keeps them in range). The two deadlines
   come from measurements on a 3090-class rig over a tailnet: final words 115 ms median,
   172 ms worst of 24 under continuous partials; first reply audio 200-275 ms for most
   sentences with outliers to about 520 ms. Moving a reply here costs the deadline plus
   this computer's own first chunk (well over a second on a small CPU), so the start
   deadline is generous on purpose.

   | Key | Default | What it bounds |
   |---|---|---|
   | `stt_final_deadline_ms` | 350 | how long the end of a turn waits for the rig's final words before this computer transcribes them |
   | `tts_start_deadline_ms` | 800 | how long a reply sentence waits for the rig's first audio before that reply moves to this computer |
   | `ahead_ms` | 3000 | how far the rig may render past what the phone has played |
   | `first_chunk_tokens` | 15 | the size of the rig's first audio chunk |
   | `allow` | (empty) | public hosts, `*.suffix`es and CIDRs the address may be (above) |

3. **Switch it on** in Settings → Voice → *Use the GPU rig when available* (the same
   `rig.enabled`). The row is greyed out with "Not set up on this computer" until the
   address and the token are there. Its status line says what is happening: used from the
   next voice chat, using the rig now, or that this computer is speaking because the rig is
   busy or not answering.

4. **Check it**: `dispatch-voice doctor` prints a `GPU rig:` line (set up or not, the
   token's source, and whether SpeechForge answers `/health` and `/ready`). The Voice lab
   shows the rig line (first audio p50, partial lag p90, the share of turns on the rig, the
   last fallback) and the rig's engines, card, limits and the deadlines in force.

## How it decides

* **One socket per voice chat.** It opens with the voice chat, beside the CPU engines'
  warm-up, and closes with it (`bye`). Until the rig has answered `ready` and a ping, every
  turn runs on this computer.
* **A turn only starts on a proven rig**: ready, a recent ping answered, its GPU lease
  held (or the rig serving *unleased*: SpeechForge loaded without a lease because its lease
  mode is off or StudioForge was away while the card was free; Settings and the Voice lab
  say so), its models loaded, no lease being given back, and no failure in the last 30
  seconds. Otherwise the turn runs exactly as it would without a rig: no deadline is
  waited for and nothing is sent to the rig.
* **A rig loading its models** (a cold start answers `state{models: "loading"}` before
  `ready`) leaves every turn here until it is ready.
* **A rig giving its GPU back** (`state{lease: "vacating"}`, before a benchmark or another
  job takes the card): the reply being spoken finishes on the rig, no new reply starts
  there, and the rig is not tried again before the deadline it announced. When it unloads,
  its `Retry-After` is honoured the same way.
* **A dead rig is noticed in under a second**: three missed pings (they go every 250 ms),
  a second with no message at all, the connection closing, a lost lease, or partial
  transcripts lagging more than 800 ms for a second while you speak.
* **Failover keeps every word.** The add-on keeps the open utterance's audio; when the rig
  goes away mid-sentence, that audio replays into the local recogniser, which continues
  live (the cost is one transcription of the whole utterance here, since the local
  recogniser was idle). A reply sentence that was being spoken re-renders on this computer
  from its start, so its opening words may be heard twice, and the rest of that reply
  stays here: one reply is always one voice engine.
* **A rig that answers with an error** for a reply (its engine failed, its models are
  gone, or it is busy twice in a row) is treated like a slow one: that reply is spoken
  here and the rig waits out the 30-second cooldown, so the next replies do not pay the
  same round trip.
* **Long replies**: the rig takes at most `limits.max_text` characters (2000 by default)
  per reply context; a longer reply continues in a fresh context with the same voice and
  seed, so it stays on the rig.
* **Coming back** happens between turns only, never within 30 seconds of a failure, and
  only after a fresh ping round trip. A connection that dropped is re-attached in the
  background (the rig keeps a dropped session for 10 seconds), but it still waits for the
  cooldown.
* **Refusals**: a refused token stops all retries until the address or the token changes;
  "busy" and "rate limited" answers are retried after the rig's `Retry-After`, between
  turns.
* **The power hold** (the performance profile) is not taken while the rig serves. It is
  taken when this computer has to do the work (a failover, a rig that is not ready yet, a
  turn that starts here) and given back when the rig is promoted again.

## Voices

The rig speaks a cloned voice from its **conditionals** (the `voice.npz` the Voice cloning
tool builds), never from the reference recording, which stays on this computer. At the
start of each voice chat the add-on checks whether the rig already has the bot's voice (and
the narrator voice) by its sha256 and uploads it once if not, over HTTP to the same server
with the same token. A bot with no voice of its own speaks what this computer would: the
default profile on the rig; with no default profile, a built-in plugin voice (here) or,
without a plugin, the stock voice (on the rig). Deleting a voice profile here deletes its
conditionals on the rig too (whenever an address and a token are set, even with the switch
off).

Built-in voices from a plugin (Kokoro, for example) stay on this computer: a bot that
speaks one gets the rig's speech recognition and end-of-turn score, but its replies are
spoken here. Acknowledgements and spoken notices also stay here (they are rendered ahead
of time and cached).

## Interruptions and what was heard

A barge-in cancels the rig's rendering (`cancel{gen}`) before the bot's turn is aborted.
A possible interruption pauses the rig's rendering with the reply, and a false alarm
resumes it. The phone's playback reports become the rig's credit, so it never renders
more than `ahead_ms` past what you heard. The rig holds the last reported position until
the next report, so the add-on re-sends it every 250 ms while a reply is open (the last
report run on in real time, never past the audio the phone has), and it never counts a
sentence as late while the rig, by that same rule, may be holding back on purpose. The rig sends word timings with its audio, so
when you cut a reply off the note of what you heard ends at the last whole word instead of
an estimate.

## Not done yet

* Speculative reply audio (rendering the first clause of a reply that may still be
  discarded) is not used: the session has no preemptive speech step to hand it.
* After a reconnect the add-on does not re-send audio or reopen speech: whatever was in
  flight had already moved to this computer.

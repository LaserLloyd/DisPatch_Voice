"""Voice profiles on disk.

    <root>/<id>/
        original.<ext>     the upload exactly as received (kept so a profile can be rebuilt)
        reference.wav      trimmed, best-window, loudness-normalised clip that was cloned
        voice.npz          the generation parameters (speech-encoder tensors)
        profile.json       metadata, quality report, consent record
        audition.wav       a fixed test line in the cloned voice
        fillers/<n>.wav    short acknowledgements in the cloned voice, played while the LLM thinks

Nothing here leaves the box, with two opt-in exceptions for a GPU rig: the
conditionals (``voice.npz``) when the rig is switched on, and ``reference.wav``
only when ``rig.send_reference_clip`` is also on and the rig's voice engine needs
it. Profiles must never be copied into a repo, a deploy allowlist or a public
backup.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from . import audio
from .engine import ChatterboxOnnx, VoiceTensors

ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")
CONSENT_KINDS = {
    "self": "This is my own voice.",
    "permission": "I have the speaker's explicit permission to clone this voice.",
    "synthetic": "This is a synthetic or stock voice, not a real person.",
}
UPLOAD_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".oga", ".opus", ".webm", ".flac", ".aac", ".mp4"}
# Deliberately unlike the recording script, so a clone that parrots its reference is audible.
AUDITION_LINE = ("Good morning! Okay, quick one before the traffic gets bad [chuckle]. "
                 "Did you know octopuses have three hearts?")
FILLERS = ("Mm-hmm.", "Right.", "Oh, nice.", "Okay.", "Hmm, good question.", "Let me think about that.")


class ProfileError(ValueError):
    """Bad request against the store. Safe to show to the user."""


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:32] or "voice"
    return f"{s}-{secrets.token_hex(3)}"


@dataclass
class Profile:
    id: str
    meta: dict

    def public(self) -> dict:
        m = dict(self.meta)
        m.pop("rig_voices", None)          # the GPU rig's bookkeeping, not the profile's
        m["id"] = self.id
        return m


class ProfileStore:
    def __init__(self, root: Path, engine: ChatterboxOnnx | None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        self.engine = engine
        self._lock = threading.Lock()
        self._voices: dict[str, VoiceTensors] = {}

    # ------------------------------------------------------------------ paths
    def _dir(self, pid: str) -> Path:
        if not ID_RE.match(pid or ""):
            raise ProfileError("Unknown voice.")
        d = (self.root / pid).resolve()
        if d.parent != self.root.resolve():
            raise ProfileError("Unknown voice.")
        return d

    def _require(self, pid: str) -> Path:
        d = self._dir(pid)
        if not (d / "profile.json").exists():
            raise KeyError(pid)
        return d

    def _write_meta(self, d: Path, meta: dict) -> None:
        tmp = d / "profile.json.tmp"
        tmp.write_text(json.dumps(meta, indent=1, sort_keys=True))
        os.replace(tmp, d / "profile.json")

    # ------------------------------------------------------------------ reads
    def list(self) -> list[dict]:
        out = []
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and ID_RE.match(d.name) and (d / "profile.json").exists():
                try:
                    out.append(Profile(d.name, json.loads((d / "profile.json").read_text())).public())
                except (OSError, json.JSONDecodeError):
                    continue
        return sorted(out, key=lambda m: m.get("created", 0), reverse=True)

    def get(self, pid: str) -> dict:
        d = self._require(pid)
        return Profile(pid, json.loads((d / "profile.json").read_text())).public()

    def file(self, pid: str, name: str) -> Path:
        d = self._require(pid)
        allowed = {"reference.wav", "audition.wav"} | {f"fillers/{i}.wav" for i in range(len(FILLERS))}
        if name not in allowed:
            raise ProfileError("Unknown file.")
        p = d / name
        if not p.exists():
            raise KeyError(name)
        return p

    def voice(self, pid: str) -> VoiceTensors:
        if pid not in self._voices:
            self._voices[pid] = VoiceTensors.load(self._require(pid) / "voice.npz")
        return self._voices[pid]

    # ----------------------------------------------------------------- writes
    def analyse(self, data: bytes) -> dict:
        """Dry run for the GUI: decode + quality report, nothing saved."""
        raw = audio.decode(data)
        return audio.analyse(raw).as_dict()

    def create(self, name: str, data: bytes, filename: str, consent: str, consent_note: str = "",
               allow_warnings: bool = True) -> dict:
        if self.engine is None:
            raise RuntimeError("voice engine not loaded")
        name = (name or "").strip()[:60]
        if not name:
            raise ProfileError("Give the voice a name.")
        if consent not in CONSENT_KINDS:
            raise ProfileError("Choose whose voice this is (consent is required).")
        ext = Path(filename or "").suffix.lower()
        if ext not in UPLOAD_EXTS:
            ext = ".bin"
        raw = audio.decode(data)
        quality = audio.analyse(raw)
        if not quality.ok:
            raise ProfileError(" ".join(c.message for c in quality.checks if c.level == "fail"))
        if not allow_warnings and any(c.level == "warn" for c in quality.checks):
            raise ProfileError("The clip has warnings; re-record or allow warnings.")
        ref = audio.prepare_reference(raw)
        t0 = time.perf_counter()
        voice = self.engine.build_voice(ref)
        build_s = time.perf_counter() - t0

        pid = slugify(name)
        d = self._dir(pid)
        tmp = self.root / f".{pid}.partial"
        shutil.rmtree(tmp, ignore_errors=True)
        (tmp / "fillers").mkdir(parents=True)
        try:
            (tmp / f"original{ext}").write_bytes(data)
            sf.write(tmp / "reference.wav", ref, audio.SAMPLE_RATE, subtype="PCM_16")
            voice.save(tmp / "voice.npz")
            t1 = time.perf_counter()
            aud = self.engine.synthesize(AUDITION_LINE, voice)
            audition_s = time.perf_counter() - t1
            sf.write(tmp / "audition.wav", aud, audio.SAMPLE_RATE, subtype="PCM_16")
            match = self.engine.similarity(ref, aud) if len(aud) > audio.SAMPLE_RATE else None
            meta = {
                "name": name,
                "created": int(time.time()),
                "engine": f"chatterbox-turbo-onnx/{self.engine.dtype}",
                "source": {"filename": Path(filename or "upload").name[:120], "bytes": len(data),
                           "sha256": hashlib.sha256(data).hexdigest()},
                "reference_s": round(len(ref) / audio.SAMPLE_RATE, 2),
                "quality": {**quality.as_dict(), "match": None if match is None else round(match, 3)},
                "consent": {"kind": consent, "statement": CONSENT_KINDS[consent],
                            "note": (consent_note or "").strip()[:300], "at": int(time.time())},
                "watermarked": self.engine.watermarking,
                "timings": {"build_s": round(build_s, 2), "audition_s": round(audition_s, 2)},
                "fillers": "pending",
                "default": False,
            }
            self._write_meta(tmp, meta)
            with self._lock:
                os.replace(tmp, d)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        self._voices[pid] = voice
        return self.get(pid)

    def render_fillers(self, pid: str) -> None:
        """Slow (a second or two each); run it in the background after create()."""
        d = self._require(pid)
        voice = self.voice(pid)
        for i, line in enumerate(FILLERS):
            wav = self.engine.synthesize(line, voice, seed=i)   # reproducible re-renders
            # tmp + rename: DisPatch may read the fillers while this runs, and
            # a half-written WAV played as a clipped acknowledgement.
            out = d / "fillers" / f"{i}.wav"
            tmp = out.with_name(f".{i}.wav.tmp")
            sf.write(tmp, wav, audio.SAMPLE_RATE, subtype="PCM_16", format="WAV")
            os.replace(tmp, out)
        with self._lock:
            meta = json.loads((d / "profile.json").read_text())
            meta["fillers"] = "ready"
            self._write_meta(d, meta)

    def update(self, pid: str, *, name: str | None = None, default: bool | None = None) -> dict:
        with self._lock:
            d = self._require(pid)
            meta = json.loads((d / "profile.json").read_text())
            if name is not None:
                if not name.strip():
                    raise ProfileError("Give the voice a name.")
                meta["name"] = name.strip()[:60]
            if default:
                for other in self.root.iterdir():
                    if other.is_dir() and other.name != pid and (other / "profile.json").exists():
                        om = json.loads((other / "profile.json").read_text())
                        if om.get("default"):
                            om["default"] = False
                            self._write_meta(other, om)
                meta["default"] = True
            elif default is False:
                meta["default"] = False
            self._write_meta(d, meta)
        return self.get(pid)

    def rig_voices(self, pid: str) -> dict:
        """``profile.json`` ``rig_voices``: ``{voice format: {"id", "transcript",
        "reference_sha256", ...}}``, the voices a GPU rig made from this profile's
        reference recording (only when the owner let the recording go there)."""
        d = self._require(pid)
        rv = json.loads((d / "profile.json").read_text()).get("rig_voices")
        return dict(rv) if isinstance(rv, dict) else {}

    def set_rig_voice(self, pid: str, fmt: str, entry: dict | None) -> None:
        """Record (or with None forget) the rig voice of one format."""
        with self._lock:
            d = self._require(pid)
            meta = json.loads((d / "profile.json").read_text())
            rv = meta.get("rig_voices") if isinstance(meta.get("rig_voices"), dict) else {}
            if entry is None:
                rv.pop(fmt, None)
            else:
                rv[fmt] = dict(entry)
            if rv:
                meta["rig_voices"] = rv
            else:
                meta.pop("rig_voices", None)
            self._write_meta(d, meta)

    def delete(self, pid: str) -> None:
        with self._lock:
            d = self._require(pid)
            shutil.rmtree(d)
        self._voices.pop(pid, None)

    def default_id(self) -> str | None:
        for m in self.list():
            if m.get("default"):
                return m["id"]
        return None

    def speak_line(self, pid: str, text: str) -> np.ndarray:
        if len(text) > 400:
            raise ProfileError("Keep test lines under 400 characters.")
        return self.engine.synthesize(text, self.voice(pid))

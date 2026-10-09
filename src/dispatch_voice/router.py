"""FastAPI router for the Voices panel.

The host app owns authentication: pass its gate in `dependencies` (DisPatch:
`[Depends(_require_full_access)]`, the unlocked tier). Nothing here is safe to
expose to a locked / Safe-Mode device, and the router refuses to build without
at least one dependency unless `allow_open=True` (tests, standalone dev on loopback).
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from importlib import resources

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.params import Depends as DependsParam
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.routing import APIRoute

from . import audio
from .profiles import CONSENT_KINDS, ProfileError, ProfileStore


def _origin_ok(request: Request) -> bool:
    """Same rule DisPatch uses on its WS routes: a browser Origin must match Host."""
    origin = request.headers.get("origin")
    if not origin:
        return True               # non-browser client (curl, the agent) — auth still applies
    host = request.headers.get("host", "")
    return origin.split("://", 1)[-1] == host


#: Room for the multipart boundaries and the small text fields (name, consent,
#: note) that travel with the file. The FILE itself is still held to
#: `max_upload` exactly, by read_upload().
FORM_OVERHEAD_BYTES = 64 * 1024


def check_body_length(request: Request, limit: int) -> None:
    """Refuse an upload from its headers, before a byte of the body is read.

    This cannot be an ordinary FastAPI dependency: FastAPI parses (and spools
    to disk) a multipart body BEFORE it resolves any dependency, so a
    dependency-level check would still let a client make the server write the
    whole body first. It runs from the route handler instead (see _capped_route).

    A body without a declared length (chunked transfer encoding) has no size to
    check up front, so it is refused rather than read to find out.
    """
    if "chunked" in request.headers.get("transfer-encoding", "").lower():
        raise HTTPException(411, "Uploads must declare their size (Content-Length).")
    raw = request.headers.get("content-length")
    if raw is None:
        raise HTTPException(411, "Uploads must declare their size (Content-Length).")
    try:
        length = int(raw)
    except ValueError as e:
        raise HTTPException(400, "Malformed Content-Length.") from e
    if length < 0:
        raise HTTPException(400, "Malformed Content-Length.")
    if length > limit:
        raise HTTPException(413, f"File is larger than {(limit - FORM_OVERHEAD_BYTES) // (1024 * 1024)} MB.")


def _capped_route(limit: int) -> type[APIRoute]:
    """An APIRoute that applies check_body_length to routes marked as uploads."""

    class CappedUploadRoute(APIRoute):
        def get_route_handler(self) -> Callable:
            handler = super().get_route_handler()
            if not getattr(self.endpoint, "_dispatch_voice_upload", False):
                return handler

            async def capped(request: Request):
                check_body_length(request, limit)
                return await handler(request)

            return capped

    return CappedUploadRoute


def _upload(fn):
    """Mark an endpoint as an upload route (body size checked from headers)."""
    fn._dispatch_voice_upload = True
    return fn


def build_router(store: ProfileStore, dependencies: Sequence[DependsParam] = (), *,
                 allow_open: bool = False, max_upload: int = audio.MAX_UPLOAD_BYTES) -> APIRouter:
    if not dependencies and not allow_open:
        raise ValueError("build_router needs an auth dependency (pass allow_open=True only for tests)")
    r = APIRouter(dependencies=list(dependencies),
                  route_class=_capped_route(max_upload + FORM_OVERHEAD_BYTES))

    def guard(request: Request) -> None:
        if not _origin_ok(request):
            raise HTTPException(403, "cross-origin request refused")

    async def read_upload(f: UploadFile) -> bytes:
        data = await f.read(max_upload + 1)
        if len(data) > max_upload:
            raise HTTPException(413, f"File is larger than {max_upload // (1024 * 1024)} MB.")
        return data

    def wav(a) -> Response:
        return Response(audio.to_wav_bytes(a), media_type="audio/wav",
                        headers={"Cache-Control": "no-store"})

    @r.get("/ui", response_class=HTMLResponse)
    def ui() -> HTMLResponse:
        html = resources.files("dispatch_voice").joinpath("static/voices.html").read_text()
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    @r.get("/ui/voices.js")
    def ui_js() -> Response:
        js = resources.files("dispatch_voice").joinpath("static/voices.js").read_text()
        return Response(js, media_type="text/javascript", headers={"Cache-Control": "no-store"})

    @r.get("/consent-kinds")
    def consent_kinds() -> dict:
        return CONSENT_KINDS

    @r.get("")
    def list_voices() -> dict:
        return {"voices": store.list(), "default": store.default_id(),
                "engine": store.engine.dtype if store.engine else None,
                "watermarking": bool(store.engine and store.engine.watermarking)}

    @r.post("/analyse")
    @_upload
    async def analyse(request: Request, file: UploadFile = File(...)) -> dict:
        guard(request)
        data = await read_upload(file)
        try:
            return await asyncio.to_thread(store.analyse, data)
        except audio.ClipError as e:
            raise HTTPException(422, str(e)) from e

    @r.post("")
    @_upload
    async def create(request: Request, background: BackgroundTasks, file: UploadFile = File(...),
                     name: str = Form(...), consent: str = Form(...), consent_note: str = Form("")) -> dict:
        guard(request)
        data = await read_upload(file)
        try:
            prof = await asyncio.to_thread(store.create, name, data, file.filename or "", consent, consent_note)
        except (audio.ClipError, ProfileError) as e:
            raise HTTPException(422, str(e)) from e
        background.add_task(store.render_fillers, prof["id"])
        return prof

    @r.get("/{pid}")
    def get_voice(pid: str) -> dict:
        try:
            return store.get(pid)
        except (KeyError, ProfileError) as e:
            raise HTTPException(404, "Unknown voice.") from e

    @r.patch("/{pid}")
    async def patch_voice(pid: str, request: Request) -> dict:
        guard(request)
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(422, "expected a JSON object")
        try:
            return store.update(pid, name=body.get("name"), default=body.get("default"))
        except KeyError as e:
            raise HTTPException(404, "Unknown voice.") from e
        except ProfileError as e:
            raise HTTPException(422, str(e)) from e

    @r.delete("/{pid}")
    def delete_voice(pid: str, request: Request) -> dict:
        guard(request)
        try:
            store.delete(pid)
        except (KeyError, ProfileError) as e:
            raise HTTPException(404, "Unknown voice.") from e
        return {"deleted": pid}

    @r.get("/{pid}/files/{name:path}")
    def voice_file(pid: str, name: str) -> FileResponse:
        try:
            return FileResponse(store.file(pid, name), media_type="audio/wav",
                                headers={"Cache-Control": "no-store"})
        except (KeyError, ProfileError) as e:
            raise HTTPException(404, "Unknown file.") from e

    @r.post("/{pid}/say")
    async def say(pid: str, request: Request) -> Response:
        """Audition any line in this voice (the GUI's 'try a line' box)."""
        guard(request)
        body = await request.json()
        text = (body or {}).get("text", "") if isinstance(body, dict) else ""
        if not text.strip():
            raise HTTPException(422, "Type something to say.")
        try:
            return wav(await asyncio.to_thread(store.speak_line, pid, text))
        except KeyError as e:
            raise HTTPException(404, "Unknown voice.") from e
        except ProfileError as e:
            raise HTTPException(422, str(e)) from e

    return r

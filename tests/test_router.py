import pytest
from conftest import speechlike, wav_bytes
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from dispatch_voice.router import build_router


def gate(request: Request):
    if request.headers.get("x-test-unlocked") != "1":
        raise HTTPException(403, "locked")


@pytest.fixture
def client(store):
    app = FastAPI()
    app.include_router(build_router(store, [Depends(gate)]), prefix="/api/voices")
    return TestClient(app, headers={"x-test-unlocked": "1"})


def upload(client, name="Host", consent="synthetic", seconds=14.0, **kw):
    return client.post("/api/voices", files={"file": ("h.wav", wav_bytes(speechlike(seconds)), "audio/wav")},
                       data={"name": name, "consent": consent}, **kw)


def test_router_requires_a_gate(store):
    with pytest.raises(ValueError):
        build_router(store)


def test_locked_device_refused_everywhere(client):
    locked = TestClient(client.app)
    for method, path in [("get", "/api/voices"), ("get", "/api/voices/ui"), ("post", "/api/voices/analyse"),
                         ("delete", "/api/voices/x-1"), ("get", "/api/voices/consent-kinds")]:
        assert getattr(locked, method)(path).status_code == 403, path


def test_full_flow(client):
    r = upload(client)
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    lst = client.get("/api/voices").json()
    assert [v["id"] for v in lst["voices"]] == [pid]
    assert client.get(f"/api/voices/{pid}").json()["fillers"] == "ready"   # BackgroundTasks ran
    assert client.get(f"/api/voices/{pid}/files/audition.wav").headers["content-type"] == "audio/wav"
    assert client.get(f"/api/voices/{pid}/files/fillers/0.wav").status_code == 200
    assert client.get(f"/api/voices/{pid}/files/voice.npz").status_code == 404
    say = client.post(f"/api/voices/{pid}/say", json={"text": "Hello there [chuckle]"})
    assert say.status_code == 200 and say.content[:4] == b"RIFF"
    assert client.patch(f"/api/voices/{pid}", json={"default": True}).json()["default"] is True
    assert client.get("/api/voices").json()["default"] == pid
    assert client.delete(f"/api/voices/{pid}").status_code == 200
    assert client.get(f"/api/voices/{pid}").status_code == 404


def test_analyse_reports_without_saving(client, store):
    r = client.post("/api/voices/analyse", files={"file": ("h.wav", wav_bytes(speechlike(3.0)), "audio/wav")})
    assert r.status_code == 200 and r.json()["ok"] is False
    assert list(store.root.iterdir()) == []


def test_bad_uploads(client):
    assert upload(client, seconds=2.0).status_code == 422
    assert upload(client, consent="nobody").status_code == 422
    r = client.post("/api/voices", files={"file": ("x.wav", b"garbage" * 100, "audio/wav")},
                    data={"name": "x", "consent": "self"})
    assert r.status_code == 422


def test_cross_origin_writes_refused(client):
    r = upload(client, headers={"origin": "https://evil.example", "x-test-unlocked": "1"})
    assert r.status_code == 403
    ok = upload(client, headers={"origin": "http://testserver", "x-test-unlocked": "1"})
    assert ok.status_code == 200


def test_upload_size_cap(store):
    app = FastAPI()
    app.include_router(build_router(store, [Depends(gate)], max_upload=1000), prefix="/v")
    c = TestClient(app, headers={"x-test-unlocked": "1"})
    r = c.post("/v", files={"file": ("h.wav", b"0" * 2000, "audio/wav")}, data={"name": "x", "consent": "self"})
    assert r.status_code == 413


def test_ui_served(client):
    html = client.get("/api/voices/ui").text
    assert "<title>Voices</title>" in html and 'src="ui/voices.js"' in html
    assert client.get("/api/voices/ui/voices.js").headers["content-type"].startswith("text/javascript")


def _capped_client(store, cap=1000):
    app = FastAPI()
    app.include_router(build_router(store, [Depends(gate)], max_upload=cap), prefix="/v")
    return TestClient(app, headers={"x-test-unlocked": "1"})


def test_oversized_upload_refused_from_headers_before_parsing(store, monkeypatch):
    """A declared Content-Length over the cap is a 413 before the form is parsed."""
    from starlette.requests import Request as StarletteRequest

    parsed = []
    real_form = StarletteRequest.form

    def spy(self, *a, **kw):
        parsed.append(self.url.path)
        return real_form(self, *a, **kw)

    monkeypatch.setattr(StarletteRequest, "form", spy)
    c = _capped_client(store)
    from dispatch_voice.router import FORM_OVERHEAD_BYTES
    body = b"0" * (1000 + FORM_OVERHEAD_BYTES + 1)
    for path in ("/v", "/v/analyse"):
        r = c.post(path, content=body, headers={"content-type": "multipart/form-data; boundary=x"})
        assert r.status_code == 413, (path, r.text)
    assert parsed == []


def test_chunked_upload_refused(store):
    c = _capped_client(store)

    def gen():
        yield b"--x\r\n"

    for path in ("/v", "/v/analyse"):
        r = c.post(path, content=gen(), headers={"content-type": "multipart/form-data; boundary=x"})
        assert r.status_code == 411, (path, r.status_code, r.text)


def test_malformed_content_length_refused(store):
    from starlette.requests import Request as StarletteRequest

    from dispatch_voice.router import check_body_length

    def req(headers):
        return StarletteRequest({"type": "http", "method": "POST", "path": "/",
                                 "headers": [(k.encode(), v.encode()) for k, v in headers.items()]})

    with pytest.raises(HTTPException) as e:
        check_body_length(req({"content-length": "abc"}), 10)
    assert e.value.status_code == 400
    with pytest.raises(HTTPException) as e:
        check_body_length(req({}), 10)
    assert e.value.status_code == 411
    with pytest.raises(HTTPException) as e:
        check_body_length(req({"content-length": "5", "transfer-encoding": "chunked"}), 10)
    assert e.value.status_code == 411
    check_body_length(req({"content-length": "10"}), 10)       # at the limit: fine


def test_cap_applies_only_to_upload_routes(client):
    """JSON routes (say/patch) are not upload routes and keep working."""
    r = client.post("/api/voices/nope/say", json={"text": "hi"})
    assert r.status_code == 404

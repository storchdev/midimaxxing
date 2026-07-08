import json
import threading
import urllib.request

import numpy as np

from keyboard_picker import run_picker_server


def _get(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


def _post(url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else b""
    req = urllib.request.Request(url, data=data, method="POST",
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def _run_server_in_thread(img, n_markers, **kwargs):
    result = {}

    def target():
        result["state"] = run_picker_server(img, n_markers, open_browser=False, **kwargs)

    t = threading.Thread(target=target)
    t.start()
    return t, result


def test_server_full_click_and_undo_flow_fixed_port():
    img = (np.random.rand(50, 100, 3) * 255).astype("uint8")
    port = 8731
    t, result = _run_server_in_thread(img, n_markers=2, port=port)

    base = f"http://127.0.0.1:{port}"
    try:
        state = _wait_for_state(base)
        assert state["frame_width"] == 100
        assert state["frame_height"] == 50
        assert state["corners"] == []

        s = _post(f"{base}/click", {"x": 10, "y": 20})
        assert s["corners"] == [[10, 20]]
        assert not s["done"]

        s = _post(f"{base}/click", {"x": 90, "y": 40})
        assert s["corners"] == [[10, 20], [90, 40]]
        assert s["markers"] == []

        s = _post(f"{base}/click", {"x": 30, "y": 0})
        assert s["markers"] == [30]
        assert not s["done"]

        s = _post(f"{base}/undo")
        assert s["markers"] == []
        assert not s["done"]

        s = _post(f"{base}/click", {"x": 30, "y": 0})
        s = _post(f"{base}/click", {"x": 70, "y": 0})
        assert s["done"]
    finally:
        t.join(timeout=5)
    assert not t.is_alive()
    assert result["state"].is_done()
    assert result["state"].markers == [30, 70]


def test_server_bbox_only_mode_zero_markers():
    img = (np.random.rand(20, 20, 3) * 255).astype("uint8")
    port = 8732
    t, result = _run_server_in_thread(img, n_markers=0, port=port)

    base = f"http://127.0.0.1:{port}"
    try:
        state = _wait_for_state(base)
        _post(f"{base}/click", {"x": 1, "y": 1})
        s = _post(f"{base}/click", {"x": 19, "y": 19})
        assert s["done"]
    finally:
        t.join(timeout=5)
    assert result["state"].bbox() == (1, 1, 19, 19)


def _wait_for_state(base, timeout=5.0):
    import time
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            return _get(f"{base}/state")
        except Exception as e:  # server not up yet
            last_err = e
            time.sleep(0.02)
    raise last_err

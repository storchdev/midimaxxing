"""Interactive keyboard-geometry picker: opens a local web GUI in the browser.

Prompts the user to click the two corners of the piano keyboard, plus the
left edge (x-only) of every C key in the given note range. Always displays
the frame "hands-bottom" for the annotator, regardless of the video's native
orientation, then inverse-transforms clicks back into the original video's
native pixel space before returning.

A tiny local HTTP server (stdlib only, no new dependency) serves the frame
and a single HTML page; clicks are posted back as JSON. This also sidesteps
needing an interactive matplotlib backend (TkAgg) or X11 -- only a browser is
required, so it works fine over SSH with a port-forward (`ssh -L 8000:localhost:PORT`).
Once every required point has been clicked, the last marker/corner stays
visible and a Submit button (or Enter) becomes active -- nothing is finalized
until the user explicitly submits, and `u` still undoes the last point even
after that button appears.

Usage:
    python keyboard_picker.py video.mp4 --hands top
    python keyboard_picker.py video.mp4 --hands bottom --lowest-note C2 --highest-note C6
    python keyboard_picker.py video.mp4 --hands top --bbox 100,200,900,300   # reduced: markers only
    python keyboard_picker.py video.mp4 --no-markers                         # bbox-only mode (used by vit.py)
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from keyboard_geometry import KeyboardGeometry, c_pitches

PathLike = str | Path


@dataclass
class PickerState:
    """Pure state machine for the click sequence, no HTTP/GUI code.

    Phases:
        "corners" -- collecting the two keyboard bbox corners (or skipped
                     entirely if a known bbox was supplied).
        "markers" -- collecting one x-click per C key, in ascending-pitch
                     (left-to-right on screen) order.
        "done"    -- all points collected.
    """
    n_markers: int
    corners: list[tuple[float, float]] = field(default_factory=list)
    markers: list[float] = field(default_factory=list)
    phase: Literal["corners", "markers", "done"] = "corners"

    def __post_init__(self):
        if self.phase == "markers" and self.n_markers == 0:
            self.phase = "done"

    def add_click(self, x: float, y: float) -> None:
        if self.phase == "done":
            return
        if self.phase == "corners":
            self.corners.append((x, y))
            if len(self.corners) == 2:
                self.phase = "markers"
                if self.n_markers == 0:
                    self.phase = "done"
        elif self.phase == "markers":
            self.markers.append(x)
            if len(self.markers) == self.n_markers:
                self.phase = "done"

    def undo(self) -> None:
        if self.phase == "markers" and self.markers:
            self.markers.pop()
        elif self.phase == "markers" and not self.markers:
            # Cross back into corners phase.
            self.phase = "corners"
            if self.corners:
                self.corners.pop()
        elif self.phase == "corners" and self.corners:
            self.corners.pop()
        elif self.phase == "done":
            self.phase = "markers"
            if self.markers:
                self.markers.pop()
            elif self.corners:
                self.phase = "corners"
                self.corners.pop()

    def is_done(self) -> bool:
        return self.phase == "done"

    def bbox(self) -> tuple[int, int, int, int]:
        (x1, y1), (x2, y2) = self.corners
        return (
            int(min(x1, x2)), int(min(y1, y2)),
            int(max(x1, x2)), int(max(y1, y2)),
        )

    def title(self) -> str:
        if self.phase == "corners":
            return f"Click TOP-LEFT then BOTTOM-RIGHT of the piano keys ({len(self.corners)}/2)  [u = undo]"
        if self.phase == "markers":
            return (f"Click the left edge of each C key, left to right "
                    f"({len(self.markers)}/{self.n_markers})  [u = undo]")
        return "All points collected -- press Submit to confirm, or u to undo the last point."


def _announce_url(url: str) -> str:
    """Format the "open this URL" message, bold+colored when stderr is a
    real terminal (plain text if redirected/piped/not a TTY)."""
    if not sys.stderr.isatty():
        return f"Keyboard picker: open {url} in your browser (or it should open automatically)"
    bold, cyan, underline, reset = "\033[1m", "\033[96m", "\033[4m", "\033[0m"
    return (
        f"\n{bold}{cyan}➤ Open {underline}{url}{reset}{bold}{cyan} in your browser{reset}\n"
        f"  (it should open automatically; press u to undo, Enter to submit once done)\n"
    )


def _decode_frame(video_path: PathLike):
    import av
    container = av.open(str(video_path))
    vs = container.streams.video[0]
    container.seek(vs.duration // 2, stream=vs)
    frame = next(container.decode(video=0))
    return frame.to_ndarray(format="rgb24")


def _encode_png(img) -> bytes:
    from matplotlib.image import imsave
    buf = io.BytesIO()
    imsave(buf, img, format="png")
    return buf.getvalue()


_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Keyboard picker</title>
<style>
  html, body { margin: 0; background: #111; color: #eee; font-family: system-ui, sans-serif; }
  #bar { padding: 10px 14px; font-size: 15px; }
  #title { margin: 0 0 10px 0; }
  #submit {
    display: block; font-size: 14px; padding: 6px 16px; border-radius: 6px; border: none;
    background: #32d74b; color: #062b0c; cursor: pointer;
  }
  #submit:disabled { background: #444; color: #888; cursor: not-allowed; }
  #wrap { position: relative; display: inline-block; }
  img { display: block; max-width: 100vw; cursor: crosshair; user-select: none; }
  svg { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
  circle { fill: #ff3b30; stroke: white; stroke-width: 1; }
  line { stroke: #32d74b; stroke-width: 2; }
</style>
</head>
<body>
<div id="bar">
  <div id="title">Loading...</div>
  <button id="submit" disabled>Submit</button>
</div>
<div id="wrap">
  <img id="frame" src="/frame.png">
  <svg id="overlay"></svg>
</div>
<script>
const img = document.getElementById('frame');
const svg = document.getElementById('overlay');
const titleEl = document.getElementById('title');
const submitBtn = document.getElementById('submit');
let frameW = 1, frameH = 1;
let submitted = false;

function svgNS(tag) { return document.createElementNS('http://www.w3.org/2000/svg', tag); }

function render(state) {
  frameW = state.frame_width;
  frameH = state.frame_height;
  titleEl.textContent = state.title;
  submitBtn.disabled = !state.done || submitted;
  svg.innerHTML = '';
  for (const [x, y] of state.corners) {
    const c = svgNS('circle');
    c.setAttribute('cx', (x / frameW * 100) + '%');
    c.setAttribute('cy', (y / frameH * 100) + '%');
    c.setAttribute('r', 6);
    svg.appendChild(c);
  }
  for (const x of state.markers) {
    const l = svgNS('line');
    l.setAttribute('x1', (x / frameW * 100) + '%');
    l.setAttribute('x2', (x / frameW * 100) + '%');
    l.setAttribute('y1', '0%');
    l.setAttribute('y2', '100%');
    svg.appendChild(l);
  }
}

async function refresh() {
  const r = await fetch('/state');
  render(await r.json());
}

img.addEventListener('click', async (e) => {
  if (submitted) return;
  const rect = img.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (img.naturalWidth / rect.width);
  const y = (e.clientY - rect.top) * (img.naturalHeight / rect.height);
  const r = await fetch('/click', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({x, y}),
  });
  render(await r.json());
});

document.addEventListener('keydown', async (e) => {
  if (submitted) return;
  if (e.key === 'u') {
    const r = await fetch('/undo', {method: 'POST'});
    render(await r.json());
  } else if (e.key === 'Enter' && !submitBtn.disabled) {
    submitBtn.click();
  }
});

submitBtn.addEventListener('click', async () => {
  if (submitBtn.disabled || submitted) return;
  const r = await fetch('/submit', {method: 'POST'});
  if (r.ok) {
    submitted = true;
    submitBtn.disabled = true;
    titleEl.textContent = 'Submitted! You can close this tab.';
  } else {
    refresh();
  }
});

refresh();
</script>
</body>
</html>
"""


def _make_handler(state: PickerState, png_bytes: bytes, frame_w: int, frame_h: int,
                   done_event: threading.Event):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass  # silence default access logging

        def _send(self, body: bytes, content_type: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, obj, code: int = 200):
            self._send(json.dumps(obj).encode(), "application/json", code)

        def _state_json(self):
            return {
                "corners": state.corners,
                "markers": state.markers,
                "done": state.is_done(),
                "title": state.title(),
                "frame_width": frame_w,
                "frame_height": frame_h,
            }

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._send(_PAGE.encode(), "text/html")
            elif path == "/frame.png":
                self._send(png_bytes, "image/png")
            elif path == "/state":
                self._send_json(self._state_json())
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b""
            with lock:
                if path == "/click":
                    data = json.loads(raw) if raw else {}
                    state.add_click(float(data["x"]), float(data["y"]))
                elif path == "/undo":
                    state.undo()
                elif path == "/submit":
                    if not state.is_done():
                        self._send_json({"error": "not all points collected yet"}, 400)
                        return
                    done_event.set()
                else:
                    self._send(b"not found", "text/plain", 404)
                    return
            self._send_json(self._state_json())

    return Handler


def run_picker_server(
    img,
    n_markers: int,
    known_bbox: tuple[int, int, int, int] | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
) -> PickerState:
    """Serve `img` in a local browser tab and block until the user submits.

    Once all required points are clicked, the page shows a Submit button
    (also triggerable with Enter); the server only shuts down and this
    returns once the user explicitly submits, so the final point stays
    visible and `u` can still undo it beforehand. If there is nothing to
    click at all (`known_bbox` given together with `n_markers == 0`),
    completes immediately without waiting. Returns the completed PickerState
    (in `img`'s pixel space).
    """
    h, w = img.shape[0], img.shape[1]
    png_bytes = _encode_png(img)

    state = PickerState(n_markers=n_markers)
    if known_bbox is not None:
        state.phase = "markers"
        if n_markers == 0:
            state.phase = "done"

    done_event = threading.Event()
    if state.is_done():
        done_event.set()

    handler_cls = _make_handler(state, png_bytes, w, h, done_event)
    httpd = HTTPServer((host, port), handler_cls)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    url = f"http://{host}:{httpd.server_address[1]}/"
    print(_announce_url(url), file=sys.stderr)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        done_event.wait()
    finally:
        httpd.shutdown()
        thread.join()

    return state


def _pick(
    video_path: PathLike,
    hands: str,
    n_markers: int,
    known_bbox: tuple[int, int, int, int] | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
) -> dict:
    img = _decode_frame(video_path)
    h, w = img.shape[0], img.shape[1]
    # Always display hands-bottom: if the native video is hands-top, rotate
    # 180 for viewing.
    display_img = img[::-1, ::-1] if hands == "top" else img

    state = run_picker_server(
        display_img, n_markers, known_bbox=known_bbox,
        host=host, port=port, open_browser=open_browser,
    )
    if not state.is_done():
        raise RuntimeError("Picker closed before all points were collected")

    display_bbox = tuple(known_bbox) if known_bbox is not None else state.bbox()

    # Inverse-transform from displayed coordinates back to native video coordinates.
    if hands == "top":
        x1, y1, x2, y2 = display_bbox
        nx1, ny1 = w - 1 - x1, h - 1 - y1
        nx2, ny2 = w - 1 - x2, h - 1 - y2
        native_bbox = (min(nx1, nx2), min(ny1, ny2), max(nx1, nx2), max(ny1, ny2))
        native_markers = [w - 1 - x for x in state.markers]
    else:
        native_bbox = display_bbox
        native_markers = list(state.markers)

    return {
        "bbox": [int(v) for v in native_bbox],
        "c_marker_xs": native_markers,
        "frame_width": w,
        "frame_height": h,
    }


def pick_keyboard_geometry(
    video_path: PathLike,
    hands: Literal["top", "bottom"],
    lowest_note: str = "A0",
    highest_note: str = "C8",
    known_bbox: tuple[int, int, int, int] | None = None,
) -> KeyboardGeometry:
    n_markers = len(c_pitches(lowest_note, highest_note))
    raw = _pick(video_path, hands, n_markers, known_bbox=known_bbox)
    return KeyboardGeometry(
        lowest_note=lowest_note,
        highest_note=highest_note,
        bbox=tuple(raw["bbox"]),
        c_marker_xs=raw["c_marker_xs"],
        hands=hands,
        frame_width=raw["frame_width"],
        frame_height=raw["frame_height"],
    )


def pick_bbox_only(video_path: PathLike) -> tuple[int, int, int, int]:
    """Bbox-only mode used by vit.py (no C markers, no --hands orientation
    logic since the corners are used exactly as clicked, native video space).
    """
    raw = _pick(video_path, hands="bottom", n_markers=0, known_bbox=None)
    return tuple(raw["bbox"])


def geometry_to_dict(geom: KeyboardGeometry) -> dict:
    return {
        "lowest_note": geom.lowest_note,
        "highest_note": geom.highest_note,
        "bbox": list(geom.bbox),
        "c_marker_xs": geom.c_marker_xs,
        "hands": geom.hands,
        "frame_width": geom.frame_width,
        "frame_height": geom.frame_height,
    }


def geometry_from_dict(d: dict) -> KeyboardGeometry:
    return KeyboardGeometry(
        lowest_note=d["lowest_note"],
        highest_note=d["highest_note"],
        bbox=tuple(d["bbox"]),
        c_marker_xs=list(d["c_marker_xs"]),
        hands=d["hands"],
        frame_width=d["frame_width"],
        frame_height=d["frame_height"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--hands", choices=["top", "bottom"], default="bottom")
    parser.add_argument("--lowest-note", default="A0")
    parser.add_argument("--highest-note", default="C8")
    parser.add_argument("--bbox", default=None, help="x1,y1,x2,y2 -- reduced mode, only C-markers get collected")
    parser.add_argument("--no-markers", action="store_true", help="bbox-only mode, used by vit.py")
    parser.add_argument("--host", default="127.0.0.1",
                         help="Interface to bind the picker's local web server to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=0, help="Port to bind to (default: pick a free one)")
    parser.add_argument("--no-browser", action="store_true",
                         help="Don't auto-open a browser tab; just print the URL")
    args = parser.parse_args()

    if args.no_markers:
        raw = _pick(args.video, hands="bottom", n_markers=0, known_bbox=None,
                    host=args.host, port=args.port, open_browser=not args.no_browser)
        print(json.dumps({"bbox": raw["bbox"]}))
        return

    known_bbox = None
    if args.bbox:
        known_bbox = tuple(int(v) for v in args.bbox.split(","))

    n_markers = len(c_pitches(args.lowest_note, args.highest_note))
    raw = _pick(args.video, hands=args.hands, n_markers=n_markers, known_bbox=known_bbox,
                host=args.host, port=args.port, open_browser=not args.no_browser)
    geom = KeyboardGeometry(
        lowest_note=args.lowest_note,
        highest_note=args.highest_note,
        bbox=tuple(raw["bbox"]),
        c_marker_xs=raw["c_marker_xs"],
        hands=args.hands,
        frame_width=raw["frame_width"],
        frame_height=raw["frame_height"],
    )
    print(json.dumps(geometry_to_dict(geom)))


if __name__ == "__main__":
    main()

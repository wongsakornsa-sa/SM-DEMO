"""Camera adapters for local devices, network streams, and JPEG snapshot URLs."""

from __future__ import annotations

import time
import threading
from urllib.request import Request, urlopen

import numpy as np


SOURCE_TYPES = ("USB / virtual camera", "Network stream", "HTTP snapshot", "Video file")
BACKENDS = ("Auto", "DirectShow", "Media Foundation", "FFmpeg")


def backend_id(cv2, name):
    return {
        "Auto": cv2.CAP_ANY,
        "DirectShow": cv2.CAP_DSHOW,
        "Media Foundation": cv2.CAP_MSMF,
        "FFmpeg": cv2.CAP_FFMPEG,
    }[name]


class CameraSource:
    def __init__(self, cv2, kind: str, value: str, backend: str = "Auto"):
        self.cv2 = cv2
        self.kind = kind
        self.value = value.strip()
        self.cap = None
        self.opened = False
        self.last_frame = None
        self.last_read = 0.0
        if kind == "HTTP snapshot":
            if not self.value.lower().startswith(("http://", "https://")):
                raise ValueError("HTTP snapshot requires an http:// or https:// URL")
            ok, _ = self._read_frame()
            self.opened = ok
            if not ok:
                raise RuntimeError("Cannot download or decode a JPEG snapshot")
        else:
            if kind == "USB / virtual camera":
                if not self.value.isdecimal():
                    raise ValueError("Camera device must be a number, such as 0, 1, or 2")
                source = int(self.value)
            else:
                source = self.value
            self.cap = cv2.VideoCapture(source, backend_id(cv2, backend))
            self.opened = self.cap.isOpened()
            if not self.opened:
                self.cap.release()
                raise RuntimeError(f"Cannot open {kind}: {self.value} using {backend}")

        self._stop = threading.Event()
        self._latest = None
        self._received_at = 0.0
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

    def _reader_loop(self):
        try:
            while not self._stop.is_set():
                ok, frame = self._read_frame()
                if ok:
                    self._latest = (time.monotonic(), frame)
                self._stop.wait(0.03 if ok else 0.1)
        finally:
            if self.cap is not None:
                self.cap.release()

    def isOpened(self):
        return self.opened

    def read(self):
        latest = self._latest
        if latest is None or time.monotonic() - latest[0] > 2:
            return False, None
        return True, latest[1]

    def _read_frame(self):
        if self.cap is not None:
            return self.cap.read()
        if self.last_frame is not None and time.monotonic() - self.last_read < 0.3:
            return True, self.last_frame
        try:
            req = Request(self.value, headers={"User-Agent": "Object360Demo/1.0"})
            with urlopen(req, timeout=5) as response:
                body = response.read(12_000_000)
            frame = self.cv2.imdecode(np.frombuffer(body, dtype=np.uint8), self.cv2.IMREAD_COLOR)
            if frame is None:
                return False, None
            self.last_frame = frame
            self.last_read = time.monotonic()
            return True, frame
        except Exception:
            return False, None

    def release(self):
        self.opened = False
        self._stop.set()


def probe_devices(cv2, limit=6):
    found = []
    for index in range(limit):
        for backend in ("DirectShow", "Media Foundation", "Auto"):
            cap = cv2.VideoCapture(index, backend_id(cv2, backend))
            try:
                if cap.isOpened():
                    ok, _ = cap.read()
                    if ok:
                        found.append((index, backend))
                        break
            finally:
                cap.release()
    return found

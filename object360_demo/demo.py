"""Eight-view object detection and tracking demo.

Press C once the manually aimed camera is steady at each view. The same eight
frames are processed immediately and then replayed after capture for comparison.
"""

from __future__ import annotations

import argparse
import csv
import importlib.resources
import json
import sys
import time
import types
from datetime import datetime
from pathlib import Path


TARGET_NAMES = ("bottle", "cup", "chair")
VIEW_COUNT = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="0", help="USB/virtual-camera index or Wi-Fi/RTSP stream URL")
    p.add_argument("--input-dir", type=Path, help="Use eight saved images instead of a camera")
    p.add_argument("--weights", default="yolo11n.pt", help="Local YOLO weights, or a model name Ultralytics can download")
    p.add_argument("--confidence", type=float, default=0.35)
    p.add_argument("--output", type=Path, default=Path("runs"))
    p.add_argument("--classes", nargs="+", default=list(TARGET_NAMES), help="Model class names to detect")
    return p.parse_args()


def require_packages():
    try:
        import cv2
        from ultralytics import YOLO
        from deep_sort_realtime.deepsort_tracker import DeepSort
    except ImportError as exc:
        raise SystemExit(
            f"Missing package: {exc}. Install with: python -m pip install -r requirements.txt"
        ) from exc
    return cv2, YOLO, DeepSort


def source_value(source: str):
    return int(source) if source.isdecimal() else source


def new_tracker(DeepSort):
    # deep-sort-realtime 1.3.2 still imports pkg_resources for its packaged
    # MobileNet weights. setuptools 82+ no longer provides that module.
    try:
        import pkg_resources  # noqa: F401
    except ModuleNotFoundError:
        shim = types.ModuleType("pkg_resources")
        shim.resource_filename = lambda package, name: str(
            importlib.resources.files(package).joinpath(name)
        )
        sys.modules["pkg_resources"] = shim
    # The packaged MobileNet embedder supplies appearance features to Deep SORT.
    return DeepSort(max_age=8, n_init=1, embedder="mobilenet", embedder_gpu=False, half=False)


def load_model(YOLO, weights: str, requested: list[str]):
    model = YOLO(weights)
    names = model.names
    name_to_id = {str(name).lower(): int(idx) for idx, name in names.items()}
    missing = [name for name in requested if name.lower() not in name_to_id]
    if missing:
        raise SystemExit(f"Model does not contain classes {missing}. Available: {list(name_to_id)}")
    return model, {name_to_id[name.lower()] for name in requested}


def process_frame(frame, view_no, mode, model, class_ids, tracker, cv2, confidence):
    started = time.perf_counter()
    result = model.predict(frame, conf=confidence, classes=sorted(class_ids), verbose=False)[0]
    detections = []
    yolo_rows = []
    for box in result.boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].cpu().tolist())
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = str(result.names[cls_id])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))
        yolo_rows.append({"view": view_no, "mode": mode, "class": label,
                          "confidence": round(conf, 4), "x1": round(x1), "y1": round(y1),
                          "x2": round(x2), "y2": round(y2)})

    tracks = tracker.update_tracks(detections, frame=frame)
    annotated = frame.copy()
    track_rows = []
    for track in tracks:
        if track.time_since_update > 0:
            continue
        bounds = track.to_ltrb(orig=True)
        if bounds is None:
            continue
        x1, y1, x2, y2 = (round(float(v)) for v in bounds)
        label = str(track.get_det_class() or "object")
        track_id = str(track.track_id)
        state = "confirmed" if track.is_confirmed() else "tentative"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (25, 210, 50), 2)
        cv2.putText(annotated, f"{label} ID {track_id}{'?' if state == 'tentative' else ''}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (25, 210, 50), 2)
        track_rows.append({"view": view_no, "mode": mode, "class": label, "track_id": track_id, "state": state,
                           "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    seconds = time.perf_counter() - started
    return annotated, yolo_rows, track_rows, seconds


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_camera(cv2, source: str, output: Path, on_capture):
    cap = cv2.VideoCapture(source_value(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera/stream: {source}")
    frames = []
    first_capture = None
    capture_end = None
    try:
        while len(frames) < VIEW_COUNT:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera stopped returning frames")
            display = frame.copy()
            cv2.putText(display, f"View {len(frames)+1}/8: aim, then press C | Q quit",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.imshow("360 object demo", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                raise KeyboardInterrupt("Capture cancelled")
            if key == ord("c"):
                if first_capture is None:
                    first_capture = time.perf_counter()
                view = len(frames) + 1
                saved = frame.copy()
                cv2.imwrite(str(output / "raw" / f"view_{view:02d}.jpg"), saved)
                frames.append(saved)
                on_capture(saved, view)
                capture_end = time.perf_counter()
                print(f"Captured view {view}/{VIEW_COUNT}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return frames, (capture_end - first_capture) if first_capture is not None else 0.0


def load_images(cv2, folder: Path, output: Path, on_capture):
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(paths) != VIEW_COUNT:
        raise ValueError(f"Expected exactly 8 images in {folder}, found {len(paths)}")
    frames = []
    started = time.perf_counter()
    for view, path in enumerate(paths, 1):
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Cannot read image: {path}")
        cv2.imwrite(str(output / "raw" / f"view_{view:02d}.jpg"), frame)
        frames.append(frame)
        on_capture(frame, view)
    return frames, time.perf_counter() - started


def main() -> int:
    args = parse_args()
    cv2, YOLO, DeepSort = require_packages()
    if not 0 < args.confidence <= 1:
        raise SystemExit("--confidence must be in (0, 1]")
    run_dir = args.output / datetime.now().strftime("demo_%Y%m%d_%H%M%S")
    for sub in ("raw", "online", "batch"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    try:
        model, class_ids = load_model(YOLO, args.weights, args.classes)
        tracker_online = new_tracker(DeepSort)
        results = {"online": {"detections": [], "tracks": [], "seconds": []},
                   "batch": {"detections": [], "tracks": [], "seconds": []}}

        def process_and_save(frame, view, mode, tracker):
            annotated, det, tracks, seconds = process_frame(
                frame, view, mode, model, class_ids, tracker, cv2, args.confidence
            )
            results[mode]["detections"].extend(det)
            results[mode]["tracks"].extend(tracks)
            results[mode]["seconds"].append(seconds)
            cv2.imwrite(str(run_dir / mode / f"view_{view:02d}.jpg"), annotated)
            print(f"{mode} view {view}: {len(det)} detections, {len(tracks)} tracks, {seconds:.3f}s")

        def on_capture(frame, view):
            process_and_save(frame, view, "online", tracker_online)

        if args.input_dir:
            frames, capture_elapsed = load_images(cv2, args.input_dir, run_dir, on_capture)
        else:
            frames, capture_elapsed = collect_camera(cv2, args.source, run_dir, on_capture)

        tracker_batch = new_tracker(DeepSort)
        for view, frame in enumerate(frames, 1):
            process_and_save(frame, view, "batch", tracker_batch)

        for mode in ("online", "batch"):
            item = results[mode]
            write_csv(run_dir / f"{mode}_detections.csv", item["detections"],
                      ["view", "mode", "class", "confidence", "x1", "y1", "x2", "y2"])
            write_csv(run_dir / f"{mode}_tracks.csv", item["tracks"],
                      ["view", "mode", "class", "track_id", "state", "x1", "y1", "x2", "y2"])
        online_sec = sum(results["online"]["seconds"])
        batch_sec = sum(results["batch"]["seconds"])
        # Both modes share the same manual capture duration. Batch total is estimated.
        manual_capture_sec = max(0.0, capture_elapsed - online_sec)
        summary = {
            "views": VIEW_COUNT, "classes": args.classes, "source": str(args.input_dir or args.source),
            "capture_phase_seconds": round(capture_elapsed, 3),
            "estimated_manual_capture_seconds": round(manual_capture_sec, 3),
            "online_processing_seconds": round(online_sec, 3),
            "batch_processing_seconds": round(batch_sec, 3),
            "online_total_seconds": round(capture_elapsed, 3),
            "estimated_batch_total_seconds": round(manual_capture_sec + batch_sec, 3),
            "timing_note": "Batch total is estimated using the same capture phase minus online processing. "
                           "Image-folder mode measures file loading instead of manual capture.",
            "detections_online": len(results["online"]["detections"]),
            "detections_batch": len(results["batch"]["detections"]),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Saved results to {run_dir.resolve()}")
        return 0
    except (RuntimeError, ValueError, KeyboardInterrupt) as exc:
        print(f"Demo stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
